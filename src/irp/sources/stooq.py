import csv
import logging
import shutil
import zipfile
from collections.abc import Iterator
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Literal

import duckdb
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

from irp.core.config import config

logger = logging.getLogger(__name__)

root_dir = config.data.root_dir
stooq_cfg = config.providers.stooq
raw_dir = root_dir / stooq_cfg.raw_dir
processed_dir = root_dir / stooq_cfg.processed_dir


# Todo: script to setup dirs, or create dirs on demand


def _is_fresh(marker: Path, *inputs: Path) -> bool:
    """True if marker exists and is newer than all inputs."""
    if not marker.exists():
        return False
    if not all(inp.exists() for inp in inputs):
        return False
    marker_mtime = marker.stat().st_mtime
    return all(inp.stat().st_mtime <= marker_mtime for inp in inputs)


def _ensure_files_available(
    files: str | Iterable[str], *, error_message: str, download_instruction: str
) -> None:
    if isinstance(files, str):
        files = [files]

    if all((raw_dir / file).is_file() for file in files):
        return

    logger.error(error_message)
    logger.info(download_instruction)
    raise FileNotFoundError(error_message)


def _unzip_bulk_files() -> None:
    """Unzips the bulk files in the raw data directory. Uses marker files to avoid re-unzipping files that have already been extracted."""
    logger.debug('Unzipping bulk files...')
    for fname in stooq_cfg.bulk_files:
        with zipfile.ZipFile(raw_dir / fname) as zf:
            zf.extractall(raw_dir)


def _iter_file_dirs(path: Path) -> Iterator[Path]:
    """Yield dirs that contain ticker files.
    Assumes file-containing dirs are terminal leaves (no mixed file+subdir dirs)."""
    entries = list(path.iterdir())
    if any(p.is_file() for p in entries):
        yield path
    else:
        for child in entries:
            if child.is_dir():
                yield from _iter_file_dirs(child)


def _category_of(leaf: Path, data_root: Path) -> Path:
    """Return the category dir (region/category structure: parts[1] below data_root)."""
    rel = leaf.relative_to(data_root)
    return data_root / rel.parts[1]


def _build_price_dataset() -> None:
    """Read extracted ticker CSVs, write bulk_prices.parquet + markets.csv, delete data/."""
    logger.debug('Building price dataset...')
    parquet_file = raw_dir / 'bulk_prices.parquet'
    data_root = raw_dir / 'data'
    market_rows: list[dict[str, str]] = []
    pq_writer: pq.ParquetWriter | None = None
    arrow_schema: pa.Schema | None = None
    ticker_count = 0
    try:
        # raw/data/freq/region/category/[sub]/ticker.txt
        for freq_dir in data_root.iterdir():
            if not freq_dir.is_dir():
                continue
            for leaf in _iter_file_dirs(freq_dir):
                market = _category_of(leaf, freq_dir).name
                for f in leaf.iterdir():
                    if not f.is_file() or f.stat().st_size == 0:
                        continue
                    df = pd.read_csv(f)
                    df['ticker'] = f.stem
                    table = pa.Table.from_pandas(df, preserve_index=False)
                    if pq_writer is None:
                        arrow_schema = table.schema
                        pq_writer = pq.ParquetWriter(parquet_file, arrow_schema)
                    pq_writer.write_table(table.cast(arrow_schema))
                    market_rows.append({'ticker': f.stem, 'market': market})
                    ticker_count += 1
    finally:
        if pq_writer:
            pq_writer.close()
    logger.debug(f'Flattened {ticker_count} tickers to {parquet_file}.')

    csv_path = raw_dir / 'markets.csv'
    with open(csv_path, 'w', newline='') as fh:
        writer = csv.DictWriter(fh, fieldnames=['ticker', 'market'])
        writer.writeheader()
        writer.writerows(market_rows)
    # logger.debug(f'Deleting {data_root}...')
    # shutil.rmtree(data_root)


@dataclass(frozen=True)
class FeedSpec:
    input_path: Path
    output_path: Path
    input_format: Literal['csv', 'parquet']
    output_format: Literal['csv', 'parquet']
    header: bool = False


FEED_SPECS: dict[str, FeedSpec] = {
    'bulk': FeedSpec(
        input_path=raw_dir / 'bulk_prices.parquet',
        output_path=processed_dir / 'bulk_prices.parquet',
        input_format='parquet',
        output_format='parquet',
    ),
    'update': FeedSpec(
        input_path=raw_dir / stooq_cfg.update_file,
        output_path=processed_dir / 'update_prices.csv',
        input_format='csv',
        output_format='csv',
    ),
}


class StooqSource:
    def fetch_bulk(self) -> None:
        """Fetches Stooq price data by unzipping bulk files and building a price dataset."""
        marker = raw_dir / '.fetched'
        zip_paths = [raw_dir / f for f in stooq_cfg.bulk_files]
        if _is_fresh(marker, *zip_paths):
            logger.info('fetch: already up to date, skipping')
            return
        logger.debug('Fetching Stooq price data...')
        _ensure_files_available(
            stooq_cfg.bulk_files,
            error_message='Bulk files not available.',
            download_instruction=stooq_cfg.bulk_download_instruction,
        )
        _unzip_bulk_files()
        _build_price_dataset()
        marker.touch()
        logger.debug('Stooq price data fetched successfully.')

    def update(self) -> None:
        _ensure_files_available(
            stooq_cfg.update_file,
            error_message='Update file not available.',
            download_instruction=stooq_cfg.update_download_instruction,
        )

    def transform(self, feed: Literal['bulk', 'update']) -> None:
        """Transforms the specified feed's raw data into a clean format and stores it in the processed directory."""
        spec = FEED_SPECS[feed]
        marker = raw_dir / f'.transformed_{feed}'
        upstream = raw_dir / '.fetched' if feed == 'bulk' else spec.input_path
        if _is_fresh(marker, upstream):
            logger.info(f'transform({feed}): already up to date, skipping')
            return
        logger.debug(f'Transforming Stooq {feed} data...')
        conn = duckdb.connect()

        if spec.input_format == 'parquet':
            source = f"read_parquet('{spec.input_path}')"
        else:
            source = f"read_csv_auto('{spec.input_path}')"

        target_format = 'PARQUET' if spec.output_format == 'parquet' else 'CSV, HEADER'

        options_sql = f'FORMAT {target_format}'

        conn.sql(f"""
            COPY (
                SELECT
                    split_part(lower(t."<TICKER>"), '.', 1) AS ticker,
                    lower(t."<TICKER>") AS src_ticker,
                    t."<PER>" AS per,
                    t."<DATE>" AS date,
                    t."<TIME>" AS time,
                    t."<OPEN>" AS open,
                    t."<HIGH>" AS high,
                    t."<LOW>" AS low,
                    t."<CLOSE>" AS close,
                    t."<VOL>" AS vol,
                    t."<OPENINT>" AS openint
                FROM {source} t
            )
            TO '{spec.output_path}'
                ({options_sql});
        """)

        logger.debug(f'Stooq {feed} data transformed successfully.')

        if feed == 'bulk':
            logger.debug('Transforming market metadata...')

            src_markets = raw_dir / 'markets.csv'
            dst_markets = processed_dir / 'markets.csv'
            conn.sql(f"""
                COPY (
                    SELECT
                        split_part(lower("ticker"), '.', 1) AS ticker,
                        ticker AS src_ticker,
                        market
                    FROM read_csv_auto('{src_markets}')
                )
                TO '{dst_markets}'
                (FORMAT CSV, HEADER);
            """)

            logger.debug('Market metadata transformed successfully.')

        marker.touch()

    def store(self, feed: Literal['bulk', 'update']) -> None:
        marker = raw_dir / f'.stored_{feed}'
        upstream = raw_dir / f'.transformed_{feed}'
        if _is_fresh(marker, upstream):
            logger.info(f'store({feed}): already up to date, skipping')
            return
        logger.debug('Storing %s data...', feed)
        spec = FEED_SPECS[feed]
        source_reader = (
            'read_parquet' if spec.output_format == 'parquet' else 'read_csv_auto'
        )

        prices_file = spec.output_path

        key_cols = [
            'src_ticker',
            'date',
        ]

        update_cols = ['per', 'time', 'open', 'high', 'low', 'close', 'vol', 'openint']

        insert_cols = [
            'ticker',
            'src_ticker',
            'per',
            'date',
            'time',
            'open',
            'high',
            'low',
            'close',
            'vol',
            'openint',
        ]

        on_clause = '\nAND '.join(f't.{col} = s.{col}' for col in key_cols)

        update_set_clause = ',\n'.join(f'{col} = s.{col}' for col in update_cols)

        insert_columns_clause = ', '.join(insert_cols)

        insert_values_clause = ', '.join(f's.{col}' for col in insert_cols)

        with duckdb.connect(config.database.path) as con:
            con.execute(f"""
                CREATE TABLE IF NOT EXISTS prices AS
                SELECT *
                FROM {source_reader}('{prices_file}')
                LIMIT 0
            """)

            con.execute(f"""
                MERGE INTO prices t
                USING (
                    SELECT *
                    FROM {source_reader}('{prices_file}')
                ) s
                ON {on_clause}

                WHEN MATCHED THEN UPDATE SET
                    {update_set_clause}

                WHEN NOT MATCHED THEN INSERT (
                    {insert_columns_clause}
                )
                VALUES (
                    {insert_values_clause}
                );
            """)

            if feed == 'bulk':
                markets_file = processed_dir / 'markets.csv'
                con.execute(f"""
                    CREATE TABLE IF NOT EXISTS markets AS
                    SELECT *
                    FROM read_csv_auto('{markets_file}')
                    LIMIT 0
                """)

                con.execute(f"""
                    MERGE INTO markets t
                    USING (
                        SELECT *
                        FROM read_csv_auto('{markets_file}')
                    ) s
                    ON t.src_ticker = s.src_ticker

                    WHEN MATCHED THEN UPDATE SET
                        market = s.market

                    WHEN NOT MATCHED THEN INSERT (
                        ticker,
                        src_ticker,
                        market
                    )
                    VALUES (
                        s.ticker,
                        s.src_ticker,
                        s.market
                    );
                """)

        marker.touch()
        logger.debug('Stooq %s data stored successfully.', feed)

    def cleanup(self) -> None:
        """Delete all intermediate files, keeping zips, markers, and the database."""
        targets = [
            raw_dir / 'bulk_prices.parquet',
            raw_dir / 'markets.csv',
            processed_dir / 'bulk_prices.parquet',
            processed_dir / 'markets.csv',
            processed_dir / 'update_prices.csv',
        ]
        for path in targets:
            if path.exists():
                path.unlink()
                logger.debug(f'Deleted {path}')
        data_dir = raw_dir / 'data'
        if data_dir.exists():
            shutil.rmtree(data_dir)
            logger.debug(f'Deleted {data_dir}')


def main():
    provider = StooqSource()
    provider.fetch_bulk()
    provider.update()
    provider.transform('bulk')
    provider.transform('update')
    provider.store('bulk')
    provider.store('update')


if __name__ == '__main__':
    main()

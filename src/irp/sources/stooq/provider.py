import csv
import logging
import shutil
from typing import Iterable, Literal
import zipfile
from pathlib import Path
from collections.abc import Iterator

import duckdb
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

from irp.core.config import config
from irp.core.logging import configure

configure()

root_dir = config.data.root_dir
stooq_cfg = config.providers.stooq
raw_dir = root_dir / stooq_cfg.raw_dir
processed_dir = root_dir / stooq_cfg.processed_dir


# Todo: script to setup dirs


def _ensure_files_available(
    files: str | Iterable[str], *, error_message: str, download_instruction: str
) -> None:
    if isinstance(files, str):
        files = [files]

    if all((raw_dir / file).is_file() for file in files):
        return

    logging.error(error_message)
    print(download_instruction)
    raise FileNotFoundError(error_message)


def _unzip_bulk_files() -> None:
    """Unzips the bulk files in the raw data directory. Uses marker files to avoid re-unzipping files that have already been extracted."""
    logging.debug('Unzipping bulk files...')
    for fname in stooq_cfg.bulk_files:
        zip_path = raw_dir / fname
        marker = raw_dir / f'.extracted_{zip_path.stem}'
        if marker.exists() and marker.stat().st_mtime >= zip_path.stat().st_mtime:
            logging.debug(f'{fname} already extracted, skipping.')
            continue
        with zipfile.ZipFile(zip_path) as zf:
            zf.extractall(raw_dir)
        marker.touch()


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
    logging.debug('Building price dataset...')
    parquet_path = raw_dir / 'bulk_prices.parquet'
    market_rows: list[dict[str, str]] = []
    pq_writer: pq.ParquetWriter | None = None
    arrow_schema: pa.Schema | None = None
    ticker_count = 0
    data_root = raw_dir / 'data'
    try:
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
                        pq_writer = pq.ParquetWriter(parquet_path, arrow_schema)
                    pq_writer.write_table(table.cast(arrow_schema))
                    market_rows.append({'ticker': f.stem, 'market': market})
                    ticker_count += 1
    finally:
        if pq_writer:
            pq_writer.close()
    csv_path = raw_dir / 'markets.csv'
    with open(csv_path, 'w', newline='') as fh:
        writer = csv.DictWriter(fh, fieldnames=['src_ticker', 'market'])
        writer.writeheader()
        writer.writerows(market_rows)
    shutil.rmtree(data_root)
    logging.debug(f'Flattened {ticker_count} tickers to {parquet_path}.')


class StooqProvider:
    def fetch(self):
        """Fetches Stooq price data by unzipping bulk files and building a price dataset."""
        logging.info('Fetching Stooq price data...')
        _ensure_files_available(
            stooq_cfg.bulk_files,
            error_message='Bulk files not available.',
            download_instruction=stooq_cfg.bulk_download_instruction,
        )
        _unzip_bulk_files()
        _build_price_dataset()
        logging.info('Stooq price data  fetched successfully.')

    def update(self):
        _ensure_files_available(
            stooq_cfg.update_file,
            error_message='Update file not available.',
            download_instruction=stooq_cfg.update_download_instruction,
        )

    def transform(self, feed: Literal['bulk', 'update']):
        logging.info(f'Transforming Stooq {feed} data...')
        if feed == 'bulk':
            from_file = raw_dir / 'bulk_prices.parquet'
            to_file = processed_dir / 'bulk_prices.parquet'
            from_claude = f"FROM read_parquet('{from_file}')"
            to_clausse = f"TO '{to_file}' (FORMAT PARQUET);"
        else:
            from_file = raw_dir / stooq_cfg.update_file
            to_file = processed_dir / stooq_cfg.update_file
            from_claude = f"FROM read_csv_auto('{stooq_cfg.update_file}')"
            to_clausse = f"TO '{to_file}' (FORMAT CSV, HEADER);"

        conn = duckdb.connect()

        conn.sql(f"""
            COPY (SELECT
                    split_part(lower("<TICKER>"), '.', 1) AS ticker,
                    lower("<TICKER>") AS src_ticker,
                    lower("<PER>") AS per,
                    "<DATE>" AS date,
                    "<TIME>" AS time,
                    "<OPEN>" AS open,
                    "<HIGH>" AS high,
                    "<LOW>" AS low,
                    "<CLOSE>" AS close,
                    "<VOL>" AS vol,
                    "<OPENINT>" AS openint
                {from_claude})
            {to_clausse}
        """)
        logging.info(f'Stooq {feed} data transformed successfully.')

        if feed == 'bulk':
            logging.info('Transforming market metadata...')  
            conn.sql(f"""
                COPY (
                    SELECT
                        split_part(lower("ticker"), '.', 1) AS ticker,
                        ticker as src_ticker,
                        market
                    FROM read_csv_auto('{raw_dir / "markets.csv"}')
                )
                TO '{processed_dir / "markets.csv"}'
                (FORMAT CSV, HEADER);
            """)
            logging.info('Market metadata transformed successfully.')

    def store(self):
        # into duckdb database
        ...


def main():
    provider = StooqProvider()
    # provider.fetch()
    # provider.update()
    provider.transform('bulk')
    # df = pd.read_parquet(processed_dir / 'bulk_prices.parquet')
    # print(df.head())
    # print(df.sample(10))


if __name__ == '__main__':
    main()

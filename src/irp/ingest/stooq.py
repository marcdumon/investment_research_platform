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
from irp.core.duckdb_merge import merge_csv
from irp.runner import Feed

logger = logging.getLogger(__name__)

root_dir = config.data.root_dir
stooq_cfg = config.providers.stooq
raw_dir = root_dir / stooq_cfg.raw_dir
processed_dir = root_dir / stooq_cfg.processed_dir


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
    """Extract all configured bulk zips into raw_dir. Caller is responsible for freshness checks."""
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
    """Read extracted ticker CSVs, write bulk_prices.parquet + markets.csv.

    Incremental parquet write: a crash mid-loop leaves a partial parquet.
    Safe to rerun because the `.fetched` marker is only touched by the
    caller after this function returns, so a partial file is overwritten
    on the next fetch_bulk(). No cleanup needed.
    """
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


@dataclass(frozen=True)
class FeedSpec:
    input_path: Path
    output_path: Path
    input_format: Literal['csv', 'parquet']
    output_format: Literal['csv', 'parquet']


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


def _transform_prices(conn: duckdb.DuckDBPyConnection, spec: FeedSpec) -> None:
    source = (
        f"read_parquet('{spec.input_path}')"
        if spec.input_format == 'parquet'
        else f"read_csv_auto('{spec.input_path}')"
    )
    fmt = 'FORMAT PARQUET' if spec.output_format == 'parquet' else 'FORMAT CSV, HEADER'
    conn.sql(f"""
        COPY (
            SELECT
                split_part(upper(t."<TICKER>"), '.', 1) AS Ticker,
                STRPTIME(CAST(t."<DATE>" AS VARCHAR), '%Y%m%d')::DATE AS Date,
                t."<OPEN>"  AS O,
                t."<HIGH>"  AS H,
                t."<LOW>"   AS L,
                t."<CLOSE>" AS C,
                t."<VOL>"   AS V,
                t."<TICKER>" AS SrcId,
                'stooq'      AS Src
            FROM {source} t
        )
        TO '{spec.output_path}' ({fmt})
    """)
    logger.debug(f'Wrote {spec.output_path}')


def _store_prices(con: duckdb.DuckDBPyConnection, spec: FeedSpec) -> None:
    merge_csv(
        con,
        'prices',
        spec.output_path,
        key_cols=['Ticker', 'SrcId', 'Date', 'Src'],
        value_cols=['O', 'H', 'L', 'C', 'V'],
    )
    logger.debug(f'Stored prices from {spec.output_path}')


class StooqSource:
    SUPPORTED_FEEDS: frozenset[Feed] = frozenset({'bulk', 'update'})

    def __init__(self) -> None:
        from irp.core.markers import MarkerSet

        self.markers = MarkerSet(raw_dir)

    def fetch_bulk(self) -> None:
        """Stooq bulk zips must be manually downloaded and placed in raw_dir. Unzips and builds the price dataset."""
        zip_paths = [raw_dir / f for f in stooq_cfg.bulk_files]
        if self.markers.is_fresh('fetched', *zip_paths):
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
        self.markers.touch('fetched')
        logger.debug('Stooq price data fetched successfully.')

    def update(self) -> None:
        """Stooq files cannot be fetched programmatically; they must be manually downloaded.
        Checks that the update file has been placed in raw_dir before continuing."""
        _ensure_files_available(
            stooq_cfg.update_file,
            error_message='Update file not available.',
            download_instruction=stooq_cfg.update_download_instruction,
        )

    def transform(self, feed: Literal['bulk', 'update']) -> None:
        spec = FEED_SPECS[feed]
        upstream = self.markers.path('fetched') if feed == 'bulk' else spec.input_path
        if self.markers.is_fresh(f'transformed_{feed}', upstream):
            logger.info(f'transform({feed}): already up to date, skipping')
            return
        conn = duckdb.connect()
        _transform_prices(conn, spec)
        self.markers.touch(f'transformed_{feed}')

    def store(self, feed: Literal['bulk', 'update']) -> None:
        if self.markers.is_fresh(f'stored_{feed}', f'transformed_{feed}'):
            logger.info(f'store({feed}): already up to date, skipping')
            return
        spec = FEED_SPECS[feed]
        from irp.core.db import write_session
        with write_session() as con:
            _store_prices(con, spec)
        self.markers.touch(f'stored_{feed}')
        logger.debug(f'Stooq {feed} data stored successfully.')

    def cleanup(self) -> None:
        """Delete all intermediate files, keeping zips, markers, and the database."""
        targets = [
            raw_dir / 'bulk_prices.parquet',
            processed_dir / 'bulk_prices.parquet',
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

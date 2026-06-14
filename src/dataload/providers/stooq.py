"""Stooq provider — produces the `prices` dataset.

Stooq bulk history and daily updates are not fetchable programmatically: the
user downloads the zips / update file and drops them in the provider's raw_dir.
`produce()` unzips + flattens the bulk tree into one parquet (and writes
`markets.csv`, the seed for the universe table), then normalizes either the
bulk parquet or the daily update file into canonical `prices`.
"""
import csv
import logging
import shutil
import zipfile
from collections.abc import Iterator, Sequence
from pathlib import Path

import duckdb
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

from dataload._sql import copy_to_parquet, reader
from dataload.context import IngestContext
from dataload.providers.base import Capability
from dataload.state import MarkerSet

logger = logging.getLogger(__name__)


def _ensure_files_available(paths: Sequence[Path], message: str) -> None:
    if all(p.is_file() for p in paths):
        return
    logger.error(message)
    raise FileNotFoundError(message)


def _unzip_bulk_files(raw_dir: Path, bulk_files: Sequence[str]) -> None:
    """Extract all configured bulk zips into raw_dir."""
    for fname in bulk_files:
        with zipfile.ZipFile(raw_dir / fname) as zf:
            zf.extractall(raw_dir)


def _iter_file_dirs(path: Path) -> Iterator[Path]:
    """Yield dirs that contain ticker files.

    Assumes file-containing dirs are terminal leaves (no mixed file+subdir dirs).
    """
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


def _build_price_dataset(raw_dir: Path) -> None:
    """Read extracted ticker CSVs, write bulk_prices.parquet + markets.csv.

    Incremental parquet write: a crash mid-loop leaves a partial parquet, which
    is harmless because the caller only touches the `.fetched` marker after this
    returns, so the next run overwrites it.
    """
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
    logger.debug(f'Flattened {ticker_count} tickers to {parquet_file}')

    with open(raw_dir / 'markets.csv', 'w', newline='') as fh:
        writer = csv.DictWriter(fh, fieldnames=['ticker', 'market'])
        writer.writeheader()
        writer.writerows(market_rows)


def _normalize_prices(con: duckdb.DuckDBPyConnection, src: Path, out: Path) -> Path:
    """Stooq raw OHLCV (`<TICKER>`, `<DATE>`, ...) -> canonical `prices` parquet."""
    return copy_to_parquet(con, f"""
        SELECT
            split_part(upper(t."<TICKER>"), '.', 1)              AS Ticker,
            STRPTIME(CAST(t."<DATE>" AS VARCHAR), '%Y%m%d')::DATE AS Date,
            t."<OPEN>"                AS Open,
            t."<HIGH>"                AS High,
            t."<LOW>"                 AS Low,
            t."<CLOSE>"               AS Close,
            CAST(t."<VOL>" AS BIGINT) AS Volume,
            'stooq'                   AS Src,
            t."<TICKER>"              AS SrcId
        FROM {reader(src)} t
    """, out)


class StooqProvider:
    name = 'stooq'

    def capabilities(self) -> dict[str, Capability]:
        # incremental = the manually-downloaded daily update file
        return {'prices': Capability(incremental=True)}

    def produce(self, ctx: IngestContext, datasets: Sequence[str], *, incremental: bool) -> dict[str, Path]:
        if 'prices' not in datasets:
            return {}
        raw = ctx.raw_dir('stooq')
        processed = ctx.processed_dir('stooq')
        cfg = ctx.cfg('stooq')

        if incremental:
            src = raw / cfg['update_file']
            _ensure_files_available([src], f'Stooq update file not available: {src}')
        else:
            self._acquire_bulk(raw, cfg)
            src = raw / 'bulk_prices.parquet'

        processed.mkdir(parents=True, exist_ok=True)
        out = processed / 'prices.parquet'
        con = duckdb.connect()
        try:
            _normalize_prices(con, src, out)
        finally:
            con.close()
        return {'prices': out}

    def _acquire_bulk(self, raw: Path, cfg: dict) -> None:
        """Unzip + flatten the bulk tree, unless the .fetched marker is fresh vs the zips."""
        markers = MarkerSet(raw)
        zips = [raw / f for f in cfg['bulk_files']]
        if markers.is_fresh('fetched', *zips):
            logger.info('stooq acquire: already up to date, skipping')
            return
        _ensure_files_available(zips, 'Stooq bulk zips not available in raw_dir.')
        _unzip_bulk_files(raw, cfg['bulk_files'])
        _build_price_dataset(raw)
        markers.touch('fetched')

    def cleanup(self, ctx: IngestContext) -> None:
        """Delete intermediates; keep zips, markers, markets.csv (universe seed), and the DB."""
        raw = ctx.raw_dir('stooq')
        processed = ctx.processed_dir('stooq')
        for path in (raw / 'bulk_prices.parquet', processed / 'prices.parquet'):
            if path.exists():
                path.unlink()
                logger.debug(f'Deleted {path}')
        data_dir = raw / 'data'
        if data_dir.exists():
            shutil.rmtree(data_dir)
            logger.debug(f'Deleted {data_dir}')

"""Load all Stooq bulk prices into DuckDB (US + world zips)."""

import logging
import time
from pathlib import Path

import pandas as pd
from dotenv import load_dotenv

import irp.config as _config
from irp._logging import configure
from irp.datasets.dataset import Dataset
from irp.sources.stooq import PRICE_SCHEMA
from irp.sources.stooq_bulk import StooqBulkSource, ensure_zips_extracted
from irp.store import Store
from irp.transforms.cleaner import Cleaner

load_dotenv()
configure()
logger = logging.getLogger(Path(__file__).stem)

_BATCH_SIZE = 5000
_PK = ["source_id", "date"]

cfg = _config.load()["stooq"]
data_dir = Path(cfg["data_dir"])

store = Store()
cleaner = Cleaner()

ensure_zips_extracted(Path(cfg["download_dir"]), data_dir)

logger.info("Indexing ticker files...")
file_index: dict[str, Path] = {
    p.stem: p
    for subdir in data_dir.iterdir()
    if subdir.is_dir()
    for p in subdir.rglob("*.txt")
}

tickers = sorted(file_index)
total = len(tickers)
logger.info("%d tickers to load", total)


errors: list[tuple[str, str]] = []
batch: list[pd.DataFrame] = []

for i, ticker in enumerate(tickers, 1):
    try:
        ds = StooqBulkSource(ticker).fetch(file_path=file_index[ticker])
        batch.append(cleaner.transform(ds).data)
    except Exception as e:
        errors.append((ticker, str(e)))

    if len(batch) >= _BATCH_SIZE:
        t0 = time.perf_counter()
        combined = pd.concat(batch, ignore_index=True)
        store.upsert(
            Dataset(
                name="prices", data=combined, schema=PRICE_SCHEMA, source="stooq_bulk"
            ),
            table="prices",
            primary_key=_PK,
        )
        logger.info(
            "%d/%d — batch flushed in %.1fs", i, total, time.perf_counter() - t0
        )
        batch.clear()

if batch:
    combined = pd.concat(batch, ignore_index=True)
    store.upsert(
        Dataset(name="prices", data=combined, schema=PRICE_SCHEMA, source="stooq_bulk"),
        table="prices",
        primary_key=_PK,
    )

logger.info("Done. %d errors.", len(errors))
for ticker, msg in errors:
    logger.warning("SKIP %s: %s", ticker, msg)

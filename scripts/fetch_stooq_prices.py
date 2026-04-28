"""Load all Stooq bulk prices into DuckDB (US + world zips)."""

import logging
from pathlib import Path

from dotenv import load_dotenv

import irp.config as _config
from irp._logging import configure
from irp.sources.stooq_bulk import StooqBulkSource, ensure_zips_extracted
from irp.store import Store
from irp.transforms.cleaner import Cleaner

load_dotenv()
configure()
logger = logging.getLogger(__name__)

cfg = _config.load()["stooq"]
data_dir = Path(cfg["data_dir"])

store = Store()
cleaner = Cleaner()

ensure_zips_extracted(Path(cfg["download_dir"]), data_dir)

tickers = sorted(p.stem for p in data_dir.rglob("*.txt"))
total = len(tickers)
logger.info("%d tickers found", total)

errors = []
for i, ticker in enumerate(tickers, 1):
    try:
        ds = StooqBulkSource(ticker).fetch()
        ds = cleaner.transform(ds)
        store.save(ds, table="prices", partition_col="ticker")
    except Exception as e:
        errors.append((ticker, str(e)))
    if i % 500 == 0:
        logger.info("  %d/%d", i, total)

logger.info("Done. %d errors.", len(errors))
for ticker, msg in errors:
    logger.warning("  SKIP %s: %s", ticker, msg)

"""Incrementally update prices table using Stooq API (new days only)."""

import logging
from datetime import datetime, timedelta

import duckdb
from dotenv import load_dotenv

import irp.config as _config
from irp._logging import configure 
from irp.sources.stooq import StooqPriceSource
from irp.store import Store
from irp.transforms.cleaner import Cleaner

load_dotenv()
configure()
logger = logging.getLogger(__name__)

cfg = _config.load()
db_path = cfg["store"]["db_path"]

store = Store()
cleaner = Cleaner()

if not store.exists("prices"):
    logger.error("No 'prices' table found. Run fetch_stooq_prices.py first.")
    raise SystemExit(1)

with duckdb.connect(db_path, read_only=True) as con:
    rows = con.execute("SELECT DISTINCT ticker, source_id FROM prices").fetchall()

ticker_map: dict[str, str] = {ticker: source_id for ticker, source_id in rows}

today = datetime.today().strftime("%Y-%m-%d")
yesterday = (datetime.today() - timedelta(days=1)).strftime("%Y-%m-%d")
tickers = sorted(ticker_map)
total = len(tickers)
logger.info("%d tickers in DB", total)

updated = skipped = 0
errors = []

for i, db_ticker in enumerate(tickers, 1):
    stooq_ticker = ticker_map[db_ticker]

    try:
        max_dt = store.max_date("prices", "date", filter_col="ticker", filter_val=db_ticker)
        if max_dt is None:
            skipped += 1
            continue

        next_day = (datetime.strptime(max_dt, "%Y-%m-%d") + timedelta(days=1)).strftime("%Y-%m-%d")
        if next_day > today:
            skipped += 1
            continue

        ds = StooqPriceSource(stooq_ticker, start=next_day, end=yesterday).fetch()
        if ds.data.empty:
            skipped += 1
            continue

        store.append(cleaner.transform(ds), table="prices", conflict_cols=["ticker", "date"])
        updated += 1

    except Exception as e:
        errors.append((db_ticker, str(e)))

    if i % 100 == 0:
        logger.info("  %d/%d — updated %d, skipped %d", i, total, updated, skipped)

logger.info("Done. %d tickers updated, %d skipped, %d errors.", updated, skipped, len(errors))
for ticker, msg in errors:
    logger.warning("  ERROR %s: %s", ticker, msg)

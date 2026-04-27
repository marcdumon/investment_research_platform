"""Incrementally update prices table using Stooq API (new days only)."""

import sys
from datetime import datetime, timedelta
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parents[1] / "src"))

from dotenv import load_dotenv

load_dotenv()

import duckdb
import irp.config as _config
from irp.sources.stooq import StooqPriceSource
from irp.store import Store
from irp.transforms.cleaner import Cleaner

cfg = _config.load()
db_path = cfg["store"]["db_path"]

store = Store()
cleaner = Cleaner()

if not store.exists("prices"):
    print("No 'prices' table found. Run fetch_stooq_prices.py first.")
    sys.exit(0)

with duckdb.connect(db_path, read_only=True) as con:
    rows = con.execute("SELECT DISTINCT ticker, source_id FROM prices").fetchall()

ticker_map: dict[str, str] = {ticker: source_id for ticker, source_id in rows}

today = datetime.today().strftime("%Y-%m-%d")
yesterday = (datetime.today() - timedelta(days=1)).strftime("%Y-%m-%d")
tickers = sorted(ticker_map)
total = len(tickers)
print(f"{total} tickers in DB", flush=True)

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
        print(f"  {i}/{total} — updated {updated}, skipped {skipped}", flush=True)

print(f"Done. {updated} tickers updated, {skipped} skipped, {len(errors)} errors.")
for ticker, msg in errors:
    print(f"  ERROR {ticker}: {msg}")

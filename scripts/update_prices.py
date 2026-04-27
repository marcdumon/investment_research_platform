"""Incrementally update prices table using Stooq API (new days only)."""

import sys
from datetime import datetime, timedelta
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parents[1] / "src"))

from dotenv import load_dotenv

load_dotenv()

import duckdb
import irp.config as _config
from irp.sources.stooq import StooqPriceSource, normalize_ticker
from irp.store import Store
from irp.transforms.cleaner import Cleaner

cfg = _config.load()
data_dir = Path(cfg["stooq"]["data_dir"])
db_path = cfg["store"]["db_path"]

store = Store()
cleaner = Cleaner()

if not store.exists("prices"):
    print("No 'prices' table found. Run fetch_stooq_prices.py first.")
    sys.exit(0)

with duckdb.connect(db_path, read_only=True) as con:
    known_tickers = sorted(r[0] for r in con.execute("SELECT DISTINCT ticker FROM prices").fetchall())

# Build reverse map db_ticker -> stooq_ticker from local txt files.
# Needed because the Stooq API requires the full symbol (e.g. "msft.us").
ticker_map: dict[str, str] = {}
for txt_path in data_dir.rglob("*.txt"):
    stooq_t = txt_path.stem
    db_t = normalize_ticker(stooq_t)
    ticker_map.setdefault(db_t, stooq_t)

today = datetime.today().strftime("%Y-%m-%d")
yesterday = (datetime.today() - timedelta(days=1)).strftime("%Y-%m-%d")
total = len(known_tickers)
print(f"{total} tickers in DB", flush=True)

updated = skipped = errors_list = 0
errors = []

for i, db_ticker in enumerate(known_tickers, 1):
    stooq_ticker = ticker_map.get(db_ticker)
    if stooq_ticker is None:
        skipped += 1
        continue

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

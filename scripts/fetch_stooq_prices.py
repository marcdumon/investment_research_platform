"""Load all Stooq bulk prices into DuckDB (US + world zips)."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parents[1] / "src"))

from dotenv import load_dotenv

load_dotenv()

import irp.config as _config
from irp.sources.stooq_bulk import StooqBulkSource
from irp.store import Store
from irp.transforms.cleaner import Cleaner

cfg = _config.load()["stooq"]
data_dir = Path(cfg["data_dir"])

store = Store()
cleaner = Cleaner()

# Trigger extraction of all zips before iterating
_dummy = StooqBulkSource.__new__(StooqBulkSource)
_dummy._download_dir = Path(cfg["download_dir"])
_dummy._data_dir = data_dir
_dummy._ensure_extracted()

tickers = sorted(p.stem for p in data_dir.rglob("*.txt"))
total = len(tickers)
print(f"{total} tickers found", flush=True)

errors = []
for i, ticker in enumerate(tickers, 1):
    try:
        ds = StooqBulkSource(ticker).fetch()
        ds = cleaner.transform(ds)
        store.save(ds, table="prices", partition_col="ticker")
    except Exception as e:
        errors.append((ticker, str(e)))
    if i % 500 == 0:
        print(f"  {i}/{total}", flush=True)

print(f"Done. {len(errors)} errors.")
for ticker, msg in errors:
    print(f"  SKIP {ticker}: {msg}")

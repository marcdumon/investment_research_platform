"""Resolve SEC EDGAR filing URLs for all (ticker, period) pairs in fundamentals.

Run once after fetch_simfin_fundamentals.py, then re-run incrementally after
update_simfin_fundamentals.py to pick up new periods.

Results are stored in the sec_filings table (ticker, period, url).
Already-resolved pairs are skipped.  To force a full re-resolve, drop the
sec_filings table first.

Q4 reports are the same filing as the annual (10-K).  The script resolves
the URL once for the canonical period (yyyyA) and writes it for both yyyyA
and yyyyQ4 rows without making a second HTTP request.
"""

import logging
import sys
import time

import pandas as pd
from dotenv import load_dotenv

from irp._logging import configure
from irp.datasets.dataset import Dataset
from irp.sources.sec_edgar import sec_filing_url
from irp.store import Store

load_dotenv()
configure()
logger = logging.getLogger(__name__)

store = Store()

if not store.exists("income"):
    logger.error("income table not found — run fetch_simfin_fundamentals.py first")
    sys.exit(1)

income = store.load("income").data
if "period" not in income.columns:
    logger.error("period column missing from income — re-run fetch_simfin_fundamentals.py")
    sys.exit(1)

pairs = (
    income[["Ticker", "period"]]
    .drop_duplicates()
    .dropna(subset=["period"])
    .rename(columns={"Ticker": "ticker"})
    .reset_index(drop=True)
)

if store.exists("sec_filings"):
    existing = store.load("sec_filings").data[["ticker", "period", "url"]]
    pairs = (
        pairs
        .merge(existing, on=["ticker", "period"], how="left")
        .query("url.isna()")
        .drop(columns=["url"])
        .reset_index(drop=True)
    )

# Q4 == annual 10-K: map yyyyQ4 → yyyyA so we resolve the URL once.
# Both the original period and the canonical period are written to the table.
pairs["canon"] = pairs["period"].apply(
    lambda p: p[:4] + "A" if p.endswith("Q4") else p
)

# Resolve unique (ticker, canonical_period) only
to_resolve = pairs[["ticker", "canon"]].drop_duplicates().reset_index(drop=True)
total = len(to_resolve)
logger.info(f"{total} unique (ticker, canonical_period) pairs to resolve")
if total == 0:
    logger.info("Nothing to do.")
    sys.exit(0)

FLUSH_EVERY = 50

url_cache: dict[tuple[str, str], str | None] = {}
error_rows: list[dict] = []
total_written = 0
total_resolved = 0


def _flush(batch: list[tuple[str, str]]) -> None:
    global total_written, total_resolved
    keys = set(batch)
    mask = [
        (t, c) in keys
        for t, c in zip(pairs["ticker"], pairs["canon"])
    ]
    chunk = pairs[mask].copy()
    chunk["url"] = [url_cache[(t, c)] for t, c in zip(chunk["ticker"], chunk["canon"])]
    rows = chunk[["ticker", "period", "url"]].to_dict("records")
    df_chunk = pd.DataFrame(rows)
    dataset = Dataset.from_df(df_chunk, name="sec_filings", source="sec_edgar")
    store.upsert(dataset, table="sec_filings", primary_key=["ticker", "period"])
    total_written += len(df_chunk)
    total_resolved += int(df_chunk["url"].notna().sum())
    logger.info(f"Flushed {len(df_chunk)} rows to DB.")


pending: list[tuple[str, str]] = []
for i, (ticker, canon) in enumerate(zip(to_resolve["ticker"], to_resolve["canon"]), 1):
    t0 = time.monotonic()
    try:
        url: str | None = sec_filing_url(ticker, canon)
        elapsed = time.monotonic() - t0
        logger.info(f"[{i}/{total}] {ticker} {canon} OK ({elapsed:.2f}s)")
    except Exception as exc:
        elapsed = time.monotonic() - t0
        logger.info(f"[{i}/{total}] {ticker} {canon} SKIP: {exc} ({elapsed:.2f}s)")
        url = None
        error_rows.append({"ticker": ticker, "period": canon, "error": str(exc)})
    url_cache[(ticker, canon)] = url
    pending.append((ticker, canon))
    if len(pending) >= FLUSH_EVERY:
        _flush(pending)
        pending.clear()

if pending:
    _flush(pending)

errors = len(error_rows)
if error_rows:
    error_path = "data/sec_filings_errors.csv"
    pd.DataFrame(error_rows).to_csv(error_path, index=False)
    logger.info(f"{errors} errors saved to {error_path}")

logger.info(f"Done. {total_written} rows written, {total_resolved} resolved, {errors} failed.")

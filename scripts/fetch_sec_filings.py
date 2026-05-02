"""Resolve SEC EDGAR filing URLs for all (ticker, period) pairs in fundamentals.

Run once after fetch_simfin_fundamentals.py, then re-run incrementally after
update_simfin_fundamentals.py to pick up new periods.

Results are stored in the sec_filings table (ticker, period, url).
Already-resolved pairs are skipped.  To force a full re-resolve, drop the
sec_filings table first.

Q4 reports are the same filing as the annual (10-K).  The script resolves
the URL once for the canonical period (yyyyFY) and writes it for both yyyyFY
and yyyyQ4 rows without making a second HTTP request.
"""
from pathlib import Path

import logging
import random
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed

import pandas as pd
from dotenv import load_dotenv

from irp._logging import configure
from irp.datasets.dataset import Dataset
from irp.sources.sec_edgar import sec_filing_url
from irp.store import Store

load_dotenv()
configure()
logger = logging.getLogger(Path(__file__).stem)

store = Store()

if not store.exists("income"):
    logger.error("income table not found — run fetch_simfin_fundamentals.py first")
    sys.exit(1)

income = store.load("income").data
if "period" not in income.columns:
    logger.error("period column missing from income — re-run fetch_simfin_fundamentals.py")
    sys.exit(1)

pairs = (
    income[["Ticker", "period", "Publish Date"]]
    .drop_duplicates()
    .dropna(subset=["Ticker", "period"])
    .rename(columns={"Ticker": "ticker", "Publish Date": "publish_date"})
    .assign(publish_date=lambda df: df["publish_date"].astype(str).str[:10])
    .reset_index(drop=True)
)

RETRY_ERRORS = False


def _exclude_done(pairs: pd.DataFrame, existing: pd.DataFrame, *, retry_errors: bool) -> pd.DataFrame:
    merged = pairs.merge(existing[["ticker", "period", "url", "error"]], on=["ticker", "period"], how="left")
    if retry_errors:
        keep = merged["url"].isna()
    else:
        keep = merged["url"].isna() & merged["error"].isna()
    return merged[keep].drop(columns=["url", "error"], errors="ignore").reset_index(drop=True)


if store.exists("sec_filings"):
    try:
        existing = store.load("sec_filings").data
        pairs = _exclude_done(pairs, existing, retry_errors=RETRY_ERRORS)
    except Exception:
        logger.warning("sec_filings table missing from DB despite metadata entry — resolving all pairs")

# Q4 == annual 10-K: map yyyyQ4 → yyyyA so we resolve the URL once.
# Both the original period and the canonical period are written to the table.
pairs["canon"] = pairs["period"].apply(
    lambda p: p[:4] + "FY" if p.endswith("Q4") else p
)

# Resolve unique (ticker, canonical_period) — use publish_date from annual row when available
to_resolve = (
    pairs.groupby(["ticker", "canon"], as_index=False)
    .agg(publish_date=("publish_date", "first"))
    .reset_index(drop=True)
)
total = len(to_resolve)
logger.info(f"{total} unique (ticker, canonical_period) pairs to resolve")
if total == 0:
    logger.info("Nothing to do.")
    sys.exit(0)

FLUSH_EVERY = 50

ERROR_PATH = "data/data_quality/sec_filings_errors.csv"

url_cache: dict[tuple[str, str], str | None] = {}
error_cache: dict[tuple[str, str], str | None] = {}
error_rows: list[dict] = []
total_written = 0
total_resolved = 0


_error_file_initialized = False


def _flush_errors(rows: list[dict]) -> None:
    global _error_file_initialized
    pd.DataFrame(rows).to_csv(
        ERROR_PATH,
        mode="w" if not _error_file_initialized else "a",
        header=not _error_file_initialized,
        index=False,
    )
    _error_file_initialized = True


def _flush(batch: list[tuple[str, str]]) -> None:
    global total_written, total_resolved
    keys = set(batch)
    mask = pd.Series(
        [(t, c) in keys for t, c in zip(pairs["ticker"], pairs["canon"])],
        index=pairs.index,
    )
    chunk = pairs[mask].copy()
    chunk["url"] = [url_cache[(t, c)] for t, c in zip(chunk["ticker"], chunk["canon"])]
    chunk["error"] = [error_cache.get((t, c)) for t, c in zip(chunk["ticker"], chunk["canon"])]
    df_chunk = chunk[["ticker", "period", "url", "error"]]
    dataset = Dataset.from_df(df_chunk, name="sec_filings", source="sec_edgar")
    store.upsert(dataset, table="sec_filings", primary_key=["ticker", "period"])
    total_written += len(df_chunk)
    total_resolved += int(df_chunk["url"].notna().sum())
    logger.info(f"Flushed {len(df_chunk)} rows to DB.")


MAX_WORKERS = 6
MAX_RETRIES = 6
MIN_REQUEST_INTERVAL = 0.12  # ~8 req/s across all threads

_rate_lock = threading.Lock()
_last_request_time = 0.0


def _throttle() -> None:
    global _last_request_time
    with _rate_lock:
        now = time.monotonic()
        wait = MIN_REQUEST_INTERVAL - (now - _last_request_time)
        if wait > 0:
            time.sleep(wait)
        _last_request_time = time.monotonic()


# Pre-warm ticker map so all threads share one HTTP call
from irp.sources.sec_edgar import _load_ticker_map  # noqa: E402
_load_ticker_map()


def _resolve(ticker: str, canon: str, publish_date: str | None) -> tuple[str, str, str | None, str | None, float]:
    t_start = time.monotonic()
    for attempt in range(MAX_RETRIES):
        _throttle()
        try:
            url: str | None = sec_filing_url(ticker, canon, publish_date)
            return ticker, canon, url, None, time.monotonic() - t_start
        except Exception as exc:
            if "429" in str(exc) and attempt < MAX_RETRIES - 1:
                wait = 2 ** attempt + random.random()
                logger.info(f"{ticker} {canon} 429 — retry {attempt + 1}/{MAX_RETRIES} in {wait:.1f}s")
                time.sleep(wait)
            else:
                return ticker, canon, None, str(exc), time.monotonic() - t_start
    return ticker, canon, None, "max retries exceeded", time.monotonic() - t_start


pending: list[tuple[str, str]] = []
done = 0
with ThreadPoolExecutor(max_workers=MAX_WORKERS) as pool:
    futures = {
        pool.submit(_resolve, ticker, canon, publish_date): (ticker, canon)
        for ticker, canon, publish_date in zip(
            to_resolve["ticker"], to_resolve["canon"], to_resolve["publish_date"]
        )
    }
    _GREEN = "\033[32m"
    _YELLOW = "\033[33m"
    _RESET = "\033[0m"
    for future in as_completed(futures):
        ticker, canon, url, err, elapsed = future.result()
        done += 1
        if err:
            status = f"{_YELLOW}SKIP: {err}{_RESET}"
        else:
            status = f"{_GREEN}OK{_RESET}"
        logger.info(f"[{done}/{total}] {ticker} {canon} {status} ({elapsed:.2f}s)")
        url_cache[(ticker, canon)] = url
        error_cache[(ticker, canon)] = err
        if err:
            error_rows.append({"ticker": ticker, "period": canon, "error": err})
            _flush_errors(error_rows[-1:])
        pending.append((ticker, canon))
        if len(pending) >= FLUSH_EVERY:
            _flush(pending)
            pending.clear()

if pending:
    _flush(pending)

errors = len(error_rows)
if errors:
    logger.info(f"{errors} errors saved to {ERROR_PATH}")

logger.info(f"Done. {total_written} rows written, {total_resolved} resolved, {errors} failed.")

"""Cross-section data loading + caching layer.

Sits between the orchestration functions (run_backtest, run_factor_decay,
run_composite_backtest) and the raw panel engine. Owns cache lookups and
the ThreadPoolExecutor used to parallelise cross-section computation.

Pure data movement — no IC math, no rebalance-date generation.
"""
import datetime
import logging
from concurrent.futures import ThreadPoolExecutor, as_completed

import pandas as pd

from irp.core.config import config
from irp.factors import cache as _cache
from irp.panel import cross_section_panel

logger = logging.getLogger(__name__)


def compute_and_cache(
    dates: list[datetime.date],
    variant: str,
    tickers: list[str] | None,
    write_cache: bool = True,
) -> dict[datetime.date, pd.DataFrame]:
    """Compute cross-sections for `dates` via the panel engine.

    Runs in a thread pool sized by `config.factors.cache_workers`.
    Writes non-empty results to the snapshot cache when `write_cache` is True.
    """
    def _one(d: datetime.date) -> tuple[datetime.date, pd.DataFrame]:
        return d, cross_section_panel(d, variant, tickers)

    result: dict[datetime.date, pd.DataFrame] = {}
    with ThreadPoolExecutor(max_workers=config.factors.cache_workers) as ex:
        futures = {ex.submit(_one, d): d for d in dates}
        for fut in as_completed(futures):
            d, xs = fut.result()
            if not xs.empty:
                result[d] = xs
                if write_cache:
                    _cache.store(d, variant, xs)
    return result


def load_cross_sections(
    dates: list[datetime.date],
    variant: str,
    tickers: list[str] | None,
) -> dict[datetime.date, pd.DataFrame]:
    """Cache-first cross-section retrieval for a list of rebalance dates.

    For full-universe runs (`tickers is None`), checks the parquet cache
    per date and only computes misses. For filtered runs, recomputes every
    date because per-date results aren't keyed by ticker subset.

    Writes the cache only for full-universe runs (filtered results would
    contaminate the full-universe key).
    """
    result: dict[datetime.date, pd.DataFrame] = {}
    to_compute: list[datetime.date] = []

    if tickers is None:
        for d in dates:
            cached = _cache.load(d, variant)
            if cached is not None:
                result[d] = cached
            else:
                to_compute.append(d)
    else:
        to_compute = list(dates)

    if to_compute:
        result.update(
            compute_and_cache(
                to_compute, variant, tickers,
                write_cache=(tickers is None),
            )
        )
    return result

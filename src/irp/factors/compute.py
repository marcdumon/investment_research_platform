"""Public factor-computation entry points.

Two foreground operations live here:
- `cross_section(date, variant, tickers)` — full-universe snapshot (cached)
- `ticker_factor_history(ticker, variant)` — one-ticker historical factors

Backtest orchestration moved to `orchestrate.py`; data-loading + caching
moved to `data_loaders.py`. Re-exports preserve the historic import surface
(`from irp.factors.compute import run_backtest, ...`).

Only this module + `cache.py` touch external state; the rest of the
package is pure computation.
"""
import datetime
import logging
from typing import Literal

import pandas as pd

from irp.factors import cache as _cache
from irp.factors._cols import REPORT_DATE
from irp.factors._pit import pit_latest, pit_price, pit_ttm
from irp.factors.momentum import compute_momentum
from irp.factors.orchestrate import (
    run_backtest,
    run_composite_backtest,
    run_factor_decay,
)
from irp.factors.profitability import compute_profitability
from irp.factors.valuation import compute_valuation
from irp.panel import cross_section_panel
from irp.query.simfin import fundamentals
from irp.query.yahoo import prices as yahoo_prices

logger = logging.getLogger(__name__)

__all__ = [
    'cross_section',
    'ticker_factor_history',
    'run_backtest',
    'run_composite_backtest',
    'run_factor_decay',
]


def cross_section(
    as_of_date: datetime.date,
    variant: Literal['A', 'Q'] = 'A',
    tickers: list[str] | None = None,
) -> pd.DataFrame:
    """Compute all factors cross-sectionally at a point in time.

    Only fundamental data with Report Date <= as_of_date and prices with
    Date <= as_of_date are used, making results PIT-safe.

    Full-universe (`tickers is None`) results are cached as parquet under
    `data/factor_cache/<variant>/<date>.parquet`. Filtered results are not
    cached.
    """
    if tickers is None:
        cached = _cache.load(as_of_date, variant)
        if cached is not None:
            return cached

    result = cross_section_panel(as_of_date, variant, tickers)
    if tickers is None and not result.empty:
        _cache.store(as_of_date, variant, result)
    return result


def ticker_factor_history(
    ticker: str,
    variant: Literal['A', 'Q'] = 'A',
) -> pd.DataFrame:
    """Factor values at each historical filing date for one ticker.

    Uses the pandas PIT pipeline (`_pit.py` + per-factor compute_* functions)
    because the panel engine is optimised for cross-section snapshots, not
    per-ticker time series. Returns one row per Report Date with valuation,
    profitability, and momentum factors.
    """
    raw_income   = fundamentals([ticker], 'income',   variant)
    raw_balance  = fundamentals([ticker], 'balance',  variant)
    raw_cashflow = fundamentals([ticker], 'cashflow', variant)
    raw_prices   = yahoo_prices([ticker])

    if any(df.empty for df in [raw_income, raw_balance, raw_cashflow, raw_prices]):
        return pd.DataFrame()

    report_dates = sorted(raw_income[REPORT_DATE].dropna().unique())
    rows = []
    for rd in report_dates:
        as_of = pd.Timestamp(rd).date()
        inc = pit_ttm(raw_income,   as_of) if variant == 'Q' else pit_latest(raw_income,   as_of)
        bal = pit_latest(raw_balance,  as_of)
        cf  = pit_ttm(raw_cashflow, as_of) if variant == 'Q' else pit_latest(raw_cashflow, as_of)
        px  = pit_price(raw_prices,    as_of)
        if inc.empty or bal.empty or cf.empty or px.empty:
            continue
        row = compute_valuation(inc, bal, cf, px).join(
            compute_profitability(inc, bal, cf), how='outer',
        ).join(
            compute_momentum(raw_prices, as_of), how='outer',
        ).reset_index()
        row[REPORT_DATE] = pd.Timestamp(rd)
        rows.append(row)

    return pd.concat(rows, ignore_index=True) if rows else pd.DataFrame()

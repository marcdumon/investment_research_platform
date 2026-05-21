"""Cross-section factor computation orchestrator.

This is the only module in irp.factors that accesses the database.
Pure computation is delegated to valuation.py and profitability.py.
"""
import datetime
from typing import Literal

import pandas as pd

from irp.factors import cache as _cache
from irp.factors._cols import REPORT_DATE, TICKER
from irp.factors._pit import pit_latest, pit_price, pit_ttm
from irp.factors.backtest import compute_backtest, compute_forward_returns
from irp.factors.momentum import compute_momentum
from irp.factors.profitability import compute_profitability
from irp.factors.valuation import compute_valuation
from irp.query.simfin import fundamentals
from irp.query.yahoo import prices as yahoo_prices


def _cross_section_from_raw(
    raw_income: pd.DataFrame,
    raw_balance: pd.DataFrame,
    raw_cashflow: pd.DataFrame,
    raw_prices: pd.DataFrame,
    as_of_date: datetime.date,
    variant: Literal['A', 'Q'],
) -> pd.DataFrame:
    """PIT cross-section from pre-fetched raw DataFrames (no DB access)."""
    if variant == 'Q':
        income   = pit_ttm(raw_income,   as_of_date)
        cashflow = pit_ttm(raw_cashflow, as_of_date)
    else:
        income   = pit_latest(raw_income,   as_of_date)
        cashflow = pit_latest(raw_cashflow, as_of_date)
    balance = pit_latest(raw_balance,  as_of_date)
    prices  = pit_price(raw_prices,    as_of_date)

    if income.empty or balance.empty or cashflow.empty or prices.empty:
        return pd.DataFrame()

    val  = compute_valuation(income, balance, cashflow, prices)
    prof = compute_profitability(income, balance, cashflow)
    mom  = compute_momentum(raw_prices, as_of_date)

    result = val.join(prof, how='outer').join(mom, how='outer')
    result.index.name = TICKER
    return result


def cross_section(
    as_of_date: datetime.date,
    variant: Literal['A', 'Q'] = 'A',
    tickers: list[str] | None = None,
) -> pd.DataFrame:
    """Compute all factors cross-sectionally at a point in time.

    Only fundamental data with Report Date <= as_of_date and prices with
    Date <= as_of_date are used, making results PIT-safe.

    Parameters
    ----------
    as_of_date : Snapshot date for the cross-section.
    variant    : 'A' for annual filings, 'Q' for quarterly.
    tickers    : Restrict to specific tickers; None = full universe.

    Returns
    -------
    DataFrame indexed by Ticker with valuation, profitability, and momentum columns.
    Tickers without sufficient data are silently absent.
    """
    if tickers is None:
        cached = _cache.load(as_of_date, variant)
        if cached is not None:
            return cached

    raw_income   = fundamentals(tickers, 'income',   variant)
    raw_balance  = fundamentals(tickers, 'balance',  variant)
    raw_cashflow = fundamentals(tickers, 'cashflow', variant)
    raw_prices   = yahoo_prices(tickers, end=as_of_date.isoformat())
    result = _cross_section_from_raw(
        raw_income, raw_balance, raw_cashflow, raw_prices, as_of_date, variant
    )
    if tickers is None and not result.empty:
        _cache.store(as_of_date, variant, result)
    return result


def ticker_factor_history(
    ticker: str,
    variant: Literal['A', 'Q'] = 'A',
) -> pd.DataFrame:
    """Factor values at each historical filing date for one ticker.

    Fetches raw data once; computes PIT-aligned factors at each Report Date.
    Returns a DataFrame with columns [Ticker, Report Date, mktcap, pe, pb, ps,
    ev_ebitda, ev_ebit, ev_sales, fcf_yield, gross_margin, op_margin, net_margin,
    roe, roa, roic, fcf_margin], indexed 0..N-1.
    Returns an empty DataFrame if any required data is unavailable.
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


def run_backtest(
    factor: str,
    horizon_days: int,
    start_date: datetime.date,
    end_date: datetime.date,
    variant: Literal['A', 'Q'] = 'A',
    tickers: list[str] | None = None,
    freq: Literal['Q', 'A'] = 'Q',
) -> dict:
    """IC series and quintile cumulative returns for one factor over a date range.

    Fetches raw data once; loops over rebalance dates in memory.

    Parameters
    ----------
    factor       : Factor column name (e.g. 'pe', 'mom_12_1').
    horizon_days : Forward-return horizon in calendar days (e.g. 252 = ~12m).
    start_date   : First rebalance date (inclusive).
    end_date     : Last rebalance date (inclusive).
    variant      : 'A' annual or 'Q' quarterly fundamentals.
    tickers      : Subset of tickers; None = full universe.
    freq         : 'Q' quarterly or 'A' annual rebalance schedule.

    Returns
    -------
    dict from compute_backtest(): ic_series, quintile_cumret, mean_ic, ic_tstat, n_dates.
    """
    if tickers is None:
        cached = _cache.load_backtest(factor, horizon_days, variant, freq, start_date, end_date)
        if cached is not None:
            return cached

    raw_income   = fundamentals(tickers, 'income',   variant)
    raw_balance  = fundamentals(tickers, 'balance',  variant)
    raw_cashflow = fundamentals(tickers, 'cashflow', variant)
    raw_prices   = yahoo_prices(tickers)

    pd_freq = 'QE' if freq == 'Q' else 'YE'
    rebalance_dates = [
        ts.date()
        for ts in pd.date_range(start_date, end_date, freq=pd_freq)
    ]

    cross_sections: dict[datetime.date, pd.DataFrame] = {}
    dates_to_compute = []

    if tickers is None:
        for d in rebalance_dates:
            cached = _cache.load(d, variant)
            if cached is not None:
                cross_sections[d] = cached
            else:
                dates_to_compute.append(d)
    else:
        dates_to_compute = list(rebalance_dates)

    for d in dates_to_compute:
        xs = _cross_section_from_raw(
            raw_income, raw_balance, raw_cashflow, raw_prices, d, variant
        )
        if not xs.empty:
            cross_sections[d] = xs
            if tickers is None:
                _cache.store(d, variant, xs)

    fwd_returns = compute_forward_returns(raw_prices, rebalance_dates, horizon_days)
    result = compute_backtest(factor, cross_sections, fwd_returns)
    if tickers is None and result['n_dates'] > 0:
        _cache.store_backtest(factor, horizon_days, variant, freq, start_date, end_date, result)
    return result

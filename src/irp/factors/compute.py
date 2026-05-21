"""Cross-section factor computation orchestrator.

This is the only module in irp.factors that accesses the database.
Pure computation is delegated to valuation.py and profitability.py.
"""
import datetime
from typing import Literal

import pandas as pd

from irp.factors._cols import REPORT_DATE, TICKER
from irp.factors._pit import pit_latest, pit_price, pit_ttm
from irp.factors.profitability import compute_profitability
from irp.factors.valuation import compute_valuation
from irp.query.simfin import fundamentals
from irp.query.yahoo import prices as yahoo_prices


def cross_section(
    as_of_date: datetime.date,
    variant: Literal['A', 'Q'] = 'A',
    tickers: list[str] | None = None,
) -> pd.DataFrame:
    """Compute all phase-1 factors cross-sectionally at a point in time.

    Only fundamental data with Report Date <= as_of_date and prices with
    Date <= as_of_date are used, making results PIT-safe.

    Parameters
    ----------
    as_of_date : Snapshot date for the cross-section.
    variant    : 'A' for annual filings, 'Q' for quarterly.
    tickers    : Restrict to specific tickers; None = full universe.

    Returns
    -------
    DataFrame indexed by Ticker. Columns from valuation:
        mktcap, pe, pb, ps, ev_ebitda, ev_ebit, ev_sales, fcf_yield
    Columns from profitability:
        gross_margin, op_margin, net_margin, roe, roa, roic, fcf_margin

    Tickers without sufficient data in any required statement, or without
    a price on or before as_of_date, are silently absent (inner-join
    semantics propagate through the merge chain in each factor module).
    """
    raw_income   = fundamentals(tickers, 'income',   variant)
    raw_balance  = fundamentals(tickers, 'balance',  variant)
    raw_cashflow = fundamentals(tickers, 'cashflow', variant)
    raw_prices   = yahoo_prices(tickers, end=as_of_date.isoformat())

    if variant == 'Q':
        income   = pit_ttm(raw_income,   as_of_date)
        cashflow = pit_ttm(raw_cashflow, as_of_date)
    else:
        income   = pit_latest(raw_income,   as_of_date)
        cashflow = pit_latest(raw_cashflow, as_of_date)
    balance  = pit_latest(raw_balance,  as_of_date)
    prices   = pit_price(raw_prices,    as_of_date)

    if income.empty or balance.empty or cashflow.empty or prices.empty:
        return pd.DataFrame()

    val  = compute_valuation(income, balance, cashflow, prices)
    prof = compute_profitability(income, balance, cashflow)

    result = val.join(prof, how='outer')
    result.index.name = TICKER
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
        ).reset_index()
        row[REPORT_DATE] = pd.Timestamp(rd)
        rows.append(row)

    return pd.concat(rows, ignore_index=True) if rows else pd.DataFrame()

"""Profitability factors: gross_margin, op_margin, net_margin, roe, roa, roic, fcf_margin.

All inputs must already be PIT-aligned (one row per ticker). No DB access.
"""
import pandas as pd

from irp.factors._cols import (
    TICKER,
    REVENUE,
    GROSS_PROFIT,
    OPERATING_INCOME,
    NET_INCOME,
    TOTAL_ASSETS,
    TOTAL_EQUITY,
    CASH_AND_ST_INVESTMENTS,
    SHORT_TERM_DEBT,
    LONG_TERM_DEBT,
    CFO,
)


def _safe_div(num: pd.Series, denom: pd.Series) -> pd.Series:
    return (num / denom).replace([float('inf'), float('-inf')], pd.NA)


def compute_profitability(
    income: pd.DataFrame,
    balance: pd.DataFrame,
    cashflow: pd.DataFrame,
) -> pd.DataFrame:
    """Compute profitability factors for each ticker.

    Parameters
    ----------
    income, balance, cashflow : PIT-aligned, one row per Ticker.

    Returns
    -------
    DataFrame indexed by Ticker with columns:
        gross_margin - Gross Profit / Revenue
        op_margin    - Operating Income / Revenue
        net_margin   - Net Income / Revenue
        roe          - Net Income / Total Equity
        roa          - Net Income / Total Assets
        roic         - Operating Income / Invested Capital  (EBIT/IC, no tax adjustment)
        fcf_margin   - CFO / Revenue

    Note: ROIC uses EBIT/IC as a simplified proxy. Tax-adjusted NOPAT/IC requires
    the effective tax rate and is deferred to phase 2.

    Tickers absent from any input frame are dropped (inner-join semantics).
    Negative ratios are kept; only inf -> NA.
    """
    w = (
        income[[TICKER, REVENUE, GROSS_PROFIT, OPERATING_INCOME, NET_INCOME]]
        .merge(
            balance[[TICKER, TOTAL_ASSETS, TOTAL_EQUITY,
                     SHORT_TERM_DEBT, LONG_TERM_DEBT, CASH_AND_ST_INVESTMENTS]],
            on=TICKER, how='inner',
        )
        .merge(
            cashflow[[TICKER, CFO]],
            on=TICKER, how='inner',
        )
        .set_index(TICKER)
    )

    st_debt = w[SHORT_TERM_DEBT].fillna(0)
    lt_debt = w[LONG_TERM_DEBT].fillna(0)
    cash    = w[CASH_AND_ST_INVESTMENTS].fillna(0)
    w['net_debt'] = st_debt + lt_debt - cash
    w['ic']       = w[TOTAL_EQUITY] + w['net_debt']

    out = pd.DataFrame(index=w.index)
    out['gross_margin'] = _safe_div(w[GROSS_PROFIT],     w[REVENUE])
    out['op_margin']    = _safe_div(w[OPERATING_INCOME], w[REVENUE])
    out['net_margin']   = _safe_div(w[NET_INCOME],       w[REVENUE])
    out['roe']          = _safe_div(w[NET_INCOME],       w[TOTAL_EQUITY])
    out['roa']          = _safe_div(w[NET_INCOME],       w[TOTAL_ASSETS])
    out['roic']         = _safe_div(w[OPERATING_INCOME], w['ic'])
    out['fcf_margin']   = _safe_div(w[CFO],              w[REVENUE])
    out.index.name      = TICKER
    return out

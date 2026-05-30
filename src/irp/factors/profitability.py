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
from irp.factors._utils import _safe_div
from irp.factors.registry import _register as register

register('gross_margin',   'Gross Margin',   pct=True, group='profitability')
register('op_margin',      'Op. Margin',     pct=True, group='profitability')
register('net_margin',     'Net Margin',     pct=True, group='profitability')
register('roe',            'ROE',            pct=True, group='profitability')
register('roa',            'ROA',            pct=True, group='profitability')
register('roic',           'ROIC',           pct=True, group='profitability')
register('fcf_margin',     'FCF Margin',     pct=True, group='profitability')
register('asset_turnover', 'Asset Turnover',           group='profitability')
register('cfo_ni_ratio',   'CFO/NI',                   group='quality')
register('accruals',       'Accruals',                 group='quality')


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
    out['gross_margin']   = _safe_div(w[GROSS_PROFIT],              w[REVENUE])
    out['op_margin']      = _safe_div(w[OPERATING_INCOME],         w[REVENUE])
    out['net_margin']     = _safe_div(w[NET_INCOME],               w[REVENUE])
    out['roe']            = _safe_div(w[NET_INCOME],               w[TOTAL_EQUITY])
    out['roa']            = _safe_div(w[NET_INCOME],               w[TOTAL_ASSETS])
    out['roic']           = _safe_div(w[OPERATING_INCOME],         w['ic'])
    out['fcf_margin']     = _safe_div(w[CFO],                      w[REVENUE])
    out['asset_turnover'] = _safe_div(w[REVENUE],                  w[TOTAL_ASSETS])
    out['cfo_ni_ratio']   = _safe_div(w[CFO],                      w[NET_INCOME])
    out['accruals']       = _safe_div(w[NET_INCOME] - w[CFO],      w[TOTAL_ASSETS])
    out.index.name        = TICKER
    return out

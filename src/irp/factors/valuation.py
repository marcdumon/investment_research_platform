"""Valuation factors: mktcap, pe, pb, ps, ev_ebitda, ev_ebit, ev_sales, fcf_yield.

All inputs must already be PIT-aligned (one row per ticker). No DB access.
"""

import numpy as np
import pandas as pd

from irp.factors._cols import (
    CASH_AND_ST_INVESTMENTS,
    CFI,
    CFO,
    DA,
    LONG_TERM_DEBT,
    NET_INCOME,
    OPERATING_INCOME,
    PRICE_CLOSE,
    REVENUE,
    SHARES_DILUTED_BAL,
    SHORT_TERM_DEBT,
    TICKER,
    TOTAL_EQUITY,
)
from irp.factors._utils import _safe_div
from irp.factors.registry import _register as register

register('mktcap',          'Mkt Cap ($B)',         group='size')
register('revenue',         'Revenue ($B)',         group='fundamentals')
register('net_income',      'Net Income ($B)',      group='fundamentals')
register('total_assets',    'Total Assets ($B)',    group='fundamentals')
register('total_equity',    'Total Equity ($B)',    group='fundamentals')
register('op_cashflow',     'Op. Cash Flow ($B)',   group='fundamentals')
register('pe',              'P/E',                  group='valuation', positive_only=True)
register('pb',              'P/B',                  group='valuation', positive_only=True)
register('ps',              'P/S',                  group='valuation')
register('ev_ebitda',       'EV/EBITDA',            group='valuation', positive_only=True)
register('ev_ebit',         'EV/EBIT',              group='valuation', positive_only=True)
register('ev_sales',        'EV/Sales',             group='valuation')
register('fcf_yield',       'FCF Yield',  pct=True, group='valuation')
register('rand',            'Random',               group='valuation')


def compute_valuation(
    income: pd.DataFrame,
    balance: pd.DataFrame,
    cashflow: pd.DataFrame,
    prices: pd.DataFrame,
) -> pd.DataFrame:
    """Compute valuation factors for each ticker.

    Parameters
    ----------
    income, balance, cashflow : PIT-aligned, one row per Ticker.
    prices : PIT-aligned price frame with columns [Ticker, Close].

    Returns
    -------
    DataFrame indexed by Ticker with columns:
        mktcap    - market capitalisation (price * diluted shares)
        pe        - P / E  (mktcap / net_income)
        pb        - P / B  (mktcap / total_equity)
        ps        - P / S  (mktcap / revenue)
        ev_ebitda - EV / EBITDA
        ev_ebit   - EV / Operating Income
        ev_sales  - EV / Revenue
        fcf_yield - (CFO + CFI) / mktcap

    Tickers absent from any input frame are dropped (inner-join semantics).
    Negative ratios (e.g. pb with negative equity) are kept; only inf -> NA.
    """
    w = (
        prices[[TICKER, PRICE_CLOSE]]
        .merge(
            income[[TICKER, NET_INCOME, REVENUE, OPERATING_INCOME]],
            on=TICKER,
            how='inner',
        )
        .merge(
            balance[
                [
                    TICKER,
                    TOTAL_EQUITY,
                    CASH_AND_ST_INVESTMENTS,
                    SHORT_TERM_DEBT,
                    LONG_TERM_DEBT,
                    SHARES_DILUTED_BAL,
                ]
            ],
            on=TICKER,
            how='inner',
        )
        .merge(
            cashflow[[TICKER, CFO, CFI, DA]],
            on=TICKER,
            how='inner',
        )
        .set_index(TICKER)
    )

    w['mktcap'] = w[PRICE_CLOSE] * w[SHARES_DILUTED_BAL]

    st_debt = w[SHORT_TERM_DEBT].fillna(0)
    lt_debt = w[LONG_TERM_DEBT].fillna(0)
    cash = w[CASH_AND_ST_INVESTMENTS].fillna(0)
    w['net_debt'] = st_debt + lt_debt - cash
    w['ev'] = w['mktcap'] + w['net_debt']

    # DA from cashflow is positive; no abs() needed.
    w['ebitda'] = w[OPERATING_INCOME] + w[DA].fillna(0)
    w['fcf'] = w[CFO] + w[CFI]

    out = pd.DataFrame(index=w.index)
    out['mktcap'] = w['mktcap']
    out['pe'] = _safe_div(w['mktcap'], w[NET_INCOME])
    out['pb'] = _safe_div(w['mktcap'], w[TOTAL_EQUITY])
    out['ps'] = _safe_div(w['mktcap'], w[REVENUE])
    out['ev_ebitda'] = _safe_div(w['ev'], w['ebitda'])
    out['ev_ebit'] = _safe_div(w['ev'], w[OPERATING_INCOME])
    out['ev_sales'] = _safe_div(w['ev'], w[REVENUE])
    out['fcf_yield'] = _safe_div(w['fcf'], w['mktcap'])
    out['rand'] = np.random.rand(out.shape[0])

    out.index.name = TICKER
    return out

"""Leverage factors: debt_equity, net_debt_ebitda, interest_coverage.

All inputs must already be PIT-aligned (one row per ticker). No DB access.
"""
import pandas as pd

from irp.factors._cols import (
    CASH_AND_ST_INVESTMENTS,
    DA,
    INTEREST_EXPENSE,
    LONG_TERM_DEBT,
    OPERATING_INCOME,
    SHORT_TERM_DEBT,
    TICKER,
    TOTAL_EQUITY,
)
from irp.factors._utils import safe_div
from irp.factors.registry import register

register('debt_equity',      'Debt/Equity',      group='leverage')
register('net_debt_ebitda',  'Net Debt/EBITDA',  group='leverage')
register('interest_coverage','Int. Coverage',     group='leverage')


def compute_leverage(
    income: pd.DataFrame,
    balance: pd.DataFrame,
    cashflow: pd.DataFrame,
) -> pd.DataFrame:
    """Compute leverage factors for each ticker.

    Parameters
    ----------
    income, balance, cashflow : PIT-aligned, one row per Ticker.

    Returns
    -------
    DataFrame indexed by Ticker with columns:
        debt_equity       - (ST Debt + LT Debt) / Total Equity
        net_debt_ebitda   - (Debt - Cash) / EBITDA  (EBITDA = EBIT + D&A)
        interest_coverage - EBIT / |Interest Expense|  (higher = safer)

    Interest Expense is stored as a negative value in SimFin income statements;
    abs() is applied before dividing so coverage is positive for profitable firms.
    Negative EBITDA or zero interest expense yield NA via safe_div.
    """
    inc_cols = [TICKER, OPERATING_INCOME]
    if INTEREST_EXPENSE in income.columns:
        inc_cols.append(INTEREST_EXPENSE)

    w = (
        income[inc_cols]
        .merge(
            balance[[TICKER, TOTAL_EQUITY, CASH_AND_ST_INVESTMENTS,
                     SHORT_TERM_DEBT, LONG_TERM_DEBT]],
            on=TICKER, how='inner',
        )
        .merge(
            cashflow[[TICKER, DA]],
            on=TICKER, how='inner',
        )
        .set_index(TICKER)
    )

    st_debt    = w[SHORT_TERM_DEBT].fillna(0)
    lt_debt    = w[LONG_TERM_DEBT].fillna(0)
    cash       = w[CASH_AND_ST_INVESTMENTS].fillna(0)
    total_debt = st_debt + lt_debt
    net_debt   = total_debt - cash
    ebitda     = w[OPERATING_INCOME] + w[DA].fillna(0)

    out = pd.DataFrame(index=w.index)
    out['debt_equity']     = safe_div(total_debt, w[TOTAL_EQUITY])
    out['net_debt_ebitda'] = safe_div(net_debt, ebitda)

    if INTEREST_EXPENSE in w.columns:
        interest_paid = w[INTEREST_EXPENSE].abs()
        out['interest_coverage'] = safe_div(w[OPERATING_INCOME], interest_paid)
    else:
        out['interest_coverage'] = pd.NA

    out.index.name = TICKER
    return out

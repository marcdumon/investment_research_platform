"""Growth factors: year-over-year revenue and earnings growth.

Takes raw (full-history) DataFrames and as_of_date, same pattern as momentum.py.
Calls pit_ttm/pit_latest twice (current + prior year) on the pre-prepared frames.
No DB access — pure computation.
"""
import datetime

import pandas as pd

from irp.factors._cols import NET_INCOME, REVENUE, TICKER
from irp.factors._pit import _pit_ttm, pit_latest
from irp.factors._utils import safe_div
from irp.factors.registry import register

register('rev_growth_1y',  'Rev Growth 1Y',  pct=True, group='growth')
register('earn_growth_1y', 'Earn Growth 1Y', pct=True, group='growth')


def compute_growth(
    raw_income: pd.DataFrame,
    raw_cashflow: pd.DataFrame,
    as_of_date: datetime.date,
    variant: str,
) -> pd.DataFrame:
    """Year-over-year growth factors.

    Parameters
    ----------
    raw_income   : Full income-statement history (all dates).
    raw_cashflow : Full cashflow history (reserved for future FCF growth factor).
    as_of_date   : Snapshot date for the cross-section.
    variant      : 'Q' → TTM comparison; 'A' → latest annual filing comparison.

    Returns
    -------
    DataFrame indexed by Ticker with columns:
        rev_growth_1y   (Revenue_curr - Revenue_prior) / |Revenue_prior|
        earn_growth_1y  (NI_curr - NI_prior) / |NI_prior|

    Only tickers with both current and prior-year data are returned.
    """
    prior_date = (pd.Timestamp(as_of_date) - pd.DateOffset(years=1)).date()
    _pit = _pit_ttm if variant == 'Q' else pit_latest

    curr_inc  = _pit(raw_income, as_of_date)
    prior_inc = _pit(raw_income, prior_date)

    if curr_inc.empty or prior_inc.empty:
        return pd.DataFrame(columns=['rev_growth_1y', 'earn_growth_1y'])

    curr = (
        curr_inc[[TICKER, REVENUE, NET_INCOME]]
        .rename(columns={REVENUE: '_rev_c', NET_INCOME: '_ni_c'})
    )
    prior = (
        prior_inc[[TICKER, REVENUE, NET_INCOME]]
        .rename(columns={REVENUE: '_rev_p', NET_INCOME: '_ni_p'})
    )
    w = curr.merge(prior, on=TICKER, how='inner').set_index(TICKER)

    out = pd.DataFrame(index=w.index)
    out['rev_growth_1y']  = safe_div(w['_rev_c'] - w['_rev_p'], w['_rev_p'].abs())
    out['earn_growth_1y'] = safe_div(w['_ni_c']  - w['_ni_p'],  w['_ni_p'].abs())
    out.index.name = TICKER
    return out

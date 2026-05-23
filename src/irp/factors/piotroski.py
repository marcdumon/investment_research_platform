"""Piotroski F-Score: 9 binary accounting signals summed to a score 0–9.

Pattern follows growth.py: takes raw (full-history) DataFrames + as_of_date,
calls pit functions at both current and prior year. No DB access.

Signals:
  Profitability:  F1 ROA>0, F2 CFO>0, F3 ΔROA>0, F4 CFO/Assets>ROA
  Leverage:       F5 ΔLT-leverage<0
  Liquidity:      F6 ΔCurrent-ratio>0
  Funding:        F7 no dilution (shares didn't increase)
  Efficiency:     F8 ΔGross-margin>0, F9 ΔAsset-turnover>0

NaN inputs propagate to NaN signals; score uses min_count=1 (partial scores
allowed when prior-year data is unavailable, e.g. newly listed companies).
"""
import datetime

import numpy as np
import pandas as pd

from irp.factors._cols import (
    CFO,
    GROSS_PROFIT,
    LONG_TERM_DEBT,
    NET_INCOME,
    REVENUE,
    SHARES_DILUTED_BAL,
    TICKER,
    TOTAL_ASSETS,
    TOTAL_CURRENT_ASSETS,
    TOTAL_CURRENT_LIABILITIES,
)
from irp.factors._pit import pit_latest, pit_ttm
from irp.factors._utils import _safe_div
from irp.factors.registry import register

register('piotroski_fscore', 'Piotroski F', group='quality')


def _gt0(s: pd.Series) -> pd.Series:
    """1.0 if s > 0, 0.0 if s <= 0, NaN if s is NaN."""
    r = (s > 0).astype(float)
    r[s.isna()] = np.nan
    return r


def _improved(curr: pd.Series, prior: pd.Series) -> pd.Series:
    """1.0 if curr > prior, 0.0 if not, NaN if either is NaN."""
    diff = curr - prior
    r = (diff > 0).astype(float)
    r[curr.isna() | prior.isna()] = np.nan
    return r


def compute_piotroski(
    raw_income: pd.DataFrame,
    raw_balance: pd.DataFrame,
    raw_cashflow: pd.DataFrame,
    as_of_date: datetime.date,
    variant: str,
) -> pd.DataFrame:
    """Compute Piotroski F-Score for each ticker.

    Parameters
    ----------
    raw_income, raw_balance, raw_cashflow : Full-history DataFrames, pit_prepare'd or raw.
    as_of_date : Snapshot date.
    variant    : 'Q' → TTM flow statements; 'A' → latest annual filing.

    Returns
    -------
    DataFrame indexed by Ticker with column `piotroski_fscore` (float 0–9 or NaN).
    """
    prior_date = (pd.Timestamp(as_of_date) - pd.DateOffset(years=1)).date()
    _pit_flow = pit_ttm if variant == 'Q' else pit_latest

    curr_inc = _pit_flow(raw_income,   as_of_date)
    curr_cf  = _pit_flow(raw_cashflow, as_of_date)
    curr_bal = pit_latest(raw_balance,  as_of_date)
    prior_inc = _pit_flow(raw_income,   prior_date)
    prior_bal = pit_latest(raw_balance,  prior_date)

    if curr_bal.empty or curr_inc.empty or curr_cf.empty:
        return pd.DataFrame(columns=['piotroski_fscore'])

    ci  = curr_inc.set_index(TICKER)
    cf  = curr_cf.set_index(TICKER)
    cb  = curr_bal.set_index(TICKER)
    pi  = prior_inc.set_index(TICKER) if not prior_inc.empty else pd.DataFrame()
    pb  = prior_bal.set_index(TICKER) if not prior_bal.empty else pd.DataFrame()

    tickers = cb.index

    def _get(frame: pd.DataFrame, col: str, fill: float | None = None) -> pd.Series:
        if frame.empty or col not in frame.columns:
            return pd.Series(np.nan, index=tickers)
        s = frame[col].reindex(tickers)
        return s.fillna(fill) if fill is not None else s

    # Current-period series
    ni   = _get(ci, NET_INCOME)
    rev  = _get(ci, REVENUE)
    gp   = _get(ci, GROSS_PROFIT)
    ta   = _get(cb, TOTAL_ASSETS)
    ltd  = _get(cb, LONG_TERM_DEBT, fill=0.0)
    ca   = _get(cb, TOTAL_CURRENT_ASSETS)
    cl   = _get(cb, TOTAL_CURRENT_LIABILITIES)
    sh   = _get(cb, SHARES_DILUTED_BAL)
    cfo  = _get(cf, CFO)

    # Prior-period series
    ni_p  = _get(pi, NET_INCOME)
    rev_p = _get(pi, REVENUE)
    gp_p  = _get(pi, GROSS_PROFIT)
    ta_p  = _get(pb, TOTAL_ASSETS)
    ltd_p = _get(pb, LONG_TERM_DEBT, fill=0.0)
    ca_p  = _get(pb, TOTAL_CURRENT_ASSETS)
    cl_p  = _get(pb, TOTAL_CURRENT_LIABILITIES)
    sh_p  = _get(pb, SHARES_DILUTED_BAL)

    # Derived ratios
    roa   = _safe_div(ni,   ta)
    roa_p = _safe_div(ni_p, ta_p)
    lev   = _safe_div(ltd,  ta)
    lev_p = _safe_div(ltd_p, ta_p)
    cr    = _safe_div(ca,   cl)
    cr_p  = _safe_div(ca_p, cl_p)
    gm    = _safe_div(gp,   rev)
    gm_p  = _safe_div(gp_p, rev_p)
    at    = _safe_div(rev,  ta)
    at_p  = _safe_div(rev_p, ta_p)

    signals = pd.DataFrame({
        'f1': _gt0(roa),
        'f2': _gt0(cfo),
        'f3': _improved(roa, roa_p),
        'f4': _gt0(_safe_div(cfo, ta) - roa),   # accruals: CFO/Assets > ROA
        'f5': _gt0(lev_p - lev),                 # leverage decreased
        'f6': _improved(cr, cr_p),
        'f7': _gt0(sh_p - sh),                   # no new shares (prior >= current)
        'f8': _improved(gm, gm_p),
        'f9': _improved(at, at_p),
    }, index=tickers)

    out = pd.DataFrame(index=tickers)
    out['piotroski_fscore'] = signals.sum(axis=1, min_count=1)
    out.index.name = TICKER
    return out

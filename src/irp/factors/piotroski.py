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
from irp.factors._pit import _pit_ttm, pit_latest
from irp.factors._utils import safe_div
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
    raw_income, raw_balance, raw_cashflow : Full-history DataFrames.
    as_of_date : Snapshot date.
    variant    : 'Q' → TTM flow statements; 'A' → latest annual filing.

    Returns
    -------
    DataFrame indexed by Ticker with column `piotroski_fscore` (float 0–9 or NaN).
    """
    prior_date = (pd.Timestamp(as_of_date) - pd.DateOffset(years=1)).date()
    _pit_flow = _pit_ttm if variant == 'Q' else pit_latest

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
    roa   = safe_div(ni,   ta)
    roa_p = safe_div(ni_p, ta_p)
    lev   = safe_div(ltd,  ta)
    lev_p = safe_div(ltd_p, ta_p)
    cr    = safe_div(ca,   cl)
    cr_p  = safe_div(ca_p, cl_p)
    gm    = safe_div(gp,   rev)
    gm_p  = safe_div(gp_p, rev_p)
    at    = safe_div(rev,  ta)
    at_p  = safe_div(rev_p, ta_p)

    signals = pd.DataFrame({
        'f1': _gt0(roa),
        'f2': _gt0(cfo),
        'f3': _improved(roa, roa_p),
        'f4': _gt0(safe_div(cfo, ta) - roa),   # accruals: CFO/Assets > ROA
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


def _compute_piotroski_panel(df: pd.DataFrame) -> pd.Series:
    """Piotroski F-Score from a combined cross-section DataFrame.

    Takes the assembled per-ticker frame used by `cross_section_panel`
    (one row per ticker, columns include current + prior year values
    already PIT-aligned). Returns a Series of f-scores 0..9 (or NaN
    when all signals are missing).

    Required columns (current): ni, ta, rev, gp, cl, ca, cfo, shares, lt_debt
    Required columns (prior):   ni_p, ta_p, rev_p, gp_p, cl_p, ca_p, shares_p, ltd_p
    """
    def _binary(condition: pd.Series, gate: pd.Series) -> pd.Series:
        out = pd.Series(np.nan, index=condition.index, dtype='float64')
        out[gate & condition] = 1.0
        out[gate & ~condition] = 0.0
        return out

    ta_ok   = df['ta'].notna()   & (df['ta']   != 0)
    tap_ok  = df['ta_p'].notna() & (df['ta_p'] != 0)
    rev_ok  = df['rev'].notna()  & (df['rev']  != 0)
    revp_ok = df['rev_p'].notna() & (df['rev_p'] != 0)
    cl_ok   = df['cl'].notna()   & (df['cl']   != 0)
    clp_ok  = df['cl_p'].notna() & (df['cl_p'] != 0)

    f1 = _binary((df['ni'] / df['ta']) > 0, ta_ok & df['ni'].notna())
    f2 = _binary(df['cfo'] > 0, df['cfo'].notna())
    f3 = _binary((df['ni'] / df['ta']) > (df['ni_p'] / df['ta_p']),
                 ta_ok & tap_ok & df['ni_p'].notna())
    f4 = _binary((df['cfo'] / df['ta']) > (df['ni'] / df['ta']),
                 ta_ok & df['cfo'].notna() & df['ni'].notna())
    f5 = _binary((df['lt_debt'] / df['ta']) < (df['ltd_p'] / df['ta_p']),
                 ta_ok & tap_ok)
    f6 = _binary((df['ca'] / df['cl']) > (df['ca_p'] / df['cl_p']),
                 cl_ok & clp_ok & df['ca'].notna() & df['ca_p'].notna())
    f7 = _binary(df['shares'] < df['shares_p'], df['shares_p'].notna())
    f8 = _binary((df['gp'] / df['rev']) > (df['gp_p'] / df['rev_p']),
                 rev_ok & revp_ok & df['gp_p'].notna())
    f9 = _binary((df['rev'] / df['ta']) > (df['rev_p'] / df['ta_p']),
                 ta_ok & tap_ok & df['rev_p'].notna())

    sig_stack = pd.concat([f1, f2, f3, f4, f5, f6, f7, f8, f9], axis=1)
    all_null = sig_stack.isna().all(axis=1)
    score = sig_stack.fillna(0).sum(axis=1)
    return score.where(~all_null, np.nan)

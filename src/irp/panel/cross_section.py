"""Factor cross-section computation on panel data.

Output schema: DataFrame indexed by Ticker with all factor columns.

Pipeline stages (all small in-memory DataFrames):
    1. `_pit_align`        — current + prior-year fundamentals (polars → pandas)
    2. `_price_snapshots`  — close at as_of plus 1m/6m/12m lookbacks + vol/ma200
    3. `_assemble`         — combine pit + prices into one wide pandas frame
    4. `_apply_formulas`   — pure pandas factor calculations
    5. Piotroski signals delegated to `irp.factors.piotroski.compute_piotroski_panel`
"""
import datetime
import logging
import warnings
from dataclasses import dataclass
from typing import Literal

import numpy as np
import pandas as pd
import polars as pl
from dateutil.relativedelta import relativedelta

from irp.panel.load import load_prices_wide, load_fundamentals
from irp.panel.pit import pit_latest, pit_ttm, pit_price_row, pit_price_at_offset

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Column lists (TTM sums — Q variant). Balance always uses latest filing.
# ---------------------------------------------------------------------------

_INCOME_TTM_COLS = [
    'Revenue', 'Gross Profit', 'Operating Income (Loss)', 'Net Income',
    'Interest Expense, Net', 'Depreciation & Amortization',
]
_CASHFLOW_TTM_COLS = [
    'Net Cash from Operating Activities', 'Net Cash from Investing Activities',
    'Depreciation & Amortization',
]

_INCOME_COLS = [
    'Ticker', 'Revenue', 'Gross Profit', 'Operating Income (Loss)',
    'Net Income', 'Interest Expense, Net', 'Depreciation & Amortization',
]
_CASHFLOW_COLS = [
    'Ticker', 'Net Cash from Operating Activities',
    'Net Cash from Investing Activities', 'Depreciation & Amortization',
]
_BALANCE_COLS = [
    'Ticker', 'Total Assets', 'Total Equity',
    'Total Current Assets', 'Total Current Liabilities',
    'Cash, Cash Equivalents & Short Term Investments',
    'Short Term Debt', 'Long Term Debt', 'Shares (Diluted)',
]

_FACTOR_COLS_ORDER = [
    'mktcap', 'revenue', 'net_income', 'total_assets', 'total_equity', 'op_cashflow',
    'pe', 'pb', 'ps', 'ev_ebitda', 'ev_ebit', 'ev_sales', 'fcf_yield', 'rand',
    'gross_margin', 'op_margin', 'net_margin', 'roe', 'roa', 'roic', 'fcf_margin',
    'asset_turnover', 'cfo_ni_ratio', 'accruals',
    'debt_equity', 'net_debt_ebitda', 'interest_coverage',
    'mom_12_1', 'mom_6_1', 'vol_21d', 'ma200_ratio',
    'rev_growth_1y', 'earn_growth_1y',
    'piotroski_fscore',
]


# ---------------------------------------------------------------------------
# Per-statement PIT (variant-aware)
# ---------------------------------------------------------------------------

def _income_pit(df: pl.DataFrame, as_of: datetime.date, variant: Literal['A', 'Q']) -> pl.DataFrame:
    if variant == 'A':
        return pit_latest(df, as_of).select(_INCOME_COLS)
    return pit_ttm(df, as_of, _INCOME_TTM_COLS, n=4)


def _cashflow_pit(df: pl.DataFrame, as_of: datetime.date, variant: Literal['A', 'Q']) -> pl.DataFrame:
    if variant == 'A':
        return pit_latest(df, as_of).select(_CASHFLOW_COLS)
    return pit_ttm(df, as_of, _CASHFLOW_TTM_COLS, n=4)


def _balance_pit(df: pl.DataFrame, as_of: datetime.date) -> pl.DataFrame:
    return pit_latest(df, as_of).select(_BALANCE_COLS)


# ---------------------------------------------------------------------------
# Stage 1 — fundamentals at as_of + prior year
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class PitFundamentals:
    income:   pd.DataFrame
    balance:  pd.DataFrame
    cashflow: pd.DataFrame


def _pit_align(
    as_of: datetime.date, variant: Literal['A', 'Q'],
) -> tuple[PitFundamentals, PitFundamentals]:
    """Return PIT-aligned fundamentals at `as_of` and one year prior."""
    inc_lf = load_fundamentals('income',   variant)
    bal_lf = load_fundamentals('balance',  variant)
    cf_lf  = load_fundamentals('cashflow', variant)

    prior = as_of - relativedelta(years=1)

    curr = PitFundamentals(
        income   = _income_pit(inc_lf, as_of, variant).to_pandas().set_index('Ticker'),
        balance  = _balance_pit(bal_lf, as_of).to_pandas().set_index('Ticker'),
        cashflow = _cashflow_pit(cf_lf, as_of, variant).to_pandas().set_index('Ticker'),
    )
    prev = PitFundamentals(
        income   = _income_pit(inc_lf, prior, variant).to_pandas().set_index('Ticker'),
        balance  = _balance_pit(bal_lf, prior).to_pandas().set_index('Ticker'),
        cashflow = _cashflow_pit(cf_lf, prior, variant).to_pandas().set_index('Ticker'),
    )
    return curr, prev




# ---------------------------------------------------------------------------
# Stage 2 — price snapshots
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class PriceSnapshots:
    p0:      dict[str, float]
    p_1m:    dict[str, float]
    p_6m:    dict[str, float]
    p_12m:   dict[str, float]
    vol_21d: dict[str, float]
    ma200:   dict[str, float]


def _price_snapshots(as_of: datetime.date) -> PriceSnapshots:
    """Close price at `as_of` + calendar-day lookbacks + window stats."""
    panel = load_prices_wide('Close')
    vol, ma = _window_stats(panel, as_of)
    return PriceSnapshots(
        p0      = pit_price_row(panel, as_of),
        p_1m    = pit_price_at_offset(panel, as_of, 30),
        p_6m    = pit_price_at_offset(panel, as_of, 182),
        p_12m   = pit_price_at_offset(panel, as_of, 365),
        vol_21d = vol,
        ma200   = ma,
    )


def _window_stats(
    panel, as_of: datetime.date,
) -> tuple[dict[str, float], dict[str, float]]:
    """vol_21d = stddev(daily log returns, last 22 trading days) × sqrt(252).
    ma200  = mean of last 200 trading-day closes. Both NaN if insufficient obs.
    Union-calendar safe: counts only non-NaN per ticker within a wider window.
    """
    i = int(np.searchsorted(panel.dates, np.datetime64(as_of, 'D'), side='right')) - 1
    if i < 0:
        return {}, {}

    # vol_21d: take last 22 non-NaN closes per ticker within a 50-row window
    vol_start = max(0, i + 1 - 50)
    vol_slice = panel.values[vol_start:i + 1].astype('float64')
    valid_c = np.isfinite(vol_slice)
    cum_c = np.flip(np.cumsum(np.flip(valid_c, axis=0), axis=0), axis=0)
    keep = (cum_c >= 1) & (cum_c <= 22) & valid_c
    n_tickers = vol_slice.shape[1]
    compact = np.full((22, n_tickers), np.nan, dtype='float64')
    for col in range(n_tickers):
        idx = np.flatnonzero(keep[:, col])
        if idx.size == 22:
            compact[:, col] = vol_slice[idx, col]
    with np.errstate(divide='ignore', invalid='ignore'):
        log_ret = np.log(compact[1:] / compact[:-1])
        log_ret[~np.isfinite(log_ret)] = np.nan
    with warnings.catch_warnings(), np.errstate(invalid='ignore'):
        warnings.simplefilter('ignore', RuntimeWarning)
        sd = np.nanstd(log_ret, axis=0, ddof=0)
    vol = sd * np.sqrt(252)
    vol[~np.isfinite(vol)] = np.nan

    # ma200: last 200 non-NaN closes per ticker within a 400-row window
    ma_start = max(0, i + 1 - 400)
    ma_slice = panel.values[ma_start:i + 1].astype('float64')
    valid = np.isfinite(ma_slice)
    cum_from_end = np.flip(np.cumsum(np.flip(valid, axis=0), axis=0), axis=0)
    mask = (cum_from_end >= 1) & (cum_from_end <= 200) & valid
    sums = np.where(mask, ma_slice, 0.0).sum(axis=0)
    counts = mask.sum(axis=0)
    with np.errstate(invalid='ignore', divide='ignore'):
        ma200 = np.where(counts >= 200, sums / counts, np.nan)

    return dict(zip(panel.tickers, vol)), dict(zip(panel.tickers, ma200))


# ---------------------------------------------------------------------------
# Stage 3 — assemble combined working frame
# ---------------------------------------------------------------------------

def _assemble(
    curr: PitFundamentals, prev: PitFundamentals, snaps: PriceSnapshots,
) -> pd.DataFrame:
    """Inner-join required statements + prices on Ticker; left-join prior/momentum."""
    p0 = pd.Series(snaps.p0, name='p0', dtype='float64')
    if p0.empty:
        return pd.DataFrame()

    tickers_common = (
        curr.income.index
        .intersection(curr.balance.index)
        .intersection(curr.cashflow.index)
        .intersection(p0.index)
    )
    if len(tickers_common) == 0:
        return pd.DataFrame()

    df = pd.DataFrame(index=tickers_common)
    df.index.name = 'Ticker'

    # Income (current)
    df['rev']  = curr.income['Revenue'].astype('float64')
    df['gp']   = curr.income['Gross Profit'].astype('float64')
    df['ebit'] = curr.income['Operating Income (Loss)'].astype('float64')
    df['ni']   = curr.income['Net Income'].astype('float64')
    df['interest_exp'] = curr.income['Interest Expense, Net'].fillna(0).astype('float64')

    # Balance (current)
    df['ta']      = curr.balance['Total Assets'].astype('float64')
    df['equity']  = curr.balance['Total Equity'].astype('float64')
    df['cash']    = curr.balance['Cash, Cash Equivalents & Short Term Investments'].fillna(0).astype('float64')
    df['st_debt'] = curr.balance['Short Term Debt'].fillna(0).astype('float64')
    df['lt_debt'] = curr.balance['Long Term Debt'].fillna(0).astype('float64')
    df['ca']      = curr.balance['Total Current Assets'].astype('float64')
    df['cl']      = curr.balance['Total Current Liabilities'].astype('float64')
    df['shares']  = curr.balance['Shares (Diluted)'].astype('float64')

    # Cashflow (current) — D&A always from cashflow
    df['cfo'] = curr.cashflow['Net Cash from Operating Activities'].astype('float64')
    df['cfi'] = curr.cashflow['Net Cash from Investing Activities'].astype('float64')
    df['da']  = curr.cashflow['Depreciation & Amortization'].fillna(0).astype('float64')

    # Prices (current + lookbacks) — LEFT join, NaN if missing
    df['p0']    = p0.reindex(df.index).astype('float64')
    df['p_1m']  = pd.Series(snaps.p_1m).reindex(df.index).astype('float64')
    df['p_6m']  = pd.Series(snaps.p_6m).reindex(df.index).astype('float64')
    df['p_12m'] = pd.Series(snaps.p_12m).reindex(df.index).astype('float64')
    df['vol_21d_raw'] = pd.Series(snaps.vol_21d).reindex(df.index).astype('float64')
    df['ma200']       = pd.Series(snaps.ma200).reindex(df.index).astype('float64')

    # Prior-year (LEFT joins — NaN if missing for that ticker)
    df['rev_p']    = prev.income['Revenue'].reindex(df.index).astype('float64')
    df['ni_p']     = prev.income['Net Income'].reindex(df.index).astype('float64')
    df['gp_p']     = prev.income['Gross Profit'].reindex(df.index).astype('float64')
    df['ta_p']     = prev.balance['Total Assets'].reindex(df.index).astype('float64')
    df['ltd_p']    = prev.balance['Long Term Debt'].fillna(0).reindex(df.index).astype('float64')
    df['ca_p']     = prev.balance['Total Current Assets'].reindex(df.index).astype('float64')
    df['cl_p']     = prev.balance['Total Current Liabilities'].reindex(df.index).astype('float64')
    df['shares_p'] = prev.balance['Shares (Diluted)'].reindex(df.index).astype('float64')
    df['cfo_p']    = prev.cashflow['Net Cash from Operating Activities'].reindex(df.index).astype('float64')

    return df


# ---------------------------------------------------------------------------
# Stage 4 — factor formulas (pure pandas/numpy)
# ---------------------------------------------------------------------------

def _safe_log_ratio(num: pd.Series, denom: pd.Series) -> pd.Series:
    valid = (num > 0) & (denom > 0)
    result = pd.Series(np.nan, index=num.index, dtype='float64')
    result[valid] = np.log(num[valid] / denom[valid])
    return result


def _div(num, denom):
    out = num / denom.replace(0, np.nan)
    return out.where(np.isfinite(out), np.nan)


def _apply_formulas(df: pd.DataFrame) -> pd.DataFrame:
    """All factor formulas on the assembled frame; returns the final cross-section."""
    out = pd.DataFrame(index=df.index)
    mktcap = df['p0'] * df['shares']
    out['mktcap']       = mktcap / 1e9
    out['revenue']      = df['rev']    / 1e9
    out['net_income']   = df['ni']     / 1e9
    out['total_assets'] = df['ta']     / 1e9
    out['total_equity'] = df['equity'] / 1e9
    out['op_cashflow']  = df['cfo']    / 1e9

    # Valuation
    out['pe']        = _div(mktcap, df['ni'])
    out['pb']        = _div(mktcap, df['equity'])
    out['ps']        = _div(mktcap, df['rev'])
    ev = mktcap + df['st_debt'] + df['lt_debt'] - df['cash']
    out['ev_ebitda'] = _div(ev, df['ebit'] + df['da'])
    out['ev_ebit']   = _div(ev, df['ebit'])
    out['ev_sales']  = _div(ev, df['rev'])
    out['fcf_yield'] = _div(df['cfo'] + df['cfi'], mktcap)
    out['rand']      = np.random.default_rng().random(len(df))

    # Profitability
    out['gross_margin']   = _div(df['gp'],   df['rev'])
    out['op_margin']      = _div(df['ebit'], df['rev'])
    out['net_margin']     = _div(df['ni'],   df['rev'])
    out['roe']            = _div(df['ni'],   df['equity'])
    out['roa']            = _div(df['ni'],   df['ta'])
    out['roic']           = _div(df['ebit'], df['equity'] + df['st_debt'] + df['lt_debt'] - df['cash'])
    out['fcf_margin']     = _div(df['cfo'],  df['rev'])
    out['asset_turnover'] = _div(df['rev'],  df['ta'])
    out['cfo_ni_ratio']   = _div(df['cfo'],  df['ni'])
    out['accruals']       = _div(df['ni'] - df['cfo'], df['ta'])

    # Leverage
    out['debt_equity']       = _div(df['st_debt'] + df['lt_debt'], df['equity'])
    out['net_debt_ebitda']   = _div(df['st_debt'] + df['lt_debt'] - df['cash'], df['ebit'] + df['da'])
    out['interest_coverage'] = _div(df['ebit'], df['interest_exp'].abs())

    # Momentum
    out['mom_12_1']    = _safe_log_ratio(df['p_1m'], df['p_12m'])
    out['mom_6_1']     = _safe_log_ratio(df['p_1m'], df['p_6m'])
    out['vol_21d']     = df['vol_21d_raw']
    out['ma200_ratio'] = _div(df['p0'], df['ma200'])

    # Growth
    out['rev_growth_1y']  = _div(df['rev'] - df['rev_p'], df['rev_p'].abs())
    out['earn_growth_1y'] = _div(df['ni']  - df['ni_p'],  df['ni_p'].abs())

    # Piotroski — shared impl with single-ticker path
    from irp.factors.piotroski import compute_piotroski_panel
    out['piotroski_fscore'] = compute_piotroski_panel(df)

    return out[_FACTOR_COLS_ORDER]


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------

def cross_section_panel(
    as_of_date: datetime.date,
    variant: Literal['A', 'Q'] = 'A',
    tickers: list[str] | None = None,
) -> pd.DataFrame:
    """Compute all factors cross-sectionally at a point in time.

    Output: DataFrame indexed by Ticker with columns in `_FACTOR_COLS_ORDER`.
    Empty DataFrame when no ticker satisfies the inner-join constraints.
    """
    logger.debug(f'cross_section_panel: {as_of_date} {variant}')
    curr, prev = _pit_align(as_of_date, variant)
    snaps = _price_snapshots(as_of_date)
    combined = _assemble(curr, prev, snaps)
    if combined.empty:
        logger.debug(f'cross_section_panel: {as_of_date} {variant} — empty (no inner-join matches)')
        return combined
    out = _apply_formulas(combined)
    if tickers is not None:
        out = out[out.index.isin(tickers)]
    logger.debug(f'cross_section_panel: {as_of_date} {variant} → {len(out)} tickers, {len(out.columns)} factors')
    return out

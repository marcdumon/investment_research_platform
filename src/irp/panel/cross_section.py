"""Factor cross-section computation on panel data.

Output schema: DataFrame indexed by Ticker with all factor columns.

Pipeline (all polars/numpy, all small DataFrames):
    1. PIT-align fundamentals (income/balance/cashflow at as_of and prior year)
    2. Price snapshots at as_of, -30d, -182d, -365d
    3. Window stats: vol_21d, ma200 from wide panel slice
    4. Combine into one DataFrame indexed by Ticker
    5. Apply factor formulas (pure pandas/numpy)
"""
import datetime
import logging
import warnings
from typing import Literal

import numpy as np
import pandas as pd
import polars as pl
from dateutil.relativedelta import relativedelta

from irp.panel.load import load_prices_wide, load_fundamentals
from irp.panel.pit import pit_latest, pit_ttm, pit_price_row, pit_price_at_offset

logger = logging.getLogger(__name__)


# Columns aggregated for TTM (Q variant) — income + cashflow only; balance uses latest filing
_INCOME_TTM_COLS = [
    'Revenue', 'Gross Profit', 'Operating Income (Loss)', 'Net Income',
    'Interest Expense, Net', 'Depreciation & Amortization',
]
_CASHFLOW_TTM_COLS = [
    'Net Cash from Operating Activities', 'Net Cash from Investing Activities',
    'Depreciation & Amortization',
]


# ---------------------------------------------------------------------------
# PIT for fundamentals (variant-aware)
# ---------------------------------------------------------------------------

def _income_pit(
    df: pl.DataFrame, as_of: datetime.date, variant: Literal['A', 'Q'],
) -> pl.DataFrame:
    if variant == 'A':
        return pit_latest(df, as_of).select(
            ['Ticker', 'Revenue', 'Gross Profit', 'Operating Income (Loss)',
             'Net Income', 'Interest Expense, Net', 'Depreciation & Amortization']
        )
    return pit_ttm(df, as_of, _INCOME_TTM_COLS, n=4)


def _cashflow_pit(
    df: pl.DataFrame, as_of: datetime.date, variant: Literal['A', 'Q'],
) -> pl.DataFrame:
    if variant == 'A':
        return pit_latest(df, as_of).select(
            ['Ticker',
             'Net Cash from Operating Activities',
             'Net Cash from Investing Activities',
             'Depreciation & Amortization']
        )
    return pit_ttm(df, as_of, _CASHFLOW_TTM_COLS, n=4)


def _balance_pit(df: pl.DataFrame, as_of: datetime.date) -> pl.DataFrame:
    """Latest balance filing — same for both A and Q."""
    return pit_latest(df, as_of).select(
        ['Ticker', 'Total Assets', 'Total Equity',
         'Total Current Assets', 'Total Current Liabilities',
         'Cash, Cash Equivalents & Short Term Investments',
         'Short Term Debt', 'Long Term Debt', 'Shares (Diluted)']
    )


# ---------------------------------------------------------------------------
# Window stats from wide price panel
# ---------------------------------------------------------------------------

def _window_stats(
    panel, as_of: datetime.date,
) -> tuple[dict[str, float], dict[str, float]]:
    """Compute vol_21d and ma200 for all tickers as of `as_of`.

    vol_21d = stddev(daily log returns, last 22 trading days) × sqrt(252).
    ma200   = mean of last 200 trading-day closes.
    Both NaN if not enough observations.
    """
    i = int(np.searchsorted(panel.dates, np.datetime64(as_of, 'D'), side='right')) - 1
    if i < 0:
        return {}, {}

    # vol_21d: stddev of 21 log returns × sqrt(252). Per ticker take last 22
    # non-NaN closes within a wider window (~50 panel rows is plenty even for
    # a US ticker with interleaved DE-only union-calendar days).
    vol_start = max(0, i + 1 - 50)
    vol_slice = panel.values[vol_start:i + 1].astype('float64')
    valid_c = np.isfinite(vol_slice)
    cum_c = np.flip(np.cumsum(np.flip(valid_c, axis=0), axis=0), axis=0)
    # Take last 22 non-NaN closes per ticker
    keep = (cum_c >= 1) & (cum_c <= 22) & valid_c
    # Compact each column: pull selected closes into a (22, n_tickers) array.
    # Equivalent vectorized form: for each col, np.take last 22 valid.
    # Build per-column index lists via argpartition on keep mask.
    n_tickers = vol_slice.shape[1]
    compact = np.full((22, n_tickers), np.nan, dtype='float64')
    # `keep` rows have variable count per column; for cols with exactly 22 keeps,
    # extract them in original order.
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
    # Cols without 22 closes already have all-NaN compact → sd NaN. Belt+braces:
    vol[~np.isfinite(vol)] = np.nan

    # ma200: mean of last 200 non-NaN closes per ticker.
    # Wide-format wrinkle: the panel uses a union calendar across markets, so a
    # row may be NaN for a US ticker on a DE-only trading day. SQL counts only
    # rows that exist per ticker. To match, pick the last 200 non-NaN per col
    # within a wider window (~400 panel rows ≈ all available history we need).
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
# Combine + compute factors
# ---------------------------------------------------------------------------

def _safe_log_ratio(num: pd.Series, denom: pd.Series) -> pd.Series:
    """log(num/denom) when both > 0, else NaN."""
    valid = (num > 0) & (denom > 0)
    result = pd.Series(np.nan, index=num.index, dtype='float64')
    result[valid] = np.log(num[valid] / denom[valid])
    return result


def _positive_only(series: pd.Series) -> pd.Series:
    """Replace non-positive values with NaN — matches SQL `CASE WHEN x > 0 THEN x END`."""
    out = series.astype('float64').copy()
    out[~(out > 0)] = np.nan
    return out


def _div(num, denom):
    """Safe division: denom==0 → NaN, also masks inf."""
    out = num / denom.replace(0, np.nan)
    return out.where(np.isfinite(out), np.nan)


def cross_section_panel(
    as_of_date: datetime.date,
    variant: Literal['A', 'Q'] = 'A',
    tickers: list[str] | None = None,
) -> pd.DataFrame:
    """Compute all factors cross-sectionally at a point in time.

    Output schema:
        index = Ticker
        columns = mktcap, pe, pb, ps, ev_ebitda, ev_ebit, ev_sales, fcf_yield, rand,
                  gross_margin, op_margin, net_margin, roe, roa, roic, fcf_margin,
                  asset_turnover, cfo_ni_ratio, accruals,
                  debt_equity, net_debt_ebitda, interest_coverage,
                  mom_12_1, mom_6_1, vol_21d, ma200_ratio,
                  rev_growth_1y, earn_growth_1y,
                  piotroski_fscore
    """
    prior_aod = as_of_date - relativedelta(years=1)

    # 1. PIT-aligned fundamentals
    inc_lf = load_fundamentals('income', variant)
    bal_lf = load_fundamentals('balance', variant)
    cf_lf  = load_fundamentals('cashflow', variant)

    inc = _income_pit(inc_lf, as_of_date, variant).to_pandas().set_index('Ticker')
    bal = _balance_pit(bal_lf, as_of_date).to_pandas().set_index('Ticker')
    cf  = _cashflow_pit(cf_lf, as_of_date, variant).to_pandas().set_index('Ticker')

    inc_p = _income_pit(inc_lf, prior_aod, variant).to_pandas().set_index('Ticker')
    bal_p = _balance_pit(bal_lf, prior_aod).to_pandas().set_index('Ticker')
    cf_p  = _cashflow_pit(cf_lf, prior_aod, variant).to_pandas().set_index('Ticker')

    # 2. Prices: snapshot + lookbacks
    panel = load_prices_wide('Close')
    p0_dict   = pit_price_row(panel, as_of_date)
    p1m_dict  = pit_price_at_offset(panel, as_of_date, 30)
    p6m_dict  = pit_price_at_offset(panel, as_of_date, 182)
    p12m_dict = pit_price_at_offset(panel, as_of_date, 365)
    vol_dict, ma_dict = _window_stats(panel, as_of_date)

    # 3. Build a single combined frame (inner join on Ticker for required fields,
    #    left join for optional/momentum/prior)
    p0 = pd.Series(p0_dict, name='p0', dtype='float64')
    if p0.empty:
        return pd.DataFrame()

    # Required: ticker must have income, balance, cashflow, and price
    tickers_common = inc.index.intersection(bal.index).intersection(cf.index).intersection(p0.index)
    if len(tickers_common) == 0:
        return pd.DataFrame()

    df = pd.DataFrame(index=tickers_common)
    df.index.name = 'Ticker'

    # Income (current)
    df['rev']  = inc['Revenue'].astype('float64')
    df['gp']   = inc['Gross Profit'].astype('float64')
    df['ebit'] = inc['Operating Income (Loss)'].astype('float64')
    df['ni']   = inc['Net Income'].astype('float64')
    df['interest_exp'] = inc['Interest Expense, Net'].fillna(0).astype('float64')

    # Balance (current)
    df['ta']      = bal['Total Assets'].astype('float64')
    df['equity']  = bal['Total Equity'].astype('float64')
    df['cash']    = bal['Cash, Cash Equivalents & Short Term Investments'].fillna(0).astype('float64')
    df['st_debt'] = bal['Short Term Debt'].fillna(0).astype('float64')
    df['lt_debt'] = bal['Long Term Debt'].fillna(0).astype('float64')
    df['ca']      = bal['Total Current Assets'].astype('float64')
    df['cl']      = bal['Total Current Liabilities'].astype('float64')
    df['shares']  = bal['Shares (Diluted)'].astype('float64')

    # Cashflow (current) — D&A always from cashflow
    df['cfo'] = cf['Net Cash from Operating Activities'].astype('float64')
    df['cfi'] = cf['Net Cash from Investing Activities'].astype('float64')
    df['da']  = cf['Depreciation & Amortization'].fillna(0).astype('float64')

    # Prices (current + lookbacks) — LEFT join (NaN if missing)
    df['p0']    = p0.reindex(df.index).astype('float64')
    df['p_1m']  = pd.Series(p1m_dict).reindex(df.index).astype('float64')
    df['p_6m']  = pd.Series(p6m_dict).reindex(df.index).astype('float64')
    df['p_12m'] = pd.Series(p12m_dict).reindex(df.index).astype('float64')
    df['vol_21d_raw']    = pd.Series(vol_dict).reindex(df.index).astype('float64')
    df['ma200']          = pd.Series(ma_dict).reindex(df.index).astype('float64')

    # Prior-year (LEFT joins — NaN if missing for that ticker)
    df['rev_p']    = inc_p['Revenue'].reindex(df.index).astype('float64')
    df['ni_p']     = inc_p['Net Income'].reindex(df.index).astype('float64')
    df['gp_p']     = inc_p['Gross Profit'].reindex(df.index).astype('float64')
    df['ta_p']     = bal_p['Total Assets'].reindex(df.index).astype('float64')
    df['ltd_p']    = bal_p['Long Term Debt'].fillna(0).reindex(df.index).astype('float64')
    df['ca_p']     = bal_p['Total Current Assets'].reindex(df.index).astype('float64')
    df['cl_p']     = bal_p['Total Current Liabilities'].reindex(df.index).astype('float64')
    df['shares_p'] = bal_p['Shares (Diluted)'].reindex(df.index).astype('float64')
    df['cfo_p']   = cf_p['Net Cash from Operating Activities'].reindex(df.index).astype('float64')

    # 4. Factor formulas
    out = pd.DataFrame(index=df.index)
    mktcap = df['p0'] * df['shares']
    out['mktcap'] = mktcap

    # Valuation
    out['pe']        = mktcap / _positive_only(df['ni'])
    out['pb']        = mktcap / _positive_only(df['equity'])
    out['ps']        = _div(mktcap, df['rev'])
    ev = mktcap + df['st_debt'] + df['lt_debt'] - df['cash']
    out['ev_ebitda'] = ev / _positive_only(df['ebit'] + df['da'])
    out['ev_ebit']   = ev / _positive_only(df['ebit'])
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

    # Piotroski signals (binary, NULL when inputs missing)
    def _binary(condition: pd.Series, gate: pd.Series) -> pd.Series:
        """1 where condition true; 0 where false; NaN where gate is False."""
        out = pd.Series(np.nan, index=condition.index, dtype='float64')
        out[gate & condition] = 1.0
        out[gate & ~condition] = 0.0
        return out

    ta_ok      = df['ta'].notna() & (df['ta'] != 0)
    tap_ok     = df['ta_p'].notna() & (df['ta_p'] != 0)
    rev_ok     = df['rev'].notna() & (df['rev'] != 0)
    revp_ok    = df['rev_p'].notna() & (df['rev_p'] != 0)
    cl_ok      = df['cl'].notna() & (df['cl'] != 0)
    clp_ok     = df['cl_p'].notna() & (df['cl_p'] != 0)

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
    out['piotroski_fscore'] = score.where(~all_null, np.nan)

    # Filter to requested tickers if any
    if tickers is not None:
        out = out[out.index.isin(tickers)]

    # Match SQL column order
    out = out[[
        'mktcap',
        'pe', 'pb', 'ps', 'ev_ebitda', 'ev_ebit', 'ev_sales', 'fcf_yield', 'rand',
        'gross_margin', 'op_margin', 'net_margin', 'roe', 'roa', 'roic', 'fcf_margin',
        'asset_turnover', 'cfo_ni_ratio', 'accruals',
        'debt_equity', 'net_debt_ebitda', 'interest_coverage',
        'mom_12_1', 'mom_6_1', 'vol_21d', 'ma200_ratio',
        'rev_growth_1y', 'earn_growth_1y',
        'piotroski_fscore',
    ]]
    return out

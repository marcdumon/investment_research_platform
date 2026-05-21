"""Factor backtest: IC series and quintile return analysis.

Pure computation — no DB access. Called by compute.run_backtest().
"""
import datetime

import numpy as np
import pandas as pd
from scipy import stats

from irp.factors._cols import PRICE_CLOSE, PRICE_DATE, PRICE_TICKER


def _price_at(prices: pd.DataFrame, target: pd.Timestamp) -> pd.DataFrame:
    """Most-recent (close, date) per ticker on or before `target`.

    Returns DataFrame indexed by Ticker with columns [Close, Date].
    """
    eligible = prices.loc[prices[PRICE_DATE] <= target]
    if eligible.empty:
        return pd.DataFrame(columns=[PRICE_CLOSE, PRICE_DATE])
    idx = eligible.groupby(PRICE_TICKER)[PRICE_DATE].idxmax()
    return eligible.loc[idx].set_index(PRICE_TICKER)[[PRICE_CLOSE, PRICE_DATE]]


def compute_forward_returns(
    prices: pd.DataFrame,
    rebalance_dates: list[datetime.date],
    horizon_days: int,
) -> pd.DataFrame:
    """Compute forward log returns at each rebalance date per ticker.

    Parameters
    ----------
    prices          : DataFrame with columns (Ticker, Date, Close, ...).
    rebalance_dates : Dates at which to enter positions.
    horizon_days    : Calendar days to hold; exit at closest price <= T+horizon.

    Returns
    -------
    DataFrame with columns [Ticker, Date, fwd_ret].
    fwd_ret = log(P_{T+horizon} / P_T). NaN when either price is unavailable
    or when no price strictly after T exists within the horizon.
    """
    if prices.empty or not rebalance_dates:
        return pd.DataFrame(columns=[PRICE_TICKER, 'Date', 'fwd_ret'])

    prices = prices.copy()
    prices[PRICE_DATE] = pd.to_datetime(prices[PRICE_DATE])

    records = []
    for d in rebalance_dates:
        t0 = pd.Timestamp(d)
        t1 = t0 + pd.Timedelta(days=horizon_days)
        p0_info = _price_at(prices, t0)
        p1_info = _price_at(prices, t1)
        if p0_info.empty or p1_info.empty:
            continue
        for ticker in p0_info.index.intersection(p1_info.index):
            entry_date = p0_info.loc[ticker, PRICE_DATE]
            exit_date  = p1_info.loc[ticker, PRICE_DATE]
            if exit_date <= entry_date:
                continue
            p0 = p0_info.loc[ticker, PRICE_CLOSE]
            p1 = p1_info.loc[ticker, PRICE_CLOSE]
            if p0 > 0 and p1 > 0:
                records.append({PRICE_TICKER: ticker, 'Date': d, 'fwd_ret': float(np.log(p1 / p0))})

    return pd.DataFrame(records, columns=[PRICE_TICKER, 'Date', 'fwd_ret'])


def compute_backtest(
    factor: str,
    cross_sections: dict[datetime.date, pd.DataFrame],
    fwd_returns: pd.DataFrame,
    n_quantiles: int = 5,
) -> dict:
    """Compute IC series and quintile cumulative returns.

    Parameters
    ----------
    factor          : Column name to evaluate (must exist in cross_sections values).
    cross_sections  : dict mapping rebalance date to cross-section DataFrame
                      (indexed by Ticker, factor column present).
    fwd_returns     : DataFrame from compute_forward_returns with columns
                      [Ticker, Date, fwd_ret].
    n_quantiles     : Number of buckets (default 5).

    Returns
    -------
    dict with keys:
        ic_series       pd.Series  DatetimeIndex → Spearman IC (NaN if < 20 obs)
        quintile_cumret pd.DataFrame  columns Q1..Q5, DatetimeIndex, cumulative log ret
        mean_ic         float
        ic_tstat        float
        n_dates         int  (number of dates with valid IC)
    """
    quantile_labels = [f'Q{i + 1}' for i in range(n_quantiles)]
    empty = {
        'ic_series': pd.Series(dtype=float),
        'quintile_cumret': pd.DataFrame(columns=quantile_labels),
        'mean_ic': float('nan'),
        'ic_tstat': float('nan'),
        'n_dates': 0,
    }
    if not cross_sections or fwd_returns.empty:
        return empty

    ic_records: list[tuple[datetime.date, float]] = []
    qret_rows: list[pd.Series] = []
    qret_dates: list[datetime.date] = []

    fwd_by_date = fwd_returns.groupby('Date')

    for d, xs in sorted(cross_sections.items()):
        if factor not in xs.columns:
            continue
        f_vals = xs[factor].dropna()
        if f_vals.empty:
            continue
        try:
            fwd_d = fwd_by_date.get_group(d).set_index(PRICE_TICKER)['fwd_ret']
        except KeyError:
            continue
        merged = pd.DataFrame({'factor': f_vals, 'fwd_ret': fwd_d}).dropna()
        if len(merged) < 20:
            ic_records.append((d, float('nan')))
            continue
        ic_val, _ = stats.spearmanr(merged['factor'], merged['fwd_ret'])
        ic_records.append((d, float(ic_val)))
        try:
            merged['q'] = pd.qcut(
                merged['factor'], n_quantiles, labels=quantile_labels, duplicates='drop'
            )
        except ValueError:
            continue
        qrets = merged.groupby('q', observed=True)['fwd_ret'].mean().reindex(quantile_labels)
        qrets.name = d
        qret_rows.append(qrets)
        qret_dates.append(d)

    if not ic_records:
        return empty

    ic_series = pd.Series(
        [v for _, v in ic_records],
        index=pd.DatetimeIndex([d for d, _ in ic_records]),
        name='IC',
    )
    valid_ic = ic_series.dropna()
    mean_ic = float(valid_ic.mean()) if not valid_ic.empty else float('nan')
    if len(valid_ic) > 1:
        std_ic = valid_ic.std()
        ic_tstat = float(mean_ic / std_ic * np.sqrt(len(valid_ic))) if std_ic > 0 else float('nan')
    else:
        ic_tstat = float('nan')

    if qret_rows:
        qdf = pd.DataFrame(qret_rows)
        qdf.index = pd.DatetimeIndex(qret_dates)
        quintile_cumret = qdf.fillna(0).cumsum()
    else:
        quintile_cumret = pd.DataFrame(columns=quantile_labels)

    return {
        'ic_series': ic_series,
        'quintile_cumret': quintile_cumret,
        'mean_ic': mean_ic,
        'ic_tstat': ic_tstat,
        'n_dates': int(len(valid_ic)),
    }

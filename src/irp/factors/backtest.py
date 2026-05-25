"""Factor backtest: IC series and quintile return analysis.

Pure computation — no DB access. Called by compute.run_backtest().
Production forward returns use `irp.panel.forward_returns_panel`;
compute_forward_returns() remains here for unit tests with synthetic data.
"""

import datetime
import math
import warnings

import numpy as np
import pandas as pd
from scipy import stats

from irp.core.config import config
from irp.factors._cols import PRICE_CLOSE, PRICE_DATE, PRICE_TICKER
from irp.factors.models import BacktestResult


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
    DataFrame with columns [Ticker, Date, fwd_ret]. fwd_ret = log(P_exit / P_entry).
    """
    if prices.empty or not rebalance_dates:
        return pd.DataFrame(columns=[PRICE_TICKER, 'Date', 'fwd_ret'])

    prices = prices.copy()
    prices[PRICE_DATE] = pd.to_datetime(prices[PRICE_DATE])

    pivot = (
        prices.pivot_table(index=PRICE_DATE, columns=PRICE_TICKER, values=PRICE_CLOSE)
        .sort_index()
    )
    entry_ts = pd.DatetimeIndex([pd.Timestamp(d) for d in rebalance_dates])
    exit_ts = entry_ts + pd.Timedelta(days=horizon_days)

    all_targets = pivot.index.union(entry_ts).union(exit_ts)
    extended = pivot.reindex(all_targets).ffill()

    p0 = extended.reindex(entry_ts).values
    p1 = extended.reindex(exit_ts).values

    log_ret = np.log(p1 / p0)
    log_ret[~np.isfinite(log_ret)] = np.nan

    result = (
        pd.DataFrame(log_ret, index=pd.Index(rebalance_dates, name='Date'), columns=pivot.columns)
        .stack()
        .dropna()
        .reset_index()
    )
    result.columns = pd.Index(['Date', PRICE_TICKER, 'fwd_ret'])
    return result


def _turnover_series(tickers_list: list[set]) -> list[float]:
    """Fraction of tickers entering/leaving a quintile at each step.

    First element is always NaN (no predecessor).
    """
    result: list[float] = [float('nan')]
    for i in range(1, len(tickers_list)):
        prev, curr = tickers_list[i - 1], tickers_list[i]
        n = max(len(prev), len(curr))
        result.append(1.0 - len(prev & curr) / n if n > 0 else float('nan'))
    return result


def compute_backtest(
    factor: str,
    cross_sections: dict[datetime.date, pd.DataFrame],
    fwd_returns: pd.DataFrame,
    n_quantiles: int = 5,
    horizon_days: int = 252,
    cost_bps: float = 0,
) -> BacktestResult:
    """Compute IC series, quintile cumulative returns, and risk metrics.

    Parameters
    ----------
    factor          : Column name to evaluate (must exist in cross_sections values).
    cross_sections  : dict mapping rebalance date to cross-section DataFrame
                      (indexed by Ticker, factor column present).
    fwd_returns     : DataFrame from compute_forward_returns with columns
                      [Ticker, Date, fwd_ret].
    n_quantiles     : Number of buckets (default 5).
    horizon_days    : Holding period in calendar days; used for annualization.
    cost_bps        : Round-trip transaction cost assumption in basis points.
                      0 = gross returns only. Applied per rebalance as
                      turnover_fraction * cost_bps / 10_000.

    Returns
    -------
    dict with keys:
        ic_series           pd.Series  DatetimeIndex → Spearman IC
        quintile_cumret     pd.DataFrame  Q1..Q5, cumulative log returns (gross)
        quintile_cumret_net pd.DataFrame | None  cost-adjusted cumret (if cost_bps>0)
        ew_cumret           pd.Series  equal-weight cumulative log return
        ls_cumret           pd.Series  Q1 − Q5 long-short cumulative log return
        mean_ic             float
        ic_tstat            float  = mean_ic / std_ic * sqrt(n)
        icir                float  = mean_ic / std_ic  (Information Ratio)
        n_dates             int
        quintile_ann_ret    dict[str, float]  annualized log return per quintile
        quintile_ann_vol    dict[str, float]  annualized volatility per quintile
        quintile_sharpe     dict[str, float]  Sharpe ratio per quintile
        quintile_max_dd     dict[str, float]  max drawdown per quintile
        turnover_q1         pd.Series  per-date turnover fraction for Q1
        turnover_q5         pd.Series  per-date turnover fraction for Q5
        mean_turnover_q1    float
        mean_turnover_q5    float
    """
    quantile_labels = [f'Q{i + 1}' for i in range(n_quantiles)]
    q_last = quantile_labels[-1]
    _nan = float('nan')
    if not cross_sections or fwd_returns.empty:
        return BacktestResult(
            quintile_cumret=pd.DataFrame(columns=quantile_labels),
        )

    ic_records: list[tuple[datetime.date, float]] = []
    qret_rows: list[pd.Series] = []
    qret_dates: list[datetime.date] = []
    all_q_tickers: dict[str, list[set]] = {q: [] for q in quantile_labels}

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
        if len(merged) < config.factors.min_ic_obs:
            ic_records.append((d, _nan))
            continue
        with warnings.catch_warnings():
            warnings.simplefilter('ignore')
            ic_val, _ = stats.spearmanr(merged['factor'], merged['fwd_ret'])
        ic_records.append((d, float(ic_val)))  # type: ignore[arg-type]
        try:
            merged['q'] = pd.qcut(
                merged['factor'], n_quantiles, labels=quantile_labels, duplicates='drop'
            )
        except ValueError:
            continue

        for q in quantile_labels:
            all_q_tickers[q].append(set(merged[merged['q'] == q].index.tolist()))

        qrets = (
            merged
            .groupby('q', observed=True)['fwd_ret']
            .mean()
            .reindex(quantile_labels)
        )
        qrets.name = d
        qret_rows.append(qrets)
        qret_dates.append(d)

    if not ic_records:
        return BacktestResult(
            quintile_cumret=pd.DataFrame(columns=quantile_labels),
        )

    ic_series = pd.Series(
        [v for _, v in ic_records],
        index=pd.DatetimeIndex([d for d, _ in ic_records]),
        name='IC',
    )
    valid_ic = ic_series.dropna()
    mean_ic = float(valid_ic.mean()) if not valid_ic.empty else _nan
    icir = _nan
    ic_tstat = _nan
    if len(valid_ic) > 1:
        std_ic = float(valid_ic.std())
        if std_ic > 0:
            icir = mean_ic / std_ic
            ic_tstat = mean_ic / std_ic * math.sqrt(len(valid_ic))

    if not qret_rows:
        return BacktestResult(
            ic_series=ic_series,
            quintile_cumret=pd.DataFrame(columns=quantile_labels),
            mean_ic=mean_ic, ic_tstat=ic_tstat, icir=icir,
            n_dates=int(len(valid_ic)),
        )

    qdf = pd.DataFrame(qret_rows)
    qdf.index = pd.DatetimeIndex(qret_dates)

    quintile_cumret = qdf.fillna(0).cumsum()
    ew_cumret = qdf.mean(axis=1).fillna(0).cumsum()

    # L/S: Q1 long, Q5 short
    ls_cumret = (qdf['Q1'] - qdf[q_last]).fillna(0).cumsum()

    # Turnover per quintile
    n_q = len(qret_dates)
    to_q1_list = _turnover_series(all_q_tickers['Q1']) if all_q_tickers['Q1'] else [_nan] * n_q
    to_q5_list = _turnover_series(all_q_tickers[q_last]) if all_q_tickers[q_last] else [_nan] * n_q
    turnover_q1 = pd.Series(to_q1_list, index=qdf.index)
    turnover_q5 = pd.Series(to_q5_list, index=qdf.index)
    to_q1_valid = turnover_q1.dropna()
    to_q5_valid = turnover_q5.dropna()
    mean_turnover_q1 = float(to_q1_valid.mean()) if not to_q1_valid.empty else _nan
    mean_turnover_q5 = float(to_q5_valid.mean()) if not to_q5_valid.empty else _nan

    # Per-quintile risk metrics
    ann_factor = 252.0 / max(horizon_days, 1)
    quintile_ann_ret: dict[str, float] = {}
    quintile_ann_vol: dict[str, float] = {}
    quintile_sharpe: dict[str, float] = {}
    quintile_max_dd: dict[str, float] = {}

    for q in quantile_labels:
        col = qdf[q].fillna(0)
        cumret_q = col.cumsum()
        per_mean = float(col.mean())
        per_std = float(col.std()) if len(col) > 1 else float('nan')
        ann_ret = per_mean * ann_factor
        ann_vol = per_std * math.sqrt(ann_factor)
        quintile_ann_ret[q] = ann_ret
        quintile_ann_vol[q] = ann_vol
        quintile_sharpe[q] = ann_ret / ann_vol if ann_vol > 0 else _nan
        quintile_max_dd[q] = (
            float((cumret_q - cumret_q.cummax()).min()) if not cumret_q.empty else _nan
        )

    # Cost-adjusted cumulative returns
    quintile_cumret_net: pd.DataFrame | None = None
    if cost_bps > 0:
        to_all = {q: _turnover_series(all_q_tickers[q]) for q in quantile_labels}
        qdf_net = qdf.copy()
        for q in quantile_labels:
            to_s = pd.Series(to_all[q], index=qdf.index).fillna(0)
            qdf_net[q] = qdf[q] - to_s * cost_bps / 10_000
        quintile_cumret_net = qdf_net.fillna(0).cumsum()

    return BacktestResult(
        ic_series=ic_series,
        quintile_cumret=quintile_cumret,
        quintile_cumret_net=quintile_cumret_net,
        ew_cumret=ew_cumret,
        ls_cumret=ls_cumret,
        mean_ic=mean_ic,
        ic_tstat=ic_tstat,
        icir=icir,
        n_dates=int(len(valid_ic)),
        quintile_ann_ret=quintile_ann_ret,
        quintile_ann_vol=quintile_ann_vol,
        quintile_sharpe=quintile_sharpe,
        quintile_max_dd=quintile_max_dd,
        turnover_q1=turnover_q1,
        turnover_q5=turnover_q5,
        mean_turnover_q1=mean_turnover_q1,
        mean_turnover_q5=mean_turnover_q5,
    )

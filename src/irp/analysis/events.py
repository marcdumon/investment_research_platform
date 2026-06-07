"""Pure event-study & seasonality math for the `/analysis` section.

Event study: align returns to a ±window around recurring events (earnings/dividend/
split), average across events (AAR per relative day) and cumulate (CAR). Seasonality:
month-of-year and day-of-week average returns. No DB, no Dash — plain pandas in/out,
unit-tested in `tests/test_events.py`.
"""
import numpy as np
import pandas as pd

_DOW = ['Mon', 'Tue', 'Wed', 'Thu', 'Fri']


def abnormal_returns(stock: pd.Series, market: pd.Series | None = None,
                     method: str = 'market') -> pd.Series:
    """Abnormal returns under the chosen model.

    `market` → `stock − market` (inner-aligned); `mean` → `stock − stock.mean()`
    (centred); `raw` → `stock` unchanged.
    """
    s = stock.dropna()
    if method == 'market' and market is not None:
        j = pd.concat([s.rename('s'), market.rename('m')], axis=1, join='inner').dropna()
        return j['s'] - j['m']
    if method == 'mean':
        return s - s.mean()
    return s


def event_window_matrix(ar: pd.Series, event_dates, pre: int, post: int) -> pd.DataFrame:
    """One row per event, columns the relative trading-day offsets `[−pre … +post]`.

    Each event date is snapped to the trading day at or just before it in `ar.index`
    (searchsorted); events without a full window (too close to either end) are dropped.
    """
    ar = ar.dropna()
    if ar.empty:
        return pd.DataFrame(columns=list(range(-pre, post + 1)))
    idx = ar.index
    vals = ar.to_numpy()
    cols = list(range(-pre, post + 1))
    rows = []
    for d in event_dates:
        d = pd.Timestamp(d)
        pos = int(np.searchsorted(idx.values, np.datetime64(d), side='right')) - 1
        if pos < pre or pos + post >= len(idx):
            continue
        rows.append(vals[pos - pre: pos + post + 1])
    if not rows:
        return pd.DataFrame(columns=cols)
    return pd.DataFrame(rows, columns=cols)


def mean_adjusted_matrix(returns: pd.Series, event_dates, pre: int, post: int,
                         est_lo: int = 60, est_hi: int = 11) -> pd.DataFrame:
    """Mean-adjusted abnormal-return window matrix with a **per-event pre-event
    estimation window** (textbook event study, no look-ahead).

    Each event's "normal" return is the mean of `returns` over `[−est_lo … −est_hi)`
    trading days *before* the event; the abnormal window `[−pre … +post]` is the raw
    return minus that pre-event mean. Events without room for the estimation window or
    the full window are dropped.
    """
    r = returns.dropna()
    cols = list(range(-pre, post + 1))
    if r.empty:
        return pd.DataFrame(columns=cols)
    idx = r.index
    vals = r.to_numpy()
    rows = []
    for d in event_dates:
        pos = int(np.searchsorted(idx.values, np.datetime64(pd.Timestamp(d)), side='right')) - 1
        if pos - est_lo < 0 or pos < pre or pos + post >= len(idx):
            continue
        est_mean = vals[pos - est_lo: pos - est_hi].mean()
        window = vals[pos - pre: pos + post + 1] - est_mean
        rows.append(window)
    if not rows:
        return pd.DataFrame(columns=cols)
    return pd.DataFrame(rows, columns=cols)


def aar_car(matrix: pd.DataFrame) -> dict:
    """Average abnormal return per relative day (AAR), cumulative (CAR), event count,
    terminal CAR and its cross-event t-statistic."""
    n = int(len(matrix))
    if n == 0 or matrix.shape[1] == 0:
        return {'aar': pd.Series(dtype=float), 'car': pd.Series(dtype=float),
                'n': n, 'car_end': np.nan, 'tstat': np.nan}
    aar = matrix.mean(axis=0)
    car = aar.cumsum()
    car_end = float(car.iloc[-1])
    per_event_car = matrix.sum(axis=1)           # each event's total window return
    sd = float(per_event_car.std(ddof=1)) if n > 1 else np.nan
    tstat = car_end / (sd / np.sqrt(n)) if sd and sd > 0 else np.nan
    return {'aar': aar, 'car': car, 'n': n, 'car_end': car_end, 'tstat': tstat}


def monthly_seasonality(rets: pd.Series) -> pd.Series:
    """Mean return by calendar month (index 1–12)."""
    r = rets.dropna()
    out = r.groupby(r.index.month).mean()
    return out.reindex(range(1, 13))


def dow_seasonality(rets: pd.Series) -> pd.Series:
    """Mean return by weekday (Mon–Fri)."""
    r = rets.dropna()
    out = r.groupby(r.index.dayofweek).mean()
    out = out.reindex(range(5))
    out.index = _DOW
    return out

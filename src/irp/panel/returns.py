"""Forward-return computation on the wide price panel.

Logic:
    For each rebalance date d:
        entry_price[ticker] = price at latest Date ≤ d
        exit_price[ticker]  = price at latest Date ≤ d + horizon_days
        fwd_ret[ticker]     = log(exit_price / entry_price)
"""
import datetime
import logging

import numpy as np
import pandas as pd

from irp.panel.load import load_prices_wide

logger = logging.getLogger(__name__)


def forward_returns_panel(
    rebalance_dates: list[datetime.date],
    horizon_days: int,
    tickers: list[str] | None = None,
) -> pd.DataFrame:
    """Compute per-ticker forward log returns at each rebalance date.

    Returns DataFrame with columns [Date, Ticker, fwd_ret].
    Date is datetime.date (matches `compute_backtest` expectation).
    """
    if not rebalance_dates:
        return pd.DataFrame(columns=['Date', 'Ticker', 'fwd_ret'])

    panel = load_prices_wide('Close')
    if tickers is not None:
        tick_arr = np.array([t for t in tickers if t in panel.ticker_to_idx])
        if len(tick_arr) == 0:
            return pd.DataFrame(columns=['Date', 'Ticker', 'fwd_ret'])
        col_idx = np.array([panel.ticker_to_idx[t] for t in tick_arr])
    else:
        tick_arr = panel.tickers
        col_idx = np.arange(len(tick_arr))

    out_rows = []
    for rd in rebalance_dates:
        rd64 = np.datetime64(rd, 'D')
        exit64 = rd64 + np.timedelta64(horizon_days, 'D')
        i_entry = int(np.searchsorted(panel.dates, rd64, side='right')) - 1
        i_exit  = int(np.searchsorted(panel.dates, exit64, side='right')) - 1
        if i_entry < 0 or i_exit < 0:
            continue

        p0 = panel.values[i_entry, col_idx]
        p1 = panel.values[i_exit,  col_idx]
        with np.errstate(divide='ignore', invalid='ignore'):
            valid = (p0 > 0) & (p1 > 0)
            ratio = np.divide(p1, p0, out=np.full_like(p0, np.nan, dtype='float64'), where=valid)
            ret = np.log(ratio, out=np.full_like(ratio, np.nan), where=valid)

        mask = np.isfinite(ret)
        if not mask.any():
            continue
        out_rows.append(pd.DataFrame({
            'Date': rd,
            'Ticker': tick_arr[mask],
            'fwd_ret': ret[mask].astype('float64'),
        }))

    if not out_rows:
        return pd.DataFrame(columns=['Date', 'Ticker', 'fwd_ret'])
    return pd.concat(out_rows, ignore_index=True)


def price_volume_panel(
    rebalance_dates: list[datetime.date],
    tickers: list[str] | None = None,
) -> pd.DataFrame:
    """PIT close + volume per ticker at each rebalance date.

    Returns DataFrame [Date, Ticker, close, volume]. Each value is the latest
    observation with price/volume Date <= the rebalance date (point-in-time).
    Rows where neither close nor volume is available are dropped.
    """
    if not rebalance_dates:
        return pd.DataFrame(columns=['Date', 'Ticker', 'close', 'volume'])

    close = load_prices_wide('Close')
    vol = load_prices_wide('Volume')

    tick_arr = np.array([t for t in tickers if t in close.ticker_to_idx]) if tickers is not None else close.tickers
    if len(tick_arr) == 0:
        return pd.DataFrame(columns=['Date', 'Ticker', 'close', 'volume'])

    c_idx = np.array([close.ticker_to_idx[t] for t in tick_arr])
    v_idx = np.array([vol.ticker_to_idx.get(t, -1) for t in tick_arr])
    v_has = v_idx >= 0

    out_rows = []
    for rd in rebalance_dates:
        rd64 = np.datetime64(rd, 'D')
        i_c = int(np.searchsorted(close.dates, rd64, side='right')) - 1
        i_v = int(np.searchsorted(vol.dates, rd64, side='right')) - 1
        if i_c < 0:
            continue
        c = close.values[i_c, c_idx]
        v = np.full(len(tick_arr), np.nan, dtype='float64')
        if i_v >= 0 and v_has.any():
            v[v_has] = vol.values[i_v, v_idx[v_has]]
        mask = np.isfinite(c) | np.isfinite(v)
        if not mask.any():
            continue
        out_rows.append(pd.DataFrame({
            'Date': rd,
            'Ticker': tick_arr[mask],
            'close': c[mask].astype('float64'),
            'volume': v[mask].astype('float64'),
        }))

    if not out_rows:
        return pd.DataFrame(columns=['Date', 'Ticker', 'close', 'volume'])
    return pd.concat(out_rows, ignore_index=True)

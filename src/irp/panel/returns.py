"""Forward-return computation on the wide price panel.

Logic:
    For each rebalance date d:
        entry_price[ticker] = price at latest Date ≤ d
        exit_price[ticker]  = price at latest Date ≤ d + horizon_days
        fwd_ret[ticker]     = log(exit_price / entry_price)
"""
import datetime
import logging
from typing import Optional

import numpy as np
import pandas as pd

from irp.panel.load import load_prices_wide

logger = logging.getLogger(__name__)


def forward_returns_panel(
    rebalance_dates: list[datetime.date],
    horizon_days: int,
    tickers: Optional[list[str]] = None,
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

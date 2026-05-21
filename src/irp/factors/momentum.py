"""Momentum and trend factors: trailing price returns, realised vol, MA ratio.

All inputs must be PIT-safe (caller passes only prices up to as_of_date).
No DB access — pure computation.
"""
import datetime

import numpy as np
import pandas as pd

from irp.factors._cols import PRICE_CLOSE, PRICE_DATE, PRICE_TICKER
from irp.factors._pit import pit_price

# Calendar-day approximations for trading-day lags
_LAG_1M  = 30    # skip last month (short-term reversal avoidance)
_LAG_6M  = 182
_LAG_12M = 365


def _lag_close(
    prices: pd.DataFrame,
    as_of_date: datetime.date,
    lag_days: int,
) -> pd.Series:
    """Closest available close per ticker on or before (as_of_date - lag_days)."""
    prices = prices.copy()
    prices[PRICE_DATE] = pd.to_datetime(prices[PRICE_DATE])
    target = pd.Timestamp(as_of_date) - pd.Timedelta(days=lag_days)
    eligible = prices.loc[prices[PRICE_DATE] <= target]
    if eligible.empty:
        return pd.Series(dtype=float, name=PRICE_CLOSE)
    idx = eligible.groupby(PRICE_TICKER)[PRICE_DATE].idxmax()
    return eligible.loc[idx].set_index(PRICE_TICKER)[PRICE_CLOSE]


def compute_momentum(
    prices: pd.DataFrame,
    as_of_date: datetime.date,
) -> pd.DataFrame:
    """Compute price-based momentum and trend factors.

    Parameters
    ----------
    prices     : Raw prices (Ticker, Date, Close, …) for all dates.
                 Caller is responsible for PIT safety (pass only dates <= as_of_date).
    as_of_date : Snapshot date.

    Returns
    -------
    DataFrame indexed by Ticker with columns:
        mom_12_1    log(P_1m / P_12m)  12m return skipping last month
        mom_6_1     log(P_1m / P_6m)   6m return skipping last month
        vol_21d     std(log_ret_1d, 21) * sqrt(252)  annualised realised vol
        ma200_ratio P_0 / mean(last 200 closes)

    Tickers with insufficient history return NaN for affected factors.
    """
    if prices.empty:
        return pd.DataFrame(columns=['mom_12_1', 'mom_6_1', 'vol_21d', 'ma200_ratio'])

    prices = prices.copy()
    prices[PRICE_DATE] = pd.to_datetime(prices[PRICE_DATE])
    cutoff = pd.Timestamp(as_of_date)
    eligible = prices.loc[prices[PRICE_DATE] <= cutoff].sort_values(
        [PRICE_TICKER, PRICE_DATE]
    )

    # Current price (P_0)
    p0_frame = pit_price(prices, as_of_date)
    p0 = p0_frame.set_index(PRICE_TICKER)[PRICE_CLOSE] if not p0_frame.empty else pd.Series(dtype=float)

    # Lag prices
    p_1m  = _lag_close(prices, as_of_date, _LAG_1M)
    p_6m  = _lag_close(prices, as_of_date, _LAG_6M)
    p_12m = _lag_close(prices, as_of_date, _LAG_12M)

    # Log momentum (NaN when either lag price is missing or non-positive)
    def _log_ret(p_num: pd.Series, p_den: pd.Series) -> pd.Series:
        ratio = p_num / p_den
        return np.log(ratio.where(ratio > 0))

    mom_12_1 = _log_ret(p_1m, p_12m)
    mom_6_1  = _log_ret(p_1m, p_6m)

    # Realised vol: std of last 21 daily log returns * sqrt(252)
    def _vol(g: pd.DataFrame) -> float:
        closes = g[PRICE_CLOSE].values
        if len(closes) < 22:
            return float('nan')
        log_rets = np.log(closes[-21:] / closes[-22:-1])
        return float(log_rets.std() * np.sqrt(252))

    vol_21d = eligible.groupby(PRICE_TICKER).apply(_vol)

    # MA200 ratio: current price / mean of last 200 closes
    def _ma200(g: pd.DataFrame) -> float:
        closes = g[PRICE_CLOSE].values
        if len(closes) < 200:
            return float('nan')
        return float(closes[-200:].mean())

    ma200 = eligible.groupby(PRICE_TICKER).apply(_ma200)
    ma200_ratio = p0 / ma200.where(ma200 > 0)

    out = pd.DataFrame({
        'mom_12_1':    mom_12_1,
        'mom_6_1':     mom_6_1,
        'vol_21d':     vol_21d,
        'ma200_ratio': ma200_ratio,
    })
    out.index.name = PRICE_TICKER
    return out

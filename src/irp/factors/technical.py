"""TA factor specs and computation using TA-Lib.

Adding a new TA factor:
  1. Add one TaSpec entry to _TA_SPECS.
  2. Re-run `build_panels()` to precompute the TA panel.
  3. Clear + rebuild the factor cache.

TaSpec.fn is the canonical implementation (TA-Lib). It is used by:
  - build_ta_panel(): precomputes the full TA panel (called once per ETL refresh).
  - ticker_factor_history(): single-ticker historical factors.
  - _ta_snapshots() fallback when ta_panel.parquet is missing.
"""
import datetime
from dataclasses import dataclass
from typing import Callable

import numpy as np
import pandas as pd
import talib

from irp.factors._cols import PRICE_CLOSE, PRICE_DATE, PRICE_TICKER
from irp.factors.registry import _register as register


# ---------------------------------------------------------------------------
# TaSpec
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class TaSpec:
    name: str
    label: str
    fn: Callable[[pd.Series], pd.Series]
    pct: bool = False
    group: str = 'momentum'


def _tl(c: pd.Series, arr: np.ndarray) -> pd.Series:
    """Wrap a TA-Lib output array as a Series with the same index as c."""
    return pd.Series(arr, index=c.index)


_TA_SPECS: list[TaSpec] = [
    TaSpec(
        name='rsi_14', label='RSI(14)',
        fn=lambda c: _tl(c, talib.RSI(c.to_numpy(dtype='float64'), timeperiod=14)),
    ),
    TaSpec(
        name='macd_hist', label='MACD Hist',
        fn=lambda c: _tl(c, talib.MACD(c.to_numpy(dtype='float64'))[2]),
    ),
    TaSpec(
        name='macd_norm', label='MACD/Price',
        fn=lambda c: _tl(c, talib.MACD(c.to_numpy(dtype='float64'))[0] / c.to_numpy(dtype='float64')),
    ),
    TaSpec(
        name='bb_pct', label='Bollinger %B',
        fn=lambda c: _tl(c, (lambda a, b: (a - b[2]) / (b[0] - b[2]))(c.to_numpy(dtype='float64'), talib.BBANDS(c.to_numpy(dtype='float64'), timeperiod=20))),
    ),
    TaSpec(
        name='ma7_ma28', label='MA7>MA28',
        fn=lambda c: _tl(c, (talib.SMA(c.to_numpy(dtype='float64'), timeperiod=7) > talib.SMA(c.to_numpy(dtype='float64'), timeperiod=28)).astype(float)),
    ),
    TaSpec(
        name='ma14_ma56', label='MA14>MA56',
        fn=lambda c: _tl(c, (talib.SMA(c.to_numpy(dtype='float64'), timeperiod=14) > talib.SMA(c.to_numpy(dtype='float64'), timeperiod=56)).astype(float)),
    ),
]

for _s in _TA_SPECS:
    register(_s.name, _s.label, pct=_s.pct, group=_s.group)


# ---------------------------------------------------------------------------
# Compute functions
# ---------------------------------------------------------------------------

def _compute_ta_snapshot(close: pd.Series) -> dict[str, float]:
    """All registered TA indicators on one Close Series; returns last-bar values."""
    result: dict[str, float] = {}
    for s in _TA_SPECS:
        try:
            val = s.fn(close).iloc[-1]
            result[s.name] = float(val) if pd.notna(val) else float('nan')
        except Exception:
            result[s.name] = float('nan')
    return result


def _compute_technical(
    prices: pd.DataFrame,
    as_of_date: datetime.date,
) -> pd.DataFrame:
    """TA factors for a single ticker up to as_of_date.

    Parameters
    ----------
    prices       : Long-format DataFrame with Ticker, Date, Close columns.
    as_of_date   : Snapshot date (PIT cutoff).

    Returns a one-row DataFrame indexed by Ticker.
    """
    if prices.empty or PRICE_CLOSE not in prices.columns:
        return pd.DataFrame(columns=[s.name for s in _TA_SPECS])

    prices = prices.copy()
    prices[PRICE_DATE] = pd.to_datetime(prices[PRICE_DATE])
    eligible = prices[prices[PRICE_DATE] <= pd.Timestamp(as_of_date)].sort_values(PRICE_DATE)

    if eligible.empty:
        return pd.DataFrame(columns=[s.name for s in _TA_SPECS])

    close = eligible[PRICE_CLOSE].reset_index(drop=True).astype(float)
    vals = _compute_ta_snapshot(close)
    ticker = prices[PRICE_TICKER].iloc[0] if PRICE_TICKER in prices.columns else 'unknown'
    return pd.DataFrame([vals], index=pd.Index([ticker], name=PRICE_TICKER))

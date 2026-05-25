"""PIT alignment helpers — polars-based.

Mirrors the semantics of `irp.factors._pit` but operates on the panel parquets.
"""
import datetime
from typing import Sequence

import numpy as np
import polars as pl

from irp.panel.load import PricePanel


def pit_latest(df: pl.DataFrame, as_of: datetime.date) -> pl.DataFrame:
    """Latest filing per Ticker where eff_date ≤ as_of.

    Tie-broken by Report Date (latest wins).
    """
    return (
        df.filter(pl.col('eff_date') <= as_of)
          .sort(['Ticker', 'Report Date'], descending=[False, True])
          .group_by('Ticker', maintain_order=True)
          .head(1)
    )


def pit_ttm(
    df: pl.DataFrame,
    as_of: datetime.date,
    sum_cols: Sequence[str],
    n: int = 4,
) -> pl.DataFrame:
    """Sum of last `n` quarterly filings per ticker where eff_date ≤ as_of.

    Returns one row per Ticker, with `sum_cols` summed; non-sum columns dropped.
    """
    last_n = (
        df.filter(pl.col('eff_date') <= as_of)
          .sort(['Ticker', 'Report Date'], descending=[False, True])
          .group_by('Ticker', maintain_order=True)
          .head(n)
    )
    return last_n.group_by('Ticker', maintain_order=True).agg(
        [pl.col(c).sum().alias(c) for c in sum_cols]
    )


def pit_price_row(panel: PricePanel, as_of: datetime.date) -> dict[str, float]:
    """Latest close per ticker at/before `as_of` (dict {ticker: close})."""
    i = int(np.searchsorted(panel.dates, np.datetime64(as_of, 'D'), side='right')) - 1
    if i < 0:
        return {}
    row = panel.values[i]
    return {t: float(v) for t, v in zip(panel.tickers, row) if np.isfinite(v)}


def pit_price_at_offset(
    panel: PricePanel, as_of: datetime.date, days_back: int,
) -> dict[str, float]:
    """Latest close per ticker at/before (as_of − days_back). Empty dict if out of range."""
    target = as_of - datetime.timedelta(days=days_back)
    return pit_price_row(panel, target)

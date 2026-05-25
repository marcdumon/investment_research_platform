"""Point-in-time (PIT) alignment utilities — pandas impl.

Used by the single-ticker `ticker_factor_history` pipeline. The canonical
algorithm spec also has a polars mirror in `irp.panel.pit` used by the
hot-path cross-section engine. Both implementations share these rules:

- Effective public date = `Publish Date` when present, else `Report Date + 60d`
- `pit_latest`: most-recent eligible row per ticker
- `pit_ttm`:    sum of last `n` eligible rows per ticker (numeric cols)
- `pit_price`:  most-recent close per ticker on or before as_of_date

Pure functions; no DB access.
"""
import datetime

import pandas as pd

from irp.factors._cols import (
    PRICE_CLOSE, PRICE_DATE, PRICE_TICKER,
    PUBLISH_DATE, REPORT_DATE, TICKER,
)


def _eff_date(df: pd.DataFrame) -> pd.Series:
    """Effective public date: Publish Date if present, else Report Date + 60 days."""
    rd = pd.to_datetime(df[REPORT_DATE])
    if PUBLISH_DATE in df.columns:
        pub = pd.to_datetime(df[PUBLISH_DATE])
        return pub.where(pub.notna(), rd + pd.Timedelta(days=60))
    return rd + pd.Timedelta(days=60)


def pit_latest(
    fundamentals: pd.DataFrame,
    as_of_date: datetime.date,
) -> pd.DataFrame:
    """Most-recent fundamental row per ticker with effective public date <= as_of_date.

    Effective date eliminates lookahead bias from using Report Date alone.
    Returns a reset-index DataFrame with the same columns as the input.
    """
    cutoff = pd.Timestamp(as_of_date)
    df = fundamentals.copy()
    df[REPORT_DATE] = pd.to_datetime(df[REPORT_DATE])
    eligible = df.loc[_eff_date(df) <= cutoff]
    if eligible.empty:
        return df.iloc[0:0].reset_index(drop=True)
    idx = eligible.groupby(TICKER)[REPORT_DATE].idxmax()
    return eligible.loc[idx].reset_index(drop=True)


def pit_ttm(
    fundamentals: pd.DataFrame,
    as_of_date: datetime.date,
    n: int = 4,
) -> pd.DataFrame:
    """Sum last-n quarterly filings per ticker up to as_of_date (TTM when n=4).

    Use for flow statements (income, cashflow) with quarterly variant so ratios
    reflect a full year of activity rather than a single quarter. Non-numeric
    columns are taken from the most recent filing. Balance sheet should still
    use pit_latest. Tickers with no eligible rows are absent from the result.
    """
    cutoff = pd.Timestamp(as_of_date)
    df = fundamentals.copy()
    df[REPORT_DATE] = pd.to_datetime(df[REPORT_DATE])
    eligible = df.loc[_eff_date(df) <= cutoff].sort_values([TICKER, REPORT_DATE])

    if eligible.empty:
        return df.iloc[0:0].reset_index(drop=True)

    numeric_cols = eligible.select_dtypes(include='number').columns.tolist()
    _rank = eligible.groupby(TICKER, sort=False).cumcount(ascending=False)
    last_n = eligible[_rank < n]
    sums = last_n.groupby(TICKER)[numeric_cols].sum()
    most_recent = eligible[_rank == 0].set_index(TICKER).copy()
    most_recent.update(sums)
    return most_recent.reset_index()


def pit_price(
    prices: pd.DataFrame,
    as_of_date: datetime.date,
) -> pd.DataFrame:
    """Closing price on the nearest trading day <= as_of_date per ticker.

    Returns columns [Ticker, Date, Close]. Tickers with no eligible price absent.
    """
    cutoff = pd.Timestamp(as_of_date)
    df = prices.copy()
    df[PRICE_DATE] = pd.to_datetime(df[PRICE_DATE])
    eligible = df.loc[df[PRICE_DATE] <= cutoff]
    if eligible.empty:
        return pd.DataFrame(columns=[PRICE_TICKER, PRICE_DATE, PRICE_CLOSE])
    idx = eligible.groupby(PRICE_TICKER)[PRICE_DATE].idxmax()
    return eligible.loc[idx, [PRICE_TICKER, PRICE_DATE, PRICE_CLOSE]].reset_index(drop=True)

"""Point-in-time (PIT) alignment utilities.

Pure functions — no DB access. Input DataFrames come from irp.query.*
"""
import datetime

import pandas as pd

from irp.factors._cols import TICKER, REPORT_DATE, PRICE_TICKER, PRICE_DATE, PRICE_CLOSE


def pit_latest(
    fundamentals: pd.DataFrame,
    as_of_date: datetime.date,
) -> pd.DataFrame:
    """Return the most-recent fundamental row per ticker where Report Date <= as_of_date.

    Uses the actual SimFin Report Date (filing date) so no fixed lag heuristic is needed.
    Tickers whose every row has Report Date > as_of_date are absent from the result.

    Returns a reset-index DataFrame with the same columns as the input.
    """
    df = fundamentals.copy()
    df[REPORT_DATE] = pd.to_datetime(df[REPORT_DATE])
    cutoff = pd.Timestamp(as_of_date)
    eligible = df.loc[df[REPORT_DATE] <= cutoff]
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
    reflect a full year of activity rather than a single quarter.
    Non-numeric columns are taken from the most recent filing.
    Balance sheet (stock) data should still use pit_latest.

    Tickers with no eligible rows are absent from the result.
    """
    df = fundamentals.copy()
    df[REPORT_DATE] = pd.to_datetime(df[REPORT_DATE])
    cutoff = pd.Timestamp(as_of_date)
    eligible = df.loc[df[REPORT_DATE] <= cutoff]
    if eligible.empty:
        return df.iloc[0:0].reset_index(drop=True)

    eligible = eligible.sort_values([TICKER, REPORT_DATE])
    numeric_cols = eligible.select_dtypes(include='number').columns.tolist()

    rows = []
    for _, g in eligible.groupby(TICKER, sort=False):
        last_n = g.tail(n)
        row = last_n.iloc[[-1]].copy()
        row[numeric_cols] = last_n[numeric_cols].sum().values
        rows.append(row)

    if not rows:
        return eligible.iloc[0:0].reset_index(drop=True)
    return pd.concat(rows, ignore_index=True)


def pit_price(
    prices: pd.DataFrame,
    as_of_date: datetime.date,
) -> pd.DataFrame:
    """Return the closing price on the nearest trading day <= as_of_date per ticker.

    Returns a DataFrame with columns [Ticker, Date, Close].
    Tickers with no price on or before as_of_date are absent.
    """
    df = prices.copy()
    df[PRICE_DATE] = pd.to_datetime(df[PRICE_DATE])
    cutoff = pd.Timestamp(as_of_date)
    eligible = df.loc[df[PRICE_DATE] <= cutoff]
    if eligible.empty:
        return pd.DataFrame(columns=[PRICE_TICKER, PRICE_DATE, PRICE_CLOSE])
    idx = eligible.groupby(PRICE_TICKER)[PRICE_DATE].idxmax()
    return eligible.loc[idx, [PRICE_TICKER, PRICE_DATE, PRICE_CLOSE]].reset_index(drop=True)

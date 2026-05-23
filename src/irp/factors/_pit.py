"""Point-in-time (PIT) alignment utilities.

Pure functions — no DB access. Input DataFrames come from irp.query.*
"""

import datetime

import pandas as pd

from irp.factors._cols import TICKER, REPORT_DATE, PUBLISH_DATE, PRICE_TICKER, PRICE_DATE, PRICE_CLOSE

_EFF_COL = '_eff'


def _eff_date(df: pd.DataFrame) -> pd.Series:
    """Effective public date: Publish Date if present, else Report Date + 60 days."""
    rd = pd.to_datetime(df[REPORT_DATE])
    if PUBLISH_DATE in df.columns:
        pub = pd.to_datetime(df[PUBLISH_DATE])
        return pub.where(pub.notna(), rd + pd.Timedelta(days=60))
    return rd + pd.Timedelta(days=60)


def pit_prepare(df: pd.DataFrame, kind: str = 'fundamental') -> pd.DataFrame:
    """Pre-process a raw DataFrame once before a multi-date rebalance loop.

    Avoids O(N × rows) repeated date parsing across N rebalance dates.
    Pass the result to pit_latest / pit_ttm / pit_price; they detect the
    pre-processed state via the '_eff' column (fundamental) or datetime dtype (price)
    and skip redundant work.

    kind='fundamental': parses REPORT_DATE + PUBLISH_DATE, adds '_eff', sorts by
                        [TICKER, REPORT_DATE].
    kind='price'      : parses PRICE_DATE only.
    """
    out = df.copy()
    if kind == 'fundamental':
        out[REPORT_DATE] = pd.to_datetime(out[REPORT_DATE])
        if PUBLISH_DATE in out.columns:
            out[PUBLISH_DATE] = pd.to_datetime(out[PUBLISH_DATE])
        out[_EFF_COL] = _eff_date(out)
        out = out.sort_values([TICKER, REPORT_DATE])
    else:
        out[PRICE_DATE] = pd.to_datetime(out[PRICE_DATE])
    return out


def pit_latest(
    fundamentals: pd.DataFrame,
    as_of_date: datetime.date,
) -> pd.DataFrame:
    """Return the most-recent fundamental row per ticker with effective public date <= as_of_date.

    Effective date = Publish Date when available, else Report Date + 60 days (conservative
    fallback for rows where the filing date is unknown). This eliminates lookahead bias
    from using Report Date alone.

    Returns a reset-index DataFrame with the same columns as the input.
    If pit_prepare() was called on the input, date parsing and copy are skipped.
    """
    cutoff = pd.Timestamp(as_of_date)
    if _EFF_COL in fundamentals.columns:
        eligible = fundamentals.loc[fundamentals[_EFF_COL] <= cutoff]
        if eligible.empty:
            return fundamentals.iloc[0:0].reset_index(drop=True)
        idx = eligible.groupby(TICKER)[REPORT_DATE].idxmax()
        return eligible.loc[idx].reset_index(drop=True)
    # slow path — pit_prepare was not called
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
    reflect a full year of activity rather than a single quarter.
    Non-numeric columns are taken from the most recent filing.
    Balance sheet (stock) data should still use pit_latest.

    Tickers with no eligible rows are absent from the result.
    If pit_prepare() was called on the input, date parsing, copy, and sort are skipped;
    the inner for-loop is replaced with a vectorized groupby.
    """
    cutoff = pd.Timestamp(as_of_date)

    if _EFF_COL in fundamentals.columns:
        eligible = fundamentals.loc[fundamentals[_EFF_COL] <= cutoff]
        empty_frame = fundamentals
    else:
        # slow path — pit_prepare was not called
        df = fundamentals.copy()
        df[REPORT_DATE] = pd.to_datetime(df[REPORT_DATE])
        eligible = df.loc[_eff_date(df) <= cutoff].sort_values([TICKER, REPORT_DATE])
        empty_frame = df

    if eligible.empty:
        return empty_frame.iloc[0:0].reset_index(drop=True)

    numeric_cols = eligible.select_dtypes(include='number').columns.tolist()
    # rank 0 = most recent row per ticker (frame sorted ascending by date)
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
    """Return the closing price on the nearest trading day <= as_of_date per ticker.

    Returns a DataFrame with columns [Ticker, Date, Close].
    Tickers with no price on or before as_of_date are absent.
    If pit_prepare() was called on the input, date parsing and copy are skipped.
    """
    cutoff = pd.Timestamp(as_of_date)
    if pd.api.types.is_datetime64_any_dtype(prices[PRICE_DATE]):
        eligible = prices.loc[prices[PRICE_DATE] <= cutoff]
        if eligible.empty:
            return pd.DataFrame(columns=[PRICE_TICKER, PRICE_DATE, PRICE_CLOSE])
        idx = eligible.groupby(PRICE_TICKER)[PRICE_DATE].idxmax()
        return eligible.loc[idx, [PRICE_TICKER, PRICE_DATE, PRICE_CLOSE]].reset_index(drop=True)
    # slow path — pit_prepare was not called
    df = prices.copy()
    df[PRICE_DATE] = pd.to_datetime(df[PRICE_DATE])
    eligible = df.loc[df[PRICE_DATE] <= cutoff]
    if eligible.empty:
        return pd.DataFrame(columns=[PRICE_TICKER, PRICE_DATE, PRICE_CLOSE])
    idx = eligible.groupby(PRICE_TICKER)[PRICE_DATE].idxmax()
    return eligible.loc[idx, [PRICE_TICKER, PRICE_DATE, PRICE_CLOSE]].reset_index(drop=True)

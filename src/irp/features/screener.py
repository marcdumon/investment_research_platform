"""Screener filter and HTML rendering logic — no widget dependencies."""
import re

import pandas as pd

from irp.features.screener_defs import METRIC_FMT, TABLE_CSS, fmt_cell


def apply_filters(
    df: pd.DataFrame,
    ticker_info: pd.DataFrame,
    sectors: list[str],
    industries: list[str],
    metric_filters: list[tuple[str, float, float]],
    sort_col: str,
    ascending: bool,
    limit: int,
) -> tuple[pd.DataFrame, int]:
    """Filter, sort, and limit metrics DataFrame.

    metric_filters: list of (col, min_val, max_val) — use float('nan') to skip a bound.
    Returns (limited_df, n_matched_before_limit).
    """
    df = df.copy()
    sel_sectors = [s for s in sectors if s != "(All)"]
    sel_industries = [i for i in industries if i != "(All)"]
    if sel_sectors or sel_industries:
        info = ticker_info.copy()
        if sel_sectors:
            info = info[info["sector"].isin(sel_sectors)]
        if sel_industries:
            info = info[info["industry"].isin(sel_industries)]
        df = df[df["ticker"].isin(info["ticker"])]

    for col, lo, hi in metric_filters:
        if col not in df.columns:
            continue
        if not pd.isna(lo):
            df = df[df[col] >= lo]
        if not pd.isna(hi):
            df = df[df[col] <= hi]

    df = df.sort_values(sort_col, ascending=ascending, na_position="last")
    n_matched = len(df)
    return df.head(limit), n_matched


def build_table_html(
    df: pd.DataFrame,
    display_cols: list[str],
    n_total: int,
    n_matched: int,
) -> str:
    """Format filtered DataFrame as styled HTML table with clickable tickers."""
    cols = ["ticker", "period"] + display_cols
    view = df[cols].copy()
    for c in display_cols:
        view[c] = view[c].apply(lambda v, k=METRIC_FMT[c]: fmt_cell(v, k))

    raw = view.to_html(index=False, classes="scr", escape=False)
    count = (
        f'<p style="color:#aaa;font-size:12px;margin:4px 0">'
        f'{n_matched} of {n_total} tickers match — click ticker to add to watchlist'
        f'</p>'
    )
    return (
        TABLE_CSS + count
        + '<div style="max-height:500px;overflow-y:auto;border:1px solid #333">'
        + _make_tickers_clickable(raw)
        + "</div>"
    )


def _make_tickers_clickable(html: str) -> str:
    if "<tbody>" not in html:
        return html
    pre, rest = html.split("<tbody>", 1)
    body, post = rest.split("</tbody>", 1)

    def _process_row(m):
        row = m.group(0)
        td_m = re.search(r'<td[^>]*>([^<]+)</td>', row)
        if not td_m:
            return row
        ticker = td_m.group(1).strip()
        return re.sub(
            r'<td[^>]*>[^<]+</td>',
            f'<td id="ticker-{ticker}" class="scr-ticker"'
            f' style="cursor:pointer;font-weight:bold;color:#58a6ff;text-decoration:underline">'
            f'{ticker}</td>',
            row, count=1,
        )

    body = re.sub(r'<tr[^>]*>.*?</tr>', _process_row, body, flags=re.DOTALL)
    return pre + "<tbody>" + body + "</tbody>" + post

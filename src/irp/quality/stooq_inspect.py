"""Stooq-anomaly inspector. Mirrors irp.quality.simfin_inspect.

Two views for a (ticker, period, rule) finding:
- `inspect()` — windowed candlestick ±pad_days around each sample date
- `inspect_long()` — longer-context candlestick covering the period
                     year ±pad_months

Staleness check:
- `staleness_scan()` — compare Stooq close vs Yahoo adjusted close across all
                       equity tickers at several historical dates; high ratio
                       means Stooq is missing post-snapshot dividend adjustments
"""
import json
import time
from pathlib import Path

import duckdb
import pandas as pd
import plotly.graph_objects as go

from irp.core.config import config

_YAHOO_ERROR_FILE = config.data.root_dir / 'data_quality' / 'stooq_yahoo_error_tickers.json'
_STALENESS_FILE = config.data.root_dir / 'data_quality' / 'stooq_staleness.csv'


def load_yahoo_errors() -> set[str]:
    """Load tickers Yahoo Finance couldn't resolve (persistent error list)."""
    if _YAHOO_ERROR_FILE.exists():
        return set(json.loads(_YAHOO_ERROR_FILE.read_text()))
    return set()


def save_yahoo_errors(errors: set[str]) -> None:
    """Persist unresolvable Yahoo tickers."""
    _YAHOO_ERROR_FILE.parent.mkdir(parents=True, exist_ok=True)
    _YAHOO_ERROR_FILE.write_text(json.dumps(sorted(errors), indent=2))


_STALENESS_COLS = ['Ticker', 'Date', 'Years_back', 'Stooq_C', 'Yahoo_C', 'Ratio']


def load_staleness_results(
    results_file: Path = _STALENESS_FILE,
    threshold: float | None = None,
) -> pd.DataFrame:
    """Load persisted staleness scan results, optionally filtered by threshold."""
    if not results_file.exists():
        return pd.DataFrame(columns=_STALENESS_COLS)
    df = pd.read_csv(results_file)
    if threshold is not None:
        df = df[df['Ratio'] >= threshold]
    return df.sort_values('Ratio', ascending=False).reset_index(drop=True)


def staleness_scan(
    market_patterns: list[str] | None = None,
    years_back: list[int] | None = None,
    threshold: float = 1.02,
    batch_size: int = 20,
    batch_sleep: float = 1.5,
    skip_yahoo_errors: bool = True,
    skip_scanned: bool = True,
    results_file: Path = _STALENESS_FILE,
) -> pd.DataFrame:
    """Compare Stooq close vs Yahoo adjusted close for equity tickers.

    Samples one Stooq trading date per entry in `years_back` (nearest actual
    trading day at or before today - N years). For each date, fetches Yahoo
    `auto_adjust=True` close in batches and computes
    `ratio = stooq_close / yahoo_adj_close`.

    Results are written to `results_file` after every batch so the scan can
    be interrupted and resumed. Returns rows with ratio >= `threshold` from
    the full accumulated file.

    Tickers that Yahoo returns nothing for (not rate-limit failures) are saved
    to `data/data_quality/stooq_yahoo_error_tickers.json` and skipped on future
    runs when `skip_yahoo_errors=True`.

    Args:
        market_patterns: LIKE patterns matched against LOWER(Market)
            (default: ['%stock%', '%etf%']).
        years_back: anchor offsets in years (default: [1, 3, 5]).
        threshold: minimum ratio to include in returned DataFrame (default: 1.02).
            All ratios are written to file regardless.
        batch_size: tickers per yfinance batch call (default: 20).
        batch_sleep: seconds to sleep between batches (default: 1.5).
        skip_yahoo_errors: skip tickers previously found unresolvable on Yahoo
            (default: True). Use False to re-probe error tickers.
        skip_scanned: skip (ticker, years_back) pairs already in `results_file`
            (default: True). Use False to re-scan everything (rows appended;
            file will contain duplicates — delete it first for a clean run).
        results_file: CSV path for incremental writes and resume
            (default: data/data_quality/stooq_staleness.csv).
    """
    try:
        import yfinance as yf
    except ImportError:
        raise ImportError("yfinance required: uv pip install yfinance")

    if market_patterns is None:
        market_patterns = ['%stock%', '%etf%']
    if years_back is None:
        years_back = [1, 3, 5]

    like_clause = ' OR '.join(['LOWER(m.Market) LIKE ?' for _ in market_patterns])
    lower_pats = [p.lower() for p in market_patterns]

    known_errors = load_yahoo_errors() if skip_yahoo_errors else set()
    new_errors: set[str] = set()

    # resume: skip (ticker, yr) pairs already written to file
    if results_file.exists():
        _existing = pd.read_csv(results_file)
        checked: set[tuple[str, int]] = {
            (t, int(y)) for t, y in zip(_existing['Ticker'], _existing['Years_back'])
        } if skip_scanned else set()
        _file_has_header = True
    else:
        checked = set()
        _file_has_header = False

    with duckdb.connect(str(config.database.path), read_only=True) as con:
        all_tickers = con.execute(
            f"SELECT DISTINCT p.Ticker FROM prices p "
            f"JOIN markets m ON m.Ticker = p.Ticker "
            f"WHERE {like_clause} ORDER BY p.Ticker",
            lower_pats,
        ).df()['Ticker'].tolist()

    today = pd.Timestamp.today().normalize()

    for yr in years_back:
        anchor_int = int((today - pd.DateOffset(years=yr)).strftime('%Y%m%d'))

        with duckdb.connect(str(config.database.path), read_only=True) as con:
            row = con.execute(
                "SELECT MAX(Date) FROM prices WHERE Date <= ?", [anchor_int]
            ).fetchone()
        if not row or row[0] is None:
            continue
        trade_date = int(row[0])
        trade_dt = pd.Timestamp(str(trade_date))
        date_str = trade_dt.strftime('%Y-%m-%d')

        with duckdb.connect(str(config.database.path), read_only=True) as con:
            stooq_df = con.execute(
                f"SELECT DISTINCT p.Ticker, p.C AS Stooq_C FROM prices p "
                f"JOIN markets m ON m.Ticker = p.Ticker "
                f"WHERE p.Date = ? AND ({like_clause})",
                [trade_date] + lower_pats,
            ).df()

        if stooq_df.empty:
            continue

        stooq_map = dict(zip(stooq_df['Ticker'], stooq_df['Stooq_C']))
        batch_tickers = [
            t for t in stooq_df['Ticker'].tolist()
            if t not in known_errors and (t, yr) not in checked
        ]
        ystart = (trade_dt - pd.Timedelta(days=4)).date()
        yend = (trade_dt + pd.Timedelta(days=4)).date()

        for i in range(0, len(batch_tickers), batch_size):
            batch = batch_tickers[i : i + batch_size]
            try:
                raw = yf.download(
                    batch, start=ystart, end=yend,
                    auto_adjust=True, progress=False, threads=False,
                )
            except Exception:
                # network / rate-limit failure — don't skip, retry next run
                time.sleep(batch_sleep)
                continue
            time.sleep(batch_sleep)
            if raw is None or raw.empty:
                new_errors.update(batch)
                save_yahoo_errors(known_errors | new_errors)
                continue
            raw.index = pd.to_datetime(raw.index)
            day = raw[raw.index.strftime('%Y-%m-%d') == date_str]
            if day.empty:
                continue
            if isinstance(raw.columns, pd.MultiIndex):
                close = day['Close']
            else:
                close = day[['Close']]
                close.columns = batch
            found: set[str] = set()
            batch_rows: list[dict] = []
            for ticker in close.columns:
                val = close[ticker].iloc[0]
                if pd.notna(val) and val > 0:
                    found.add(str(ticker))
                    stooq_c = stooq_map.get(str(ticker))
                    if stooq_c is not None:
                        batch_rows.append({
                            'Ticker': str(ticker),
                            'Date': date_str,
                            'Years_back': yr,
                            'Stooq_C': float(stooq_c),
                            'Yahoo_C': float(val),
                            'Ratio': float(stooq_c) / float(val),
                        })
            batch_unknown = {t for t in batch if t not in found}
            if batch_unknown:
                new_errors.update(batch_unknown)
                save_yahoo_errors(known_errors | new_errors)
            if batch_rows:
                results_file.parent.mkdir(parents=True, exist_ok=True)
                pd.DataFrame(batch_rows, columns=_STALENESS_COLS).to_csv(
                    results_file, mode='a', header=not _file_has_header, index=False,
                )
                _file_has_header = True

    if new_errors:
        save_yahoo_errors(known_errors | new_errors)

    return load_staleness_results(results_file, threshold=threshold)


def _query_bars(ticker: str, dates: list[int]) -> pd.DataFrame:
    if not dates:
        return pd.DataFrame()
    placeholders = ','.join(str(d) for d in sorted(dates))
    with duckdb.connect(str(config.database.path), read_only=True) as con:
        return con.execute(
            f"SELECT Date, O, H, L, C, V FROM prices "
            f"WHERE Ticker = ? AND Date IN ({placeholders}) ORDER BY Date",
            [ticker],
        ).df()


def _query_range(ticker: str, start: int, end: int) -> pd.DataFrame:
    with duckdb.connect(str(config.database.path), read_only=True) as con:
        return con.execute(
            "SELECT Date, O, H, L, C, V FROM prices "
            "WHERE Ticker = ? AND Date >= ? AND Date <= ? ORDER BY Date",
            [ticker, start, end],
        ).df()


def flagged_bars(ticker: str, sample_dates: list[int]) -> pd.DataFrame:
    """OHLCV rows for each flagged date (Date as YYYY-MM-DD string)."""
    if not sample_dates:
        return pd.DataFrame()
    df = _query_bars(ticker, [int(d) for d in sample_dates])
    if df.empty:
        return df
    df['Date'] = _date_str(df)
    return df[['Date', 'O', 'H', 'L', 'C', 'V']].reset_index(drop=True)


def yahoo_bars(ticker: str, sample_dates: list[int]) -> pd.DataFrame:
    """OHLCV from Yahoo Finance for each flagged date. Returns empty DataFrame
    if yfinance can't resolve the ticker or has no data. Same column shape
    as `flagged_bars` so the two tables can be displayed side-by-side.
    Note: ticker symbols are passed through as-is — Yahoo and Stooq often
    agree for US-listed equities but diverge for indices, ADRs, non-US.
    """
    if not sample_dates:
        return pd.DataFrame()
    try:
        import yfinance as yf
    except ImportError:
        return pd.DataFrame()
    dates = sorted(int(d) for d in sample_dates)
    try:
        start = pd.Timestamp(str(dates[0])).date()
        end = pd.Timestamp(str(dates[-1])).date() + pd.Timedelta(days=1)
    except (TypeError, ValueError):
        return pd.DataFrame()
    # auto_adjust=True so Close/OHLC reflect split + dividend adjustments,
    # matching Stooq's adjusted feed for a meaningful side-by-side comparison.
    try:
        df = yf.download(ticker, start=start, end=end, progress=False, auto_adjust=True)
    except Exception:
        return pd.DataFrame()
    if df is None or df.empty:
        return pd.DataFrame()
    df = df.reset_index()
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = [c[0] if isinstance(c, tuple) else c for c in df.columns]
    df['Date'] = pd.to_datetime(df['Date']).dt.strftime('%Y-%m-%d')
    rename = {'Open': 'O', 'High': 'H', 'Low': 'L', 'Close': 'C', 'Volume': 'V'}
    df = df.rename(columns=rename)
    cols = ['Date'] + [c for c in ('O', 'H', 'L', 'C', 'V') if c in df.columns]
    targets = {f'{str(d)[:4]}-{str(d)[4:6]}-{str(d)[6:8]}' for d in dates}
    return df.loc[df['Date'].isin(targets), cols].reset_index(drop=True)


def yahoo_url(ticker: str, period_str: str | None = None) -> str:
    """Yahoo Finance history page URL for a ticker, optionally scoped to a
    +/- 6 month window around `period_str` (calendar year as string).
    Ticker symbols mostly match across feeds but not always — indices, ADRs,
    and non-US listings may need adjustment.
    """
    base = f'https://finance.yahoo.com/quote/{ticker}/history'
    if not period_str:
        return base
    try:
        year = int(period_str)
    except (TypeError, ValueError):
        return base
    p1 = int(pd.Timestamp(f'{year - 1}-07-01').timestamp())
    p2 = int(pd.Timestamp(f'{year + 1}-06-30').timestamp())
    return f'{base}?period1={p1}&period2={p2}&interval=1d'


def _date_str(df: pd.DataFrame) -> pd.Series:
    return df['Date'].astype(str).str.replace(
        r'^(\d{4})(\d{2})(\d{2})$', r'\1-\2-\3', regex=True,
    )


def inspect(
    rule_name: str,
    ticker: str,
    period_str: str,
    sample_dates: list[int],
    pad_days: int = 5,
) -> go.Figure | None:
    """Windowed candlestick: bars within ±pad_days of each sample date."""
    if not sample_dates:
        return None
    windows: set[int] = set()
    for d in sample_dates:
        ds = str(int(d))
        base = pd.Timestamp(f'{ds[:4]}-{ds[4:6]}-{ds[6:8]}')
        for k in range(-pad_days, pad_days + 1):
            windows.add(int((base + pd.Timedelta(days=k)).strftime('%Y%m%d')))
    bars = _query_bars(ticker, list(windows))
    if bars.empty:
        return None
    bars['date_str'] = _date_str(bars)
    flagged = {int(d) for d in sample_dates}
    fig = go.Figure(data=[go.Candlestick(
        x=bars['date_str'],
        open=bars['O'], high=bars['H'], low=bars['L'], close=bars['C'],
        name='OHLC',
    )])
    flagged_df = bars[bars['Date'].isin(flagged)]
    if len(flagged_df):
        fig.add_trace(go.Scatter(
            x=flagged_df['date_str'],
            y=flagged_df['L'] * 0.99,
            mode='markers',
            marker={'symbol': 'triangle-up', 'size': 12, 'color': 'red'},
            name='flagged',
        ))
    fig.update_layout(
        title=f'{ticker} {period_str} — {rule_name} (±{pad_days}d)',
        height=380, width=560,
        xaxis_rangeslider_visible=False,
        margin={'l': 20, 'r': 20, 't': 40, 'b': 20},
    )
    return fig


def inspect_long(
    ticker: str,
    period_str: str,
    sample_dates: list[int] | None = None,
    pad_months: int = 6,
) -> go.Figure | None:
    """Longer-context candlestick: the period year plus ±pad_months padding.
    If `sample_dates` is given, flagged bars are marked with red triangles."""
    try:
        year = int(period_str)
    except (TypeError, ValueError):
        return None
    start_dt = pd.Timestamp(f'{year}-01-01') - pd.DateOffset(months=pad_months)
    end_dt = pd.Timestamp(f'{year}-12-31') + pd.DateOffset(months=pad_months)
    start = int(start_dt.strftime('%Y%m%d'))
    end = int(end_dt.strftime('%Y%m%d'))
    bars = _query_range(ticker, start, end)
    if bars.empty:
        return None
    bars['date_str'] = _date_str(bars)
    fig = go.Figure(data=[go.Candlestick(
        x=bars['date_str'],
        open=bars['O'], high=bars['H'], low=bars['L'], close=bars['C'],
        name='OHLC',
    )])
    if sample_dates:
        flagged = {int(d) for d in sample_dates}
        flagged_df = bars[bars['Date'].isin(flagged)]
        if len(flagged_df):
            fig.add_trace(go.Scatter(
                x=flagged_df['date_str'],
                y=flagged_df['L'] * 0.99,
                mode='markers',
                marker={'symbol': 'triangle-up', 'size': 8, 'color': 'red'},
                name='flagged',
            ))
    fig.update_layout(
        title=f'{ticker} context — {period_str} ±{pad_months}mo',
        height=380, width=720,
        xaxis_rangeslider_visible=False,
        margin={'l': 20, 'r': 20, 't': 40, 'b': 20},
    )
    return fig

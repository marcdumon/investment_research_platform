"""Yahoo provider — produces `prices`, `dividends`, `splits` via yfinance.

Acquire is resume-safe through JSON sets in raw_dir (`queried_prices.json`,
`queried_actions.json`, `error_tickers.json`): interrupt any time and rerun.

The `incremental` flag replaces the old bulk/update split:
  - full (incremental=False): skip already-queried/errored tickers (fast resume).
  - incremental (True): prices fetched since each ticker's last stored Date;
    actions re-fetched in full so new dividend/split events are caught (the
    merge dedups). No separate session-marker state machine.
"""
import logging
import time
from collections.abc import Sequence
from datetime import date, timedelta
from pathlib import Path
from typing import Any

import duckdb
import pandas as pd

from dataload._sql import copy_to_parquet, reader
from dataload.context import IngestContext
from dataload.providers.base import Capability
from dataload.state import JsonSet

logger = logging.getLogger(__name__)

_ACTIONS_COLS = ['Ticker', 'Date', 'Type', 'Value']
_LOOKBACK_DAYS = 30
_ACTIONS_UPDATE_SESSION = '.actions_update_session'


# --------------------------------------------------------------------------- #
# pure helpers
# --------------------------------------------------------------------------- #
def _todo(ticker_map: dict[str, str], queried: set[str], errors: set[str]) -> list[str]:
    """Tickers still to fetch: in the map, not already queried, not known-bad."""
    return [t for t in ticker_map if t not in queried and t not in errors]


def _batch_start_kwargs(starts: list[date | None]) -> dict:
    """yf.download date kwargs for a price batch.

    Uses the earliest known last-date (+1 day) as the shared start, so the batch
    refetches only recent bars. Full history (period='max') is used only when NO
    ticker in the batch has a stored date — a single new (date-less) ticker no
    longer drags the whole batch into a multi-decade refetch.
    """
    known = [s for s in starts if s is not None]
    if known:
        return {'start': (min(known) + timedelta(days=1)).isoformat()}
    return {'period': 'max'}


def _actions_to_long(ticker: str, actions: pd.DataFrame) -> pd.DataFrame:
    """yfinance `.actions` (DatetimeIndex + Dividends / Stock Splits columns) ->
    long-form `(Ticker, Date, Type, Value)` rows where Value > 0."""
    df = actions.reset_index()
    date_col = df.columns[0]
    df['Date'] = pd.to_datetime(df[date_col]).dt.strftime('%Y-%m-%d')
    parts: list[pd.DataFrame] = []
    if 'Dividends' in df.columns:
        div = df.loc[df['Dividends'] > 0, ['Date', 'Dividends']].rename(columns={'Dividends': 'Value'})
        div['Type'] = 'dividend'
        parts.append(div)
    if 'Stock Splits' in df.columns:
        spl = df.loc[df['Stock Splits'] > 0, ['Date', 'Stock Splits']].rename(columns={'Stock Splits': 'Value'})
        spl['Type'] = 'split'
        parts.append(spl)
    if not parts:
        return pd.DataFrame(columns=_ACTIONS_COLS)
    out = pd.concat(parts, ignore_index=True)
    if out.empty:
        return pd.DataFrame(columns=_ACTIONS_COLS)
    out.insert(0, 'Ticker', ticker)
    return out[_ACTIONS_COLS]


def _prices_to_long(ticker: str, hist: pd.DataFrame) -> pd.DataFrame:
    """yfinance `.history()` OHLCV -> long-form `(Date, Ticker, Open..Volume)` rows."""
    df = hist[['Open', 'High', 'Low', 'Close', 'Volume']].copy()
    df.index = pd.to_datetime(df.index).strftime('%Y-%m-%d')
    df.index.name = 'Date'
    df.insert(0, 'Ticker', ticker)
    return df.reset_index()


def _extract_batch_slice(raw: pd.DataFrame, ticker: str, single_ticker: bool) -> pd.DataFrame | None:
    try:
        df = raw if single_ticker else raw[ticker]
    except KeyError:
        return None
    if df.empty or 'Close' not in df.columns:
        return None
    df = df.dropna(subset=['Close'])
    return df if not df.empty else None


def _append_csv(df: pd.DataFrame, path: Path, has_header: bool) -> bool:
    df.to_csv(path, mode='a', header=not has_header, index=False)
    return True


# --------------------------------------------------------------------------- #
# DB reads (universe map + incremental anchors)
# --------------------------------------------------------------------------- #
def _load_ticker_map(ctx: IngestContext) -> dict[str, str]:
    """Canonical -> yahoo_ticker from `universe`, excluding configured markets."""
    excludes = [m.lower() for m in ctx.cfg('yahoo').get('markets_exclude', [])]
    placeholders = ', '.join('?' for _ in excludes)
    where_excl = f'AND LOWER(Market) NOT IN ({placeholders})' if excludes else ''
    with ctx.connect() as con:
        df = con.execute(
            f'SELECT Ticker, yahoo_ticker FROM universe '
            f'WHERE yahoo_ticker IS NOT NULL {where_excl} ORDER BY Ticker',
            excludes,
        ).df()
    return dict(zip(df['Ticker'], df['yahoo_ticker'], strict=False))


def _load_last_prices_dates(ctx: IngestContext) -> dict[str, date]:
    """Per-ticker last stored Yahoo Date in the unified `prices` table."""
    try:
        with ctx.connect() as con:
            rows = con.execute(
                "SELECT Ticker, MAX(Date) FROM prices WHERE Src = 'yahoo' GROUP BY Ticker"
            ).fetchall()
        return {ticker: d for ticker, d in rows}
    except duckdb.CatalogException:
        return {}


# --------------------------------------------------------------------------- #
# acquire (network)
# --------------------------------------------------------------------------- #
def _fetch_actions(ctx: IngestContext, yf: Any, raw: Path, ticker_map: dict[str, str], incremental: bool) -> None:
    """Fetch dividends + splits per ticker.

    Actions have no incremental API, so an `incremental` run re-scans every
    ticker to catch new events — but it does so under a *session* tracker
    (`queried_actions_update.json` + a `.actions_update_session` marker) that is
    reset only at the start of a fresh session, so an interrupted run resumes
    instead of restarting from zero. A `bulk` run uses the persistent
    `queried_actions.json`, skipping tickers already fetched across runs.
    """
    errors_set = JsonSet(raw / 'error_tickers.json')
    actions_file = raw / 'actions.csv'
    sleep = ctx.cfg('yahoo').get('actions_sleep', 0.5)

    if incremental:
        tracker = JsonSet(raw / 'queried_actions_update.json')
        session = raw / _ACTIONS_UPDATE_SESSION
        if not session.exists():
            tracker.save(set())   # fresh session — re-scan everything
            session.touch()
    else:
        tracker = JsonSet(raw / 'queried_actions.json')
        session = None

    queried = tracker.load()
    errors = errors_set.load()
    has_header = actions_file.exists()
    todo = _todo(ticker_map, queried, errors)
    logger.info(f'Yahoo actions: {len(todo)} tickers')
    completed = True
    for ticker in todo:
        if ctx.is_cancelled():
            completed = False
            break
        try:
            actions = yf.Ticker(ticker_map[ticker]).actions
        except Exception as e:  # noqa: BLE001 — yfinance raises many error types
            logger.warning(f'{ticker}: {type(e).__name__}: {e}')
            errors.add(ticker)
            errors_set.save(errors)
            time.sleep(sleep)
            continue
        time.sleep(sleep)
        if actions is not None and not actions.empty:
            rows = _actions_to_long(ticker, actions)
            if not rows.empty:
                has_header = _append_csv(rows, actions_file, has_header)
        queried.add(ticker)
        tracker.save(queried)

    if session is not None and completed:
        session.unlink(missing_ok=True)   # clean finish — next session re-scans


def _fetch_prices(ctx: IngestContext, yf: Any, raw: Path, ticker_map: dict[str, str],
                  skip_queried: bool, last_dates: dict[str, date] | None) -> None:
    queried_set = JsonSet(raw / 'queried_prices.json')
    errors_set = JsonSet(raw / 'error_tickers.json')
    prices_file = raw / 'prices.csv'
    cfg = ctx.cfg('yahoo')
    bsize = cfg.get('prices_batch_size', 50)
    sleep = cfg.get('batch_sleep', 1.0)

    queried = queried_set.load() if skip_queried else set()
    errors = errors_set.load()
    has_header = prices_file.exists()
    todo = _todo(ticker_map, queried, errors)
    n_batches = (len(todo) + bsize - 1) // bsize
    logger.info(f'Yahoo prices (batch size={bsize}): {len(todo)} tickers, {n_batches} batches')

    for i in range(0, len(todo), bsize):
        if ctx.is_cancelled():
            break
        batch = todo[i:i + bsize]
        batch_yahoo = [ticker_map[t] for t in batch]
        starts = [last_dates.get(t) for t in batch] if last_dates is not None else []
        dl_kwargs = _batch_start_kwargs(starts)
        logger.info(f'Batch {i // bsize + 1}/{n_batches}: {len(batch)} tickers, start={dl_kwargs.get("start", "max")}')
        try:
            data = yf.download(batch_yahoo, **dl_kwargs, auto_adjust=True, progress=False, group_by='ticker')
        except Exception as e:  # noqa: BLE001
            logger.warning(f'batch [{i}:{i + len(batch)}] failed: {type(e).__name__}: {e}')
            time.sleep(sleep)
            continue
        time.sleep(sleep)
        if data is None or data.empty:
            errors.update(batch)
            errors_set.save(errors)
            continue
        for ticker in batch:
            df = _extract_batch_slice(data, ticker_map[ticker], single_ticker=(len(batch) == 1))
            if df is None or df.empty:
                errors.add(ticker)
                errors_set.save(errors)
                continue
            has_header = _append_csv(_prices_to_long(ticker, df), prices_file, has_header)
            queried.add(ticker)
            queried_set.save(queried)


# --------------------------------------------------------------------------- #
# normalize (raw csv -> canonical parquet)
# --------------------------------------------------------------------------- #
def _normalize_prices(con: duckdb.DuckDBPyConnection, src: Path, out: Path) -> Path:
    return copy_to_parquet(con, f"""
        SELECT Ticker, CAST(Date AS DATE) AS Date,
               Open, High, Low, Close, CAST(Volume AS BIGINT) AS Volume,
               'yahoo' AS Src, Ticker AS SrcId
        FROM {reader(src)}
    """, out)


def _normalize_actions(con: duckdb.DuckDBPyConnection, src: Path, div_out: Path, spl_out: Path) -> None:
    copy_to_parquet(con, f"""
        SELECT Ticker, CAST(Date AS DATE) AS Date, Value AS Amount,
               'yahoo' AS Src, Ticker AS SrcId
        FROM {reader(src)} WHERE Type = 'dividend'
    """, div_out)
    copy_to_parquet(con, f"""
        SELECT Ticker, CAST(Date AS DATE) AS Date, Value AS Ratio,
               'yahoo' AS Src, Ticker AS SrcId
        FROM {reader(src)} WHERE Type = 'split'
    """, spl_out)


class YahooProvider:
    name = 'yahoo'

    def capabilities(self) -> dict[str, Capability]:
        return {
            'prices': Capability(incremental=True),
            'dividends': Capability(incremental=True),
            'splits': Capability(incremental=True),
        }

    def produce(self, ctx: IngestContext, datasets: Sequence[str], *, incremental: bool) -> dict[str, Path]:
        want_prices = 'prices' in datasets
        want_div = 'dividends' in datasets
        want_spl = 'splits' in datasets
        want_actions = want_div or want_spl
        if not (want_prices or want_actions):
            return {}

        raw = ctx.raw_dir('yahoo')
        processed = ctx.processed_dir('yahoo')
        raw.mkdir(parents=True, exist_ok=True)
        processed.mkdir(parents=True, exist_ok=True)
        ticker_map = _load_ticker_map(ctx)

        import yfinance as yf
        if want_actions:
            _fetch_actions(ctx, yf, raw, ticker_map, incremental=incremental)
        if want_prices:
            last_dates = None
            if incremental:
                last_dates = {t: d - timedelta(days=_LOOKBACK_DAYS)
                              for t, d in _load_last_prices_dates(ctx).items()}
            _fetch_prices(ctx, yf, raw, ticker_map, skip_queried=not incremental, last_dates=last_dates)

        return self._normalize(raw, processed, want_prices, want_div, want_spl)

    def _normalize(self, raw: Path, processed: Path, want_prices: bool,
                   want_div: bool, want_spl: bool) -> dict[str, Path]:
        out: dict[str, Path] = {}
        con = duckdb.connect()
        try:
            if want_prices and (raw / 'prices.csv').exists():
                out['prices'] = _normalize_prices(con, raw / 'prices.csv', processed / 'prices.parquet')
            if (want_div or want_spl) and (raw / 'actions.csv').exists():
                div_out = processed / 'dividends.parquet'
                spl_out = processed / 'splits.parquet'
                _normalize_actions(con, raw / 'actions.csv', div_out, spl_out)
                if want_div:
                    out['dividends'] = div_out
                if want_spl:
                    out['splits'] = spl_out
        finally:
            con.close()
        return out

    def cleanup(self, ctx: IngestContext) -> None:
        """Delete normalized parquets; keep raw csvs, JSON resume-state, markers, DB."""
        processed = ctx.processed_dir('yahoo')
        for name in ('prices.parquet', 'dividends.parquet', 'splits.parquet'):
            p = processed / name
            if p.exists():
                p.unlink()

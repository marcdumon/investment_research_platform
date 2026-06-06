import logging
import time
from datetime import date, timedelta
from pathlib import Path
from typing import Literal

import duckdb
import pandas as pd

from irp.core.config import config
from irp.core.duckdb_merge import merge_csv
from irp.core.freshness import is_fresh
from irp.core.jsonset import JsonSet
from irp.runner import Feed

logger = logging.getLogger(__name__)

root_dir = config.data.root_dir
yahoo_cfg = config.providers.yahoo
raw_dir = root_dir / yahoo_cfg.raw_dir
processed_dir = root_dir / yahoo_cfg.processed_dir

_ACTIONS_FILE = raw_dir / 'actions.csv'
_PRICES_FILE = raw_dir / 'prices.csv'
# Resume-state JSON sets (sorted string arrays on disk):
#   queried_actions.json — tickers whose actions (dividends + splits) have been fetched
#   queried_prices.json  — tickers whose OHLCV history has been fetched
#   error_tickers.json   — tickers that raised an exception during any yfinance call
#                          (shared across both feeds; a failing ticker is skipped for both)
# These are the source of truth for fetch progress. The `catalog` table in DuckDB
# mirrors them as BOOLEAN columns but is a derived snapshot, not live state.
_QUERIED_ACTIONS = JsonSet(raw_dir / 'queried_actions.json')
_QUERIED_PRICES = JsonSet(raw_dir / 'queried_prices.json')
_ERRORS = JsonSet(raw_dir / 'error_tickers.json')
_ACTIONS_COLS = ['Ticker', 'Date', 'Type', 'Value']


def _load_yahoo_tickers() -> dict[str, str]:
    """Canonical->yahoo_ticker map from `universe`, excluding configured Market types.

    Filters to rows where yahoo_ticker IS NOT NULL (instruments with no Yahoo
    equivalent are excluded automatically). Uses a short-lived local connection
    so the read-only singleton is never created before store() opens its
    read-write connection — DuckDB disallows mixing both in one process.
    """
    excludes = [m.lower() for m in yahoo_cfg.markets_exclude]
    placeholders = ', '.join(['?' for _ in excludes])
    try:
        with duckdb.connect(str(config.database.path), read_only=True) as con:
            df = con.execute(
                f'SELECT Ticker, yahoo_ticker FROM universe '
                f'WHERE yahoo_ticker IS NOT NULL '
                f'AND LOWER(Market) NOT IN ({placeholders}) '
                f'ORDER BY Ticker',
                excludes,
            ).df()
    except duckdb.CatalogException:
        logger.error(
            'universe table not found in DB. '
            'Run the seed-universe and universe steps first: '
            'uv run irp -> Steps -> seed-universe, universe'
        )
        raise
    return dict(zip(df['Ticker'], df['yahoo_ticker'], strict=False))


def _load_last_prices_dates() -> dict[str, date]:
    """Per-ticker last stored date in yahoo_prices. Returns {} if table absent."""
    try:
        with duckdb.connect(str(config.database.path), read_only=True) as con:
            rows = con.execute(
                'SELECT Ticker, MAX(Date) FROM yahoo_prices GROUP BY Ticker'
            ).fetchall()
        return {ticker: d for ticker, d in rows}
    except duckdb.CatalogException:
        # yahoo_prices table absent on first-time load
        return {}
    except duckdb.IOException as exc:
        logger.warning(
            f'_load_last_prices_dates: DB unreachable, treating as empty: {exc}'
        )
        return {}


def _actions_to_long(ticker: str, actions: pd.DataFrame) -> pd.DataFrame:
    """Vectorized melt of yfinance `.actions` (DatetimeIndex with Dividends
    and/or Stock Splits columns — yfinance omits whichever has no events)
    into long-form `(Ticker, Date, Type, Value)` rows where Value > 0."""
    df = actions.reset_index()
    date_col = df.columns[0]
    df['Date'] = pd.to_datetime(df[date_col]).dt.strftime('%Y-%m-%d')
    parts: list[pd.DataFrame] = []
    if 'Dividends' in df.columns:
        div = df.loc[df['Dividends'] > 0, ['Date', 'Dividends']].rename(
            columns={'Dividends': 'Value'}
        )
        div['Type'] = 'dividend'
        parts.append(div)
    if 'Stock Splits' in df.columns:
        spl = df.loc[df['Stock Splits'] > 0, ['Date', 'Stock Splits']].rename(
            columns={'Stock Splits': 'Value'}
        )
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
    """yfinance `.history()` OHLCV → long-form `(Ticker, Date, Open, High, Low,
    Close, Volume)` rows."""
    df = hist[['Open', 'High', 'Low', 'Close', 'Volume']].copy()
    df.index = pd.to_datetime(df.index).strftime('%Y-%m-%d')
    df.index.name = 'Date'
    df.insert(0, 'Ticker', ticker)
    return df.reset_index()


def _append_csv(df: pd.DataFrame, path: Path, has_header: bool) -> bool:
    """Append df to path. Returns updated has_header flag."""
    df.to_csv(path, mode='a', header=not has_header, index=False)
    return True


def _fetch_actions_per_ticker(
    yf,
    ticker_map: dict[str, str],
    queried: set[str],
    known_errors: set[str],
    new_errors: set[str],
    has_header: bool,
) -> bool:
    from irp.core.cancel import is_cancelled

    todo = [t for t in ticker_map if t not in known_errors and t not in queried]
    logger.info(f'Yahoo actions: {len(todo)} tickers')
    for ticker in todo:
        if is_cancelled():
            break
        yahoo = ticker_map[ticker]
        logger.debug(f'Actions: {ticker} (yahoo={yahoo})')
        try:
            actions = yf.Ticker(yahoo).actions
        except Exception as e:
            logger.warning(f'{ticker}: {type(e).__name__}: {e}')
            new_errors.add(ticker)
            _ERRORS.save(known_errors | new_errors)
            time.sleep(yahoo_cfg.actions_sleep)
            continue
        time.sleep(yahoo_cfg.actions_sleep)
        if actions is not None and not actions.empty:
            rows_df = _actions_to_long(ticker, actions)
            if not rows_df.empty:
                has_header = _append_csv(rows_df, _ACTIONS_FILE, has_header)
        queried.add(ticker)
        _QUERIED_ACTIONS.save(queried)
    return has_header


def _fetch_prices_per_ticker(
    yf,
    ticker_map: dict[str, str],
    queried: set[str],
    known_errors: set[str],
    new_errors: set[str],
    has_header: bool,
    last_dates: dict[str, date] | None = None,
) -> bool:
    from irp.core.cancel import is_cancelled

    todo = [t for t in ticker_map if t not in known_errors and t not in queried]
    logger.info(f'Yahoo prices (per-ticker): {len(todo)} tickers')
    for ticker in todo:
        if is_cancelled():
            break
        yahoo = ticker_map[ticker]
        logger.debug(f'Prices: {ticker} (yahoo={yahoo})')
        last = last_dates.get(ticker) if last_dates is not None else None
        kwargs = (
            {'start': (last + timedelta(days=1)).isoformat(), 'auto_adjust': True}
            if last is not None
            else {'period': 'max', 'auto_adjust': True}
        )
        try:
            hist = yf.Ticker(yahoo).history(**kwargs)
        except Exception as e:
            logger.warning(f'{ticker}: {type(e).__name__}: {e}')
            new_errors.add(ticker)
            _ERRORS.save(known_errors | new_errors)
            time.sleep(yahoo_cfg.ticker_sleep)
            continue
        time.sleep(yahoo_cfg.ticker_sleep)
        if hist is not None and not hist.empty:
            rows_df = _prices_to_long(ticker, hist)
            has_header = _append_csv(rows_df, _PRICES_FILE, has_header)
        queried.add(ticker)
        _QUERIED_PRICES.save(queried)
    return has_header


def _fetch_prices_batched(
    yf,
    ticker_map: dict[str, str],
    queried: set[str],
    known_errors: set[str],
    new_errors: set[str],
    has_header: bool,
    last_dates: dict[str, date] | None = None,
) -> bool:
    """Batch download via yf.download. Each batch is one HTTP request to
    Yahoo's chart API; invalid tickers come back as all-NaN columns (not
    failures), so per-ticker success is detected by checking the Close column.
    Whole-batch failures (rate-limit, network) are retried on next run.

    ticker_map keys are canonical Stooq tickers; values are Yahoo Finance
    symbols. yf.download uses the Yahoo symbols; data is stored under
    canonical tickers so all DB tables share the same key.

    When `last_dates` is provided, each batch uses the minimum last_date of
    the group as the shared start date (incremental update). Batches containing
    any ticker absent from `last_dates` fall back to `period='max'`."""
    from irp.core.cancel import is_cancelled

    todo = [t for t in ticker_map if t not in known_errors and t not in queried]
    bsize = yahoo_cfg.prices_batch_size
    n_batches = (len(todo) + bsize - 1) // bsize
    logger.info(
        f'Yahoo prices (batched, size={bsize}): {len(todo)} tickers, {n_batches} batches'
    )
    for i in range(0, len(todo), bsize):
        if is_cancelled():
            break
        batch = todo[i : i + bsize]
        batch_yahoo = [ticker_map[t] for t in batch]
        batch_starts = (
            [last_dates.get(t) for t in batch] if last_dates is not None else []
        )
        non_null_starts: list[date] = [s for s in batch_starts if s is not None]
        if non_null_starts and len(non_null_starts) == len(batch_starts):
            dl_kwargs: dict = {
                'start': (min(non_null_starts) + timedelta(days=1)).isoformat()
            }
        else:
            dl_kwargs = {'period': 'max'}
        batch_num = i // bsize + 1
        logger.info(
            f'Batch {batch_num}/{n_batches}: {len(batch)} tickers, start={dl_kwargs.get("start", "max")}'
        )
        try:
            raw = yf.download(
                batch_yahoo,
                **dl_kwargs,
                auto_adjust=True,
                threads=False,
                progress=False,
                group_by='ticker',
            )
        except Exception as e:
            logger.warning(
                f'batch [{i}:{i + len(batch)}] failed: {type(e).__name__}: {e}'
            )
            time.sleep(yahoo_cfg.batch_sleep)
            continue
        time.sleep(yahoo_cfg.batch_sleep)
        if raw is None or raw.empty:
            new_errors.update(batch)
            _ERRORS.save(known_errors | new_errors)
            continue
        for ticker in batch:
            yahoo = ticker_map[ticker]
            df = _extract_batch_slice(raw, yahoo, single_ticker=(len(batch) == 1))
            if df is None or df.empty:
                new_errors.add(ticker)
                _ERRORS.save(known_errors | new_errors)
                continue
            rows_df = _prices_to_long(ticker, df)  # store under canonical ticker
            has_header = _append_csv(rows_df, _PRICES_FILE, has_header)
            queried.add(ticker)
            _QUERIED_PRICES.save(queried)
    return has_header


def _extract_batch_slice(
    raw: pd.DataFrame, ticker: str, single_ticker: bool
) -> pd.DataFrame | None:
    """Pull one ticker's OHLCV out of yf.download's result. With multiple
    tickers + group_by='ticker', columns are MultiIndex ('ticker', field).
    With a single ticker the result is flat-columned."""
    try:
        df = raw if single_ticker else raw[ticker]
    except KeyError:
        return None
    if df.empty or 'Close' not in df.columns:
        return None
    df = df.dropna(subset=['Close'])  # type: ignore (pandas typing issue)
    return df if not df.empty else None


def _fetch_ticker_data(
    skip_errors: bool = True,
    skip_queried: bool = True,
    fetch_actions: bool = True,
    fetch_prices: bool = True,
    prices_mode: Literal['batch', 'ticker'] = 'batch',
    last_dates: dict[str, date] | None = None,
) -> None:
    """Pull dividends, splits, and/or OHLCV prices from yfinance.

    `prices_mode='batch'` (default) uses yf.download with batch_size from
    config — ~10x faster than per-ticker. `prices_mode='ticker'` falls back
    to per-ticker yf.Ticker.history (slower, finer error attribution).

    Actions always go per-ticker (no batch endpoint).

    `last_dates` maps Ticker -> last stored date in yahoo_prices. When set,
    prices are fetched from last_date+1 day instead of full history. Tickers
    absent from the map get full history. Batched mode uses the minimum
    last_date across the batch as the shared start.

    Resume-safe: skips tickers already in queried_actions.json /
    queried_prices.json / error_tickers.json.
    """
    import yfinance as yf

    raw_dir.mkdir(parents=True, exist_ok=True)
    ticker_map = _load_yahoo_tickers()
    known_errors = _ERRORS.load() if skip_errors else set()
    queried_actions = _QUERIED_ACTIONS.load() if skip_queried else set()
    queried_prices = _QUERIED_PRICES.load() if skip_queried else set()
    new_errors: set[str] = set()

    actions_has_header = _ACTIONS_FILE.exists()
    prices_has_header = _PRICES_FILE.exists()

    if fetch_actions:
        actions_has_header = _fetch_actions_per_ticker(
            yf,
            ticker_map,
            queried_actions,
            known_errors,
            new_errors,
            actions_has_header,
        )

    if fetch_prices:
        fn = (
            _fetch_prices_batched
            if prices_mode == 'batch'
            else _fetch_prices_per_ticker
        )
        prices_has_header = fn(
            yf,
            ticker_map,
            queried_prices,
            known_errors,
            new_errors,
            prices_has_header,
            last_dates=last_dates,
        )


def _transform_actions(conn: duckdb.DuckDBPyConnection) -> None:
    src = _ACTIONS_FILE
    if not src.exists():
        logger.warning(f'No raw actions file at {src}')
        return
    processed_dir.mkdir(parents=True, exist_ok=True)
    div_out = processed_dir / 'dividends.csv'
    spl_out = processed_dir / 'splits.csv'

    conn.sql(f"""
        COPY (
            SELECT
                Ticker,
                CAST(Date AS DATE) AS Date,
                Value AS Amount,
                Ticker AS SrcId,
                'yahoo' AS Src
            FROM read_csv_auto('{src}')
            WHERE Type = 'dividend'
        )
        TO '{div_out}' (FORMAT CSV, HEADER)
    """)
    logger.debug(f'Wrote {div_out}')

    conn.sql(f"""
        COPY (
            SELECT
                Ticker,
                CAST(Date AS DATE) AS Date,
                Value AS Ratio,
                Ticker AS SrcId,
                'yahoo' AS Src
            FROM read_csv_auto('{src}')
            WHERE Type = 'split'
        )
        TO '{spl_out}' (FORMAT CSV, HEADER)
    """)
    logger.debug(f'Wrote {spl_out}')


def _transform_prices(conn: duckdb.DuckDBPyConnection) -> None:
    src = _PRICES_FILE
    if not src.exists():
        logger.warning(f'No raw prices file at {src}')
        return
    processed_dir.mkdir(parents=True, exist_ok=True)
    out = processed_dir / 'prices.csv'

    conn.sql(f"""
        COPY (
            SELECT
                Ticker,
                CAST(Date AS DATE) AS Date,
                Open, High, Low, Close,
                CAST(Volume AS BIGINT) AS Volume
            FROM read_csv_auto('{src}')
        )
        TO '{out}' (FORMAT CSV, HEADER)
    """)
    logger.debug(f'Wrote {out}')


def _store_dividends(con: duckdb.DuckDBPyConnection) -> None:
    src = processed_dir / 'dividends.csv'
    if not src.exists():
        return
    merge_csv(
        con,
        'dividends',
        src,
        key_cols=['Ticker', 'Date'],
        value_cols=['Amount'],
        extra_insert_cols=['SrcId', 'Src'],
    )
    logger.debug('Stored dividends')


def _store_splits(con: duckdb.DuckDBPyConnection) -> None:
    src = processed_dir / 'splits.csv'
    if not src.exists():
        return
    merge_csv(
        con,
        'splits',
        src,
        key_cols=['Ticker', 'Date'],
        value_cols=['Ratio'],
        extra_insert_cols=['SrcId', 'Src'],
    )
    logger.debug('Stored splits')


def _store_prices(con: duckdb.DuckDBPyConnection) -> None:
    src = processed_dir / 'prices.csv'
    if not src.exists():
        return
    merge_csv(
        con,
        'yahoo_prices',
        src,
        key_cols=['Ticker', 'Date'],
        value_cols=['Open', 'High', 'Low', 'Close', 'Volume'],
    )
    logger.debug('Stored yahoo_prices')


class YahooSource:
    SUPPORTED_FEEDS: frozenset[Feed] = frozenset({'bulk', 'update'})

    def __init__(
        self,
        fetch_actions: bool = True,
        fetch_prices: bool = True,
        prices_mode: Literal['batch', 'ticker'] = 'batch',
    ) -> None:
        from irp.core.markers import MarkerSet

        self._fetch_actions = fetch_actions
        self._fetch_prices = fetch_prices
        self._prices_mode = prices_mode
        self.markers = MarkerSet(raw_dir)

    def fetch_bulk(self) -> None:
        """Pull dividends, splits, and/or OHLCV prices for every eligible ticker.

        Target tickers come from universe.yahoo_ticker (NOT NULL, Market NOT IN
        markets_exclude). yfinance is called with the yahoo_ticker value; data
        is stored under the canonical Ticker key so all DB tables share the same
        primary key.

        Resume-safe: progress is tracked in `queried_actions.json`,
        `queried_prices.json`, and `error_tickers.json` under raw_dir. Interrupt
        any time with Ctrl-C; rerun skips already-fetched and known-error tickers."""
        marker = self.markers.path('fetched')
        actions_fresh = not self._fetch_actions or (
            is_fresh(marker, _QUERIED_ACTIONS.path) and _ACTIONS_FILE.exists()
        )
        prices_fresh = not self._fetch_prices or (
            is_fresh(marker, _QUERIED_PRICES.path) and _PRICES_FILE.exists()
        )
        if actions_fresh and prices_fresh:
            logger.info('fetch: already up to date, skipping')
            return
        logger.debug('Fetching Yahoo ticker data...')
        _fetch_ticker_data(
            fetch_actions=self._fetch_actions,
            fetch_prices=self._fetch_prices,
            prices_mode=self._prices_mode,  # type: ignore (pylance widens instance attribute type)
        )
        self.markers.touch('fetched')
        logger.debug('Yahoo ticker data fetched.')

    def update(self) -> None:
        """Incremental refresh: fetches only dates after the last stored date per ticker.

        Prices use per-ticker or batched start dates from yahoo_prices. Batches
        use the minimum last_date of the group, so new tickers in a batch pull
        full history. Actions always re-fetch full history (no incremental API).
        Merge dedupes on (Ticker, Date) so reruns are idempotent."""
        _LOOKBACK_DAYS = 30
        last_dates_raw = _load_last_prices_dates() if self._fetch_prices else None
        last_dates = (
            {t: d - timedelta(days=_LOOKBACK_DAYS) for t, d in last_dates_raw.items()}
            if last_dates_raw is not None
            else None
        )
        _fetch_ticker_data(
            skip_queried=False,
            fetch_actions=self._fetch_actions,
            fetch_prices=self._fetch_prices,
            prices_mode=self._prices_mode,  # type: ignore (pylance widens instance attribute type)
            last_dates=last_dates,
        )
        self.markers.touch('fetched')

    def transform(self, feed: Literal['bulk', 'update']) -> None:
        if self.markers.is_fresh(f'transformed_{feed}', 'fetched'):
            logger.info(f'transform({feed}): already up to date, skipping')
            return
        conn = duckdb.connect()
        _transform_actions(conn)
        _transform_prices(conn)
        self.markers.touch(f'transformed_{feed}')

    def store(self, feed: Literal['bulk', 'update']) -> None:
        if self.markers.is_fresh(f'stored_{feed}', f'transformed_{feed}'):
            logger.info(f'store({feed}): already up to date, skipping')
            return
        from irp.core.db import write_session
        with write_session() as con:
            _store_dividends(con)
            _store_splits(con)
            _store_prices(con)
        self.markers.touch(f'stored_{feed}')
        logger.debug(f'Yahoo {feed} data stored.')

    def cleanup(self) -> None:
        """Delete intermediates, keeping raw files + markers + database."""
        targets = [
            processed_dir / 'dividends.csv',
            processed_dir / 'splits.csv',
            processed_dir / 'prices.csv',
        ]
        for path in targets:
            if path.exists():
                path.unlink()
                logger.debug(f'Deleted {path}')

import logging
import time
from pathlib import Path
from typing import Literal

import duckdb
import pandas as pd

from irp.core.config import config
from irp.core.duckdb_merge import merge_csv
from irp.core.freshness import is_fresh
from irp.core.jsonset import JsonSet

logger = logging.getLogger(__name__)

root_dir = config.data.root_dir
yahoo_cfg = config.providers.yahoo
raw_dir = root_dir / yahoo_cfg.raw_dir
processed_dir = root_dir / yahoo_cfg.processed_dir

_ACTIONS_FILE = raw_dir / 'actions.csv'
_PRICES_FILE = raw_dir / 'prices.csv'
_QUERIED_ACTIONS = JsonSet(raw_dir / 'queried_actions.json')
_QUERIED_PRICES = JsonSet(raw_dir / 'queried_prices.json')
_ERRORS = JsonSet(raw_dir / 'error_tickers.json')
_ACTIONS_COLS = ['Ticker', 'Date', 'Type', 'Value']


def _load_target_tickers() -> list[str]:
    """Tickers from the `markets` table, excluding configured Market types.

    Uses a short-lived local connection (not the global db() singleton) so
    the read-only singleton is never created in the CLI process before store()
    opens its read-write connection — DuckDB disallows mixing both in one process.
    """
    excludes = [m.lower() for m in yahoo_cfg.markets_exclude]
    placeholders = ', '.join(['?' for _ in excludes])
    with duckdb.connect(str(config.database.path), read_only=True) as con:
        df = con.execute(
            f'SELECT DISTINCT Ticker FROM markets '
            f'WHERE LOWER(Market) NOT IN ({placeholders}) '
            f'ORDER BY Ticker',
            excludes,
        ).df()
    return df['Ticker'].tolist()


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
    tickers: list[str],
    queried: set[str],
    known_errors: set[str],
    new_errors: set[str],
    has_header: bool,
) -> bool:
    todo = [t for t in tickers if t not in known_errors and t not in queried]
    logger.info(f'Yahoo actions: {len(todo)} tickers')
    for ticker in todo:
        logger.debug(f'Actions: {ticker}')
        try:
            actions = yf.Ticker(ticker).actions
        except Exception as e:
            logger.warning(f'{ticker}: {type(e).__name__}: {e}')
            new_errors.add(ticker)
            _ERRORS.save(known_errors | new_errors)
            time.sleep(yahoo_cfg.batch_sleep)
            continue
        time.sleep(yahoo_cfg.batch_sleep)
        if actions is not None and not actions.empty:
            rows_df = _actions_to_long(ticker, actions)
            if not rows_df.empty:
                has_header = _append_csv(rows_df, _ACTIONS_FILE, has_header)
        queried.add(ticker)
        _QUERIED_ACTIONS.save(queried)
    return has_header


def _fetch_prices_per_ticker(
    yf,
    tickers: list[str],
    queried: set[str],
    known_errors: set[str],
    new_errors: set[str],
    has_header: bool,
) -> bool:
    todo = [t for t in tickers if t not in known_errors and t not in queried]
    logger.info(f'Yahoo prices (per-ticker): {len(todo)} tickers')
    for ticker in todo:
        logger.debug(f'Prices: {ticker}')
        try:
            hist = yf.Ticker(ticker).history(period='max', auto_adjust=True)
        except Exception as e:
            logger.warning(f'{ticker}: {type(e).__name__}: {e}')
            new_errors.add(ticker)
            _ERRORS.save(known_errors | new_errors)
            time.sleep(yahoo_cfg.batch_sleep)
            continue
        time.sleep(yahoo_cfg.batch_sleep)
        if hist is not None and not hist.empty:
            rows_df = _prices_to_long(ticker, hist)
            has_header = _append_csv(rows_df, _PRICES_FILE, has_header)
        queried.add(ticker)
        _QUERIED_PRICES.save(queried)
    return has_header


def _fetch_prices_batched(
    yf,
    tickers: list[str],
    queried: set[str],
    known_errors: set[str],
    new_errors: set[str],
    has_header: bool,
) -> bool:
    """Batch download via yf.download. Each batch is one HTTP request to
    Yahoo's chart API; invalid tickers come back as all-NaN columns (not
    failures), so per-ticker success is detected by checking the Close
    column. Whole-batch failures (rate-limit, network) are retried on next
    run."""
    todo = [t for t in tickers if t not in known_errors and t not in queried]
    bsize = yahoo_cfg.prices_batch_size
    logger.info(f'Yahoo prices (batched, size={bsize}): {len(todo)} tickers')
    for i in range(0, len(todo), bsize):
        batch = todo[i : i + bsize]
        logger.debug(f'Batch {i // bsize + 1}: {len(batch)} tickers')
        try:
            raw = yf.download(
                batch,
                period='max',
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
            df = _extract_batch_slice(raw, ticker, single_ticker=(len(batch) == 1))
            if df is None or df.empty:
                new_errors.add(ticker)
                _ERRORS.save(known_errors | new_errors)
                continue
            rows_df = _prices_to_long(ticker, df)
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
    df = df.dropna(subset=['Close']) # type: ignore (pandas typing issue)
    return df if not df.empty else None


def _fetch_ticker_data(
    skip_errors: bool = True,
    skip_queried: bool = True,
    fetch_actions: bool = True,
    fetch_prices: bool = True,
    prices_mode: Literal['batch', 'ticker'] = 'batch',
) -> None:
    """Pull dividends, splits, and/or OHLCV prices from yfinance.

    `prices_mode='batch'` (default) uses yf.download with batch_size from
    config — ~10x faster than per-ticker. `prices_mode='ticker'` falls back
    to per-ticker yf.Ticker.history (slower, finer error attribution).

    Actions always go per-ticker (no batch endpoint).

    Resume-safe: skips tickers already in queried_actions.json /
    queried_prices.json / error_tickers.json.
    """
    import yfinance as yf

    raw_dir.mkdir(parents=True, exist_ok=True)
    tickers = _load_target_tickers()
    known_errors = _ERRORS.load() if skip_errors else set()
    queried_actions = _QUERIED_ACTIONS.load() if skip_queried else set()
    queried_prices = _QUERIED_PRICES.load() if skip_queried else set()
    new_errors: set[str] = set()

    actions_has_header = _ACTIONS_FILE.exists()
    prices_has_header = _PRICES_FILE.exists()

    if fetch_actions:
        actions_has_header = _fetch_actions_per_ticker(
            yf,
            tickers,
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
            tickers,
            queried_prices,
            known_errors,
            new_errors,
            prices_has_header,
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
    SUPPORTED_FEEDS = frozenset({'bulk', 'update'})

    def __init__(
        self,
        fetch_actions: bool = True,
        fetch_prices: bool = True,
        prices_mode: Literal['batch', 'ticker'] = 'batch',
    ) -> None:
        self._fetch_actions = fetch_actions
        self._fetch_prices = fetch_prices
        self._prices_mode = prices_mode

    def fetch_bulk(self) -> None:
        """Pull dividends, splits, and/or OHLCV prices for every eligible ticker.

        Resume-safe via `queried_tickers.json` and `error_tickers.json` in raw_dir.
        Interrupt with kernel signal at any time; rerun continues."""
        marker = raw_dir / '.fetched'
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
            prices_mode=self._prices_mode, # type: ignore (pylance widens instance attribute type)
        )
        marker.touch()
        logger.debug('Yahoo ticker data fetched.')

    def update(self) -> None:
        """Re-fetch for all eligible tickers (forces re-query).

        yfinance has no incremental endpoint; full per-ticker refresh is the
        only safe path. Merge dedupes on (Ticker, Date) so reruns are idempotent."""
        _fetch_ticker_data(
            skip_queried=False,
            fetch_actions=self._fetch_actions,
            fetch_prices=self._fetch_prices,
            prices_mode=self._prices_mode, # type: ignore (pylance widens instance attribute type)
        )
        (raw_dir / '.fetched').touch()

    def transform(self, feed: Literal['bulk', 'update']) -> None:
        marker = raw_dir / f'.transformed_{feed}'
        upstream = raw_dir / '.fetched'
        if is_fresh(marker, upstream):
            logger.info(f'transform({feed}): already up to date, skipping')
            return
        conn = duckdb.connect()
        _transform_actions(conn)
        _transform_prices(conn)
        marker.touch()

    def store(self, feed: Literal['bulk', 'update']) -> None:
        marker = raw_dir / f'.stored_{feed}'
        upstream = raw_dir / f'.transformed_{feed}'
        if is_fresh(marker, upstream):
            logger.info(f'store({feed}): already up to date, skipping')
            return
        with duckdb.connect(config.database.path) as con:
            _store_dividends(con)
            _store_splits(con)
            _store_prices(con)
        marker.touch()
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

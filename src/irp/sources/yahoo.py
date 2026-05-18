import json
import logging
import time
from pathlib import Path
from typing import Literal

import duckdb
import pandas as pd

from irp.core.config import config
from irp.core.freshness import is_fresh

logger = logging.getLogger(__name__)

root_dir = config.data.root_dir
yahoo_cfg = config.providers.yahoo
raw_dir = root_dir / yahoo_cfg.raw_dir
processed_dir = root_dir / yahoo_cfg.processed_dir

_ACTIONS_FILE = raw_dir / 'actions.csv'
_PRICES_FILE = raw_dir / 'prices.csv'
_QUERIED_ACTIONS_FILE = raw_dir / 'queried_actions.json'
_QUERIED_PRICES_FILE = raw_dir / 'queried_prices.json'
_ERROR_FILE = raw_dir / 'error_tickers.json'
_ACTIONS_COLS = ['Ticker', 'Date', 'Type', 'Value']
_PRICES_COLS = ['Ticker', 'Date', 'Open', 'High', 'Low', 'Close', 'Volume']


def _load_target_tickers() -> list[str]:
    """Tickers from the `markets` table, excluding configured Market types."""
    excludes = [m.lower() for m in yahoo_cfg.markets_exclude]
    placeholders = ', '.join(['?' for _ in excludes])
    with duckdb.connect(str(config.database.path), read_only=True) as con:
        df = con.execute(
            f"SELECT DISTINCT Ticker FROM markets "
            f"WHERE LOWER(Market) NOT IN ({placeholders}) "
            f"ORDER BY Ticker",
            excludes,
        ).df()
    return df['Ticker'].tolist()


def _load_queried(path: Path) -> set[str]:
    if path.exists():
        return set(json.loads(path.read_text()))
    return set()


def _save_queried(queried: set[str], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(sorted(queried), indent=2))


def _load_errors() -> set[str]:
    if _ERROR_FILE.exists():
        return set(json.loads(_ERROR_FILE.read_text()))
    return set()


def _save_errors(errors: set[str]) -> None:
    _ERROR_FILE.parent.mkdir(parents=True, exist_ok=True)
    _ERROR_FILE.write_text(json.dumps(sorted(errors), indent=2))


def _fetch_ticker_data(
    skip_errors: bool = True,
    skip_queried: bool = True,
    fetch_actions: bool = True,
    fetch_prices: bool = True,
) -> None:
    """Pull dividends, splits, and/or OHLCV prices per ticker from yfinance.

    Resume-safe: tickers already in `queried_tickers.json` or `error_tickers.json`
    are skipped on rerun. Interrupt with kernel signal at any time.
    """
    import yfinance as yf

    raw_dir.mkdir(parents=True, exist_ok=True)
    tickers = _load_target_tickers()
    known_errors = _load_errors() if skip_errors else set()
    queried_actions = _load_queried(_QUERIED_ACTIONS_FILE) if skip_queried else set()
    queried_prices = _load_queried(_QUERIED_PRICES_FILE) if skip_queried else set()
    new_errors: set[str] = set()

    todo = [
        t for t in tickers
        if t not in known_errors and (
            (fetch_actions and t not in queried_actions) or
            (fetch_prices and t not in queried_prices)
        )
    ]
    logger.info(f'Yahoo fetch: {len(todo)} of {len(tickers)} tickers to process')

    actions_has_header = _ACTIONS_FILE.exists()
    prices_has_header = _PRICES_FILE.exists()

    for ticker in todo:
        need_actions = fetch_actions and ticker not in queried_actions
        need_prices = fetch_prices and ticker not in queried_prices
        logger.debug(f'Fetching {ticker}')
        try:
            obj = yf.Ticker(ticker)
            actions = obj.actions if need_actions else None
            hist = obj.history(period='max', auto_adjust=True) if need_prices else None
        except Exception as e:
            logger.warning(f'{ticker}: {e}')
            new_errors.add(ticker)
            _save_errors(known_errors | new_errors)
            time.sleep(yahoo_cfg.batch_sleep)
            continue
        time.sleep(yahoo_cfg.batch_sleep)

        # --- actions ---
        if need_actions:
            if actions is not None and not actions.empty:
                rows: list[dict] = []
                for date, row in actions.iterrows():
                    d = pd.Timestamp(date).strftime('%Y-%m-%d')
                    div = float(row.get('Dividends', 0) or 0)
                    sp = float(row.get('Stock Splits', 0) or 0)
                    if div > 0:
                        rows.append({'Ticker': ticker, 'Date': d, 'Type': 'dividend', 'Value': div})
                    if sp > 0:
                        rows.append({'Ticker': ticker, 'Date': d, 'Type': 'split', 'Value': sp})
                if rows:
                    pd.DataFrame(rows, columns=_ACTIONS_COLS).to_csv(
                        _ACTIONS_FILE, mode='a', header=not actions_has_header, index=False,
                    )
                    actions_has_header = True
            queried_actions.add(ticker)
            _save_queried(queried_actions, _QUERIED_ACTIONS_FILE)

        # --- prices ---
        if need_prices:
            if hist is not None and not hist.empty:
                price_df = hist[['Open', 'High', 'Low', 'Close', 'Volume']].copy()
                price_df.index = pd.to_datetime(price_df.index).strftime('%Y-%m-%d')
                price_df.index.name = 'Date'
                price_df.insert(0, 'Ticker', ticker)
                price_df.reset_index().to_csv(
                    _PRICES_FILE, mode='a', header=not prices_has_header, index=False,
                )
                prices_has_header = True
            queried_prices.add(ticker)
            _save_queried(queried_prices, _QUERIED_PRICES_FILE)


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
                CAST(REPLACE(Date, '-', '') AS INTEGER) AS Date,
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
                CAST(REPLACE(Date, '-', '') AS INTEGER) AS Date,
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
                CAST(REPLACE(Date, '-', '') AS INTEGER) AS Date,
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
    con.execute(
        f"CREATE TABLE IF NOT EXISTS dividends AS SELECT * FROM read_csv_auto('{src}') LIMIT 0"
    )
    con.execute(f"""
        MERGE INTO dividends t
        USING (
            SELECT DISTINCT ON (Ticker, Date) *
            FROM read_csv_auto('{src}')
        ) s
        ON t.Ticker = s.Ticker AND t.Date = s.Date
        WHEN MATCHED THEN UPDATE SET Amount = s.Amount
        WHEN NOT MATCHED THEN INSERT (Ticker, Date, Amount, SrcId, Src)
        VALUES (s.Ticker, s.Date, s.Amount, s.SrcId, s.Src)
    """)
    logger.debug('Stored dividends')


def _store_splits(con: duckdb.DuckDBPyConnection) -> None:
    src = processed_dir / 'splits.csv'
    if not src.exists():
        return
    con.execute(
        f"CREATE TABLE IF NOT EXISTS splits AS SELECT * FROM read_csv_auto('{src}') LIMIT 0"
    )
    con.execute(f"""
        MERGE INTO splits t
        USING (
            SELECT DISTINCT ON (Ticker, Date) *
            FROM read_csv_auto('{src}')
        ) s
        ON t.Ticker = s.Ticker AND t.Date = s.Date
        WHEN MATCHED THEN UPDATE SET Ratio = s.Ratio
        WHEN NOT MATCHED THEN INSERT (Ticker, Date, Ratio, SrcId, Src)
        VALUES (s.Ticker, s.Date, s.Ratio, s.SrcId, s.Src)
    """)
    logger.debug('Stored splits')


def _store_prices(con: duckdb.DuckDBPyConnection) -> None:
    src = processed_dir / 'prices.csv'
    if not src.exists():
        return
    con.execute(
        f"CREATE TABLE IF NOT EXISTS yahoo_prices AS SELECT * FROM read_csv_auto('{src}') LIMIT 0"
    )
    con.execute(f"""
        MERGE INTO yahoo_prices t
        USING (
            SELECT DISTINCT ON (Ticker, Date) *
            FROM read_csv_auto('{src}')
        ) s
        ON t.Ticker = s.Ticker AND t.Date = s.Date
        WHEN MATCHED THEN UPDATE SET
            Open = s.Open, High = s.High, Low = s.Low,
            Close = s.Close, Volume = s.Volume
        WHEN NOT MATCHED THEN INSERT (Ticker, Date, Open, High, Low, Close, Volume)
        VALUES (s.Ticker, s.Date, s.Open, s.High, s.Low, s.Close, s.Volume)
    """)
    logger.debug('Stored yahoo_prices')


class YahooSource:
    def __init__(self, fetch_actions: bool = True, fetch_prices: bool = True) -> None:
        self._fetch_actions = fetch_actions
        self._fetch_prices = fetch_prices

    def fetch_bulk(self) -> None:
        """Pull dividends, splits, and/or OHLCV prices for every eligible ticker.

        Resume-safe via `queried_tickers.json` and `error_tickers.json` in raw_dir.
        Interrupt with kernel signal at any time; rerun continues."""
        marker = raw_dir / '.fetched'
        actions_fresh = not self._fetch_actions or (
            is_fresh(marker, _QUERIED_ACTIONS_FILE) and _ACTIONS_FILE.exists()
        )
        prices_fresh = not self._fetch_prices or (
            is_fresh(marker, _QUERIED_PRICES_FILE) and _PRICES_FILE.exists()
        )
        if actions_fresh and prices_fresh:
            logger.info('fetch: already up to date, skipping')
            return
        logger.debug('Fetching Yahoo ticker data...')
        _fetch_ticker_data(fetch_actions=self._fetch_actions, fetch_prices=self._fetch_prices)
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

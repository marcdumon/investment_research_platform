from unittest.mock import MagicMock

import duckdb
import pandas as pd
import pytest

import irp.sources.stooq.provider as mod
from irp.sources.stooq.provider import StooqProvider


_BULK_PRICES = pd.DataFrame([
    {"ticker": "aapl", "src_ticker": "aapl", "per": "D", "date": 20230101,
     "time": 0, "open": 150.0, "high": 155.0, "low": 148.0, "close": 152.0,
     "vol": 1_000_000, "openint": 0},
    {"ticker": "eurusd", "src_ticker": "eurusd", "per": "D", "date": 20230101,
     "time": 0, "open": 1.1, "high": 1.11, "low": 1.09, "close": 1.105,
     "vol": 0, "openint": 0},
])

_MARKETS = pd.DataFrame([
    {"ticker": "aapl", "src_ticker": "aapl.us", "market": "nasdaq stocks"},
    {"ticker": "eurusd", "src_ticker": "eurusd", "market": "currencies"},
])


@pytest.fixture
def store_env(tmp_path, monkeypatch):
    processed = tmp_path / "processed"
    processed.mkdir()
    _BULK_PRICES.to_parquet(processed / "bulk_prices.parquet", index=False)
    _MARKETS.to_csv(processed / "markets.csv", index=False)
    db_path = tmp_path / "test.duckdb"
    cfg = MagicMock()
    cfg.database.path = db_path
    monkeypatch.setattr(mod, "processed_dir", processed)
    monkeypatch.setattr(mod, "config", cfg)
    return processed, db_path


def _tables(db_path):
    with duckdb.connect(db_path) as con:
        return con.execute("SHOW TABLES").df()["name"].tolist()


def _query(db_path, sql):
    with duckdb.connect(db_path) as con:
        return con.execute(sql).df()


# --- bulk ---

def test_store_bulk_creates_prices_table(store_env):
    _, db_path = store_env
    StooqProvider().store(feed="bulk")
    assert "prices" in _tables(db_path)


def test_store_bulk_creates_markets_table(store_env):
    _, db_path = store_env
    StooqProvider().store(feed="bulk")
    assert "markets" in _tables(db_path)


def test_store_bulk_prices_row_count(store_env):
    _, db_path = store_env
    StooqProvider().store(feed="bulk")
    count = _query(db_path, "SELECT COUNT(*) AS n FROM prices")["n"][0]
    assert count == len(_BULK_PRICES)


def test_store_bulk_markets_row_count(store_env):
    _, db_path = store_env
    StooqProvider().store(feed="bulk")
    count = _query(db_path, "SELECT COUNT(*) AS n FROM markets")["n"][0]
    assert count == len(_MARKETS)


def test_store_bulk_idempotent(store_env):
    """Storing twice must not duplicate rows (MERGE key: src_ticker + date)."""
    _, db_path = store_env
    StooqProvider().store(feed="bulk")
    StooqProvider().store(feed="bulk")
    count = _query(db_path, "SELECT COUNT(*) AS n FROM prices")["n"][0]
    assert count == len(_BULK_PRICES)


# --- update ---

_UPDATE_CSV = (
    "ticker,src_ticker,per,date,time,open,high,low,close,vol,openint\n"
    "aapl,aapl,D,20230101,0,999.0,999.0,999.0,999.0,0,0\n"   # existing key → update
    "msft,msft,D,20230101,0,200.0,205.0,198.0,202.0,500000,0\n"  # new key → insert
)


@pytest.fixture
def store_env_with_update(store_env):
    processed, db_path = store_env
    (processed / "update_prices.csv").write_text(_UPDATE_CSV)
    StooqProvider().store(feed="bulk")
    return processed, db_path


def test_store_update_inserts_new_ticker(store_env_with_update):
    _, db_path = store_env_with_update
    StooqProvider().store(feed="update")
    tickers = _query(db_path, "SELECT DISTINCT ticker FROM prices")["ticker"].tolist()
    assert "msft" in tickers


def test_store_update_overwrites_existing_row(store_env_with_update):
    """MERGE must update OHLCV for matching (src_ticker, date)."""
    _, db_path = store_env_with_update
    StooqProvider().store(feed="update")
    close = _query(
        db_path,
        "SELECT close FROM prices WHERE src_ticker = 'aapl' AND date = 20230101",
    )["close"][0]
    assert close == 999.0


def test_store_update_does_not_duplicate(store_env_with_update):
    """Update must not add duplicate rows for matched keys."""
    _, db_path = store_env_with_update
    StooqProvider().store(feed="update")
    count = _query(
        db_path,
        "SELECT COUNT(*) AS n FROM prices WHERE src_ticker = 'aapl' AND date = 20230101",
    )["n"][0]
    assert count == 1

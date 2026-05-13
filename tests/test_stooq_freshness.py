"""Tests for the _is_fresh marker-based checkpointing in each pipeline step."""
import os
import time
import zipfile
from unittest.mock import MagicMock, patch

import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
import pytest

import irp.sources.stooq as mod
from irp.sources.stooq import FeedSpec, StooqSource


ZIP_NAME = "d_us_txt.zip"
TICKER_CSV = (
    "<TICKER>,<PER>,<DATE>,<TIME>,<OPEN>,<HIGH>,<LOW>,<CLOSE>,<VOL>,<OPENINT>\n"
    "AAPL,D,20230101,0,150.0,155.0,148.0,152.0,1000000,0\n"
)


def _touch(path, mtime: float):
    path.touch()
    os.utime(path, (mtime, mtime))


# --- fetch ---

@pytest.fixture
def fetch_env(tmp_path, monkeypatch):
    raw = tmp_path / "raw"
    raw.mkdir()
    with zipfile.ZipFile(raw / ZIP_NAME, "w") as zf:
        zf.writestr("data/daily/us/nasdaq stocks/aapl.us.txt", TICKER_CSV)
    cfg = MagicMock()
    cfg.bulk_files = [ZIP_NAME]
    cfg.bulk_download_instruction = ""
    monkeypatch.setattr(mod, "raw_dir", raw)
    monkeypatch.setattr(mod, "stooq_cfg", cfg)
    return raw


def test_fetch_skips_when_marker_is_fresh(fetch_env):
    raw = fetch_env
    now = time.time()
    _touch(raw / ZIP_NAME, now - 10)
    _touch(raw / ".fetched", now)
    with patch.object(mod, "_build_price_dataset") as mock_build:
        StooqSource().fetch_bulk()
    mock_build.assert_not_called()


def test_fetch_runs_when_zip_is_newer(fetch_env):
    raw = fetch_env
    now = time.time()
    _touch(raw / ".fetched", now - 10)
    _touch(raw / ZIP_NAME, now)
    with patch.object(mod, "_build_price_dataset") as mock_build:
        with patch.object(mod, "_unzip_bulk_files"):
            StooqSource().fetch_bulk()
    mock_build.assert_called_once()


def test_fetch_runs_when_marker_missing(fetch_env):
    with patch.object(mod, "_build_price_dataset") as mock_build:
        with patch.object(mod, "_unzip_bulk_files"):
            StooqSource().fetch_bulk()
    mock_build.assert_called_once()


def test_fetch_writes_marker(fetch_env):
    raw = fetch_env
    with patch.object(mod, "_unzip_bulk_files"):
        with patch.object(mod, "_build_price_dataset"):
            StooqSource().fetch_bulk()
    assert (raw / ".fetched").exists()


# --- transform ---

@pytest.fixture
def transform_env(tmp_path, monkeypatch):
    raw = tmp_path / "raw"
    processed = tmp_path / "processed"
    raw.mkdir()
    processed.mkdir()
    df = pd.DataFrame([{
        "<TICKER>": "AAPL", "<PER>": "D", "<DATE>": 20230101, "<TIME>": 0,
        "<OPEN>": 150.0, "<HIGH>": 155.0, "<LOW>": 148.0, "<CLOSE>": 152.0,
        "<VOL>": 1_000_000, "<OPENINT>": 0, "ticker": "aapl.us",
    }])
    pq.write_table(pa.Table.from_pandas(df, preserve_index=False), raw / "bulk_prices.parquet")
    pd.DataFrame([{"ticker": "aapl.us", "market": "nasdaq stocks"}]).to_csv(
        raw / "markets.csv", index=False
    )
    monkeypatch.setattr(mod, "raw_dir", raw)
    monkeypatch.setattr(mod, "processed_dir", processed)
    monkeypatch.setattr(mod, "FEED_SPECS", {
        **mod.FEED_SPECS,
        "bulk": FeedSpec(
            input_path=raw / "bulk_prices.parquet",
            output_path=processed / "bulk_prices.parquet",
            input_format="parquet",
            output_format="parquet",
        ),
    })
    return raw, processed


def test_transform_skips_when_marker_is_fresh(transform_env):
    raw, _ = transform_env
    now = time.time()
    _touch(raw / ".fetched", now - 10)
    _touch(raw / ".transformed_bulk", now)
    with patch.object(mod.duckdb, "connect") as mock_conn:
        StooqSource().transform(feed="bulk")
    mock_conn.assert_not_called()


def test_transform_runs_when_upstream_is_newer(transform_env):
    raw, _ = transform_env
    now = time.time()
    _touch(raw / ".transformed_bulk", now - 10)
    _touch(raw / ".fetched", now)
    called = []
    original = mod.duckdb.connect
    def spy(*a, **kw):
        called.append(True)
        return original(*a, **kw)
    with patch.object(mod.duckdb, "connect", side_effect=spy):
        StooqSource().transform(feed="bulk")
    assert called


def test_transform_writes_marker(transform_env):
    raw, _ = transform_env
    _touch(raw / ".fetched", time.time() - 10)
    StooqSource().transform(feed="bulk")
    assert (raw / ".transformed_bulk").exists()


# --- store ---

@pytest.fixture
def store_env(tmp_path, monkeypatch):
    raw = tmp_path / "raw"
    processed = tmp_path / "processed"
    raw.mkdir()
    processed.mkdir()
    df = pd.DataFrame([{
        "ticker": "aapl", "src_ticker": "aapl", "per": "D", "date": 20230101,
        "time": 0, "open": 150.0, "high": 155.0, "low": 148.0, "close": 152.0,
        "vol": 1_000_000, "openint": 0,
    }])
    df.to_parquet(processed / "bulk_prices.parquet", index=False)
    pd.DataFrame([{"ticker": "aapl", "src_ticker": "aapl.us", "market": "nasdaq stocks"}]).to_csv(
        processed / "markets.csv", index=False
    )
    db_path = tmp_path / "test.duckdb"
    cfg = MagicMock()
    cfg.database.path = db_path
    monkeypatch.setattr(mod, "raw_dir", raw)
    monkeypatch.setattr(mod, "processed_dir", processed)
    monkeypatch.setattr(mod, "config", cfg)
    monkeypatch.setattr(mod, "FEED_SPECS", {
        **mod.FEED_SPECS,
        "bulk": FeedSpec(
            input_path=processed / "bulk_prices.parquet",
            output_path=processed / "bulk_prices.parquet",
            input_format="parquet",
            output_format="parquet",
        ),
    })
    return raw, processed, db_path


def test_store_skips_when_marker_is_fresh(store_env):
    raw, _, _ = store_env
    now = time.time()
    _touch(raw / ".transformed_bulk", now - 10)
    _touch(raw / ".stored_bulk", now)
    with patch.object(mod.duckdb, "connect") as mock_conn:
        StooqSource().store(feed="bulk")
    mock_conn.assert_not_called()


def test_store_runs_when_upstream_is_newer(store_env):
    raw, _, _ = store_env
    now = time.time()
    _touch(raw / ".stored_bulk", now - 10)
    _touch(raw / ".transformed_bulk", now)
    StooqSource().store(feed="bulk")
    assert (raw / ".stored_bulk").stat().st_mtime >= now - 1


def test_store_writes_marker(store_env):
    raw, _, _ = store_env
    _touch(raw / ".transformed_bulk", time.time() - 10)
    StooqSource().store(feed="bulk")
    assert (raw / ".stored_bulk").exists()


# --- cleanup ---

def test_cleanup_deletes_intermediate_files(store_env):
    raw, processed, _ = store_env
    StooqSource().cleanup()
    assert not (raw / "bulk_prices.parquet").exists()
    assert not (raw / "markets.csv").exists()
    assert not (processed / "bulk_prices.parquet").exists()
    assert not (processed / "markets.csv").exists()


def test_cleanup_keeps_markers(store_env):
    raw, _, _ = store_env
    now = time.time()
    _touch(raw / ".fetched", now)
    _touch(raw / ".transformed_bulk", now)
    StooqSource().cleanup()
    assert (raw / ".fetched").exists()
    assert (raw / ".transformed_bulk").exists()

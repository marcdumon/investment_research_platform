import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
import pytest

import irp.ingest.stooq as mod
from irp.ingest.stooq import StooqSource, FeedSpec


_RAW_BULK_ROWS = [
    {
        "<TICKER>": "AAPL", "<PER>": "D", "<DATE>": 20230101, "<TIME>": 0,
        "<OPEN>": 150.0, "<HIGH>": 155.0, "<LOW>": 148.0, "<CLOSE>": 152.0,
        "<VOL>": 1_000_000, "<OPENINT>": 0, "ticker": "aapl.us",
    },
    {
        "<TICKER>": "EURUSD", "<PER>": "D", "<DATE>": 20230101, "<TIME>": 0,
        "<OPEN>": 1.1, "<HIGH>": 1.11, "<LOW>": 1.09, "<CLOSE>": 1.105,
        "<VOL>": 0, "<OPENINT>": 0, "ticker": "eurusd",
    },
]

_UPDATE_CSV = (
    "<TICKER>,<PER>,<DATE>,<TIME>,<OPEN>,<HIGH>,<LOW>,<CLOSE>,<VOL>,<OPENINT>\n"
    "AAPL,D,20230102,0,152.0,157.0,150.0,154.0,900000,0\n"
)

_EXPECTED_COLUMNS = frozenset(
    {"ticker", "src_ticker", "per", "date", "time", "open", "high", "low", "close", "vol", "openint"}
)
_UPDATE_FILE = "d_us_update.csv"


@pytest.fixture
def bulk_raw(tmp_path, monkeypatch):
    raw = tmp_path / "raw"
    processed = tmp_path / "processed"
    raw.mkdir()
    processed.mkdir()
    pq.write_table(
        pa.Table.from_pandas(pd.DataFrame(_RAW_BULK_ROWS), preserve_index=False),
        raw / "bulk_prices.parquet",
    )
    pd.DataFrame([
        {"ticker": "aapl.us", "market": "nasdaq stocks"},
        {"ticker": "eurusd", "market": "currencies"},
    ]).to_csv(raw / "markets.csv", index=False)
    monkeypatch.setattr(mod, "raw_dir", raw)
    monkeypatch.setattr(mod, "processed_dir", processed)
    monkeypatch.setattr(mod, "TRANSFORMS", {
        **mod.FEED_SPECS,
        "bulk": FeedSpec(
            input_path=raw / "bulk_prices.parquet",
            output_path=processed / "bulk_prices.parquet",
            input_format="parquet",
            output_format="parquet",
        ),
    })
    return raw, processed


@pytest.fixture
def update_raw(tmp_path, monkeypatch):
    raw = tmp_path / "raw"
    processed = tmp_path / "processed"
    raw.mkdir()
    processed.mkdir()
    (raw / _UPDATE_FILE).write_text(_UPDATE_CSV)
    monkeypatch.setattr(mod, "TRANSFORMS", {
        **mod.FEED_SPECS,
        "update": FeedSpec(
            input_path=raw / _UPDATE_FILE,
            output_path=processed / _UPDATE_FILE,
            input_format="csv",
            output_format="csv",
            # header=True,
        ),
    })
    return raw, processed


# --- bulk (parquet) ---

def test_transform_bulk_creates_output(bulk_raw):
    _, processed = bulk_raw
    StooqSource().transform(feed="bulk")
    assert (processed / "bulk_prices.parquet").exists()


def test_transform_bulk_columns(bulk_raw):
    _, processed = bulk_raw
    StooqSource().transform(feed="bulk")
    df = pd.read_parquet(processed / "bulk_prices.parquet")
    assert set(df.columns) == _EXPECTED_COLUMNS


def test_transform_bulk_ticker_lowercase(bulk_raw):
    """ticker = split_part(lower(<TICKER>), '.', 1): aapl, eurusd."""
    _, processed = bulk_raw
    StooqSource().transform(feed="bulk")
    df = pd.read_parquet(processed / "bulk_prices.parquet")
    assert set(df["ticker"]) == {"aapl", "eurusd"}


def test_transform_bulk_src_ticker(bulk_raw):
    """src_ticker = lower(<TICKER>)."""
    _, processed = bulk_raw
    StooqSource().transform(feed="bulk")
    df = pd.read_parquet(processed / "bulk_prices.parquet")
    assert set(df["src_ticker"]) == {"aapl", "eurusd"}


# --- update (csv) ---

def test_transform_update_creates_output(update_raw):
    _, processed = update_raw
    StooqSource().transform(feed="update")
    assert (processed / _UPDATE_FILE).exists()


def test_transform_update_columns(update_raw):
    _, processed = update_raw
    StooqSource().transform(feed="update")
    df = pd.read_csv(processed / _UPDATE_FILE)
    assert set(df.columns) == _EXPECTED_COLUMNS


def test_transform_update_ticker(update_raw):
    _, processed = update_raw
    StooqSource().transform(feed="update")
    df = pd.read_csv(processed / _UPDATE_FILE)
    assert "aapl" in df["ticker"].values

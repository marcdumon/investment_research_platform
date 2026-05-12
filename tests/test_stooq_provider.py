import zipfile
from pathlib import Path
from unittest.mock import MagicMock

import pandas as pd
import pytest

import irp.sources.stooq as mod
from irp.sources.stooq import StooqSource


TICKER_CSV = (
    "<TICKER>,<PER>,<DATE>,<TIME>,<OPEN>,<HIGH>,<LOW>,<CLOSE>,<VOL>,<OPENINT>\n"
    "AAPL,D,20230101,0,150.0,155.0,148.0,152.0,1000000,0\n"
)
ZIP_NAME = "d_us_txt.zip"
ZIP_INNER = "data/daily/us/nasdaq stocks/aapl.us.txt"


@pytest.fixture
def raw_dir(tmp_path):
    raw = tmp_path / "raw"
    raw.mkdir()
    with zipfile.ZipFile(raw / ZIP_NAME, "w") as zf:
        zf.writestr(ZIP_INNER, TICKER_CSV)
    return raw


@pytest.fixture
def patched(raw_dir, monkeypatch):
    cfg = MagicMock()
    cfg.bulk_files = [ZIP_NAME]
    cfg.raw_dir = "stooq/raw"
    monkeypatch.setattr(mod, "raw_dir", raw_dir)
    monkeypatch.setattr(mod, "stooq_cfg", cfg)
    return raw_dir


def test_fetch_creates_outputs(patched):
    StooqSource().fetch()
    assert (patched / "ticker_prices.parquet").exists()
    assert (patched / "ticker_markets.csv").exists()


def test_fetch_deletes_data_dir(patched):
    StooqSource().fetch()
    assert not (patched / "data").exists()


def test_fetch_parquet_contains_ticker(patched):
    StooqSource().fetch()
    df = pd.read_parquet(patched / "ticker_prices.parquet")
    assert "aapl.us" in df["ticker"].values


def test_fetch_markets_csv_contains_market(patched):
    StooqSource().fetch()
    df = pd.read_csv(patched / "ticker_markets.csv")
    assert "nasdaq stocks" in df["market"].values


def test_fetch_nested_shards_and_subcategories(tmp_path, monkeypatch):
    """Numeric shard dirs and named subcategory dirs both resolve to correct market."""
    raw = tmp_path / "raw"
    raw.mkdir()

    ticker_csv2 = (
        "<TICKER>,<PER>,<DATE>,<TIME>,<OPEN>,<HIGH>,<LOW>,<CLOSE>,<VOL>,<OPENINT>\n"
        "MSFT,D,20230101,0,200.0,205.0,198.0,202.0,500000,0\n"
    )
    eurusd_csv = (
        "<TICKER>,<PER>,<DATE>,<TIME>,<OPEN>,<HIGH>,<LOW>,<CLOSE>,<VOL>,<OPENINT>\n"
        "EURUSD,D,20230101,0,1.1,1.11,1.09,1.105,0,0\n"
    )

    with zipfile.ZipFile(raw / ZIP_NAME, "w") as zf:
        zf.writestr("data/daily/us/nasdaq stocks/1/aapl.us.txt", TICKER_CSV)   # numeric shard
        zf.writestr("data/daily/us/nasdaq stocks/2/msft.us.txt", ticker_csv2)  # numeric shard
        zf.writestr("data/daily/world/currencies/major/eurusd.txt", eurusd_csv)  # named subcat

    cfg = MagicMock()
    cfg.bulk_files = [ZIP_NAME]
    cfg.raw_dir = "stooq/raw"
    monkeypatch.setattr(mod, "raw_dir", raw)
    monkeypatch.setattr(mod, "stooq_cfg", cfg)

    StooqSource().fetch()

    markets = pd.read_csv(raw / "ticker_markets.csv").set_index("ticker")["market"]
    assert markets["aapl.us"] == "nasdaq stocks"
    assert markets["msft.us"] == "nasdaq stocks"
    assert markets["eurusd"] == "currencies"


def test_fetch_raises_when_zips_missing(tmp_path, monkeypatch):
    raw = tmp_path / "raw"
    raw.mkdir()
    cfg = MagicMock()
    cfg.bulk_files = [ZIP_NAME]
    monkeypatch.setattr(mod, "raw_dir", raw)
    monkeypatch.setattr(mod, "stooq_cfg", cfg)
    with pytest.raises(FileNotFoundError):
        StooqSource().fetch()

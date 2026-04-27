import zipfile
from pathlib import Path
from unittest.mock import patch

import pytest

from irp.sources.stooq_bulk import StooqBulkSource

_CSV_US = "Date,Open,High,Low,Close,Volume\n20240102,374.0,376.5,372.0,375.0,20000000\n20240103,375.0,378.0,373.0,377.0,21000000\n"
_CSV_WORLD = "Date,Open,High,Low,Close,Volume\n20240102,4800.0,4820.0,4790.0,4810.0,0\n"


def _make_zips(tmp_path: Path) -> None:
    with zipfile.ZipFile(tmp_path / "d_us_txt.zip", "w") as zf:
        zf.writestr("data/daily/us/nasdaq stocks/1/msft.us.txt", _CSV_US)
    with zipfile.ZipFile(tmp_path / "d_world_txt.zip", "w") as zf:
        zf.writestr("data/daily/world/indices/^spx.txt", _CSV_WORLD)


def _cfg(tmp_path: Path) -> dict:
    return {
        "stooq": {
            "download_dir": str(tmp_path),
            "data_dir": str(tmp_path / "daily"),
        },
        "simfin": {"data_dir": "data/simfin"},
    }


@pytest.fixture
def bulk_config(tmp_path):
    _make_zips(tmp_path)
    with patch("irp.sources.stooq_bulk._config.load", return_value=_cfg(tmp_path)):
        yield


def test_fetch_us_ticker(bulk_config):
    ds = StooqBulkSource("msft.us").fetch()
    assert ds.name == "msft.us"
    assert ds.source == "stooq_bulk"
    assert list(ds.data.columns) == ["ticker", "date", "open", "high", "low", "close", "volume"]
    assert len(ds.data) == 2


def test_fetch_world_ticker(bulk_config):
    ds = StooqBulkSource("^spx").fetch()
    assert ds.name == "^spx"
    assert len(ds.data) == 1


def test_date_normalized(bulk_config):
    ds = StooqBulkSource("msft.us").fetch()
    assert ds.data["date"].iloc[0] == "2024-01-02"


def test_date_filter(bulk_config):
    ds = StooqBulkSource("msft.us", start="2024-01-03").fetch()
    assert len(ds.data) == 1
    assert ds.data["date"].iloc[0] == "2024-01-03"


def test_validates(bulk_config):
    StooqBulkSource("msft.us").fetch().validate()


def test_no_zips_raises(tmp_path):
    with patch("irp.sources.stooq_bulk._config.load", return_value=_cfg(tmp_path)):
        with pytest.raises(FileNotFoundError, match="No zip files"):
            StooqBulkSource("msft.us").fetch()


def test_missing_ticker_raises(bulk_config):
    with pytest.raises(FileNotFoundError, match="unknown.us"):
        StooqBulkSource("unknown.us").fetch()


def test_extraction_marker_prevents_reextract(bulk_config, tmp_path):
    StooqBulkSource("msft.us").fetch()
    marker = tmp_path / "daily" / ".extracted_d_us_txt"
    assert marker.exists()

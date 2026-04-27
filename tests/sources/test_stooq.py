
import io
from unittest.mock import MagicMock, patch

import pytest

from irp.sources.stooq import StooqPriceSource

_FIXTURE_CSV = "Date,Open,High,Low,Close,Volume\n2024-01-02,374.0,376.5,372.0,375.0,20000000\n2024-01-03,375.0,378.0,373.0,377.0,21000000\n"


def _mock_urlopen(url):
    resp = MagicMock()
    resp.read.return_value = _FIXTURE_CSV.encode()
    resp.__enter__ = lambda s: s
    resp.__exit__ = MagicMock(return_value=False)
    return resp


@pytest.fixture(autouse=True)
def stooq_env(monkeypatch):
    monkeypatch.setenv("STOOQ_API_KEY", "testkey")


def test_missing_api_key(monkeypatch):
    monkeypatch.delenv("STOOQ_API_KEY", raising=False)
    with pytest.raises(KeyError):
        StooqPriceSource("msft.us", "2024-01-01", "2024-01-31")


def test_fetch_returns_dataset():
    with patch("irp.sources.stooq.urllib.request.urlopen", side_effect=_mock_urlopen):
        ds = StooqPriceSource("msft.us", "2024-01-01", "2024-01-31").fetch()

    assert ds.name == "msft.us"
    assert ds.source == "stooq"
    assert list(ds.data.columns) == ["ticker", "date", "open", "high", "low", "close", "volume"]
    assert len(ds.data) == 2


def test_fetch_schema_keys():
    with patch("irp.sources.stooq.urllib.request.urlopen", side_effect=_mock_urlopen):
        ds = StooqPriceSource("msft.us", "2024-01-01", "2024-01-31").fetch()

    ds.validate()


import io
import urllib.error
from unittest.mock import MagicMock, patch

import pytest

from irp.sources.stooq import StooqPriceSource, StooqRateLimitError

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
    assert list(ds.data.columns) == ["ticker", "source_id", "source", "date", "open", "high", "low", "close", "volume"]
    assert ds.data["ticker"].iloc[0] == "MSFT"
    assert ds.data["source_id"].iloc[0] == "msft.us"
    assert ds.data["source"].iloc[0] == "stooq"
    assert len(ds.data) == 2


def test_fetch_schema_keys():
    with patch("irp.sources.stooq.urllib.request.urlopen", side_effect=_mock_urlopen):
        ds = StooqPriceSource("msft.us", "2024-01-01", "2024-01-31").fetch()

    ds.validate()


def _mock_urlopen_html(url):
    resp = MagicMock()
    resp.read.return_value = b"<html><body>Exceeded the daily number</body></html>"
    resp.__enter__ = lambda s: s
    resp.__exit__ = MagicMock(return_value=False)
    return resp


def test_rate_limit_html_response_raises():
    with patch("irp.sources.stooq.urllib.request.urlopen", side_effect=_mock_urlopen_html):
        with pytest.raises(StooqRateLimitError):
            StooqPriceSource("msft.us", "2024-01-01", "2024-01-31").fetch()


def test_rate_limit_http_403_raises():
    err = urllib.error.HTTPError(url="", code=403, msg="Forbidden", hdrs=None, fp=None)
    with patch("irp.sources.stooq.urllib.request.urlopen", side_effect=err):
        with pytest.raises(StooqRateLimitError):
            StooqPriceSource("msft.us", "2024-01-01", "2024-01-31").fetch()


def test_normalize_ticker():
    from irp.sources.stooq import normalize_ticker
    assert normalize_ticker("msft.us") == "MSFT"
    assert normalize_ticker("aapl.us") == "AAPL"
    assert normalize_ticker("eurusd.fx") == "EURUSD"
    assert normalize_ticker("^spx") == "^SPX"

from unittest.mock import patch

import pandas as pd
import pytest

from irp.sources.simfin import SimFinFundamentalsSource

_FIXTURE_DF = pd.DataFrame(
    {
        "SimFinId": [1001, 1002],
        "Ticker": ["MSFT", "AAPL"],
        "Report Date": ["2023-12-31", "2023-09-30"],
        "Fiscal Year": [2023, 2023],
        "Fiscal Period": ["FY", "FY"],
        "Revenue": [211915e6, 383285e6],
        "Net Income": [72361e6, 96995e6],
    }
).set_index(["SimFinId", "Ticker", "Report Date"])

_MOCK_LOADERS = {"income": lambda **_: _FIXTURE_DF}


@pytest.fixture(autouse=True)
def simfin_env(monkeypatch):
    monkeypatch.setenv("SIMFIN_API_KEY", "testkey")
    monkeypatch.setenv("SIMFIN_DATA_DIR", "/tmp/simfin_test")


def test_missing_api_key(monkeypatch):
    monkeypatch.delenv("SIMFIN_API_KEY", raising=False)
    with pytest.raises(KeyError):
        SimFinFundamentalsSource("income")


def test_fetch_income_returns_dataset():
    with (
        patch("irp.sources.simfin.simfin.set_api_key"),
        patch("irp.sources.simfin.simfin.set_data_dir"),
        patch("irp.sources.simfin._LOADERS", _MOCK_LOADERS),
    ):
        ds = SimFinFundamentalsSource("income").fetch()

    assert ds.name == "income"
    assert ds.source == "simfin"
    assert "source_id" in ds.data.columns
    assert "source" in ds.data.columns
    assert ds.data["source"].iloc[0] == "simfin"
    assert "Ticker" in ds.data.columns
    assert "Revenue" in ds.data.columns
    assert len(ds.data) == 2


def test_schema_built_from_columns():
    with (
        patch("irp.sources.simfin.simfin.set_api_key"),
        patch("irp.sources.simfin.simfin.set_data_dir"),
        patch("irp.sources.simfin._LOADERS", _MOCK_LOADERS),
    ):
        ds = SimFinFundamentalsSource("income").fetch()

    assert set(ds.schema.keys()) == set(ds.data.columns)

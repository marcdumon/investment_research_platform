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


def _make_loader(rows: list[dict]):
    df = pd.DataFrame(rows).set_index(["SimFinId", "Ticker", "Report Date"])
    return lambda **_: df


def _fetch_annual(rows):
    loader = _make_loader(rows)
    with (
        patch("irp.sources.simfin.simfin.set_api_key"),
        patch("irp.sources.simfin.simfin.set_data_dir"),
        patch("irp.sources.simfin._LOADERS", {"income": loader}),
    ):
        return SimFinFundamentalsSource("income", variant="annual").fetch().data


def _fetch_quarterly(rows):
    loader = _make_loader(rows)
    with (
        patch("irp.sources.simfin.simfin.set_api_key"),
        patch("irp.sources.simfin.simfin.set_data_dir"),
        patch("irp.sources.simfin._LOADERS", {"income": loader}),
    ):
        return SimFinFundamentalsSource("income", variant="quarterly").fetch().data


def test_annual_dec_fy_period():
    # Dec FY: Fiscal Year matches Report Date year → "2023FY"
    df = _fetch_annual([{
        "SimFinId": 1, "Ticker": "MSFT",
        "Report Date": "2023-12-31", "Fiscal Year": 2023, "Fiscal Period": "FY",
        "Revenue": 100.0,
    }])
    assert df.loc[df["Ticker"] == "MSFT", "period"].iloc[0] == "2023FY"


def test_annual_non_dec_fy_period():
    # Mar FY: SimFin Fiscal Year=2021, Report Date=2022-03-31 → SEC says 2022FY
    df = _fetch_annual([{
        "SimFinId": 2, "Ticker": "BENF",
        "Report Date": "2022-03-31", "Fiscal Year": 2021, "Fiscal Period": "FY",
        "Revenue": 100.0,
    }])
    assert df.loc[df["Ticker"] == "BENF", "period"].iloc[0] == "2022FY"


def test_annual_sep_fy_period():
    # Sep FY: Fiscal Year=2023, Report Date=2023-09-30 → "2023FY" (no correction needed)
    df = _fetch_annual([{
        "SimFinId": 3, "Ticker": "AAPL",
        "Report Date": "2023-09-30", "Fiscal Year": 2023, "Fiscal Period": "FY",
        "Revenue": 100.0,
    }])
    assert df.loc[df["Ticker"] == "AAPL", "period"].iloc[0] == "2023FY"


def test_quarterly_non_dec_fy_period_propagated():
    # Mar FY: Q4 Report Date=2022-03-31 (year=2022) propagates to Q1/Q2/Q3
    rows = [
        {"SimFinId": 4, "Ticker": "BENF", "Report Date": "2021-06-30",
         "Fiscal Year": 2021, "Fiscal Period": "Q1", "Revenue": 10.0},
        {"SimFinId": 4, "Ticker": "BENF", "Report Date": "2021-09-30",
         "Fiscal Year": 2021, "Fiscal Period": "Q2", "Revenue": 10.0},
        {"SimFinId": 4, "Ticker": "BENF", "Report Date": "2021-12-31",
         "Fiscal Year": 2021, "Fiscal Period": "Q3", "Revenue": 10.0},
        {"SimFinId": 4, "Ticker": "BENF", "Report Date": "2022-03-31",
         "Fiscal Year": 2021, "Fiscal Period": "Q4", "Revenue": 10.0},
    ]
    df = _fetch_quarterly(rows)
    benf = df[df["Ticker"] == "BENF"].set_index("period")
    assert "2022Q1" in benf.index
    assert "2022Q2" in benf.index
    assert "2022Q3" in benf.index
    assert "2022Q4" in benf.index

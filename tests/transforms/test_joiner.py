import pandas as pd
import pytest

from irp.datasets.dataset import Dataset
from irp.transforms.joiner import Joiner


def make_prices() -> Dataset:
    df = pd.DataFrame({
        "ticker": ["MSFT", "MSFT"],
        "date": ["2024-01-01", "2024-01-02"],
        "close": [374.0, 376.0],
    })
    return Dataset(name="prices", data=df, schema={}, source="test")


def make_fund() -> Dataset:
    df = pd.DataFrame({
        "ticker": ["MSFT", "MSFT"],
        "date": ["2024-01-01", "2024-01-02"],
        "revenue": [100.0, 100.0],
    })
    return Dataset(name="fund", data=df, schema={}, source="test")


def make_companies() -> Dataset:
    df = pd.DataFrame({
        "ticker": ["MSFT"],
        "sector": ["Technology"],
    })
    return Dataset(name="companies", data=df, schema={}, source="test")


def test_join_on_ticker_date():
    ds = Joiner(right=make_fund(), on=["ticker", "date"]).transform(make_prices())
    assert "revenue" in ds.data.columns
    assert "close" in ds.data.columns
    assert len(ds.data) == 2


def test_static_join_on_ticker_only():
    ds = Joiner(right=make_companies(), on=["ticker"]).transform(make_prices())
    assert "sector" in ds.data.columns
    assert len(ds.data) == 2


def test_output_name():
    ds = Joiner(right=make_fund(), on=["ticker", "date"]).transform(make_prices())
    assert ds.name == "prices+fund"


def test_source_is_join():
    ds = Joiner(right=make_fund(), on=["ticker", "date"]).transform(make_prices())
    assert ds.source == "join"


def test_inner_join_drops_unmatched():
    fund = make_fund()
    fund.data.__class__  # just reference
    prices = make_prices()
    # add a row to prices with no match in fund
    extra = pd.concat([prices.data, pd.DataFrame([{"ticker": "AAPL", "date": "2024-01-01", "close": 180.0}])], ignore_index=True)
    prices_extra = Dataset(name="prices", data=extra, schema={}, source="test")
    ds = Joiner(right=fund, on=["ticker", "date"], how="inner").transform(prices_extra)
    assert len(ds.data) == 2

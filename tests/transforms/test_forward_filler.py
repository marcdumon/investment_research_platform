import pandas as pd
import pytest

from irp.datasets.dataset import Dataset
from irp.transforms.forward_filler import ForwardFiller


def make_ds() -> Dataset:
    df = pd.DataFrame({
        "date": pd.to_datetime(["2024-01-01", "2024-01-04"]),
        "Ticker": ["MSFT", "MSFT"],
        "revenue": [100.0, 110.0],
    })
    return Dataset(name="test", data=df, schema={}, source="test")


def test_fills_gaps():
    target = pd.date_range("2024-01-01", "2024-01-05")
    ds = ForwardFiller(target_dates=target).transform(make_ds())
    msft = ds.data[ds.data["Ticker"] == "MSFT"].sort_values("date")
    assert len(msft) == 5
    # Jan 2-3 filled with 100.0 (from Jan 1)
    assert msft.iloc[1]["revenue"] == 100.0
    assert msft.iloc[2]["revenue"] == 100.0
    # Jan 4 onward = 110.0
    assert msft.iloc[3]["revenue"] == 110.0


def test_limit_caps_fill():
    target = pd.date_range("2024-01-01", "2024-01-10")
    ds = ForwardFiller(target_dates=target, limit=1).transform(make_ds())
    msft = ds.data[ds.data["Ticker"] == "MSFT"].sort_values("date")
    # only 1 day filled after Jan 1 → Jan 3 should be NaN
    jan3 = msft[msft["date"] == pd.Timestamp("2024-01-03")]["revenue"].iloc[0]
    assert pd.isna(jan3)


def test_output_has_all_target_dates():
    target = pd.date_range("2024-01-01", "2024-01-05")
    ds = ForwardFiller(target_dates=target).transform(make_ds())
    dates = ds.data["date"].sort_values().unique()
    assert len(dates) == 5

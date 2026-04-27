import pandas as pd
import pytest

from irp.datasets.dataset import Dataset
from irp.transforms.date_aligner import DateAligner


def make_ds() -> Dataset:
    df = pd.DataFrame({
        "date": pd.to_datetime(["2024-01-01", "2024-01-03"]),
        "close": [10.0, 12.0],
    })
    return Dataset(name="test", data=df, schema={"date": "datetime64[ns]", "close": "float64"}, source="test")


def test_adds_missing_dates_with_ffill():
    target = pd.DatetimeIndex(pd.date_range("2024-01-01", "2024-01-03"))
    ds = DateAligner(target_dates=target, ticker_col=None).transform(make_ds())
    assert len(ds.data) == 3
    assert ds.data.iloc[1]["close"] == 10.0  # Jan 2 filled from Jan 1


def test_drops_extra_dates():
    target = pd.DatetimeIndex([pd.Timestamp("2024-01-01")])
    ds = DateAligner(target_dates=target, ticker_col=None).transform(make_ds())
    assert len(ds.data) == 1


def test_method_none_leaves_nan():
    target = pd.DatetimeIndex(pd.date_range("2024-01-01", "2024-01-03"))
    ds = DateAligner(target_dates=target, ticker_col=None, method="none").transform(make_ds())
    assert pd.isna(ds.data.iloc[1]["close"])

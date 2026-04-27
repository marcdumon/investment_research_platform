import pandas as pd
import pytest

from irp.datasets.dataset import Dataset
from irp.transforms.date_parser import DateParser


def make_ds(dates: list[str], col: str = "date") -> Dataset:
    df = pd.DataFrame({col: dates, "close": [1.0] * len(dates)})
    return Dataset(name="test", data=df, schema={col: "object", "close": "float64"}, source="test")


def test_default_col_parsed():
    ds = DateParser().transform(make_ds(["2024-01-01", "2024-01-02"]))
    assert pd.api.types.is_datetime64_any_dtype(ds.data["date"])


def test_custom_col_parsed():
    ds = DateParser(col="Publish Date").transform(make_ds(["2024-01-01"], col="Publish Date"))
    assert pd.api.types.is_datetime64_any_dtype(ds.data["Publish Date"])


def test_schema_updated():
    ds = DateParser().transform(make_ds(["2024-01-01"]))
    assert ds.schema["date"] == "datetime64[ns]"


def test_other_cols_unchanged():
    ds = DateParser().transform(make_ds(["2024-01-01"]))
    assert ds.data["close"].dtype == "float64"

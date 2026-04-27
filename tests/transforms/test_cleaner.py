import numpy as np
import pandas as pd
import pytest

from irp.datasets.dataset import Dataset
from irp.transforms.cleaner import Cleaner


def make_ds(data: dict) -> Dataset:
    df = pd.DataFrame(data)
    schema = {c: str(df[c].dtype) for c in df.columns}
    return Dataset(name="test", data=df, schema=schema, source="test")


def test_inf_replaced_with_nan():
    ds = make_ds({"v": [1.0, np.inf, -np.inf]})
    out = Cleaner(drop_all_nan=False).transform(ds)
    assert out.data["v"].isna().sum() == 2


def test_all_nan_rows_dropped():
    ds = make_ds({"a": [1.0, np.nan], "b": [2.0, np.nan]})
    out = Cleaner(drop_all_nan=True).transform(ds)
    assert len(out.data) == 1


def test_partial_nan_row_kept():
    ds = make_ds({"a": [1.0, np.nan], "b": [2.0, 3.0]})
    out = Cleaner(drop_all_nan=True).transform(ds)
    assert len(out.data) == 2


def test_clip_percentile():
    ds = make_ds({"v": [1.0, 2.0, 3.0, 4.0, 100.0]})
    out = Cleaner(drop_all_nan=False, clip_percentile=0.1).transform(ds)
    assert out.data["v"].max() < 100.0


def test_schema_preserved():
    ds = make_ds({"v": [1.0, 2.0]})
    out = Cleaner().transform(ds)
    assert out.schema == ds.schema

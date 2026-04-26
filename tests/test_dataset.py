import dataclasses

import pandas as pd
import pytest

from irp.datasets.dataset import Dataset


def make_dataset(**kwargs) -> Dataset:
    defaults = dict(
        name="test",
        data=pd.DataFrame({"close": [1.0, 2.0], "volume": [100, 200]}),
        schema={"close": "float64", "volume": "int64"},
        source="test_source",
    )
    return Dataset(**{**defaults, **kwargs})


def test_dataset_immutable():
    ds = make_dataset()
    with pytest.raises(dataclasses.FrozenInstanceError):
        ds.name = "other"  # type: ignore[misc]


def test_dataset_data_is_copy():
    df = pd.DataFrame({"close": [1.0]})
    ds = make_dataset(data=df, schema={"close": "float64"})
    df["close"] = 999.0
    assert ds.data["close"].iloc[0] == 1.0


def test_validate_passes():
    make_dataset().validate()


def test_validate_fails_missing_column():
    ds = make_dataset(schema={"close": "float64", "missing_col": "float64"})
    with pytest.raises(ValueError, match="missing columns"):
        ds.validate()

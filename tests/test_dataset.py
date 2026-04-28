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


def test_validate_fails_dtype_mismatch():
    df = pd.DataFrame({"close": [1, 2]})  # int64, not float64
    ds = Dataset(name="t", data=df, schema={"close": "float64"}, source="s")
    with pytest.raises(ValueError, match="dtype mismatch"):
        ds.validate()


def test_from_df_infers_schema():
    df = pd.DataFrame({"close": [1.0], "ticker": ["MSFT"]})
    ds = Dataset.from_df(df, name="prices", source="stooq")
    assert ds.schema == {"close": "float64", "ticker": "str"}
    assert ds.name == "prices"
    assert ds.source == "stooq"


def test_evolve_preserves_metadata():
    ds = make_dataset()
    df2 = pd.DataFrame({"close": [9.0], "volume": [50]})
    evolved = ds.evolve(data=df2)
    assert evolved.name == ds.name
    assert evolved.source == ds.source
    assert evolved.captured_at == ds.captured_at
    assert evolved.data["close"].iloc[0] == 9.0


def test_evolve_overrides_fields():
    ds = make_dataset()
    evolved = ds.evolve(name="renamed", source="other")
    assert evolved.name == "renamed"
    assert evolved.source == "other"
    assert evolved.schema == ds.schema

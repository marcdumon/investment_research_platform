import pandas as pd
import pytest

from irp.datasets.dataset import Dataset
from irp.store import Store


def make_ds(name: str = "test_prices") -> Dataset:
    df = pd.DataFrame({
        "date": pd.to_datetime(["2024-01-01", "2024-01-02"]),
        "close": [374.0, 376.0],
    })
    return Dataset(name=name, data=df, schema={"date": "datetime64[ns]", "close": "float64"}, source="stooq")


@pytest.fixture
def store(tmp_path):
    return Store(db_path=tmp_path / "test.duckdb")


def test_save_and_load(store):
    ds = make_ds()
    store.save(ds)
    loaded = store.load("test_prices")
    assert loaded.name == ds.name
    assert loaded.source == ds.source
    assert set(loaded.data.columns) == set(ds.data.columns)
    assert len(loaded.data) == 2


def test_schema_roundtrip(store):
    ds = make_ds()
    store.save(ds)
    loaded = store.load("test_prices")
    assert loaded.schema == ds.schema


def test_overwrite(store):
    store.save(make_ds())
    ds2 = Dataset(
        name="test_prices",
        data=pd.DataFrame({"date": pd.to_datetime(["2024-01-03"]), "close": [380.0]}),
        schema={"date": "datetime64[ns]", "close": "float64"},
        source="stooq",
    )
    store.save(ds2)
    loaded = store.load("test_prices")
    assert len(loaded.data) == 1


def test_list(store):
    store.save(make_ds("ds_a"))
    store.save(make_ds("ds_b"))
    assert set(store.list()) == {"ds_a", "ds_b"}


def test_delete(store):
    store.save(make_ds())
    store.delete("test_prices")
    assert not store.exists("test_prices")


def test_load_missing_raises(store):
    with pytest.raises(KeyError, match="not found"):
        store.load("nonexistent")


def test_exists(store):
    assert not store.exists("test_prices")
    store.save(make_ds())
    assert store.exists("test_prices")

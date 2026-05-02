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


def make_str_date_ds(name: str = "prices", ticker: str = "MSFT") -> Dataset:
    df = pd.DataFrame({
        "ticker": [ticker, ticker],
        "date": ["2024-01-01", "2024-01-02"],
        "close": [100.0, 101.0],
    })
    return Dataset(name=name, data=df, schema={"ticker": "object", "date": "object", "close": "float64"}, source="test")


@pytest.fixture
def store(tmp_path):
    return Store(db_path=tmp_path / "test.duckdb")


def test_save_and_load(store):
    ds = make_ds()
    store.save(ds)
    loaded = store.load("test_prices")
    assert loaded.name == ds.name
    assert loaded.source == ds.source
    assert set(ds.data.columns).issubset(set(loaded.data.columns))
    assert "inserted_at" in loaded.data.columns
    assert len(loaded.data) == 2


def test_schema_roundtrip(store):
    ds = make_ds()
    store.save(ds)
    loaded = store.load("test_prices")
    assert ds.schema.items() <= loaded.schema.items()
    assert "inserted_at" in loaded.schema


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


def test_get_max_date_nonexistent_table(store):
    assert store.get_max_date("prices", "date") is None


def test_get_max_date_no_filter(store):
    store.upsert(make_str_date_ds(), table="prices", primary_key=["ticker", "date"])
    assert store.get_max_date("prices", "date") == "2024-01-02"


def test_get_max_date_with_filter(store):
    store.upsert(make_str_date_ds(ticker="MSFT"), table="prices", primary_key=["ticker", "date"])
    store.upsert(make_str_date_ds(ticker="AAPL"), table="prices", primary_key=["ticker", "date"])
    assert store.get_max_date("prices", "date", filter_col="ticker", filter_val="MSFT") == "2024-01-02"
    assert store.get_max_date("prices", "date", filter_col="ticker", filter_val="GOOG") is None


def test_upsert_creates_table(store):
    store.upsert(make_str_date_ds(), table="prices", primary_key=["ticker", "date"])
    assert store.exists("prices")
    assert len(store.load("prices").data) == 2


def test_upsert_adds_rows(store):
    store.upsert(make_str_date_ds(ticker="MSFT"), table="prices", primary_key=["ticker", "date"])
    df2 = pd.DataFrame({"ticker": ["AAPL"], "date": ["2024-01-03"], "close": [180.0]})
    ds2 = Dataset(name="prices", data=df2, schema={"ticker": "object", "date": "object", "close": "float64"}, source="test")
    store.upsert(ds2, table="prices", primary_key=["ticker", "date"])
    loaded = store.load("prices")
    assert len(loaded.data) == 3
    assert set(loaded.data["ticker"]) == {"MSFT", "AAPL"}


def test_upsert_no_duplicates(store):
    ds = make_str_date_ds(ticker="MSFT")
    store.upsert(ds, table="prices", primary_key=["ticker", "date"])
    store.upsert(ds, table="prices", primary_key=["ticker", "date"])
    assert len(store.load("prices").data) == 2


def test_upsert_updates_rows(store):
    ds1 = make_str_date_ds(ticker="MSFT")
    store.upsert(ds1, table="prices", primary_key=["ticker", "date"])
    df2 = pd.DataFrame({
        "ticker": ["MSFT", "MSFT"],
        "date": ["2024-01-01", "2024-01-02"],
        "close": [200.0, 201.0],
    })
    ds2 = Dataset(name="prices", data=df2, schema={"ticker": "object", "date": "object", "close": "float64"}, source="test")
    store.upsert(ds2, table="prices", primary_key=["ticker", "date"])
    loaded = store.load("prices")
    assert len(loaded.data) == 2
    assert set(loaded.data["close"]) == {200.0, 201.0}

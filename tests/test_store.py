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


def make_price_ds(ticker: str) -> Dataset:
    df = pd.DataFrame({
        "ticker": [ticker, ticker],
        "date": pd.to_datetime(["2024-01-01", "2024-01-02"]),
        "close": [100.0, 101.0],
    })
    return Dataset(name=ticker, data=df, schema={"ticker": "object", "date": "datetime64[ns]", "close": "float64"}, source="stooq_bulk")


def test_save_partition_creates_shared_table(store):
    ds = make_price_ds("msft.us")
    store.save(ds, table="prices", partition_col="ticker")
    assert store.exists("prices")
    loaded = store.load("prices")
    assert len(loaded.data) == 2
    assert set(loaded.data["ticker"]) == {"msft.us"}


def test_save_partition_two_tickers(store):
    store.save(make_price_ds("msft.us"), table="prices", partition_col="ticker")
    store.save(make_price_ds("aapl.us"), table="prices", partition_col="ticker")
    loaded = store.load("prices")
    assert len(loaded.data) == 4
    assert set(loaded.data["ticker"]) == {"msft.us", "aapl.us"}


def test_save_partition_upsert_replaces(store):
    store.save(make_price_ds("msft.us"), table="prices", partition_col="ticker")
    store.save(make_price_ds("msft.us"), table="prices", partition_col="ticker")
    loaded = store.load("prices")
    assert len(loaded.data) == 2  # not 4


def test_exists_partition(store):
    assert not store.exists_partition("prices", "ticker", "msft.us")
    store.save(make_price_ds("msft.us"), table="prices", partition_col="ticker")
    assert store.exists_partition("prices", "ticker", "msft.us")
    assert not store.exists_partition("prices", "ticker", "aapl.us")


def test_load_partition(store):
    store.save(make_price_ds("msft.us"), table="prices", partition_col="ticker")
    store.save(make_price_ds("aapl.us"), table="prices", partition_col="ticker")
    ds = store.load_partition("prices", "ticker", "msft.us")
    assert len(ds.data) == 2
    assert set(ds.data["ticker"]) == {"msft.us"}


def make_str_date_ds(name: str = "prices", ticker: str = "MSFT") -> Dataset:
    df = pd.DataFrame({
        "ticker": [ticker, ticker],
        "date": ["2024-01-01", "2024-01-02"],
        "close": [100.0, 101.0],
    })
    return Dataset(name=name, data=df, schema={"ticker": "object", "date": "object", "close": "float64"}, source="test")


def test_max_date_nonexistent_table(store):
    assert store.max_date("prices", "date") is None


def test_max_date_no_filter(store):
    store.save(make_str_date_ds(), table="prices", partition_col="ticker")
    assert store.max_date("prices", "date") == "2024-01-02"


def test_max_date_with_filter(store):
    store.save(make_str_date_ds(ticker="MSFT"), table="prices", partition_col="ticker")
    store.save(make_str_date_ds(ticker="AAPL"), table="prices", partition_col="ticker")
    assert store.max_date("prices", "date", filter_col="ticker", filter_val="MSFT") == "2024-01-02"
    assert store.max_date("prices", "date", filter_col="ticker", filter_val="GOOG") is None


def test_append_creates_table(store):
    ds = make_str_date_ds()
    store.append(ds, table="prices")
    assert store.exists("prices")
    assert len(store.load("prices").data) == 2


def test_append_adds_rows(store):
    store.save(make_str_date_ds(ticker="MSFT"), table="prices", partition_col="ticker")
    df2 = pd.DataFrame({"ticker": ["AAPL"], "date": ["2024-01-03"], "close": [180.0]})
    ds2 = Dataset(name="prices", data=df2, schema={"ticker": "object", "date": "object", "close": "float64"}, source="test")
    store.append(ds2, table="prices")
    loaded = store.load("prices")
    assert len(loaded.data) == 3
    assert set(loaded.data["ticker"]) == {"MSFT", "AAPL"}


def test_append_does_not_delete_existing(store):
    store.save(make_str_date_ds(ticker="MSFT"), table="prices", partition_col="ticker")
    ds2 = make_str_date_ds(ticker="MSFT")
    store.append(ds2, table="prices")
    assert len(store.load("prices").data) == 4  # 2 original + 2 appended


def test_append_conflict_cols_skips_duplicates(store):
    ds = make_str_date_ds(ticker="MSFT")
    store.append(ds, table="prices", conflict_cols=["ticker", "date"])
    store.append(ds, table="prices", conflict_cols=["ticker", "date"])
    assert len(store.load("prices").data) == 2  # no doubles


def test_append_conflict_cols_inserts_new_rows(store):
    store.append(make_str_date_ds(ticker="MSFT"), table="prices", conflict_cols=["ticker", "date"])
    df2 = pd.DataFrame({"ticker": ["MSFT"], "date": ["2024-01-03"], "close": [380.0]})
    ds2 = Dataset(name="prices", data=df2, schema={"ticker": "object", "date": "object", "close": "float64"}, source="test")
    store.append(ds2, table="prices", conflict_cols=["ticker", "date"])
    assert len(store.load("prices").data) == 3

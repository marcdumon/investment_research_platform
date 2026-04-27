from unittest.mock import MagicMock

import pandas as pd
import pytest

from irp.datasets.dataset import Dataset
from irp.store import Store
from irp.transforms.caching import CachingTransformer
from irp.transforms.date_parser import DateParser


def make_ds(name: str = "raw") -> Dataset:
    df = pd.DataFrame({"date": ["2024-01-01"], "close": [374.0]})
    return Dataset(name=name, data=df, schema={"date": "object", "close": "float64"}, source="test")


@pytest.fixture
def store(tmp_path):
    return Store(db_path=tmp_path / "test.duckdb")


def test_runs_transform_and_saves(store):
    ct = CachingTransformer(DateParser("date"), store, name="parsed_prices")
    result = ct.transform(make_ds())
    assert result.name == "parsed_prices"
    assert store.exists("parsed_prices")


def test_returns_cache_on_second_call(store):
    ct = CachingTransformer(DateParser("date"), store, name="parsed_prices")
    ct.transform(make_ds())

    # replace transformer with a spy to verify it is NOT called again
    spy = MagicMock(wraps=DateParser("date"))
    ct.transformer = spy
    ct.transform(make_ds())
    spy.transform.assert_not_called()


def test_force_reruns_transform(store):
    ct = CachingTransformer(DateParser("date"), store, name="parsed_prices")
    ct.transform(make_ds())

    spy = MagicMock(wraps=DateParser("date"))
    spy.transform.return_value = make_ds("parsed_prices")
    ct.transformer = spy
    ct.force = True
    ct.transform(make_ds())
    spy.transform.assert_called_once()


def test_cached_result_name(store):
    ct = CachingTransformer(DateParser("date"), store, name="my_dataset")
    result = ct.transform(make_ds())
    assert result.name == "my_dataset"

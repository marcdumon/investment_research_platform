from unittest.mock import MagicMock

import pandas as pd
import pytest

from irp.datasets.dataset import Dataset
from irp.pipeline import Pipeline
from irp.store import Store
from irp.transforms.cleaner import Cleaner
from irp.transforms.date_parser import DateParser


def make_ds() -> Dataset:
    df = pd.DataFrame({"date": ["2024-01-01", "2024-01-02"], "close": [374.0, 376.0]})
    return Dataset(name="raw", data=df, schema={"date": "object", "close": "float64"}, source="test")


@pytest.fixture
def store(tmp_path):
    return Store(db_path=tmp_path / "test.duckdb")


def test_runs_all_steps(store):
    result = (
        Pipeline(store)
        .step(DateParser("date"), name="parsed")
        .step(Cleaner(), name="clean")
        .run(make_ds())
    )
    assert result.name == "clean"
    assert store.exists("parsed")
    assert store.exists("clean")


def test_skips_cached_steps(store):
    pipeline = (
        Pipeline(store)
        .step(DateParser("date"), name="parsed")
        .step(Cleaner(), name="clean")
    )
    pipeline.run(make_ds())

    spy = MagicMock(wraps=DateParser("date"))
    pipeline._steps[0].transformer = spy
    pipeline.run(make_ds())
    spy.transform.assert_not_called()


def test_force_pipeline_reruns_all(store):
    pipeline = Pipeline(store, force=True).step(DateParser("date"), name="parsed")
    pipeline.run(make_ds())

    spy = MagicMock(wraps=DateParser("date"))
    spy.transform.return_value = make_ds()
    pipeline._steps[0].transformer = spy
    pipeline.run(make_ds())
    spy.transform.assert_called_once()


def test_force_single_step(store):
    pipeline = (
        Pipeline(store)
        .step(DateParser("date"), name="parsed", force=True)
    )
    pipeline.run(make_ds())
    spy = MagicMock(wraps=DateParser("date"))
    spy.transform.return_value = make_ds()
    pipeline._steps[0].transformer = spy
    pipeline.run(make_ds())
    spy.transform.assert_called_once()


def test_reset_single_step(store):
    pipeline = Pipeline(store).step(DateParser("date"), name="parsed")
    pipeline.run(make_ds())
    pipeline.reset("parsed")
    assert not store.exists("parsed")


def test_reset_all(store):
    pipeline = (
        Pipeline(store)
        .step(DateParser("date"), name="parsed")
        .step(Cleaner(), name="clean")
    )
    pipeline.run(make_ds())
    pipeline.reset()
    assert not store.exists("parsed")
    assert not store.exists("clean")


def test_result_name_is_last_step(store):
    result = Pipeline(store).step(DateParser("date"), name="parsed").run(make_ds())
    assert result.name == "parsed"

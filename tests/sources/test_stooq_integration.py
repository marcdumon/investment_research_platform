import os

import pytest

from irp.sources.stooq import PRICE_SCHEMA, StooqPriceSource

pytestmark = pytest.mark.skipif(
    not os.environ.get("STOOQ_API_KEY"),
    reason="STOOQ_API_KEY not set",
)


def test_real_fetch_columns():
    ds = StooqPriceSource("msft.us", "2024-01-02", "2024-01-05").fetch()
    assert set(ds.data.columns) == set(PRICE_SCHEMA)


def test_real_fetch_dtypes():
    ds = StooqPriceSource("msft.us", "2024-01-02", "2024-01-05").fetch()
    for col in ("open", "high", "low", "close", "volume"):
        assert ds.data[col].dtype == "float64", f"{col} dtype mismatch: {ds.data[col].dtype}"


def test_real_fetch_nonempty():
    ds = StooqPriceSource("msft.us", "2024-01-02", "2024-01-05").fetch()
    assert len(ds.data) > 0


def test_real_fetch_validates():
    ds = StooqPriceSource("msft.us", "2024-01-02", "2024-01-05").fetch()
    ds.validate()

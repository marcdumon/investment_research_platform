"""Tests for point-in-time alignment in irp.factors._pit."""
import datetime

import pandas as pd
import pytest

from irp.factors._cols import (
    PRICE_CLOSE,
    PRICE_DATE,
    PRICE_TICKER,
    PUBLISH_DATE,
    REPORT_DATE,
    TICKER,
)
from irp.factors._pit import pit_latest, pit_price


def _fund(*rows: dict) -> pd.DataFrame:
    return pd.DataFrame(rows)


def _prices(*rows: dict) -> pd.DataFrame:
    return pd.DataFrame(rows)


class TestPitLatest:
    # ── Fallback path (no Publish Date column) ────────────────────────

    def test_fallback_uses_report_date_plus_60(self):
        # Report Date 2024-02-15 → effective = 2024-04-15
        df = _fund({TICKER: 'AAPL', REPORT_DATE: '2024-02-15', 'Revenue': 100})
        # as_of = 2024-04-14: not yet public
        assert pit_latest(df, datetime.date(2024, 4, 14)).empty
        # as_of = 2024-04-15: just became public
        assert len(pit_latest(df, datetime.date(2024, 4, 15))) == 1

    def test_fallback_returns_most_recent_per_ticker(self):
        df = _fund(
            {TICKER: 'AAPL', REPORT_DATE: '2024-02-15', 'Revenue': 100},
            {TICKER: 'AAPL', REPORT_DATE: '2023-08-10', 'Revenue': 90},
            {TICKER: 'MSFT', REPORT_DATE: '2024-01-20', 'Revenue': 200},
        )
        result = pit_latest(df, datetime.date(2024, 6, 1))
        assert len(result) == 2
        aapl = result.loc[result[TICKER] == 'AAPL'].iloc[0]
        assert aapl['Revenue'] == 100

    def test_fallback_excludes_row_not_yet_public(self):
        # Report Date 2024-07-01 → effective 2024-08-30; as_of 2024-06-01 too early
        df = _fund(
            {TICKER: 'AAPL', REPORT_DATE: '2024-07-01', 'Revenue': 120},
            {TICKER: 'AAPL', REPORT_DATE: '2024-02-15', 'Revenue': 100},
        )
        result = pit_latest(df, datetime.date(2024, 6, 1))
        assert len(result) == 1
        assert result.iloc[0]['Revenue'] == 100

    def test_fallback_returns_empty_when_all_too_recent(self):
        df = _fund({TICKER: 'AAPL', REPORT_DATE: '2025-01-01', 'Revenue': 100})
        result = pit_latest(df, datetime.date(2024, 6, 1))
        assert result.empty

    def test_fallback_ticker_absent_when_all_rows_too_recent(self):
        df = _fund(
            {TICKER: 'AAPL', REPORT_DATE: '2024-02-01', 'Revenue': 100},
            {TICKER: 'GOOG', REPORT_DATE: '2025-01-01', 'Revenue': 999},
        )
        result = pit_latest(df, datetime.date(2024, 6, 1))
        assert 'GOOG' not in result[TICKER].values
        assert 'AAPL' in result[TICKER].values

    def test_does_not_mutate_input(self):
        df = _fund({TICKER: 'AAPL', REPORT_DATE: '2024-02-01', 'Revenue': 100})
        original_dtype = df[REPORT_DATE].dtype
        pit_latest(df, datetime.date(2024, 6, 1))
        assert df[REPORT_DATE].dtype == original_dtype

    def test_preserves_all_columns(self):
        df = _fund({TICKER: 'AAPL', REPORT_DATE: '2024-02-01', 'Revenue': 100, 'NetIncome': 20})
        result = pit_latest(df, datetime.date(2024, 6, 1))
        assert set(df.columns) == set(result.columns)

    # ── Publish Date path ─────────────────────────────────────────────

    def test_publish_date_used_when_present(self):
        # Report Date 2024-01-01 but Publish Date 2024-03-15
        df = _fund({TICKER: 'AAPL', REPORT_DATE: '2024-01-01', PUBLISH_DATE: '2024-03-15', 'Revenue': 100})
        assert pit_latest(df, datetime.date(2024, 3, 14)).empty
        assert len(pit_latest(df, datetime.date(2024, 3, 15))) == 1

    def test_publish_date_earlier_than_fallback(self):
        # Filed quickly: Publish Date < Report Date + 60d
        df = _fund({TICKER: 'AAPL', REPORT_DATE: '2024-01-01', PUBLISH_DATE: '2024-01-20', 'Revenue': 100})
        assert len(pit_latest(df, datetime.date(2024, 1, 20))) == 1

    def test_null_publish_date_falls_back_to_report_plus_60(self):
        df = _fund({TICKER: 'AAPL', REPORT_DATE: '2024-01-01', PUBLISH_DATE: None, 'Revenue': 100})
        # effective = 2024-01-01 + 60 = 2024-03-01
        assert pit_latest(df, datetime.date(2024, 2, 28)).empty
        assert len(pit_latest(df, datetime.date(2024, 3, 1))) == 1

    def test_mixed_publish_date_and_null(self):
        df = _fund(
            {TICKER: 'AAPL', REPORT_DATE: '2024-01-01', PUBLISH_DATE: '2024-01-20', 'Revenue': 100},
            {TICKER: 'MSFT', REPORT_DATE: '2024-01-01', PUBLISH_DATE: None, 'Revenue': 200},
        )
        # 2024-01-25: AAPL published, MSFT not yet (fallback = Mar 1)
        result = pit_latest(df, datetime.date(2024, 1, 25))
        assert list(result[TICKER]) == ['AAPL']

    def test_publish_date_picks_correct_row_per_ticker(self):
        df = _fund(
            {TICKER: 'AAPL', REPORT_DATE: '2024-07-01', PUBLISH_DATE: '2024-07-30', 'Revenue': 120},
            {TICKER: 'AAPL', REPORT_DATE: '2024-01-01', PUBLISH_DATE: '2024-01-20', 'Revenue': 100},
        )
        result = pit_latest(df, datetime.date(2024, 6, 1))
        assert result.iloc[0]['Revenue'] == 100
        result2 = pit_latest(df, datetime.date(2024, 8, 1))
        assert result2.iloc[0]['Revenue'] == 120


class TestPitPrice:
    def test_returns_closest_price_on_or_before_as_of(self):
        df = _prices(
            {PRICE_TICKER: 'AAPL', PRICE_DATE: '2024-05-30', PRICE_CLOSE: 180.0},
            {PRICE_TICKER: 'AAPL', PRICE_DATE: '2024-05-31', PRICE_CLOSE: 182.0},
            {PRICE_TICKER: 'AAPL', PRICE_DATE: '2024-06-03', PRICE_CLOSE: 190.0},
        )
        result = pit_price(df, datetime.date(2024, 6, 1))
        assert result.iloc[0][PRICE_CLOSE] == pytest.approx(182.0)

    def test_inclusive_on_exact_as_of_date(self):
        df = _prices({PRICE_TICKER: 'AAPL', PRICE_DATE: '2024-06-01', PRICE_CLOSE: 180.0})
        result = pit_price(df, datetime.date(2024, 6, 1))
        assert len(result) == 1

    def test_returns_empty_when_all_prices_after_as_of(self):
        df = _prices({PRICE_TICKER: 'AAPL', PRICE_DATE: '2025-01-01', PRICE_CLOSE: 200.0})
        result = pit_price(df, datetime.date(2024, 6, 1))
        assert result.empty

    def test_one_row_per_ticker(self):
        df = _prices(
            {PRICE_TICKER: 'AAPL', PRICE_DATE: '2024-05-01', PRICE_CLOSE: 170.0},
            {PRICE_TICKER: 'AAPL', PRICE_DATE: '2024-05-30', PRICE_CLOSE: 180.0},
            {PRICE_TICKER: 'MSFT', PRICE_DATE: '2024-05-30', PRICE_CLOSE: 400.0},
        )
        result = pit_price(df, datetime.date(2024, 6, 1))
        assert len(result) == 2

    def test_output_contains_required_columns(self):
        df = _prices({PRICE_TICKER: 'AAPL', PRICE_DATE: '2024-05-30', PRICE_CLOSE: 180.0})
        result = pit_price(df, datetime.date(2024, 6, 1))
        assert {PRICE_TICKER, PRICE_DATE, PRICE_CLOSE}.issubset(set(result.columns))

    def test_does_not_mutate_input(self):
        df = _prices({PRICE_TICKER: 'AAPL', PRICE_DATE: '2024-05-30', PRICE_CLOSE: 180.0})
        original_dtype = df[PRICE_DATE].dtype
        pit_price(df, datetime.date(2024, 6, 1))
        assert df[PRICE_DATE].dtype == original_dtype

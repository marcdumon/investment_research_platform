"""Yahoo fetch internals: batch-start logic + actions-update resume (no network)."""
from contextlib import contextmanager
from datetime import date

import duckdb
import pandas as pd

from dataload.context import IngestContext
from dataload.providers.yahoo import _batch_start_kwargs, _fetch_actions
from dataload.state import JsonSet


# --- fix #1: batch start ---------------------------------------------------- #
def test_batch_start_full_history_when_no_dates() -> None:
    assert _batch_start_kwargs([]) == {'period': 'max'}
    assert _batch_start_kwargs([None, None]) == {'period': 'max'}


def test_batch_start_uses_min_known_date_plus_one_day() -> None:
    assert _batch_start_kwargs([date(2024, 1, 5), date(2024, 1, 2)]) == {'start': '2024-01-03'}


def test_batch_start_mixed_does_not_fall_back_to_full_history() -> None:
    """THE fix: one new (date-less) ticker must not force the whole batch to period='max'."""
    assert _batch_start_kwargs([date(2024, 1, 10), None]) == {'start': '2024-01-11'}


# --- fix #2: actions-update resume ----------------------------------------- #
class _FakeTicker:
    @property
    def actions(self) -> pd.DataFrame:
        return pd.DataFrame()  # no events; keeps the test offline


class _FakeYf:
    def __init__(self) -> None:
        self.called: list[str] = []

    def Ticker(self, sym: str) -> _FakeTicker:
        self.called.append(sym)
        return _FakeTicker()


def _ctx(tmp_path) -> IngestContext:
    @contextmanager
    def connect():
        yield duckdb.connect()

    return IngestContext(tmp_path, {'yahoo': {'raw_dir': 'yahoo/raw', 'processed_dir': 'yahoo/processed',
                                              'actions_sleep': 0}}, connect)


def _raw(ctx) -> object:
    raw = ctx.raw_dir('yahoo')
    raw.mkdir(parents=True, exist_ok=True)
    return raw


def test_actions_incremental_fresh_session_fetches_all_then_clears_marker(tmp_path) -> None:
    ctx = _ctx(tmp_path)
    raw = _raw(ctx)
    yf = _FakeYf()
    _fetch_actions(ctx, yf, raw, {'A': 'A', 'B': 'B'}, incremental=True)
    assert set(yf.called) == {'A', 'B'}
    assert not (raw / '.actions_update_session').exists()  # removed on clean completion


def test_actions_incremental_resumes_within_an_open_session(tmp_path) -> None:
    ctx = _ctx(tmp_path)
    raw = _raw(ctx)
    # an interrupted session: marker present + A already done
    (raw / '.actions_update_session').touch()
    JsonSet(raw / 'queried_actions_update.json').save({'A'})
    yf = _FakeYf()
    _fetch_actions(ctx, yf, raw, {'A': 'A', 'B': 'B', 'C': 'C'}, incremental=True)
    assert set(yf.called) == {'B', 'C'}  # A skipped — resumed, not reset


def test_actions_bulk_uses_persistent_tracker(tmp_path) -> None:
    ctx = _ctx(tmp_path)
    raw = _raw(ctx)
    JsonSet(raw / 'queried_actions.json').save({'A'})
    yf = _FakeYf()
    _fetch_actions(ctx, yf, raw, {'A': 'A', 'B': 'B'}, incremental=False)
    assert set(yf.called) == {'B'}  # bulk skips already-queried across runs

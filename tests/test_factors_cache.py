"""Tests for irp.factors.cache — parquet-based cross-section snapshot and backtest cache."""
import datetime
import math
from pathlib import Path
from unittest.mock import MagicMock, patch

import pandas as pd
import pytest

import irp.factors.cache as cache_mod


@pytest.fixture(autouse=True)
def patch_cache_root(tmp_path, monkeypatch):
    """Redirect _CACHE_ROOT to tmp_path for every test."""
    monkeypatch.setattr(cache_mod, '_CACHE_ROOT', tmp_path / 'factor_cache')


_DATE = datetime.date(2024, 12, 31)
_DF = pd.DataFrame(
    {'pe': [10.0, 20.0], 'pb': [1.5, 2.5]},
    index=pd.Index(['AAPL', 'MSFT'], name='Ticker'),
)


def test_store_and_load_roundtrip():
    cache_mod.store(_DATE, 'A', _DF)
    result = cache_mod.load(_DATE, 'A')
    assert result is not None
    pd.testing.assert_frame_equal(result, _DF)


def test_load_returns_none_on_miss():
    assert cache_mod.load(_DATE, 'A') is None


def test_index_preserved():
    cache_mod.store(_DATE, 'Q', _DF)
    result = cache_mod.load(_DATE, 'Q')
    assert result.index.name == 'Ticker'
    assert list(result.index) == ['AAPL', 'MSFT']


def test_clear_variant_removes_only_that_variant():
    cache_mod.store(_DATE, 'A', _DF)
    cache_mod.store(_DATE, 'Q', _DF)
    n = cache_mod.clear('A')
    assert n == 1
    assert cache_mod.load(_DATE, 'A') is None
    assert cache_mod.load(_DATE, 'Q') is not None


def test_clear_all_removes_everything():
    cache_mod.store(_DATE, 'A', _DF)
    cache_mod.store(_DATE, 'Q', _DF)
    n = cache_mod.clear()
    assert n == 2
    assert cache_mod.load(_DATE, 'A') is None
    assert cache_mod.load(_DATE, 'Q') is None


def test_clear_missing_returns_zero():
    n = cache_mod.clear()
    assert n == 0


def test_clear_variant_missing_returns_zero():
    n = cache_mod.clear('A')
    assert n == 0


def test_cross_section_writes_to_cache(tmp_path, monkeypatch):
    """cross_section() stores result on first call when tickers=None."""
    import irp.factors.compute as compute_mod

    monkeypatch.setattr(cache_mod, '_CACHE_ROOT', tmp_path / 'fc')

    dummy = _DF.copy()
    with (
        patch.object(compute_mod, 'cross_section_panel', return_value=dummy),
        patch.object(compute_mod, '_cache', cache_mod),
    ):
        result = compute_mod.cross_section(_DATE, 'A', tickers=None)

    pd.testing.assert_frame_equal(result, dummy)
    assert cache_mod.load(_DATE, 'A') is not None


def test_cross_section_uses_cache_without_db(tmp_path, monkeypatch):
    """cross_section() returns cached value and skips SQL on second call."""
    import irp.factors.compute as compute_mod

    monkeypatch.setattr(cache_mod, '_CACHE_ROOT', tmp_path / 'fc')
    cache_mod.store(_DATE, 'A', _DF)

    mock_sql = MagicMock()
    with (
        patch.object(compute_mod, 'cross_section_panel', mock_sql),
        patch.object(compute_mod, '_cache', cache_mod),
    ):
        result = compute_mod.cross_section(_DATE, 'A', tickers=None)

    mock_sql.assert_not_called()
    pd.testing.assert_frame_equal(result, _DF)


# ── Backtest cache ─────────────────────────────────────────────────────────────

_START = datetime.date(2020, 1, 1)
_END   = datetime.date(2022, 12, 31)

_IC = pd.Series(
    [0.05, 0.12, -0.03, 0.08],
    index=[
        datetime.date(2020, 3, 31), datetime.date(2020, 6, 30),
        datetime.date(2020, 9, 30), datetime.date(2020, 12, 31),
    ],
    name='ic',
)
_QCR = pd.DataFrame(
    {'Q1': [0.0, -0.1], 'Q2': [0.0, 0.05], 'Q3': [0.0, 0.02],
     'Q4': [0.0, 0.08], 'Q5': [0.0, 0.15]},
    index=[datetime.date(2020, 3, 31), datetime.date(2020, 6, 30)],
)
_BT_RESULT = {
    'ic_series': _IC,
    'quintile_cumret': _QCR,
    'mean_ic': 0.055,
    'ic_tstat': 1.8,
    'n_dates': 4,
}
_BT_KWARGS = dict(factor='pe', horizon_days=252, variant='A', freq='Q',
                  start_date=_START, end_date=_END)


def test_store_and_load_backtest_roundtrip():
    cache_mod.store_backtest(**_BT_KWARGS, result=_BT_RESULT)
    r = cache_mod.load_backtest(**_BT_KWARGS)
    assert r is not None
    pd.testing.assert_series_equal(r['ic_series'], _IC)
    pd.testing.assert_frame_equal(r['quintile_cumret'], _QCR)
    assert abs(r['mean_ic'] - 0.055) < 1e-9
    assert abs(r['ic_tstat'] - 1.8) < 1e-9
    assert r['n_dates'] == 4


def test_load_backtest_returns_none_on_miss():
    assert cache_mod.load_backtest(**_BT_KWARGS) is None


def test_store_backtest_nan_scalars():
    result = {**_BT_RESULT, 'mean_ic': float('nan'), 'ic_tstat': float('nan')}
    cache_mod.store_backtest(**_BT_KWARGS, result=result)
    r = cache_mod.load_backtest(**_BT_KWARGS)
    assert r is not None
    assert math.isnan(r['mean_ic'])
    assert math.isnan(r['ic_tstat'])


def test_clear_all_removes_backtest():
    cache_mod.store_backtest(**_BT_KWARGS, result=_BT_RESULT)
    n = cache_mod.clear()
    assert n > 0
    assert cache_mod.load_backtest(**_BT_KWARGS) is None


def test_clear_variant_preserves_backtest():
    """clear('A') removes only cross-section snapshots, not backtest results."""
    cache_mod.store(_DATE, 'A', _DF)
    cache_mod.store_backtest(**_BT_KWARGS, result=_BT_RESULT)
    cache_mod.clear('A')
    assert cache_mod.load(_DATE, 'A') is None
    assert cache_mod.load_backtest(**_BT_KWARGS) is not None

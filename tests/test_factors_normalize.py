import numpy as np
import pandas as pd
import pytest

from irp.features.normalize import rank_norm, sector_neutral, zscore


@pytest.fixture
def xs() -> pd.DataFrame:
    """Small cross-section with two factor columns and some NaN."""
    return pd.DataFrame(
        {'pe': [10.0, 20.0, 30.0, 40.0, 50.0], 'roe': [0.1, 0.2, float('nan'), 0.4, 0.5]},
        index=pd.Index(['A', 'B', 'C', 'D', 'E'], name='Ticker'),
    )


def test_zscore_mean_zero(xs):
    out = zscore(xs)
    assert abs(out['pe'].mean()) < 1e-10


def test_zscore_std_one(xs):
    out = zscore(xs)
    assert abs(out['pe'].std() - 1.0) < 1e-6


def test_zscore_clips(xs):
    extreme = xs.copy()
    extreme.loc['A', 'pe'] = 1e9
    out = zscore(extreme)
    assert out['pe'].max() <= 3.0
    assert out['pe'].min() >= -3.0


def test_zscore_preserves_nan(xs):
    out = zscore(xs)
    assert pd.isna(out['roe']['C'])


def test_zscore_col_filter(xs):
    out = zscore(xs, cols=['pe'])
    # 'roe' unchanged (use .equals() so NaN==NaN)
    assert out['roe'].equals(xs['roe'])


def test_rank_norm_range(xs):
    out = rank_norm(xs)
    valid = out['pe'].dropna()
    assert valid.min() >= -0.5
    assert valid.max() <= 0.5


def test_rank_norm_preserves_nan(xs):
    out = rank_norm(xs)
    assert pd.isna(out['roe']['C'])


def test_rank_norm_monotone(xs):
    out = rank_norm(xs)
    # pe: 10 < 20 < 30 < 40 < 50 → ranks strictly increasing
    assert list(out['pe']) == sorted(out['pe'])


def test_sector_neutral_basic():
    df = pd.DataFrame(
        {'val': [1.0, 2.0, 3.0, 10.0, 20.0, 30.0]},
        index=pd.Index(['A', 'B', 'C', 'D', 'E', 'F'], name='Ticker'),
    )
    sector = pd.Series({'A': 'Tech', 'B': 'Tech', 'C': 'Tech', 'D': 'Finance', 'E': 'Finance', 'F': 'Finance'})
    out = sector_neutral(df, sector, cols=['val'])
    tech    = out.loc[['A', 'B', 'C'], 'val']
    finance = out.loc[['D', 'E', 'F'], 'val']
    assert abs(tech.mean()) < 1e-10
    assert abs(finance.mean()) < 1e-10


def test_sector_neutral_missing_sector_is_nan():
    df = pd.DataFrame({'val': [1.0, 2.0, 3.0]}, index=pd.Index(['A', 'B', 'C'], name='Ticker'))
    sector = pd.Series({'A': 'Tech', 'B': 'Tech'})  # C has no sector
    out = sector_neutral(df, sector, cols=['val'])
    assert pd.isna(out.loc['C', 'val'])


def test_sector_neutral_rank_method():
    df = pd.DataFrame(
        {'val': [1.0, 2.0, 3.0, 10.0, 20.0, 30.0]},
        index=pd.Index(['A', 'B', 'C', 'D', 'E', 'F'], name='Ticker'),
    )
    sector = pd.Series({'A': 'X', 'B': 'X', 'C': 'X', 'D': 'Y', 'E': 'Y', 'F': 'Y'})
    out = sector_neutral(df, sector, cols=['val'], method='rank')
    valid = out['val'].dropna()
    assert valid.min() >= -0.5
    assert valid.max() <= 0.5

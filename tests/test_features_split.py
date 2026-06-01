import datetime

import numpy as np
import pandas as pd

from irp.features import engineering as eng


def _grid_panel(n_dates=10, tickers=('A', 'B', 'C', 'D')):
    rows = []
    for i in range(n_dates):
        d = datetime.date(2010 + i, 12, 31)
        for t in tickers:
            rows.append({'Date': d, 'Ticker': t, 'f': float(i)})
    return pd.DataFrame(rows)


def test_assign_split_date_chronological_disjoint():
    df = _grid_panel(10)
    df = df.assign(split=eng.assign_split(df, 'date', (0.7, 0.15, 0.15)).to_numpy())
    tr = df[df.split == 'train']['Date']
    va = df[df.split == 'valid']['Date']
    te = df[df.split == 'test']['Date']
    assert tr.max() < va.min() <= va.max() < te.min()    # train < valid < test
    assert tr.nunique() == 7                              # 70% of 10 dates


def test_assign_split_date_all_rows_of_a_date_share_split():
    df = _grid_panel(8)
    df = df.assign(split=eng.assign_split(df, 'date', (0.5, 0.25, 0.25)).to_numpy())
    assert (df.groupby('Date')['split'].nunique() == 1).all()


def test_assign_split_ticker_leaveout_disjoint():
    df = _grid_panel(5, tickers=tuple(f'T{i}' for i in range(10)))
    df = df.assign(split=eng.assign_split(df, 'ticker', (0.6, 0.2, 0.2), seed=0).to_numpy())
    assert (df.groupby('Ticker')['split'].nunique() == 1).all()   # whole ticker per split
    sets = {k: set(v['Ticker'].unique()) for k, v in df.groupby('split')}
    assert not (sets.get('train', set()) & sets.get('test', set()))
    assert not (sets.get('train', set()) & sets.get('valid', set()))
    assert df.groupby('split')['Ticker'].nunique()['train'] == 6   # 60% of 10


def test_assign_split_ticker_deterministic():
    df = _grid_panel(3, tickers=tuple(f'T{i}' for i in range(12)))
    a = eng.assign_split(df, 'ticker', (0.5, 0.25, 0.25), seed=7)
    b = eng.assign_split(df, 'ticker', (0.5, 0.25, 0.25), seed=7)
    assert (a.to_numpy() == b.to_numpy()).all()


def test_assign_split_small_range_all_three_nonempty():
    # n=3 dates, default 0.7/0.15/0.15 → naive rounding gives valid=0; all 3 must be present
    df = _grid_panel(3)
    s = eng.assign_split(df, 'date', (0.7, 0.15, 0.15))
    assert set(s.unique()) == {'train', 'valid', 'test'}


def test_assign_split_skewed_ratios_keep_valid_nonempty():
    # valid frac rounds to 0 over 10 units, but train+valid<n → valid must still get 1
    df = _grid_panel(10)
    s = eng.assign_split(df, 'date', (0.85, 0.05, 0.10))
    assert (s == 'valid').any()


def test_assign_split_ratios_normalized():
    # ratios not summing to 1 are normalized
    df = _grid_panel(10)
    s = eng.assign_split(df, 'date', (7, 1.5, 1.5))
    assert set(s.unique()) <= {'train', 'valid', 'test'}
    assert (s == 'train').sum() == (df['Date'].isin(
        sorted(df['Date'].unique())[:7])).sum()

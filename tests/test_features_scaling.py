import datetime

import numpy as np
import pandas as pd

from irp.features import engineering as eng

_D2018 = datetime.date(2018, 12, 31)
_D2020 = datetime.date(2020, 12, 31)


def _panel():
    """2 dates × 3 tickers; feature 'f' + reserved cols (fwd_ret, label)."""
    rows = []
    for y, base in ((2018, 0.0), (2020, 10.0)):
        d = datetime.date(y, 12, 31)
        for i, tk in enumerate(['A', 'B', 'C']):
            rows.append({'Date': d, 'Ticker': tk, 'f': base + i,
                         'fwd_ret': 0.01 * i, 'label': i})
    return pd.DataFrame(rows)


def test_scale_minmax_per_date():
    out = eng.scale_features(_panel(), ['f'], method='minmax', scope='date')
    for _, g in out.groupby('Date'):
        assert g['f'].min() == 0.0 and g['f'].max() == 1.0


def _f32_panel():
    """float32 feature whose minmax scaling yields non-dyadic fractions (1/3, 2/3)
    that float32 cannot hold losslessly — triggers pandas LossySetitemError on
    partial-row .loc assignment if the column is not upcast first."""
    d = datetime.date(2020, 12, 31)
    df = pd.DataFrame({
        'Date': [d, d, d, d],
        'Ticker': ['A', 'B', 'C', 'D'],
        'f': np.array([0, 1, 2, 3], dtype='float32'),
        'fwd_ret': 0.0, 'label': 0,
    })
    return df


def test_scale_float32_column_date_scope():
    # Dense panels carry float32 columns; partial-row .loc assignment of float64
    # scaled values must not raise pandas LossySetitemError.
    out = eng.scale_features(_f32_panel(), ['f'], method='minmax', scope='date')
    assert out['f'].min() == 0.0 and out['f'].max() == 1.0


def test_scale_float32_column_ticker_scope():
    out = eng.scale_features(_f32_panel(), ['f'], method='minmax', scope='ticker',
                             train_cutoff=2020)
    assert out['f'].notna().all()


def test_scale_never_touches_reserved_cols():
    df = _panel()
    out = eng.scale_features(df, ['f'], 'minmax', 'date')
    assert (out['label'] == df['label']).all()
    assert (out['fwd_ret'] == df['fwd_ret']).all()
    assert (out['Ticker'] == df['Ticker']).all()


def test_scale_per_date_is_leakfree():
    df = _panel()
    out1 = eng.scale_features(df, ['f'], 'minmax', 'date')
    df2 = df.copy()
    df2.loc[df2['Date'] == _D2020, 'f'] *= 100        # perturb the other date
    out2 = eng.scale_features(df2, ['f'], 'minmax', 'date')
    a = out1[out1['Date'] == _D2018]['f'].to_numpy()
    b = out2[out2['Date'] == _D2018]['f'].to_numpy()
    assert np.allclose(a, b)                           # 2018 unaffected by 2020


def test_scale_global_cutoff_fits_on_train_only():
    # cutoff 2018 → train = 2018 rows (f in {0,1,2}); minmax center 0 scale 2
    out = eng.scale_features(_panel(), ['f'], 'minmax', 'global', train_cutoff=2018)
    train = sorted(out[out['Date'] == _D2018]['f'])
    test = sorted(out[out['Date'] == _D2020]['f'])
    assert np.allclose(train, [0.0, 0.5, 1.0])
    assert np.allclose(test, [5.0, 5.5, 6.0])          # uses train params → exceeds 1


def test_scale_global_leakfree_wrt_test():
    df = _panel()
    out1 = eng.scale_features(df, ['f'], 'minmax', 'global', train_cutoff=2018)
    df2 = df.copy()
    df2.loc[df2['Date'] == _D2020, 'f'] += 999         # perturb test rows
    out2 = eng.scale_features(df2, ['f'], 'minmax', 'global', train_cutoff=2018)
    a = out1[out1['Date'] == _D2018]['f'].to_numpy()
    b = out2[out2['Date'] == _D2018]['f'].to_numpy()
    assert np.allclose(a, b)                            # train scaling unaffected


def test_scale_per_ticker_independent():
    df = _panel()
    out1 = eng.scale_features(df, ['f'], 'minmax', 'ticker', train_cutoff=2020)
    df2 = df.copy()
    df2.loc[df2['Ticker'] == 'C', 'f'] *= 50           # perturb ticker C
    out2 = eng.scale_features(df2, ['f'], 'minmax', 'ticker', train_cutoff=2020)
    a = out1[out1['Ticker'] == 'A']['f'].to_numpy()
    b = out2[out2['Ticker'] == 'A']['f'].to_numpy()
    assert np.allclose(a, b)                            # ticker A unaffected by C


def test_scale_robust_constant_column_is_zero():
    df = _panel()
    df['f'] = 5.0                                       # zero IQR
    out = eng.scale_features(df, ['f'], method='robust', scope='date')
    assert (out['f'] == 0.0).all()                      # no inf / NaN


def test_scale_preserves_nan_input():
    df = _panel()
    df.loc[0, 'f'] = np.nan
    out = eng.scale_features(df, ['f'], 'minmax', 'date')
    assert pd.isna(out.loc[0, 'f'])


def test_scale_empty_train_falls_back_full_sample():
    # cutoff before all data → no train rows → fall back to each ticker's full sample
    out = eng.scale_features(_panel(), ['f'], 'minmax', 'ticker', train_cutoff=2010)
    for _, g in out.groupby('Ticker'):
        assert g['f'].min() == 0.0 and g['f'].max() == 1.0


def test_scale_global_fits_on_train_mask():
    # train_mask (e.g. from a split) overrides cutoff: fit on 2018 rows only
    df = _panel()
    mask = df['Date'] == _D2018
    out = eng.scale_features(df, ['f'], 'minmax', 'global', train_mask=mask)
    train = sorted(out[df['Date'] == _D2018]['f'])
    test = sorted(out[df['Date'] == _D2020]['f'])
    assert np.allclose(train, [0.0, 0.5, 1.0])     # train params from 2018
    assert np.allclose(test, [5.0, 5.5, 6.0])       # 2020 uses train params


def test_scale_robust_per_date_median_centered():
    out = eng.scale_features(_panel(), ['f'], method='robust', scope='date')
    # per date f=[base,base+1,base+2]: median=base+1, IQR=(base+1.5)-(base+0.5)=1
    for _, g in out.groupby('Date'):
        assert np.isclose(sorted(g['f'])[1], 0.0)       # middle value → 0


def test_signed_log_monotonic_and_handles_neg_zero():
    s = pd.Series([-100.0, -1.0, 0.0, 1.0, 100.0])
    out = eng.signed_log(s)
    assert out.iloc[2] == 0.0                      # log1p(0) == 0
    assert (out.iloc[0] < 0) and (out.iloc[4] > 0)  # sign preserved
    assert out.is_monotonic_increasing              # monotonic


def test_detect_heavy_tailed_flags_fat_ignores_bounded_and_constant():
    rng = np.random.default_rng(0)
    n = 2000
    df = pd.DataFrame({
        'Date': [datetime.date(2020, 1, 1)] * n,
        'Ticker': [f'T{i}' for i in range(n)],
        'fat': rng.standard_cauchy(n) * 1e6,        # heavy tails
        'bounded': rng.uniform(-1, 1, n),           # well-behaved
        'const': np.ones(n),                        # zero IQR
    })
    flagged = eng.detect_heavy_tailed(df, ['fat', 'bounded', 'const'])
    assert 'fat' in flagged
    assert 'bounded' not in flagged
    assert 'const' not in flagged


def _tame_panel():
    """2 dates × 3 tickers; 'f' has a big outlier on the later (test) date."""
    rows = []
    for y, vals in ((2018, [1.0, 2.0, 3.0]), (2020, [4.0, 5.0, 1000.0])):
        d = datetime.date(y, 12, 31)
        for tk, v in zip(['A', 'B', 'C'], vals):
            rows.append({'Date': d, 'Ticker': tk, 'f': v, 'fwd_ret': 0.0, 'label': 0})
    return pd.DataFrame(rows)


def test_tame_clip_fits_on_train_no_leak():
    df = _tame_panel()
    train_mask = pd.to_datetime(df['Date']).dt.year <= 2018   # 2018 rows only
    out = eng.tame_columns(df, ['f'], 'clip', p=0.0, train_mask=train_mask)
    # train (2018) max is 3.0 -> cap is 3.0; the 1000.0 test outlier clips to 3.0
    assert out['f'].max() == 3.0
    # perturbing the test outlier higher must not move the (train-fit) cap
    df2 = df.copy()
    df2.loc[df2['f'] == 1000.0, 'f'] = 1e9
    out2 = eng.tame_columns(df2, ['f'], 'clip', p=0.0, train_mask=train_mask)
    assert out2['f'].max() == 3.0


def test_tame_log_applies_signed_log():
    df = _tame_panel()
    out = eng.tame_columns(df, ['f'], 'log')
    assert np.isclose(out.loc[out['f'].idxmax(), 'f'], np.log1p(1000.0))


def test_tame_drop_removes_col_never_reserved():
    df = _tame_panel()
    out = eng.tame_columns(df, ['f', 'label'], 'drop')   # 'label' is reserved
    assert 'f' not in out.columns
    assert 'label' in out.columns                        # reserved never dropped


def test_tame_none_or_empty_is_noop():
    df = _tame_panel()
    assert eng.tame_columns(df, [], 'clip').equals(df)
    assert eng.tame_columns(df, ['f'], 'none').equals(df)


def test_residual_scale_flags_fires_on_large_silent_on_small():
    df = pd.DataFrame({
        'Date': [datetime.date(2020, 1, 1)] * 100,
        'Ticker': [f'T{i}' for i in range(100)],
        'huge': np.linspace(-500.0, 1000.0, 100),
        'small': np.linspace(-1.0, 1.0, 100),
    })
    flags = eng.residual_scale_flags(df, ['huge', 'small'])
    assert 'huge' in flags and flags['huge'] > 10
    assert 'small' not in flags
    assert eng.residual_scale_flags(df, ['Ticker']) == {}   # reserved never flagged


def _report_panel():
    """200 well-behaved tickers + one 'BAD' ticker owning extreme roe values."""
    rng = np.random.default_rng(7)
    d = datetime.date(2020, 12, 31)
    rows = [{'Date': d, 'Ticker': f'T{i}', 'roe': float(rng.normal()),
             'bounded': float(rng.uniform(-1, 1)), 'fwd_ret': 0.0, 'label': 0}
            for i in range(200)]
    for j, dd in enumerate([datetime.date(2019, 12, 31), datetime.date(2020, 12, 31)]):
        rows.append({'Date': dd, 'Ticker': 'BAD', 'roe': 1e6 * (j + 1),
                     'bounded': 0.0, 'fwd_ret': 0.0, 'label': 0})
    return pd.DataFrame(rows)


def test_heavy_tail_report_flags_col_and_points_to_bad_ticker():
    rep = eng.heavy_tail_report(_report_panel())
    assert 'roe' in set(rep.summary['col'])
    assert 'bounded' not in set(rep.summary['col'])          # well-behaved ignored
    roe_row = rep.summary[rep.summary['col'] == 'roe'].iloc[0]
    assert roe_row['worst_ticker'] == 'BAD'
    assert rep.offenders.sort_values('z', key=lambda s: s.abs(), ascending=False)\
        .iloc[0]['Ticker'] == 'BAD'                          # worst offender is BAD
    bt = rep.by_ticker[(rep.by_ticker['col'] == 'roe') & (rep.by_ticker['Ticker'] == 'BAD')]
    assert len(bt) == 1 and bt.iloc[0]['max_abs_z'] > 100


def test_apply_tame_plan_mixed_actions_clip_train_fit():
    df = pd.DataFrame({
        'Date': [datetime.date(2020, 12, 31)] * 5, 'Ticker': list('ABCDE'),
        'dropme': [1.0, 2, 3, 4, 5], 'logme': [0.0, 10, 100, 1000, 10000],
        'clipme': [1.0, 2, 3, 4, 1000.0], 'fwd_ret': 0.0, 'label': 0,
    })
    plan = [{'col': 'dropme', 'action': 'drop'},
            {'col': 'logme', 'action': 'log'},
            {'col': 'clipme', 'action': 'clip', 'p': 0.0}]
    mask = df['Ticker'] != 'E'                               # clip caps fit on A-D only
    out = eng.apply_tame_plan(df, plan, train_mask=mask)
    assert 'dropme' not in out.columns
    assert np.isclose(out['logme'].max(), np.log1p(10000.0))
    assert out['clipme'].max() == 4.0                        # E's 1000 clipped to train max 4
    assert 'label' in out.columns and 'Ticker' in out.columns

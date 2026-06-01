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

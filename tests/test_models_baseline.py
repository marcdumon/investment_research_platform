import datetime

import numpy as np
import pandas as pd

from irp.models import baseline as bl


def _synthetic(n_dates=20, n_tickers=60, signal=1.0, seed=0):
    """Panel where fwd_ret = signal * f1 + noise, so a linear model should learn it."""
    rng = np.random.default_rng(seed)
    rows = []
    for i in range(n_dates):
        d = datetime.date(2015, 1, 1) + datetime.timedelta(days=90 * i)
        f1 = rng.normal(size=n_tickers)
        f2 = rng.normal(size=n_tickers)
        fwd = signal * f1 + 0.1 * rng.normal(size=n_tickers)
        rows.append(pd.DataFrame({
            'Date': d, 'Ticker': [f'T{j:03d}' for j in range(n_tickers)],
            'f1_z': f1, 'f2_z': f2, 'fwd_ret': fwd,
        }))
    return pd.concat(rows, ignore_index=True), ['f1_z', 'f2_z']


def test_walk_forward_recovers_signal():
    df, feats = _synthetic(signal=1.0)
    res = bl.walk_forward_linear(df, feats, min_train_dates=5)
    assert res.mean_ic > 0.8          # strong linear signal recovered OOS
    assert res.r2_oos > 0.5
    # the predictive feature gets the larger coefficient
    assert abs(res.coefs['f1_z']) > abs(res.coefs['f2_z'])


def test_no_signal_gives_low_ic():
    df, feats = _synthetic(signal=0.0)
    res = bl.walk_forward_linear(df, feats, min_train_dates=5)
    assert abs(res.mean_ic) < 0.2     # nothing to learn


def test_predictions_are_out_of_sample():
    """No prediction date should appear in its own training window."""
    df, feats = _synthetic()
    res = bl.walk_forward_linear(df, feats, min_train_dates=5)
    pred_dates = sorted(res.predictions['Date'].unique())
    all_dates = sorted(df['Date'].unique())
    # first 5 dates are train-only, never predicted
    assert pred_dates[0] > all_dates[4]


def test_quintile_cumret_shape():
    df, feats = _synthetic()
    res = bl.walk_forward_linear(df, feats, min_train_dates=5, n_quantiles=5)
    assert list(res.quintile_cumret.columns) == ['Q1', 'Q2', 'Q3', 'Q4', 'Q5']
    # top quintile beats bottom when signal is positive
    assert res.quintile_cumret['Q5'].iloc[-1] > res.quintile_cumret['Q1'].iloc[-1]

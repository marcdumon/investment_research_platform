import datetime

import numpy as np
import pandas as pd

from irp.models import classifier as clf


def _synthetic(n_dates=24, n_tickers=90, signal=1.0, seed=0):
    """label = tercile of (signal*f1 + noise); fwd_ret tracks the same driver."""
    rng = np.random.default_rng(seed)
    rows = []
    for i in range(n_dates):
        d = datetime.date(2015, 1, 1) + datetime.timedelta(days=90 * i)
        f1 = rng.normal(size=n_tickers)
        f2 = rng.normal(size=n_tickers)
        driver = signal * f1 + 0.1 * rng.normal(size=n_tickers)
        label = pd.qcut(driver, 3, labels=False)
        rows.append(pd.DataFrame({
            'Date': d, 'Ticker': [f'T{j:03d}' for j in range(n_tickers)],
            'f1_z': f1, 'f2_z': f2, 'label': label, 'fwd_ret': driver,
        }))
    return pd.concat(rows, ignore_index=True), ['f1_z', 'f2_z']


def test_classifier_learns_separable_label():
    df, feats = _synthetic(signal=1.0)
    res = clf.walk_forward_classifier(df, feats, target='label', min_train_dates=5)
    assert res.accuracy > 0.6                 # well above 1/3 chance
    assert res.confusion.shape == (3, 3)
    assert res.mean_ic > 0.5                  # score ranks fwd_ret


def test_classifier_no_signal_near_chance():
    df, feats = _synthetic(signal=0.0)
    res = clf.walk_forward_classifier(df, feats, target='label', min_train_dates=5)
    assert res.accuracy < 0.45                 # ~1/3 for 3 balanced classes


def test_classifier_predictions_out_of_sample():
    df, feats = _synthetic()
    res = clf.walk_forward_classifier(df, feats, target='label', min_train_dates=5)
    first_pred = sorted(res.predictions['Date'].unique())[0]
    assert first_pred > sorted(df['Date'].unique())[4]


def test_missing_target_raises():
    df, feats = _synthetic()
    import pytest
    with pytest.raises(ValueError, match='no "label"'):
        clf.walk_forward_classifier(df.drop(columns=['label']), feats, target='label')

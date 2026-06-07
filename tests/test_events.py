import numpy as np
import pandas as pd

from irp.analysis import events as ev


def _idx(n):
    return pd.bdate_range('2015-01-01', periods=n)


def test_abnormal_returns_methods():
    idx = _idx(50)
    stock = pd.Series(np.linspace(0.0, 0.049, 50), index=idx)
    market = pd.Series(0.01, index=idx)
    mkt = ev.abnormal_returns(stock, market, method='market')
    assert np.allclose(mkt.to_numpy(), (stock - 0.01).to_numpy())
    mean = ev.abnormal_returns(stock, None, method='mean')
    assert np.isclose(mean.mean(), 0.0, atol=1e-12)
    raw = ev.abnormal_returns(stock, None, method='raw')
    assert np.allclose(raw.to_numpy(), stock.to_numpy())


def test_event_window_matrix_aligns_day_zero():
    idx = _idx(200)
    ar = pd.Series(0.0, index=idx)
    # plant a +5% bump on three event days
    event_pos = [50, 100, 150]
    for p in event_pos:
        ar.iloc[p] = 0.05
    event_dates = [idx[p] for p in event_pos]
    mat = ev.event_window_matrix(ar, event_dates, pre=5, post=5)
    assert mat.shape == (3, 11)                       # 3 events × (-5..+5)
    assert list(mat.columns) == list(range(-5, 6))
    assert np.allclose(mat[0].to_numpy(), 0.05)       # day-0 column is the bump
    assert np.allclose(mat[1].to_numpy(), 0.0)        # day +1 is flat


def test_event_window_drops_incomplete():
    idx = _idx(60)
    ar = pd.Series(0.0, index=idx)
    # event near the start can't fill a -10 window
    dates = [idx[2], idx[30]]
    mat = ev.event_window_matrix(ar, dates, pre=10, post=10)
    assert mat.shape[0] == 1                           # only the middle event survives


def test_mean_adjusted_matrix_uses_pre_event_window():
    idx = _idx(400)
    r = pd.Series(0.002, index=idx)            # baseline "normal" return = 0.002
    pos = 200
    r.iloc[pos] = 0.05                          # event-day bump
    dates = [idx[pos]]
    mat = ev.mean_adjusted_matrix(r, dates, pre=5, post=5, est_lo=60, est_hi=11)
    assert mat.shape == (1, 11)
    # abnormal = raw - pre-event mean(0.002); day 0 = 0.05-0.002, other days ~0
    assert np.isclose(mat[0].iloc[0], 0.05 - 0.002)   # event day: bump minus normal
    assert np.isclose(mat[1].iloc[0], 0.0)            # normal day: 0.002 - 0.002 = 0


def test_mean_adjusted_matrix_drops_without_estimation_window():
    idx = _idx(120)
    r = pd.Series(0.001, index=idx)
    # event too early: no room for a 60-day pre-event estimation window
    mat = ev.mean_adjusted_matrix(r, [idx[40]], pre=5, post=5, est_lo=60, est_hi=11)
    assert mat.shape[0] == 0


def test_aar_car():
    mat = pd.DataFrame(
        [[0.0, 0.02, 0.01], [0.0, 0.04, 0.03]], columns=[-1, 0, 1])
    res = ev.aar_car(mat)
    assert np.allclose(res['aar'].to_numpy(), [0.0, 0.03, 0.02])
    assert np.allclose(res['car'].to_numpy(), [0.0, 0.03, 0.05])
    assert res['n'] == 2
    assert np.isclose(res['car_end'], 0.05)


def test_monthly_seasonality_planted_january():
    idx = pd.bdate_range('2010-01-01', periods=2000)
    rng = np.random.default_rng(0)
    r = pd.Series(rng.normal(0, 0.001, len(idx)), index=idx)
    r[r.index.month == 1] += 0.01                      # strong January
    m = ev.monthly_seasonality(r)
    assert len(m) == 12
    assert m.idxmax() == 1


def test_dow_seasonality_planted_monday():
    idx = pd.bdate_range('2010-01-01', periods=2000)
    rng = np.random.default_rng(1)
    r = pd.Series(rng.normal(0, 0.001, len(idx)), index=idx)
    r[r.index.dayofweek == 0] += 0.01                  # strong Monday
    d = ev.dow_seasonality(r)
    assert d.idxmax() == 'Mon'

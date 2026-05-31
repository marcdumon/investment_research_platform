import datetime

import pandas as pd

from irp.ui.services import features_service as svc


def test_build_panel_cold_cache_returns_missing_dates(monkeypatch):
    """Fully cold cache must short-circuit with missing_dates, not compute inline."""
    monkeypatch.setattr(svc.universe_service, '_filter_tickers', lambda *a, **k: None)
    # every snapshot.load returns None -> all dates missing
    monkeypatch.setattr(svc._snapshot, 'load', lambda d, v: None)

    panel, missing = svc.build_panel(
        2020, 2021, 'A', 'A', steps=[], label_cfg={'mode': 'none'},
    )
    assert panel.empty
    assert len(missing) == 2  # 2020, 2021 year-ends
    assert all(isinstance(m, str) for m in missing)


def test_build_panel_partial_cache_skips_uncached(monkeypatch):
    """When some dates are cached, build from them and report the rest as skipped."""
    tickers = ['AAA', 'BBB']

    def _fake_load(d, v):
        # only the 2021 year-end is cached; 2020 is a hole
        if d.year == 2021:
            return pd.DataFrame({'roe': [0.1, 0.2]},
                                index=pd.Index(tickers, name='Ticker'))
        return None

    monkeypatch.setattr(svc.universe_service, '_filter_tickers', lambda *a, **k: None)
    monkeypatch.setattr(svc._snapshot, 'load', _fake_load)

    panel, missing = svc.build_panel(
        2020, 2021, 'A', 'A',
        steps=[{'op': 'base', 'col': 'roe'}], label_cfg={'mode': 'none'},
    )
    assert not panel.empty
    assert set(panel['Date'].map(lambda d: d.year)) == {2021}  # only cached date built
    assert missing == ['2020-12-31']  # the hole reported, non-blocking


def test_build_panel_warm_cache_assembles_and_labels(monkeypatch):
    tickers = ['AAA', 'BBB']

    def _fake_load(d, v):
        return pd.DataFrame(
            {'roe': [0.1, 0.2], 'revenue': [100.0, 50.0]},
            index=pd.Index(tickers, name='Ticker'),
        )

    def _fake_fwd(dates, horizon, tickers=None):
        rows = [(d, t, 0.05) for d in dates for t in tickers or ['AAA', 'BBB']]
        return pd.DataFrame(rows, columns=['Date', 'Ticker', 'fwd_ret'])

    monkeypatch.setattr(svc.universe_service, '_filter_tickers', lambda *a, **k: None)
    monkeypatch.setattr(svc._snapshot, 'load', _fake_load)
    monkeypatch.setattr(svc, '_forward_returns', _fake_fwd)

    panel, missing = svc.build_panel(
        2020, 2021, 'A', 'A',
        steps=[{'op': 'lag', 'col': 'roe', 'k': 1}],
        label_cfg={'mode': 'continuous', 'horizon_days': 252},
    )
    assert missing == []
    assert 'roe_lag1' in panel.columns
    assert 'fwd_ret' in panel.columns
    assert len(panel) == 4  # 2 tickers × 2 dates


def test_build_panel_keeps_only_selected_features(monkeypatch):
    """Output = Date/Ticker + step-produced columns + label only (not all 39 base)."""
    tickers = ['AAA', 'BBB']

    def _fake_load(d, v):
        return pd.DataFrame(
            {'fcf_margin': [0.1, 0.2], 'revenue': [100.0, 50.0], 'pe': [10.0, 20.0]},
            index=pd.Index(tickers, name='Ticker'),
        )

    monkeypatch.setattr(svc.universe_service, '_filter_tickers', lambda *a, **k: None)
    monkeypatch.setattr(svc._snapshot, 'load', _fake_load)

    base_steps = [{'op': 'diff', 'col': 'fcf_margin', 'k': 1}]
    panel_b, _ = svc.build_panel(2020, 2021, 'A', 'A', base_steps, {'mode': 'none'})
    assert set(panel_b.columns) == {'Date', 'Ticker', 'fcf_margin_diff1'}
    # source column not leaked into output
    assert 'fcf_margin' not in panel_b.columns
    assert 'revenue' not in panel_b.columns

    # adding `base fcf_margin` must change the output (extra column)
    a_steps = base_steps + [{'op': 'base', 'col': 'fcf_margin'}]
    panel_a, _ = svc.build_panel(2020, 2021, 'A', 'A', a_steps, {'mode': 'none'})
    assert set(panel_a.columns) == {'Date', 'Ticker', 'fcf_margin_diff1', 'fcf_margin'}
    assert set(panel_a.columns) != set(panel_b.columns)


def test_build_panel_adds_close_volume(monkeypatch):
    tickers = ['AAA', 'BBB']

    def _fake_load(d, v):
        return pd.DataFrame({'roe': [0.1, 0.2]}, index=pd.Index(tickers, name='Ticker'))

    def _fake_pv(dates, tickers=None):
        rows = [(d, t, 10.0, 1000.0) for d in dates for t in ['AAA', 'BBB']]
        return pd.DataFrame(rows, columns=['Date', 'Ticker', 'close', 'volume'])

    monkeypatch.setattr(svc.universe_service, '_filter_tickers', lambda *a, **k: None)
    monkeypatch.setattr(svc._snapshot, 'load', _fake_load)
    monkeypatch.setattr(svc, '_price_volume', _fake_pv)

    panel, _ = svc.build_panel(
        2020, 2021, 'A', 'A',
        steps=[{'op': 'base', 'col': 'close'}, {'op': 'pct_change', 'col': 'close', 'k': 1}],
        label_cfg={'mode': 'none'},
    )
    assert 'close' in panel.columns and 'close_pct1' in panel.columns


def test_available_columns_includes_price_volume():
    cols = svc.available_columns()
    assert 'roe' in cols and 'pe' in cols
    assert 'close' in cols and 'volume' in cols


def test_available_columns_dense_drops_price_factors():
    cols = svc.available_columns('D')
    assert 'roe' in cols and 'close' in cols and 'volume' in cols
    assert 'pe' not in cols and 'mktcap' not in cols  # price-dependent dropped


def test_dense_build_dispatch_and_carry_forward(monkeypatch):
    """Dense freq → dense path: dense close + carried (PIT) fundamentals."""
    import datetime as _dt

    def _fake_pv(grid, tickers=None):
        rows = [(d, 'AAA', 10.0 + i, 100.0) for i, d in enumerate(grid)]
        return pd.DataFrame(rows, columns=['Date', 'Ticker', 'close', 'volume'])

    def _fake_snap_long(sy, ey, variant, cols):
        # one filing at 2019-12-31 with roe=0.3
        return pd.DataFrame({'Date': [pd.Timestamp('2019-12-31')],
                             'Ticker': ['AAA'], 'roe': [0.3]})

    def _fake_ta(grid, tickers):
        return pd.DataFrame(columns=['Date', 'Ticker'] + svc._DENSE_TA_COLS)

    monkeypatch.setattr(svc.universe_service, '_filter_tickers', lambda *a, **k: None)
    monkeypatch.setattr(svc, '_price_volume', _fake_pv)
    monkeypatch.setattr(svc, '_load_snapshots_long', _fake_snap_long)
    monkeypatch.setattr(svc, '_dense_ta', _fake_ta)

    panel, notes = svc.build_panel(
        2020, 2020, 'M', 'A',
        steps=[{'op': 'base', 'col': 'close'}, {'op': 'base', 'col': 'roe'}],
        label_cfg={'mode': 'none'},
    )
    assert 'close' in panel.columns and 'roe' in panel.columns
    assert panel['Ticker'].nunique() == 1
    assert panel['close'].nunique() > 1          # dense sequence
    assert (panel['roe'] == 0.3).all()           # carried forward to every month


def test_panel_head_survives_dash_json(snapshots_dates_panel):
    """Date objects in the preview records must serialize for dcc.Store."""
    from dash._utils import to_json

    from irp.features import engineering as eng
    panel = eng.assemble_panel(snapshots_dates_panel)
    head = panel.head(50).to_dict('records')
    out = to_json({'data': {'head': head}})
    assert '2020-12-31' in out  # date rendered as ISO string


import datetime as _dt  # noqa: E402

import pytest  # noqa: E402


@pytest.fixture
def snapshots_dates_panel():
    tickers = ['AAA', 'BBB']
    return {
        _dt.date(2020, 12, 31): pd.DataFrame(
            {'roe': [0.1, 0.2]}, index=pd.Index(tickers, name='Ticker')
        )
    }

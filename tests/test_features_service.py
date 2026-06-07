import datetime

import numpy as np
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


def test_build_skips_steps_with_missing_inputs(monkeypatch):
    """A step on a column absent from the panel is skipped + warned, not crashed."""
    tickers = ['AAA', 'BBB']

    def _fake_load(d, v):
        return pd.DataFrame({'roe': [0.1, 0.2]}, index=pd.Index(tickers, name='Ticker'))

    monkeypatch.setattr(svc.universe_service, '_filter_tickers', lambda *a, **k: None)
    monkeypatch.setattr(svc._snapshot, 'load', _fake_load)
    monkeypatch.setattr(svc, '_forward_returns',
                        lambda *a, **k: pd.DataFrame(columns=['Date', 'Ticker', 'fwd_ret']))

    panel, notes = svc.build_panel(
        2020, 2021, 'A', 'A',
        steps=[{'op': 'base', 'col': 'roe'}, {'op': 'base', 'col': 'nonexistent'}],
        label_cfg={'mode': 'none'},
    )
    assert 'roe' in panel.columns
    assert 'nonexistent' not in panel.columns
    assert any('nonexistent' in n for n in notes)  # warned, did not raise


def test_build_panel_scales_features_per_date(monkeypatch):
    tickers = ['AAA', 'BBB']

    def _fake_load(d, v):
        val = [0.1, 0.2] if d.year == 2020 else [0.3, 0.5]
        return pd.DataFrame({'roe': val}, index=pd.Index(tickers, name='Ticker'))

    monkeypatch.setattr(svc.universe_service, '_filter_tickers', lambda *a, **k: None)
    monkeypatch.setattr(svc._snapshot, 'load', _fake_load)
    monkeypatch.setattr(svc, '_price_volume',
                        lambda dates, tickers=None: pd.DataFrame(
                            columns=['Date', 'Ticker', 'close', 'volume']))

    panel, notes = svc.build_panel(
        2020, 2021, 'A', 'A',
        steps=[{'op': 'base', 'col': 'roe'}], label_cfg={'mode': 'none'},
        scale_cfg={'method': 'minmax', 'scope': 'date'},
    )
    for _, g in panel.groupby('Date'):
        assert g['roe'].min() == 0.0 and g['roe'].max() == 1.0


def test_build_panel_scale_global_no_cutoff_warns(monkeypatch):
    tickers = ['AAA', 'BBB']

    def _fake_load(d, v):
        return pd.DataFrame({'roe': [0.1, 0.2]}, index=pd.Index(tickers, name='Ticker'))

    monkeypatch.setattr(svc.universe_service, '_filter_tickers', lambda *a, **k: None)
    monkeypatch.setattr(svc._snapshot, 'load', _fake_load)
    monkeypatch.setattr(svc, '_price_volume',
                        lambda dates, tickers=None: pd.DataFrame(
                            columns=['Date', 'Ticker', 'close', 'volume']))

    _, notes = svc.build_panel(
        2020, 2021, 'A', 'A',
        steps=[{'op': 'base', 'col': 'roe'}], label_cfg={'mode': 'none'},
        scale_cfg={'method': 'minmax', 'scope': 'global'},
    )
    assert any('look-ahead' in n for n in notes)


def test_build_panel_scale_does_not_touch_label(monkeypatch):
    tickers = ['AAA', 'BBB']

    def _fake_load(d, v):
        return pd.DataFrame({'roe': [0.1, 0.9]}, index=pd.Index(tickers, name='Ticker'))

    def _fake_fwd(dates, horizon, tickers=None):
        return pd.DataFrame([(d, t, 0.05) for d in dates for t in ['AAA', 'BBB']],
                            columns=['Date', 'Ticker', 'fwd_ret'])

    monkeypatch.setattr(svc.universe_service, '_filter_tickers', lambda *a, **k: None)
    monkeypatch.setattr(svc._snapshot, 'load', _fake_load)
    monkeypatch.setattr(svc, '_forward_returns', _fake_fwd)
    monkeypatch.setattr(svc, '_price_volume',
                        lambda dates, tickers=None: pd.DataFrame(
                            columns=['Date', 'Ticker', 'close', 'volume']))

    panel, _ = svc.build_panel(
        2020, 2021, 'A', 'A',
        steps=[{'op': 'base', 'col': 'roe'}],
        label_cfg={'mode': 'continuous', 'horizon_days': 252},
        scale_cfg={'method': 'minmax', 'scope': 'date'},
    )
    assert (panel['fwd_ret'] == 0.05).all()   # label untouched by scaling


def test_build_panel_date_split_adds_column(monkeypatch):
    tickers = ['AAA', 'BBB']
    monkeypatch.setattr(svc.universe_service, '_filter_tickers', lambda *a, **k: None)
    monkeypatch.setattr(svc._snapshot, 'load', lambda d, v: pd.DataFrame(
        {'roe': [0.1, 0.2]}, index=pd.Index(tickers, name='Ticker')))
    monkeypatch.setattr(svc, '_price_volume', lambda dates, tickers=None: pd.DataFrame(
        columns=['Date', 'Ticker', 'close', 'volume']))

    panel, _ = svc.build_panel(
        2015, 2020, 'A', 'A',
        steps=[{'op': 'base', 'col': 'roe'}], label_cfg={'mode': 'none'},
        split_cfg={'method': 'date', 'train': 0.5, 'valid': 0.25, 'test': 0.25},
    )
    assert 'split' in panel.columns
    assert set(panel['split'].unique()) <= {'train', 'valid', 'test'}
    assert panel[panel.split == 'train']['Date'].max() < panel[panel.split == 'test']['Date'].min()


def test_build_panel_scaler_fits_on_train_split(monkeypatch):
    tickers = ['AAA', 'BBB']

    def _load(d, v):
        base = (d.year - 2015) / 10.0           # grows over time
        return pd.DataFrame({'roe': [base, base + 0.01]},
                            index=pd.Index(tickers, name='Ticker'))

    monkeypatch.setattr(svc.universe_service, '_filter_tickers', lambda *a, **k: None)
    monkeypatch.setattr(svc._snapshot, 'load', _load)
    monkeypatch.setattr(svc, '_price_volume', lambda dates, tickers=None: pd.DataFrame(
        columns=['Date', 'Ticker', 'close', 'volume']))

    panel, _ = svc.build_panel(
        2015, 2020, 'A', 'A',
        steps=[{'op': 'base', 'col': 'roe'}], label_cfg={'mode': 'none'},
        scale_cfg={'method': 'minmax', 'scope': 'global'},
        split_cfg={'method': 'date', 'train': 0.5, 'valid': 0.25, 'test': 0.25},
    )
    # scaler fit on train → later (test) roe exceeds the train [0,1] range
    assert panel[panel.split == 'test']['roe'].max() > 1.0
    assert panel[panel.split == 'train']['roe'].max() <= 1.0 + 1e-9


def test_export_panel_writes_three_split_files(monkeypatch, tmp_path):
    monkeypatch.setattr(svc, '_EXPORT_DIR', tmp_path)
    df = pd.DataFrame({
        'Date': pd.to_datetime(['2015-12-31', '2018-12-31', '2020-12-31']).date if False
        else ['2015-12-31', '2018-12-31', '2020-12-31'],
        'Ticker': ['A', 'A', 'A'], 'roe': [0.1, 0.2, 0.3],
        'split': ['train', 'valid', 'test'],
    })
    out = svc.export_panel(df, 'parquet', name='ds')
    assert isinstance(out, list) and len(out) == 3
    for p in out:
        d = pd.read_parquet(p)
        assert 'split' not in d.columns         # split column dropped from each file
    names = sorted(p.stem.split('_')[-1] for p in out)
    assert names == ['test', 'train', 'valid']


def test_available_columns_dense_drops_price_factors():
    cols = svc.available_columns('D')
    assert 'roe' in cols and 'close' in cols and 'volume' in cols
    assert 'pe' not in cols and 'mktcap' not in cols  # price-dependent dropped


def test_dense_build_dispatch_and_carry_forward(monkeypatch):
    """Dense freq → dense path: dense close + carried (PIT) fundamentals."""
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


def _panel_with_fat_col():
    rng = np.random.default_rng(2)
    rows = []
    for y in (2018, 2020):
        d = datetime.date(y, 12, 31)
        for i in range(200):
            rows.append({'Date': d, 'Ticker': f'T{i}',
                         'roe': rng.standard_cauchy() * 1e6,
                         'fwd_ret': 0.0, 'label': 0})
    return pd.DataFrame(rows)


def test_split_and_scale_tames_then_scales_with_notes():
    panel = _panel_with_fat_col()
    scale_cfg = {'method': 'minmax', 'scope': 'date',
                 'tame_action': 'clip', 'tame_cols': ['roe'], 'tame_p': 0.01}
    out, notes = svc._split_and_scale(panel, scale_cfg, None)
    # detection named the fat column up front
    assert any('roe' in n for n in notes)
    # clip + minmax-per-date bounds the column to [0, 1] — no residual warning
    assert not any('still large' in n for n in notes)
    assert out['roe'].abs().quantile(0.99) < 10


def test_split_and_scale_clip_fits_train_split_not_full_date_scope():
    """Default date scope + by-date split + Clip: clip caps MUST come from the train
    rows, never the full sample (the user's train-only rule). The latest date lands in
    the test split with large varying roe; a train-fit clip cap collapses it to a
    constant -> minmax-per-date yields all zeros. A leaky full-sample clip leaves it
    varying."""
    rows = []
    for yr in range(2010, 2018):          # train-ish dates, small roe
        d = datetime.date(yr, 12, 31)
        for i in range(10):
            rows.append({'Date': d, 'Ticker': f'T{i}', 'roe': float(i),
                         'fwd_ret': 0.0, 'label': 0})
    dt = datetime.date(2019, 12, 31)      # most-recent date -> test split; huge, varying
    for i in range(10):
        rows.append({'Date': dt, 'Ticker': f'T{i}', 'roe': 1000.0 * (i + 1),
                     'fwd_ret': 0.0, 'label': 0})
    panel = pd.DataFrame(rows)
    cfg = {'method': 'minmax', 'scope': 'date',
           'tame_action': 'clip', 'tame_cols': ['roe'], 'tame_p': 0.0}
    split = {'method': 'date', 'train': 0.7, 'valid': 0.15, 'test': 0.15}
    out, _ = svc._split_and_scale(panel, cfg, split)
    huge = out[out['Date'] == datetime.date(2019, 12, 31)]
    assert len(huge) == 10 and (huge['split'] == 'test').all()   # sanity: it is the test split
    assert np.allclose(huge['roe'].to_numpy(), 0.0)        # train-fit clip -> constant -> 0


def test_prepare_features_excludes_tames_scales_splits():
    rows = []
    for yr in range(2012, 2020):
        d = datetime.date(yr, 12, 31)
        for i in range(20):
            rows.append({'Date': d, 'Ticker': f'T{i}', 'roe': float(i),
                         'fwd_ret': 0.0, 'label': 0})
        rows.append({'Date': d, 'Ticker': 'BAD', 'roe': 1e9,   # one junk ticker
                     'fwd_ret': 0.0, 'label': 0})
    df = pd.DataFrame(rows)
    out, notes = svc.prepare_features(
        df,
        tame_plan=[{'col': 'roe', 'action': 'clip', 'p': 0.01}],
        exclude_tickers=['BAD'],
        scale_cfg={'method': 'minmax', 'scope': 'date'},
        split_cfg={'method': 'date', 'train': 0.7, 'valid': 0.15, 'test': 0.15},
    )
    assert 'BAD' not in set(out['Ticker'])                 # excluded
    assert 'split' in out.columns                          # split applied
    assert out['roe'].abs().max() <= 1.0 + 1e-9            # clipped + minmax-bounded
    assert any('excluded' in n for n in notes)


def test_column_preview_clip_log_drop_exclude():
    rows = []
    for yr in range(2015, 2020):
        d = datetime.date(yr, 12, 31)
        for i in range(20):
            rows.append({'Date': d, 'Ticker': f'T{i}', 'roe': float(i),
                         'fwd_ret': 0.0, 'label': 0})
        rows.append({'Date': d, 'Ticker': 'BAD', 'roe': 1e9, 'fwd_ret': 0.0, 'label': 0})
    df = pd.DataFrame(rows)

    s, rep = svc.column_preview(df, 'roe')                 # no action
    assert rep is not None and s.max() == 1e9
    s2, _ = svc.column_preview(df, 'roe', {'col': 'roe', 'action': 'clip', 'p': 0.2})
    assert s2.max() < 1e9                                  # clipped
    s3, _ = svc.column_preview(df, 'roe', {'col': 'roe', 'action': 'log'})
    assert np.isclose(s3.max(), np.log1p(1e9))             # signed-log
    s4, rep4 = svc.column_preview(df, 'roe', {'col': 'roe', 'action': 'drop'})
    assert s4 is None and rep4 is None                     # dropped → nothing to show
    s5, _ = svc.column_preview(df, 'roe', exclude_tickers=['BAD'])
    assert s5.max() == 19.0                                # BAD rows gone


def test_column_series_log_exclude_drop():
    df = pd.DataFrame({
        'Date': [datetime.date(2020, 12, 31)] * 4, 'Ticker': list('ABCD'),
        'roe': [1.0, 2.0, 3.0, 1e6], 'fwd_ret': 0.0, 'label': 0,
    })
    assert svc.column_series(df, 'roe').max() == 1e6                  # raw (clip/none)
    assert np.isclose(svc.column_series(df, 'roe', {'action': 'log'}).max(),
                      np.log1p(1e6))                                  # log applied
    assert svc.column_series(df, 'roe', exclude_tickers=['D']).max() == 3.0  # D excluded
    assert svc.column_series(df, 'roe', {'action': 'drop'}) is None   # dropped

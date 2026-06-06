import pandas as pd

from irp.ui.services import screener_service as svc


def _df():
    return pd.DataFrame({
        'Ticker': ['A', 'B', 'C', 'D'],
        'pe': [5.0, 15.0, 25.0, 35.0],
    })


def test_apply_step_range_both_bounds():
    out = svc.apply_step(_df(), {'type': 'range', 'col': 'pe', 'min': 10, 'max': 30})
    assert list(out['Ticker']) == ['B', 'C']


def test_apply_step_range_open_upper():
    out = svc.apply_step(_df(), {'type': 'range', 'col': 'pe', 'min': 20, 'max': None})
    assert list(out['Ticker']) == ['C', 'D']


def test_apply_step_range_unknown_col_is_noop():
    out = svc.apply_step(_df(), {'type': 'range', 'col': 'missing', 'min': 0, 'max': 1})
    assert list(out['Ticker']) == ['A', 'B', 'C', 'D']


def test_apply_step_keep_and_remove():
    keep = svc.apply_step(_df(), {'type': 'keep', 'tickers': ['A', 'C']})
    assert list(keep['Ticker']) == ['A', 'C']
    rem = svc.apply_step(_df(), {'type': 'remove', 'tickers': ['A', 'C']})
    assert list(rem['Ticker']) == ['B', 'D']


def test_apply_steps_compose_in_order():
    steps = [
        {'type': 'range', 'col': 'pe', 'min': 10, 'max': None},   # B, C, D
        {'type': 'remove', 'tickers': ['D']},                     # B, C
    ]
    assert list(svc.apply_steps(_df(), steps)['Ticker']) == ['B', 'C']


def test_auto_name_from_range_labels_and_date():
    steps = [{'type': 'range', 'label': 'pe ≥ 10'}]
    assert svc.auto_name(steps, '2023-01-15T00:00:00') == 'pege10_2023-01-15'


def test_auto_name_date_only_when_no_range_steps():
    assert svc.auto_name([], '2023-01-15') == '2023-01-15'


def test_auto_name_screener_fallback_when_fully_empty():
    assert svc.auto_name([], None) == 'screener_'


def test_build_summary_prefixes_by_type():
    steps = [
        {'type': 'range', 'label': 'pe ≥ 10'},
        {'type': 'keep', 'label': 'tech'},
        {'type': 'remove', 'label': 'banks'},
        {'type': 'range', 'label': ''},          # no label → skipped
    ]
    assert svc.build_summary(steps) == 'pe ≥ 10; +tech; -banks'

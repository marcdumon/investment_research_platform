import numpy as np
import pandas as pd

from irp.features.composite import PRESETS
from irp.ui.services import today_service as ts


def _xsection():
    idx = ['AAA', 'BBB', 'CCC', 'DDD']
    return pd.DataFrame({
        'roe': [0.40, 0.10, 0.05, 0.20],
        'roic': [0.35, 0.08, 0.04, 0.18],
        'gross_margin': [0.60, 0.30, 0.20, 0.45],
        'fcf_margin': [0.25, 0.05, 0.02, 0.15],
        'Sector': ['Tech', 'Tech', 'Energy', 'Energy'],
        'Company Name': ['A Co', 'B Co', 'C Co', 'D Co'],
    }, index=idx)


def test_rank_by_composite_orders_best_first():
    df = _xsection()
    out = ts._rank_by_composite(df, PRESETS['quality'], n=3)
    assert len(out) == 3
    assert out.index[0] == 'AAA'                 # strongest quality name on top
    assert list(out['Rank']) == [1, 2, 3]
    assert 'score' in out.columns


def test_rank_by_composite_ignores_missing_factor_cols():
    df = _xsection()                             # has no momentum columns
    out = ts._rank_by_composite(df, PRESETS['momentum'], n=4)
    assert out.empty                             # no usable factor -> empty, no raise


def test_playbook_maps_every_label_to_valid_preset():
    for label in ('risk_on', 'neutral', 'risk_off', 'unknown'):
        preset = ts.playbook_preset(label)
        assert preset in PRESETS


def test_playbook_preset_risk_off_is_defensive():
    assert ts.playbook_preset('risk_off') == 'quality'
    assert ts.playbook_preset('risk_on') == 'momentum'


def test_universe_floor_excludes_small_caps():
    df = _xsection()
    df['mktcap'] = [5.0, 0.2, 50.0, 0.5]          # $B
    kept = ts._apply_universe_floor(df, 1.0)
    assert list(kept.index) == ['AAA', 'CCC']     # BBB/DDD below $1B dropped
    # and the floor flows through ranking: a sub-threshold name can't be a top pick
    top = ts._rank_by_composite(kept, PRESETS['quality'], n=10)
    assert 'BBB' not in top.index and 'DDD' not in top.index


def test_universe_floor_noop_without_mktcap():
    df = _xsection()                              # no mktcap column
    assert len(ts._apply_universe_floor(df, 1.0)) == len(df)


def test_rank_by_composite_drops_nan_scores():
    df = _xsection()
    df.loc['CCC', ['roe', 'roic', 'gross_margin', 'fcf_margin']] = np.nan
    out = ts._rank_by_composite(df, PRESETS['quality'], n=10)
    assert 'CCC' not in out.index

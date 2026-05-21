import pandas as pd
import pytest

from irp.features.composite import PRESETS, build_composite


@pytest.fixture
def xs() -> pd.DataFrame:
    """Minimal cross-section with all preset factor columns."""
    n = 30
    import numpy as np
    rng = np.random.default_rng(42)
    data = {
        'pe': rng.uniform(5, 50, n),
        'pb': rng.uniform(0.5, 5, n),
        'ps': rng.uniform(0.5, 10, n),
        'fcf_yield': rng.uniform(-0.05, 0.1, n),
        'roe': rng.uniform(-0.2, 0.5, n),
        'roic': rng.uniform(-0.1, 0.4, n),
        'gross_margin': rng.uniform(0.1, 0.8, n),
        'fcf_margin': rng.uniform(-0.1, 0.3, n),
        'mom_12_1': rng.uniform(-0.5, 0.5, n),
        'mom_6_1': rng.uniform(-0.3, 0.3, n),
    }
    tickers = [f'T{i:03d}' for i in range(n)]
    return pd.DataFrame(data, index=pd.Index(tickers, name='Ticker'))


def test_build_composite_returns_series(xs):
    result = build_composite(xs, PRESETS['value'])
    assert isinstance(result, pd.Series)
    assert result.name == 'composite'


def test_build_composite_index_matches(xs):
    result = build_composite(xs, PRESETS['composite'])
    assert list(result.index) == list(xs.index)


def test_build_composite_missing_col_raises(xs):
    with pytest.raises(ValueError, match='Missing factor columns'):
        build_composite(xs, {'nonexistent_col': 1.0})


def test_build_composite_zscore_normalized(xs):
    result = build_composite(xs, PRESETS['value'], normalize='zscore')
    # After re-normalization the result should have mean ~0 and std ~1
    valid = result.dropna()
    assert abs(valid.mean()) < 0.1
    assert abs(valid.std() - 1.0) < 0.1


def test_build_composite_rank_normalized(xs):
    result = build_composite(xs, PRESETS['value'], normalize='rank')
    valid = result.dropna()
    assert valid.min() >= -0.5
    assert valid.max() <= 0.5


def test_all_presets_run(xs):
    for name, weights in PRESETS.items():
        result = build_composite(xs, weights)
        assert len(result) == len(xs), f'Preset {name} returned wrong length'


def test_sector_neutral(xs):
    import pandas as pd
    sector = pd.Series({t: ('Tech' if i < 15 else 'Finance') for i, t in enumerate(xs.index)})
    result = build_composite(xs, PRESETS['composite'], sector=sector)
    assert isinstance(result, pd.Series)
    assert len(result) == len(xs)


def test_normalize_none(xs):
    result = build_composite(xs, PRESETS['value'], normalize='none')
    assert isinstance(result, pd.Series)
    assert len(result.dropna()) > 0

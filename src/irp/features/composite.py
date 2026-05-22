"""Multi-factor composite signal construction.

No DB access. Call build_composite() with a cross-section DataFrame and
a weights dict to get a normalized composite score.
"""
import pandas as pd

from irp.features.normalize import rank_norm, zscore

PRESETS: dict[str, dict[str, float]] = {
    'value':     {'pe': -1, 'pb': -1, 'ps': -1, 'fcf_yield': 1},
    'quality':   {'roe': 1, 'roic': 1, 'gross_margin': 1, 'fcf_margin': 1},
    'momentum':  {'mom_12_1': 1, 'mom_6_1': 0.5},
    'composite': {'pe': -1, 'pb': -0.5, 'roe': 1, 'roic': 1, 'mom_12_1': 1},
}


def _norm_series(s: pd.Series, method: str) -> pd.Series:
    tmp = s.to_frame('__s__')
    if method == 'zscore':
        return zscore(tmp, ['__s__'])['__s__']
    if method == 'rank':
        return rank_norm(tmp, ['__s__'])['__s__']
    return s


def build_composite(
    df: pd.DataFrame,
    weights: dict[str, float],
    normalize: str = 'zscore',
    sector: pd.Series | None = None,
) -> pd.Series:
    """Weighted sum of normalized factors.

    Parameters
    ----------
    df        : Cross-section DataFrame indexed by Ticker.
    weights   : Factor column → weight. Negative weight = short that signal.
    normalize : 'zscore' | 'rank' | 'none'. Applied per factor before weighting.
    sector    : Optional pd.Series (indexed by Ticker) for sector-neutral normalization.

    Returns
    -------
    pd.Series indexed by Ticker. Re-normalized composite score (z-scored or ranked).

    Raises
    ------
    ValueError if any weight key is absent from df.columns.
    """
    cols = list(weights.keys())
    missing = [c for c in cols if c not in df.columns]
    if missing:
        raise ValueError(f'Missing factor columns: {missing}')

    if normalize == 'none' or sector is None:
        fn = zscore if normalize == 'zscore' else (rank_norm if normalize == 'rank' else None)
        normed = fn(df[cols], cols) if fn is not None else df[cols].copy()
    else:
        from irp.features.normalize import sector_neutral
        normed = sector_neutral(df[cols], sector, cols=cols, method=normalize)

    score = pd.DataFrame({c: normed[c] * w for c, w in weights.items()}).sum(axis=1)
    return _norm_series(score, normalize if normalize != 'none' else 'zscore').rename('composite')

"""Cross-sectional factor normalization utilities.

Pure functions; no DB access. All operate per-snapshot (one date at a time).
"""
import pandas as pd

from irp.factors.registry import _all_factors as all_factors

FACTOR_COLS: list[str] = [f.name for f in all_factors()]


def _cols(df: pd.DataFrame, cols: list[str] | None) -> list[str]:
    return cols if cols is not None else [c for c in FACTOR_COLS if c in df.columns]


def zscore(df: pd.DataFrame, cols: list[str] | None = None) -> pd.DataFrame:
    """Cross-sectional z-score per column. Clips at ±3 to suppress outliers."""
    out = df.copy()
    for c in _cols(df, cols):
        s = out[c]
        std = s.std() if s.count() > 1 else 0.0
        if std > 0:
            out[c] = ((s - s.mean()) / std).clip(-3, 3)
    return out


def rank_norm(df: pd.DataFrame, cols: list[str] | None = None) -> pd.DataFrame:
    """Rank each column within the cross-section, scaled to [-0.5, +0.5].

    Uses average rank for ties; NaN propagated.
    """
    out = df.copy()
    for c in _cols(df, cols):
        s = out[c]
        n = s.count()
        if n > 1:
            out[c] = s.rank(method='average', na_option='keep') / (n + 1) - 0.5
    return out


def sector_neutral(
    df: pd.DataFrame,
    sector: pd.Series,
    cols: list[str] | None = None,
    method: str = 'zscore',
) -> pd.DataFrame:
    """Normalize each column within its sector group (vectorized).

    Parameters
    ----------
    df     : Cross-section DataFrame indexed by Ticker.
    sector : pd.Series indexed by Ticker with sector labels.
    cols   : Columns to normalize; defaults to standard factor cols.
    method : 'zscore' or 'rank'.

    Tickers absent from sector get NaN for all cols. Singleton sectors get
    NaN (zero variance) for z-score, and centered rank for rank-norm.
    """
    resolved = _cols(df, cols)
    out = df.copy()
    aligned = sector.reindex(df.index)
    groups = aligned.dropna()

    if method == 'zscore':
        for c in resolved:
            mean = df[c].groupby(groups).transform('mean')
            std  = df[c].groupby(groups).transform('std')
            out[c] = ((df[c] - mean) / std).clip(-3, 3)
    else:
        for c in resolved:
            n = df[c].groupby(groups).transform('count')
            ranks = df[c].groupby(groups).rank(method='average', na_option='keep')
            out[c] = ranks / (n + 1) - 0.5

    out.loc[aligned.isna(), resolved] = float('nan')
    return out

"""Cross-sectional factor normalization utilities.

Pure functions; no DB access. All operate per-snapshot (one date at a time).
"""
import pandas as pd

FACTOR_COLS: list[str] = [
    'pe', 'pb', 'ps', 'ev_ebitda', 'ev_ebit', 'ev_sales', 'fcf_yield',
    'gross_margin', 'op_margin', 'net_margin', 'roe', 'roa', 'roic', 'fcf_margin',
    'mom_12_1', 'mom_6_1', 'vol_21d', 'ma200_ratio',
]


def _cols(df: pd.DataFrame, cols: list[str] | None) -> list[str]:
    return cols if cols is not None else [c for c in FACTOR_COLS if c in df.columns]


def zscore(df: pd.DataFrame, cols: list[str] | None = None) -> pd.DataFrame:
    """Cross-sectional z-score per column. Clips at ±3 to suppress outliers."""
    out = df.copy()
    for c in _cols(df, cols):
        s = out[c]
        std = s.std()
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
    """Normalize each column within its sector group.

    Parameters
    ----------
    df     : Cross-section DataFrame indexed by Ticker.
    sector : pd.Series indexed by Ticker with sector labels.
    cols   : Columns to normalize; defaults to standard factor cols.
    method : 'zscore' or 'rank'.

    Tickers absent from sector or in a singleton sector get NaN for all cols.
    """
    resolved = _cols(df, cols)
    out = df.copy()
    aligned = sector.reindex(df.index)
    fn = zscore if method == 'zscore' else rank_norm

    for sect in aligned.dropna().unique():
        mask = aligned == sect
        normed = fn(out.loc[mask], resolved)
        out.loc[mask, resolved] = normed[resolved]

    out.loc[aligned.isna(), resolved] = float('nan')
    return out

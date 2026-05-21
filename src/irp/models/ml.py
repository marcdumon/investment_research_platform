"""ML-based factor model: walk-forward backtest using sklearn-compatible estimators.

No DB access. Works directly from cached cross-section snapshots and
forward return DataFrames produced by backtest.compute_forward_returns().

Suggested models:
    Ridge(alpha=1.0)                              -- regularised linear, fast, interpretable
    LGBMRegressor(n_estimators=100, lr=0.05)      -- tree-based, captures non-linear interactions

Both require sklearn (Ridge) or lightgbm (LGBMRegressor) — install separately if needed.
"""
import datetime
import logging

import numpy as np
import pandas as pd

from irp.factors._cols import PRICE_TICKER
from irp.features.normalize import FACTOR_COLS, rank_norm, zscore

logger = logging.getLogger(__name__)


def build_dataset(
    cross_sections: dict[datetime.date, pd.DataFrame],
    fwd_returns: pd.DataFrame,
    normalize: str = 'rank',
    feature_cols: list[str] | None = None,
) -> pd.DataFrame:
    """Long-format (date, Ticker) panel with factor features + fwd_ret label.

    Parameters
    ----------
    cross_sections : Dict mapping date → cross-section DataFrame indexed by Ticker.
    fwd_returns    : DataFrame with columns [Ticker, Date, fwd_ret] from compute_forward_returns().
    normalize      : 'rank' | 'zscore' | 'none'. Applied cross-sectionally per date.
    feature_cols   : Factor columns to include; defaults to all standard factor cols present.

    Returns
    -------
    DataFrame with MultiIndex (date, Ticker). Columns: feature_cols + 'fwd_ret'.
    Rows with NaN fwd_ret dropped.
    """
    fn = rank_norm if normalize == 'rank' else (zscore if normalize == 'zscore' else None)
    rows = []
    for d, xs in sorted(cross_sections.items()):
        cols = feature_cols or [c for c in FACTOR_COLS if c in xs.columns]
        sub = fn(xs[cols]) if fn is not None else xs[cols].copy()
        sub = sub.copy()
        sub['date'] = d
        rows.append(sub.reset_index())

    if not rows:
        return pd.DataFrame()

    panel = pd.concat(rows, ignore_index=True)
    fwd_long = fwd_returns.rename(columns={PRICE_TICKER: 'Ticker', 'Date': 'date'})
    panel = panel.merge(fwd_long[['date', 'Ticker', 'fwd_ret']], on=['date', 'Ticker'], how='left')
    panel = panel.dropna(subset=['fwd_ret'])
    return panel.set_index(['date', 'Ticker'])


def walk_forward_splits(
    dates: list[datetime.date],
    n_train: int,
    n_test: int,
) -> list[tuple[list[datetime.date], list[datetime.date]]]:
    """Non-overlapping walk-forward splits.

    Parameters
    ----------
    dates   : Sorted list of rebalance dates.
    n_train : Number of periods in training window.
    n_test  : Number of periods in test window.

    Returns
    -------
    List of (train_dates, test_dates) tuples. Test always follows train.
    """
    splits = []
    i = 0
    while i + n_train + n_test <= len(dates):
        train = dates[i: i + n_train]
        test  = dates[i + n_train: i + n_train + n_test]
        splits.append((train, test))
        i += n_test
    return splits


def run_ml_backtest(
    model,
    cross_sections: dict[datetime.date, pd.DataFrame],
    fwd_returns: pd.DataFrame,
    n_train: int = 20,
    n_test: int = 4,
    normalize: str = 'rank',
    feature_cols: list[str] | None = None,
) -> dict:
    """Walk-forward backtest using an sklearn-compatible estimator.

    Fits the model on a rolling training window, generates return-rank predictions
    on the subsequent test window, then evaluates with compute_backtest().

    Parameters
    ----------
    model        : sklearn-compatible estimator with fit(X, y) and predict(X).
    cross_sections : Dict mapping date → cross-section DataFrame.
    fwd_returns  : Output of compute_forward_returns().
    n_train      : Training window size in rebalance periods.
    n_test       : Test window size in rebalance periods.
    normalize    : Normalization applied to features before fitting.
    feature_cols : Factor columns to use; defaults to all standard cols.

    Returns
    -------
    Same dict shape as compute_backtest(): ic_series, quintile_cumret,
    mean_ic, ic_tstat, n_dates. Compatible with existing plotting code.
    """
    from irp.factors.backtest import compute_backtest

    panel = build_dataset(cross_sections, fwd_returns, normalize=normalize, feature_cols=feature_cols)
    if panel.empty:
        return {
            'ic_series': pd.Series(dtype=float),
            'quintile_cumret': pd.DataFrame(),
            'mean_ic': float('nan'),
            'ic_tstat': float('nan'),
            'n_dates': 0,
        }

    dates = sorted(cross_sections.keys())
    splits = walk_forward_splits(dates, n_train, n_test)
    if not splits:
        logger.warning(f'Not enough dates for walk-forward (need {n_train + n_test}, have {len(dates)})')
        return {
            'ic_series': pd.Series(dtype=float),
            'quintile_cumret': pd.DataFrame(),
            'mean_ic': float('nan'),
            'ic_tstat': float('nan'),
            'n_dates': 0,
        }

    cols = feature_cols or [c for c in FACTOR_COLS if c in panel.columns]
    predicted_cross_sections: dict[datetime.date, pd.DataFrame] = {}

    for train_dates, test_dates in splits:
        train = panel.loc[panel.index.get_level_values('date').isin(train_dates)]
        X_train = train[cols].fillna(0)
        y_train = train['fwd_ret']
        model.fit(X_train, y_train)
        for d in test_dates:
            if d not in cross_sections:
                continue
            xs = cross_sections[d]
            feat_cols = [c for c in cols if c in xs.columns]
            fn = rank_norm if normalize == 'rank' else (zscore if normalize == 'zscore' else None)
            sub = fn(xs[feat_cols]) if fn is not None else xs[feat_cols].copy()
            X_test = sub.reindex(columns=cols).fillna(0)
            preds = model.predict(X_test)
            xs_out = xs.copy()
            xs_out['__ml__'] = pd.Series(preds, index=sub.index)
            predicted_cross_sections[d] = xs_out
        logger.info(f'ML: trained on {len(train_dates)} periods, predicted {len(test_dates)} test dates')

    return compute_backtest('__ml__', predicted_cross_sections, fwd_returns)

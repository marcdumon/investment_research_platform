"""Baseline cross-sectional classifier over a /features export.

Walk-forward (expanding window) logistic regression predicting the quantile /
binary `label` column from the feature panel. Mirrors `baseline.py` (regression)
and reuses its loader, IC, quintile and theme helpers — keep notebooks thin.

A ranking `score` (expected bucket index from predicted class probabilities) is
also produced, so the same IC + quintile diagnostics as the regression baseline
apply when the export carries `fwd_ret`.
"""

from dataclasses import dataclass, field

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from sklearn.base import clone
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix

from irp.models import baseline as bl

# re-export so the notebook only imports this module
list_exports = bl.list_exports
load_export = bl.load_export
nb_template = bl.nb_template


@dataclass
class ClassifierResult:
    predictions: pd.DataFrame  # Date, Ticker, label, pred_class, pred (score), fwd_ret?
    classes: list
    accuracy: float
    accuracy_by_date: pd.Series
    confusion: np.ndarray  # rows=true, cols=pred
    feature_cols: list[str]
    ic_series: pd.Series = field(default=None)  # score vs fwd_ret (if available)
    quintile_cumret: pd.DataFrame = field(default=None)
    ls_cumret: pd.Series = field(default=None)
    mean_ic: float = field(default=np.nan)
    icir: float = field(default=np.nan)


def walk_forward_classifier(
    df: pd.DataFrame,
    feature_cols: list[str],
    target: str = 'label',
    model=None,
    min_train_dates: int = 12,
    n_quantiles: int = 5,
) -> ClassifierResult:
    """Expanding-window classification.

    At each date d (after `min_train_dates`), fit `model` on rows with Date < d
    and predict the cross-section at d. `pred` = expected bucket index from
    `predict_proba` (a continuous ranking score). Out-of-sample throughout.
    """
    if target not in df.columns:
        raise ValueError(
            f'no "{target}" column — build a quantile/binary label on /features.'
        )
    model = model or LogisticRegression(max_iter=1000)
    has_ret = 'fwd_ret' in df.columns
    dates = sorted(df['Date'].unique())
    need = list(feature_cols) + [target]

    preds = []
    for i, d in enumerate(dates):
        if i < min_train_dates:
            continue
        train = df[df['Date'] < d].dropna(subset=need)
        test = df[df['Date'] == d].dropna(subset=feature_cols)
        if train[target].nunique() < 2 or len(train) < 50 or test.empty:
            continue
        m = clone(model).fit(train[feature_cols].to_numpy(), train[target].to_numpy())
        proba = m.predict_proba(test[feature_cols].to_numpy())
        score = proba @ np.asarray(m.classes_, dtype=float)  # expected bucket
        t = test[['Date', 'Ticker', target]].rename(columns={target: 'label'}).copy()
        t['pred_class'] = m.classes_[proba.argmax(axis=1)]
        t['pred'] = score
        if has_ret:
            t['fwd_ret'] = test['fwd_ret'].to_numpy()
        preds.append(t)

    if not preds:
        return ClassifierResult(
            pd.DataFrame(),
            [],
            np.nan,
            pd.Series(dtype=float),
            np.empty((0, 0)),
            feature_cols,
        )

    pred_df = pd.concat(preds, ignore_index=True)
    classes = sorted(pred_df['label'].dropna().unique().tolist())
    acc_by_date = pred_df.groupby('Date').apply(
        lambda g: (g['pred_class'] == g['label']).mean()
    )
    acc = float((pred_df['pred_class'] == pred_df['label']).mean())
    cm = confusion_matrix(pred_df['label'], pred_df['pred_class'], labels=classes)

    res = ClassifierResult(
        predictions=pred_df,
        classes=classes,
        accuracy=acc,
        accuracy_by_date=acc_by_date,
        confusion=cm,
        feature_cols=feature_cols,
    )
    if has_ret:
        res.ic_series = bl._ic_series(pred_df)
        res.quintile_cumret, res.ls_cumret = bl._quintile_cumret(pred_df, n_quantiles)
        res.mean_ic = float(res.ic_series.mean())
        res.icir = (
            float(res.ic_series.mean() / res.ic_series.std())
            if res.ic_series.std()
            else np.nan
        )
    return res


# ── reporting ─────────────────────────────────────────────────────────


def summary(res: ClassifierResult) -> pd.DataFrame:
    base = (
        res.predictions['label'].value_counts(normalize=True).max()
        if len(res.predictions)
        else np.nan
    )
    out = pd.DataFrame([
        {
            'accuracy': res.accuracy,
            'baseline_acc (majority)': base,
            'n_classes': len(res.classes),
            'n_dates': len(res.accuracy_by_date),
            'n_preds': len(res.predictions),
            'score_IC': res.mean_ic,
            'score_ICIR': res.icir,
        }
    ])
    print(out.T.to_string(header=False))
    return out


# ── plots ─────────────────────────────────────────────────────────────


def plot_confusion(res: ClassifierResult, template: str = 'plotly_dark') -> go.Figure:
    cm = res.confusion
    norm = cm / cm.sum(axis=1, keepdims=True) if cm.size else cm
    labels = [str(c) for c in res.classes]
    fig = go.Figure(
        go.Heatmap(
            z=norm,
            x=labels,
            y=labels,
            colorscale='Blues',
            zmin=0,
            zmax=1,
            text=cm,
            texttemplate='%{text}',
            hovertemplate='true %{y} → pred %{x}<extra></extra>',
        )
    )
    fig.update_layout(
        bl._layout(
            template,
            title='Confusion (row-normalized)',
            xaxis_title='predicted',
            yaxis_title='true',
        )
    )
    fig.update_yaxes(autorange='reversed')
    return fig


def plot_accuracy(res: ClassifierResult, template: str = 'plotly_dark') -> go.Figure:
    a = res.accuracy_by_date
    fig = go.Figure(go.Bar(x=list(a.index), y=a.values, marker_color=bl._ACCENT))
    fig.add_hline(
        y=res.accuracy,
        line_dash='dash',
        line_color='#fff',
        annotation_text=f'mean {res.accuracy:.3f}',
    )
    fig.update_layout(
        bl._layout(
            template, title='Out-of-sample accuracy per date', yaxis_title='accuracy'
        )
    )
    return fig


def plot_quintiles(res: ClassifierResult, template: str = 'plotly_dark') -> go.Figure:
    if res.quintile_cumret is None or res.quintile_cumret.empty:
        return go.Figure(
            layout=bl._layout(
                template, title='No fwd_ret in export — quintiles unavailable'
            )
        )
    return bl.plot_quintiles(res, template)  # duck-typed (same attrs)

"""Baseline cross-sectional linear model over the feature panel.

Walk-forward (expanding window) linear regression predicting forward returns
from normalized factors. Keep notebooks thin: build the dataset, run the
backtest, and plot — all here.

Data path reuses the `/features` builder (`features_service.build_panel`), so the
notebook trains on exactly the panel the UI exports.
"""
import datetime
from dataclasses import dataclass, field

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from scipy import stats
from sklearn.base import clone
from sklearn.linear_model import Ridge

_ACCENT = '#58a6ff'
_MUTED = '#7d8590'
_QCOLORS = ['#d65a5a', '#d6a05a', '#c9c95a', '#7ec97e', '#4ec94e']  # Q1..Q5


@dataclass
class BaselineResult:
    predictions: pd.DataFrame          # Date, Ticker, fwd_ret, pred
    ic_series: pd.Series               # Spearman IC per date
    quintile_cumret: pd.DataFrame      # Q1..Q5 cumulative return
    coefs: pd.Series                   # mean coefficient per feature
    feature_cols: list[str]
    mean_ic: float = field(default=np.nan)
    icir: float = field(default=np.nan)
    r2_oos: float = field(default=np.nan)
    ls_cumret: pd.Series = field(default=None)


# ── dataset (load an export from the /features page) ──────────────────

_NON_FEATURE = {'Date', 'Ticker', 'fwd_ret', 'label'}


def _export_dir():
    from irp.ui.services import features_service
    return features_service._EXPORT_DIR


def list_exports() -> pd.DataFrame:
    """Datasets exported from the /features page, newest first."""
    d = _export_dir()
    files = list(d.glob('*.parquet')) + list(d.glob('*.csv')) if d.exists() else []
    rows = [{'file': f.name, 'modified': datetime.datetime.fromtimestamp(f.stat().st_mtime),
             'mb': round(f.stat().st_size / 1e6, 2)} for f in files]
    if not rows:
        return pd.DataFrame(columns=['file', 'modified', 'mb'])
    return pd.DataFrame(rows).sort_values('modified', ascending=False).reset_index(drop=True)


def load_export(
    path=None,
    feature_cols: list[str] | None = None,
    target: str = 'fwd_ret',
) -> tuple[pd.DataFrame, list[str]]:
    """Load a dataset built on the /features page (parquet/CSV).

    `path=None` loads the most recently exported file from `data/feature_exports/`.
    `feature_cols=None` infers features as every numeric column except
    Date/Ticker/fwd_ret/label. The export must carry the `target` column — build
    it with a label (Return / Up-Down / Quantile) on the /features page.
    """
    from pathlib import Path

    if path is None:
        files = list(_export_dir().glob('*.parquet')) + list(_export_dir().glob('*.csv'))
        if not files:
            raise FileNotFoundError(
                'No exports in data/feature_exports/. Build + Export a dataset on '
                'the /features page first (with a forward-return label).')
        path = max(files, key=lambda f: f.stat().st_mtime)
    path = Path(path)
    df = pd.read_csv(path) if path.suffix == '.csv' else pd.read_parquet(path)
    print(f'loaded {path.name}  {df.shape[0]:,} rows × {df.shape[1]} cols')

    if target not in df.columns:
        raise ValueError(
            f'export has no "{target}" column — rebuild on /features with a label '
            f'(Target type = Return / Up-Down / Quantile).')
    if feature_cols is None:
        feature_cols = [c for c in df.columns
                        if c not in _NON_FEATURE and pd.api.types.is_numeric_dtype(df[c])]
    return df, feature_cols


# ── walk-forward backtest ─────────────────────────────────────────────

def walk_forward_linear(
    df: pd.DataFrame,
    feature_cols: list[str],
    model=None,
    min_train_dates: int = 12,
    n_quantiles: int = 5,
) -> BaselineResult:
    """Expanding-window linear backtest.

    At each date d (after `min_train_dates`), fit `model` on all rows with
    Date < d, predict the cross-section at d. Out-of-sample throughout.
    """
    model = model or Ridge(alpha=1.0)
    dates = sorted(df['Date'].unique())
    need = list(feature_cols) + ['fwd_ret']

    preds, coefs = [], []
    for i, d in enumerate(dates):
        if i < min_train_dates:
            continue
        train = df[df['Date'] < d].dropna(subset=need)
        test = df[df['Date'] == d].dropna(subset=feature_cols)
        if len(train) < 50 or test.empty:
            continue
        m = clone(model).fit(train[feature_cols].to_numpy(), train['fwd_ret'].to_numpy())
        t = test[['Date', 'Ticker', 'fwd_ret']].copy()
        t['pred'] = m.predict(test[feature_cols].to_numpy())
        preds.append(t)
        coefs.append(m.coef_)

    if not preds:
        empty = pd.DataFrame(columns=['Date', 'Ticker', 'fwd_ret', 'pred'])
        return BaselineResult(empty, pd.Series(dtype=float),
                              pd.DataFrame(), pd.Series(dtype=float), feature_cols)

    pred_df = pd.concat(preds, ignore_index=True)
    ic = _ic_series(pred_df)
    qcr, ls = _quintile_cumret(pred_df, n_quantiles)
    valid = pred_df.dropna(subset=['fwd_ret', 'pred'])
    r2 = _r2(valid['fwd_ret'].to_numpy(), valid['pred'].to_numpy()) if len(valid) else np.nan
    coef = pd.Series(np.mean(coefs, axis=0), index=feature_cols)

    return BaselineResult(
        predictions=pred_df, ic_series=ic, quintile_cumret=qcr, coefs=coef,
        feature_cols=feature_cols, mean_ic=float(ic.mean()),
        icir=float(ic.mean() / ic.std()) if ic.std() else np.nan,
        r2_oos=float(r2), ls_cumret=ls,
    )


def _ic_series(pred_df: pd.DataFrame) -> pd.Series:
    def _spear(g):
        g = g.dropna(subset=['fwd_ret', 'pred'])
        if len(g) < 5:
            return np.nan
        return stats.spearmanr(g['pred'], g['fwd_ret']).correlation
    return pred_df.groupby('Date').apply(_spear).dropna()


def _quintile_cumret(pred_df: pd.DataFrame, n: int) -> tuple[pd.DataFrame, pd.Series]:
    rows = []
    for d, g in pred_df.groupby('Date'):
        g = g.dropna(subset=['fwd_ret', 'pred'])
        if len(g) < n:
            continue
        q = pd.qcut(g['pred'].rank(method='first'), n, labels=False)
        means = g['fwd_ret'].groupby(q).mean()
        rows.append(pd.Series(means, name=d))
    if not rows:
        return pd.DataFrame(), pd.Series(dtype=float)
    per_date = pd.DataFrame(rows).sort_index()      # rows=date, cols=quintile 0..n-1
    per_date.columns = [f'Q{int(c) + 1}' for c in per_date.columns]
    cum = np.exp(per_date.fillna(0).cumsum())       # fwd_ret is log return
    ls = np.exp((per_date[f'Q{n}'] - per_date['Q1']).fillna(0).cumsum())
    return cum, ls


def _r2(y, yhat) -> float:
    ss_res = np.nansum((y - yhat) ** 2)
    ss_tot = np.nansum((y - np.nanmean(y)) ** 2)
    return 1 - ss_res / ss_tot if ss_tot else np.nan


# ── reporting ─────────────────────────────────────────────────────────

def summary(res: BaselineResult) -> pd.DataFrame:
    """Print headline metrics; return them as a one-row DataFrame."""
    n_q = res.quintile_cumret.shape[1] if not res.quintile_cumret.empty else 0
    q_top = res.quintile_cumret.iloc[-1].iloc[-1] if n_q else np.nan
    q_bot = res.quintile_cumret.iloc[-1].iloc[0] if n_q else np.nan
    out = pd.DataFrame([{
        'mean_IC': res.mean_ic, 'ICIR': res.icir, 'R2_oos': res.r2_oos,
        'n_dates': len(res.ic_series), 'n_preds': len(res.predictions),
        'topQ_x': q_top, 'botQ_x': q_bot,
        'LS_x': res.ls_cumret.iloc[-1] if res.ls_cumret is not None and len(res.ls_cumret) else np.nan,
    }])
    print(out.T.to_string(header=False))
    return out


# ── plots (return go.Figure; notebook displays) ───────────────────────

def nb_template() -> str:
    """Plotly template matching the VS Code theme (dark by default)."""
    import json
    from pathlib import Path
    try:
        cfg = json.loads(Path.home().joinpath('.config/Code/User/settings.json').read_text())
        return 'plotly_white' if 'light' in str(cfg.get('workbench.colorTheme', '')).lower() else 'plotly_dark'
    except Exception:
        return 'plotly_dark'



def _layout(template: str, **extra) -> dict:
    base = dict(template=template, margin=dict(l=50, r=20, t=40, b=40),
                paper_bgcolor='rgba(0,0,0,0)', font=dict(color=_MUTED))
    base.update(extra)
    return base


def plot_ic(res: BaselineResult, template: str = 'plotly_dark') -> go.Figure:
    ic = res.ic_series
    fig = go.Figure(go.Bar(x=list(ic.index), y=ic.values, marker_color=_ACCENT, name='IC'))
    fig.add_hline(y=res.mean_ic, line_dash='dash', line_color='#fff',
                  annotation_text=f'mean {res.mean_ic:.3f}')
    fig.update_layout(_layout(template, title='Out-of-sample IC per date',
                              yaxis_title='Spearman IC'))
    return fig


def plot_quintiles(res: BaselineResult, template: str = 'plotly_dark') -> go.Figure:
    qcr = res.quintile_cumret
    fig = go.Figure()
    for i, col in enumerate(qcr.columns):
        fig.add_trace(go.Scatter(x=list(qcr.index), y=qcr[col], mode='lines',
                                 name=col, line=dict(color=_QCOLORS[i % len(_QCOLORS)])))
    if res.ls_cumret is not None and len(res.ls_cumret):
        fig.add_trace(go.Scatter(x=list(res.ls_cumret.index), y=res.ls_cumret.values,
                                 mode='lines', name='L/S (Q5-Q1)',
                                 line=dict(color='#fff', dash='dash')))
    fig.update_layout(_layout(template, title='Quintile cumulative return (by predicted score)',
                              yaxis_title='Growth of 1 (log-cumulative)'))
    return fig


def plot_coefs(res: BaselineResult, template: str = 'plotly_dark') -> go.Figure:
    c = res.coefs.sort_values()
    colors = [_QCOLORS[0] if v < 0 else _QCOLORS[-1] for v in c.values]
    fig = go.Figure(go.Bar(x=c.values, y=list(c.index), orientation='h', marker_color=colors))
    fig.update_layout(_layout(template, title='Mean linear coefficients',
                              xaxis_title='coefficient'))
    return fig


def plot_pred_vs_actual(res: BaselineResult, sample: int = 5000,
                        template: str = 'plotly_dark') -> go.Figure:
    d = res.predictions.dropna(subset=['fwd_ret', 'pred'])
    if len(d) > sample:
        d = d.sample(sample, random_state=0)
    fig = go.Figure(go.Scattergl(x=d['pred'], y=d['fwd_ret'], mode='markers',
                                 marker=dict(color=_ACCENT, size=4, opacity=0.4)))
    fig.update_layout(_layout(template, title='Predicted vs actual forward return (log ret)',
                              xaxis_title='predicted log ret', yaxis_title='actual log ret'))
    return fig

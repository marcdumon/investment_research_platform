"""Model-prediction export/import — the bridge from notebook models back to the UI.

Notebooks save a walk-forward prediction frame here; `/today` (via `today_service`) reads the
latest file and surfaces the most recent date's top-ranked names as "model picks". Mirrors the
`feature_exports` convention. Pure filesystem + pandas; no Dash, no DB.
"""
import datetime
from pathlib import Path

import pandas as pd

from irp.core.config import config

_DIR = Path(config.data.root_dir) / 'model_predictions'


def _rank_col(df: pd.DataFrame) -> str:
    """The column to rank by: `pred` (regression) or `score` (classifier ranking)."""
    if 'pred' in df.columns:
        return 'pred'
    if 'score' in df.columns:
        return 'score'
    raise ValueError("prediction frame needs a 'pred' or 'score' column")


def save_predictions(pred_df: pd.DataFrame, name: str) -> Path:
    """Write a walk-forward prediction frame (Date, Ticker, pred/score [, fwd_ret]) to
    `data/model_predictions/<name>_<ts>.parquet`. Returns the path."""
    missing = {'Date', 'Ticker'} - set(pred_df.columns)
    if missing:
        raise ValueError(f'prediction frame missing columns: {sorted(missing)}')
    _rank_col(pred_df)                         # validates pred/score present
    _DIR.mkdir(parents=True, exist_ok=True)
    ts = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
    path = _DIR / f'{name}_{ts}.parquet'
    pred_df.to_parquet(path)
    return path


def list_predictions() -> list[Path]:
    """Exported prediction files, newest first."""
    if not _DIR.exists():
        return []
    return sorted(_DIR.glob('*.parquet'), key=lambda p: p.stat().st_mtime, reverse=True)


def load_predictions(path: str | Path | None = None) -> pd.DataFrame:
    """Load a prediction frame (most recent export when `path=None`)."""
    if path is None:
        files = list_predictions()
        if not files:
            raise FileNotFoundError('no exports in data/model_predictions/ — run a model '
                                    'notebook and call save_predictions(res.predictions, name)')
        path = files[0]
    df = pd.read_parquet(path)
    df['Date'] = pd.to_datetime(df['Date'])
    return df


def latest_picks(pred_df: pd.DataFrame, n: int = 20) -> pd.DataFrame:
    """Top-`n` names on the **most recent** prediction date, ranked by predicted return.

    Indexed by Ticker, carries a leading `Rank`. The latest date is the model's freshest
    out-of-sample cross-section — the actionable slice."""
    if pred_df.empty:
        return pd.DataFrame()
    col = _rank_col(pred_df)
    last_date = pred_df['Date'].max()
    cur = pred_df[pred_df['Date'] == last_date].copy()
    cur = cur.dropna(subset=[col]).sort_values(col, ascending=False).head(n)
    cur = cur.set_index('Ticker')
    cur.insert(0, 'Rank', range(1, len(cur) + 1))
    return cur

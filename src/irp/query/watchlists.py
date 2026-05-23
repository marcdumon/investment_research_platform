"""Named ticker subset persistence as JSON on disk.

Avoids DuckDB write-lock complexity for a simple key/value store.
"""
import datetime
import json
from pathlib import Path

import pandas as pd

from irp.core.config import config


def _path() -> Path:
    return Path(config.data.root_dir) / 'watchlists.json'


def _load_all() -> dict:
    p = _path()
    if not p.exists():
        return {}
    try:
        with open(p) as f:
            return json.load(f)
    except (json.JSONDecodeError, OSError):
        return {}


def _save_all(data: dict) -> None:
    p = _path()
    p.parent.mkdir(parents=True, exist_ok=True)
    with open(p, 'w') as f:
        json.dump(data, f, indent=2, default=str)


def list_watchlists() -> pd.DataFrame:
    """Return DataFrame with columns [name, n, created, summary], newest first."""
    data = _load_all()
    rows = [
        {
            'name': k,
            'n': v.get('n', len(v.get('tickers', []))),
            'created': v.get('created', ''),
            'summary': v.get('summary', ''),
        }
        for k, v in data.items()
    ]
    if not rows:
        return pd.DataFrame(columns=['name', 'n', 'created', 'summary'])
    return (
        pd.DataFrame(rows)
        .sort_values('created', ascending=False)
        .reset_index(drop=True)
    )


def load_watchlist(name: str) -> list[str]:
    """Return ticker list for a saved watchlist. Raises KeyError if not found."""
    data = _load_all()
    if name not in data:
        raise KeyError(name)
    return data[name]['tickers']


def save_watchlist(name: str, tickers: list[str], summary: str = '') -> None:
    """Create or replace a watchlist."""
    data = _load_all()
    data[name] = {
        'tickers': tickers,
        'n': len(tickers),
        'summary': summary,
        'created': datetime.date.today().isoformat(),
    }
    _save_all(data)


def delete_watchlist(name: str) -> None:
    """Remove a watchlist (no-op if name does not exist)."""
    data = _load_all()
    if name in data:
        data.pop(name)
        _save_all(data)

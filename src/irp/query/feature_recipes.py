"""Named feature-recipe persistence as JSON on disk.

A recipe is the full feature spec (grid params + step stack + label config +
universe filters) needed to re-materialize a dataset. Same key/value-on-disk
pattern as `irp.query.watchlists`.
"""
import datetime
import json
from pathlib import Path

import pandas as pd

from irp.core.config import config


def _path() -> Path:
    return Path(config.data.root_dir) / 'feature_recipes.json'


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


def list_recipes() -> pd.DataFrame:
    """Return DataFrame [name, n_steps, created], newest first."""
    data = _load_all()
    rows = [
        {
            'name': k,
            'n_steps': len(v.get('steps', [])),
            'created': v.get('created', ''),
        }
        for k, v in data.items()
    ]
    if not rows:
        return pd.DataFrame(columns=['name', 'n_steps', 'created'])
    return (
        pd.DataFrame(rows)
        .sort_values('created', ascending=False)
        .reset_index(drop=True)
    )


def load_recipe(name: str) -> dict:
    """Return the full spec dict for a recipe. Raises KeyError if not found."""
    data = _load_all()
    if name not in data:
        raise KeyError(name)
    return data[name]


def save_recipe(name: str, spec: dict) -> None:
    """Create or replace a recipe. `spec` is stored as-is plus a created date."""
    data = _load_all()
    data[name] = {**spec, 'created': datetime.date.today().isoformat()}
    _save_all(data)


def delete_recipe(name: str) -> None:
    """Remove a recipe (no-op if name does not exist)."""
    data = _load_all()
    if name in data:
        data.pop(name)
        _save_all(data)

import tomllib
from pathlib import Path

import pandas as pd

_PATH = Path(__file__).parents[3] / "data" / "anomaly_whitelist.toml"
_KEYS = ["ticker", "period", "rule", "column"]
_COLS = _KEYS + ["note"]


def load(path: str | Path | None = None) -> pd.DataFrame:
    """Load exceptions from TOML file. Returns empty DataFrame if file absent."""
    p = Path(path or _PATH)
    if not p.exists():
        return pd.DataFrame(columns=_COLS)
    with open(p, "rb") as f:
        data = tomllib.load(f)
    rows = data.get("exceptions", [])
    if not rows:
        return pd.DataFrame(columns=_COLS)
    df = pd.DataFrame(rows)
    for col in _COLS:
        if col not in df.columns:
            df[col] = ""
    return df[_COLS].fillna("")


def suppress(findings: pd.DataFrame, path: str | Path | None = None) -> pd.DataFrame:
    """Remove findings that match an entry in the exceptions file."""
    exc = load(path)
    if exc.empty:
        return findings
    keys = set(zip(exc["ticker"], exc["period"], exc["rule"], exc["column"]))
    mask = [
        (t, p, r, c) not in keys
        for t, p, r, c in zip(
            findings["ticker"], findings["period"], findings["rule"], findings["column"]
        )
    ]
    return findings[mask].reset_index(drop=True)


def add(
    ticker: str,
    period: str,
    rule: str,
    column: str,
    note: str = "",
    path: str | Path | None = None,
) -> bool:
    """Append one [[exceptions]] entry to the TOML file.

    Preserves all existing content and comments. Returns False if entry already exists.
    """
    p = Path(path or _PATH)
    p.parent.mkdir(parents=True, exist_ok=True)

    # check for duplicate
    exc = load(p)
    key = (ticker, period, rule, column)
    if not exc.empty:
        existing = set(zip(exc["ticker"], exc["period"], exc["rule"], exc["column"]))
        if key in existing:
            return False

    block = (
        f"\n[[exceptions]]\n"
        f'ticker = "{ticker}"\n'
        f'period = "{period}"\n'
        f'rule   = "{rule}"\n'
        f'column = "{column}"\n'
        f'note   = "{note}"\n'
    )
    with open(p, "a") as f:
        f.write(block)
    return True

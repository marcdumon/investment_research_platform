"""EDGAR correction storage for SimFin line items.

Each correction records that a SimFin line item value differs from the value
reported in the EDGAR filing. Key: (ticker, period, stmt_kind, line_item).
save_corrections replaces the full set for a triplet atomically (rewrite).
"""
import tomllib
from datetime import date

import pandas as pd

from irp.core.config import config

CORRECTIONS_PATH = config.data.root_dir / 'data_quality' / 'simfin_edgar_corrections.toml'

_COLS = ['ticker', 'period', 'stmt_kind', 'line_item', 'simfin_value', 'edgar_value', 'annotated_at']


def _load_raw() -> list[dict]:
    if not CORRECTIONS_PATH.exists():
        return []
    with open(CORRECTIONS_PATH, 'rb') as f:
        return tomllib.load(f).get('corrections', [])


def _esc(s: str) -> str:
    return s.replace('\\', '\\\\').replace('"', '\\"')


def _write_all(rows: list[dict]) -> None:
    CORRECTIONS_PATH.parent.mkdir(parents=True, exist_ok=True)
    header = (
        '# EDGAR corrections for SimFin line items\n'
        '# Each [[corrections]] entry records a SimFin vs EDGAR discrepancy.\n'
        '# Key: (ticker, period, stmt_kind, line_item)\n'
    )
    blocks = []
    for r in rows:
        sf = r.get('simfin_value')
        sf_str = str(float(sf)) if sf is not None else 'null'
        blocks.append(
            '\n[[corrections]]\n'
            f'ticker       = "{_esc(r["ticker"])}"\n'
            f'period       = "{_esc(r["period"])}"\n'
            f'stmt_kind    = "{_esc(r["stmt_kind"])}"\n'
            f'line_item    = "{_esc(r["line_item"])}"\n'
            f'simfin_value = {sf_str}\n'
            f'edgar_value  = {float(r["edgar_value"])}\n'
            f'annotated_at = "{r.get("annotated_at", date.today().isoformat())}"\n'
        )
    CORRECTIONS_PATH.write_text(header + ''.join(blocks))


def load_corrections(ticker: str, period: str, stmt_kind: str) -> dict[str, float]:
    """Return {line_item: edgar_value} for a (ticker, period, stmt_kind) triplet."""
    return {
        r['line_item']: float(r['edgar_value'])
        for r in _load_raw()
        if (r.get('ticker') == ticker
            and r.get('period') == period
            and r.get('stmt_kind') == stmt_kind
            and r.get('edgar_value') is not None)
    }


def save_corrections(
    ticker: str,
    period: str,
    stmt_kind: str,
    edgar_values: dict[str, float],
    simfin_values: dict[str, float] | None = None,
) -> None:
    """Replace all corrections for (ticker, period, stmt_kind) with edgar_values.

    edgar_values: {line_item: edgar_float}
    simfin_values: optional {line_item: simfin_float} stored for reporting.
    """
    existing = [
        r for r in _load_raw()
        if not (r.get('ticker') == ticker
                and r.get('period') == period
                and r.get('stmt_kind') == stmt_kind)
    ]
    today = date.today().isoformat()
    new_rows = [
        {
            'ticker': ticker,
            'period': period,
            'stmt_kind': stmt_kind,
            'line_item': item,
            'simfin_value': (simfin_values or {}).get(item),
            'edgar_value': val,
            'annotated_at': today,
        }
        for item, val in edgar_values.items()
    ]
    _write_all(existing + new_rows)


def get_all_corrections() -> pd.DataFrame:
    """All corrections across all tickers/periods as a DataFrame."""
    rows = _load_raw()
    if not rows:
        return pd.DataFrame(columns=_COLS)
    df = pd.DataFrame(rows)
    for col in _COLS:
        if col not in df.columns:
            df[col] = None
    return df[_COLS].reset_index(drop=True)

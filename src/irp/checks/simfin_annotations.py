"""Unified annotation storage for SimFin data quality.

One file, grouped by (ticker, period, stmt_kind, rule). Each entry holds:
  - review fields: rule, status, note, reviewed_at  (flat, at statement level)
  - corrections sub-array: [[statements.corrections]]  (per line item)

Replaces simfin_reviews.py + simfin_corrections.py.
Migrates legacy data on first access.
"""
import tomllib
from datetime import date

import pandas as pd

from irp.core.config import config

ANNOTATIONS_PATH         = config.data.root_dir / 'data_quality' / 'simfin_annotations.toml'

VALID_STATUS  = ('ok', 'data_error', 'to_check')
MANUAL_RULE   = 'manual_check'


# ── helpers ───────────────────────────────────────────────────────────────────

def _esc(s: str) -> str:
    return str(s).replace('\\', '\\\\').replace('"', '\\"').replace('\n', '\\n').replace('\r', '')


def _stmt_kind_for_rule(rule: str) -> str:
    from irp.checks.simfin_rules import REGISTRY
    for r in REGISTRY:
        if r.name == rule:
            return r.statement
    return 'cross'


def _key(s: dict) -> tuple:
    return (s.get('ticker', ''), s.get('period', ''),
            s.get('stmt_kind', ''), s.get('rule', ''))


# ── I/O ───────────────────────────────────────────────────────────────────────

def _load_raw() -> list[dict]:
    if not ANNOTATIONS_PATH.exists():
        return []
    with open(ANNOTATIONS_PATH, 'rb') as f:
        return tomllib.load(f).get('statements', [])


def _write_all(stmts: list[dict]) -> None:
    ANNOTATIONS_PATH.parent.mkdir(parents=True, exist_ok=True)
    lines = [
        '# SimFin data quality annotations\n',
        '# Grouped by (ticker, period, stmt_kind, rule).\n',
        '# review fields are flat at statement level; corrections are sub-array.\n',
    ]
    for s in stmts:
        lines.append('\n[[statements]]\n')
        lines.append(f'ticker      = "{_esc(s["ticker"])}"\n')
        lines.append(f'period      = "{_esc(s["period"])}"\n')
        lines.append(f'stmt_kind   = "{_esc(s["stmt_kind"])}"\n')
        lines.append(f'rule        = "{_esc(s.get("rule", ""))}"\n')
        lines.append(f'status      = "{_esc(s.get("status", "to_check"))}"\n')
        lines.append(f'note        = "{_esc(s.get("note", ""))}"\n')
        lines.append(f'reviewed_at = "{_esc(s.get("reviewed_at", date.today().isoformat()))}"\n')
        for corr in s.get('corrections', []):
            lines.append('\n  [[statements.corrections]]\n')
            lines.append(f'  line_item    = "{_esc(corr["line_item"])}"\n')
            sf = corr.get('simfin_value')
            if sf is not None:
                lines.append(f'  simfin_value = {float(sf)}\n')
            lines.append(f'  edgar_value  = {float(corr["edgar_value"])}\n')
            lines.append(f'  note         = "{_esc(corr.get("note", ""))}"\n')
            lines.append(f'  annotated_at = "{_esc(corr.get("annotated_at", date.today().isoformat()))}"\n')
    ANNOTATIONS_PATH.write_text(''.join(lines))



# ── reviews API ───────────────────────────────────────────────────────────────

def _load_reviews() -> set[tuple[str, str, str]]:
    """Set of (ticker, period, rule) used to suppress rule-detected findings."""
    return {
        (s['ticker'], s['period'], s['rule'])
        for s in _load_raw()
        if s.get('rule') and s['rule'] not in ('', MANUAL_RULE)
    }


def _load_reviews_df() -> pd.DataFrame:
    cols = ['ticker', 'period', 'stmt_kind', 'rule', 'status', 'note', 'reviewed_at']
    rows = [
        {c: s.get(c, '') for c in cols}
        for s in _load_raw()
        if s.get('rule') not in ('', None)
    ]
    return pd.DataFrame(rows, columns=cols) if rows else pd.DataFrame(columns=cols)


def _load_flags_df() -> pd.DataFrame:
    df = _load_reviews_df()
    if df.empty:
        return df
    return df[df['rule'] == MANUAL_RULE].reset_index(drop=True)


def _add_review(ticker: str, period: str, rule: str, stmt_kind: str,
                status: str, note: str) -> None:
    if status not in VALID_STATUS:
        raise ValueError(f'status must be one of {VALID_STATUS}')
    stmts = _load_raw()
    stmts = [s for s in stmts
             if not (s['ticker'] == ticker and s['period'] == period
                     and s['stmt_kind'] == stmt_kind and s['rule'] == rule)]
    stmts.append({
        'ticker': ticker, 'period': period, 'stmt_kind': stmt_kind,
        'rule': rule, 'status': status, 'note': note,
        'reviewed_at': date.today().isoformat(), 'corrections': [],
    })
    _write_all(stmts)


def _add_flag(ticker: str, period: str, subject: str, status: str, note: str) -> None:
    stmts = _load_raw()
    stmts = [s for s in stmts
             if not (s['ticker'] == ticker and s['period'] == period
                     and s['rule'] == MANUAL_RULE and s.get('stmt_kind', '') == subject)]
    stmts.append({
        'ticker': ticker, 'period': period, 'stmt_kind': subject,
        'rule': MANUAL_RULE, 'status': status, 'note': note,
        'reviewed_at': date.today().isoformat(), 'corrections': [],
    })
    _write_all(stmts)


# ── corrections API ───────────────────────────────────────────────────────────

def _load_corrections(ticker: str, period: str, stmt_kind: str) -> dict[str, float]:
    for s in _load_raw():
        if s['ticker'] == ticker and s['period'] == period and s['stmt_kind'] == stmt_kind:
            return {c['line_item']: float(c['edgar_value'])
                    for c in s.get('corrections', [])
                    if c.get('edgar_value') is not None}
    return {}


def _load_correction_notes(ticker: str, period: str, stmt_kind: str) -> dict[str, str]:
    for s in _load_raw():
        if s['ticker'] == ticker and s['period'] == period and s['stmt_kind'] == stmt_kind:
            return {c['line_item']: str(c['note'])
                    for c in s.get('corrections', [])
                    if c.get('note')}
    return {}


def _save_statement(
    ticker: str,
    period: str,
    stmt_kind: str,
    rule: str,
    status: str,
    note: str,
    edgar_values: dict[str, float],
    simfin_values: dict[str, float] | None = None,
    line_notes: dict[str, str] | None = None,
) -> None:
    """Save or replace the full statement entry (review + corrections)."""
    today = date.today().isoformat()
    corrections = [
        {
            'line_item':    item,
            'simfin_value': (simfin_values or {}).get(item),
            'edgar_value':  val,
            'note':         (line_notes or {}).get(item, ''),
            'annotated_at': today,
        }
        for item, val in edgar_values.items()
    ]
    stmts = _load_raw()
    # Replace entry with same (ticker, period, stmt_kind, rule)
    stmts = [s for s in stmts
             if not (s['ticker'] == ticker and s['period'] == period
                     and s['stmt_kind'] == stmt_kind and s['rule'] == rule)]
    stmts.append({
        'ticker': ticker, 'period': period, 'stmt_kind': stmt_kind,
        'rule': rule, 'status': status, 'note': note,
        'reviewed_at': today, 'corrections': corrections,
    })
    _write_all(stmts)


# kept for backward compat — routes to save_statement with no corrections
def _save_corrections(
    ticker: str,
    period: str,
    stmt_kind: str,
    edgar_values: dict[str, float],
    simfin_values: dict[str, float] | None = None,
    notes: dict[str, str] | None = None,
) -> None:
    stmts = _load_raw()
    entry = next((s for s in stmts
                  if s['ticker'] == ticker and s['period'] == period
                  and s['stmt_kind'] == stmt_kind), None)
    if entry is None:
        entry = {'ticker': ticker, 'period': period, 'stmt_kind': stmt_kind,
                 'rule': '', 'status': 'to_check', 'note': '',
                 'reviewed_at': date.today().isoformat(), 'corrections': []}
        stmts.append(entry)
    today = date.today().isoformat()
    entry['corrections'] = [
        {'line_item': item, 'simfin_value': (simfin_values or {}).get(item),
         'edgar_value': val, 'note': (notes or {}).get(item, ''),
         'annotated_at': today}
        for item, val in edgar_values.items()
    ]
    _write_all(stmts)


def _get_all_corrections() -> pd.DataFrame:
    cols = ['ticker', 'period', 'stmt_kind', 'line_item',
            'simfin_value', 'edgar_value', 'note', 'annotated_at']
    rows = []
    for s in _load_raw():
        for c in s.get('corrections', []):
            rows.append({
                'ticker':       s['ticker'],
                'period':       s['period'],
                'stmt_kind':    s['stmt_kind'],
                'line_item':    c.get('line_item', ''),
                'simfin_value': c.get('simfin_value'),
                'edgar_value':  c.get('edgar_value'),
                'note':         c.get('note', ''),
                'annotated_at': c.get('annotated_at', ''),
            })
    return pd.DataFrame(rows, columns=cols) if rows else pd.DataFrame(columns=cols)

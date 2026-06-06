"""Pure filter-stack + naming logic for the /screener page.

Page boundary: the screener page imports these instead of defining the
filter/naming logic inside its callbacks. No Dash, no DB — pure pandas/str so
the behaviour is unit-testable on its own.

A *filter step* is a dict with a `type`:
  - `range`  — keep rows with `col` in [`min`, `max`] (either bound optional)
  - `keep`   — keep only rows whose Ticker is in `tickers`
  - `remove` — drop rows whose Ticker is in `tickers`
Each step also carries a human `label` used for naming.
"""
import pandas as pd


def apply_step(df: pd.DataFrame, step: dict) -> pd.DataFrame:
    """Apply one filter step to the working frame."""
    if step['type'] == 'range':
        col, lo, hi = step['col'], step.get('min'), step.get('max')
        if col in df.columns:
            if lo is not None:
                df = df[df[col] >= lo]
            if hi is not None:
                df = df[df[col] <= hi]
    elif step['type'] == 'keep':
        df = df[df['Ticker'].isin(step['tickers'])]
    elif step['type'] == 'remove':
        df = df[~df['Ticker'].isin(step['tickers'])]
    return df


def apply_steps(df: pd.DataFrame, steps: list[dict]) -> pd.DataFrame:
    """Apply an ordered list of filter steps."""
    for step in steps:
        df = apply_step(df, step)
    return df


def auto_name(steps: list[dict], as_of_date: str | None) -> str:
    """Suggest a watchlist name from the range filters + as-of date."""
    range_parts = [
        s['label'].replace(' ', '').replace('≥', 'ge').replace('≤', 'le')
        for s in (steps or [])
        if s.get('type') == 'range'
    ]
    suffix = (as_of_date or '')[:10]
    parts = range_parts + ([suffix] if suffix else [])
    return '_'.join(parts) if parts else f'screener_{suffix}'


def build_summary(steps: list) -> str:
    """One-line human summary of a filter stack (range plain, +keep, -remove)."""
    parts = []
    for s in steps:
        label = s.get('label', '')
        if not label:
            continue
        t = s.get('type', '')
        if t == 'range':
            parts.append(label)
        elif t == 'keep':
            parts.append(f'+{label}')
        elif t == 'remove':
            parts.append(f'-{label}')
    return '; '.join(parts)

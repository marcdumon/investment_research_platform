import tomllib
from datetime import date

import pandas as pd

from irp.core.config import config

REVIEWS_PATH = config.data.root_dir / 'data_quality' / 'simfin_anomaly_reviews.toml'
VALID_STATUS = ('ok', 'data_error', 'to_check')
MANUAL_RULE = 'manual'


def period_str(fy: int, fp: str, period: str) -> str:
    """(2024, 'FY', 'A') -> '2024FY'; (2024, 'Q1', 'Q') -> '2024Q1'."""
    return f'{fy}FY' if period == 'A' else f'{fy}{fp}'


def _load_raw() -> list[dict]:
    if not REVIEWS_PATH.exists():
        return []
    with open(REVIEWS_PATH, 'rb') as f:
        return tomllib.load(f).get('reviews', [])


_REQUIRED = ('ticker', 'period', 'rule')


def _valid(r: dict) -> bool:
    return all(k in r for k in _REQUIRED)


def load_reviews() -> set[tuple[str, str, str]]:
    """Set of (ticker, period, rule) keys used to suppress rule-detected findings.
    Manual flags (rule == 'manual') are not included — they never filter the queue."""
    return {
        (r['ticker'], r['period'], r['rule'])
        for r in _load_raw()
        if _valid(r) and r['rule'] != MANUAL_RULE
    }


def load_reviews_df() -> pd.DataFrame:
    """All entries (rule-detected reviews + manual flags)."""
    rows = [r for r in _load_raw() if _valid(r)]
    if not rows:
        return pd.DataFrame(columns=['ticker', 'period', 'rule', 'subject', 'status', 'note', 'reviewed_at'])
    df = pd.DataFrame(rows)
    # back-compat: old entries used 'column' instead of 'subject'
    if 'column' in df.columns and 'subject' not in df.columns:
        df = df.rename(columns={'column': 'subject'})
    elif 'column' in df.columns and 'subject' in df.columns:
        df['subject'] = df['subject'].fillna(df['column'])
        df = df.drop(columns=['column'])
    return df


def load_flags_df() -> pd.DataFrame:
    """Manual flags only (rule == 'manual')."""
    df = load_reviews_df()
    if df.empty or 'rule' not in df.columns:
        return df
    return df[df['rule'] == MANUAL_RULE].reset_index(drop=True)


def _write(ticker: str, period: str, rule: str, status: str, note: str, subject: str = '') -> None:
    if status not in VALID_STATUS:
        raise ValueError(f'status must be one of {VALID_STATUS}, got {status!r}')
    REVIEWS_PATH.parent.mkdir(parents=True, exist_ok=True)
    if not REVIEWS_PATH.exists():
        REVIEWS_PATH.write_text(
            '# Quality-finding reviews\n'
            '# Each [[reviews]] entry suppresses a rule-detected finding (rule = registered name),\n'
            '# or records a manual side-flag (rule = "manual", subject populated).\n'
            '# subject scope can be a single field, a full statement, or anything in between.\n'
            '# status: ok | data_error | to_check\n'
        )
    def esc(s: str) -> str:
        return s.replace('\\', '\\\\').replace('"', '\\"')
    block = (
        '\n[[reviews]]\n'
        f'ticker      = "{ticker}"\n'
        f'period      = "{period}"\n'
        f'rule        = "{rule}"\n'
        f'subject     = "{esc(subject)}"\n'
        f'status      = "{status}"\n'
        f'note        = "{esc(note)}"\n'
        f'reviewed_at = "{date.today().isoformat()}"\n'
    )
    with open(REVIEWS_PATH, 'a') as f:
        f.write(block)


def add_review(ticker: str, period: str, rule: str, status: str, note: str) -> None:
    """Record a review for a rule-detected finding."""
    _write(ticker, period, rule, status, note, subject='')


def add_flag(ticker: str, period: str, subject: str, status: str, note: str) -> None:
    """Record a manually-spotted finding (not auto-detected by a rule).
    subject can be a single field name, a statement name, or anything in between."""
    _write(ticker, period, MANUAL_RULE, status, note, subject=subject)

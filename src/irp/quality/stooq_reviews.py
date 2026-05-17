"""Stooq-anomaly review storage. Mirrors irp.quality.simfin_reviews.

Key shape: (ticker, period, rule). For stooq, `period` is the calendar year
as a string (e.g. '2024'). The string-typed `period` field is shared with
simfin's TOML schema so review-set lookups have a uniform tuple type.
"""
import tomllib
from datetime import date

import pandas as pd

from irp.core.config import config

REVIEWS_PATH = config.data.root_dir / 'data_quality' / 'stooq_anomaly_reviews.toml'
VALID_STATUS = ('ok', 'data_error', 'to_check')

_REQUIRED = ('ticker', 'period', 'rule')


def _load_raw() -> list[dict]:
    if not REVIEWS_PATH.exists():
        return []
    with open(REVIEWS_PATH, 'rb') as f:
        return tomllib.load(f).get('reviews', [])


def _valid(r: dict) -> bool:
    return all(k in r for k in _REQUIRED)


def load_reviews() -> set[tuple[str, str, str]]:
    """Set of (ticker, period, rule) keys to suppress in future runs."""
    return {(r['ticker'], str(r['period']), r['rule']) for r in _load_raw() if _valid(r)}


def load_reviews_df() -> pd.DataFrame:
    rows = [r for r in _load_raw() if _valid(r)]
    if not rows:
        return pd.DataFrame(columns=['ticker', 'period', 'rule', 'status', 'note', 'reviewed_at'])
    return pd.DataFrame(rows)


def add_review(ticker: str, period: str, rule: str, status: str, note: str) -> None:
    """Append a review entry to the stooq anomaly TOML.
    `period` should be the calendar year as a string (e.g. '2024')."""
    if status not in VALID_STATUS:
        raise ValueError(f'status must be one of {VALID_STATUS}, got {status!r}')
    REVIEWS_PATH.parent.mkdir(parents=True, exist_ok=True)
    if not REVIEWS_PATH.exists():
        REVIEWS_PATH.write_text(
            '# Stooq anomaly reviews\n'
            '# Each [[reviews]] entry suppresses a (ticker, period, rule) finding.\n'
            '# period: calendar year as string, e.g. "2024".\n'
            '# status: ok | data_error | to_check\n'
        )
    note_escaped = note.replace('\\', '\\\\').replace('"', '\\"')
    block = (
        '\n[[reviews]]\n'
        f'ticker      = "{ticker}"\n'
        f'period      = "{period}"\n'
        f'rule        = "{rule}"\n'
        f'status      = "{status}"\n'
        f'note        = "{note_escaped}"\n'
        f'reviewed_at = "{date.today().isoformat()}"\n'
    )
    with open(REVIEWS_PATH, 'a') as f:
        f.write(block)

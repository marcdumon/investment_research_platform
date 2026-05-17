import tomllib
from datetime import date

import pandas as pd

from irp.core.config import config

REVIEWS_PATH = config.data.root_dir / 'data_quality' / 'anomaly_reviews.toml'


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
    """Set of (ticker, period, rule) keys for reviewed items."""
    return {(r['ticker'], r['period'], r['rule']) for r in _load_raw() if _valid(r)}


def load_reviews_df() -> pd.DataFrame:
    rows = [r for r in _load_raw() if _valid(r)]
    if not rows:
        return pd.DataFrame(columns=['ticker', 'period', 'rule', 'status', 'note', 'reviewed_at'])
    return pd.DataFrame(rows)


def add_review(ticker: str, period: str, rule: str, status: str, note: str) -> None:
    REVIEWS_PATH.parent.mkdir(parents=True, exist_ok=True)
    if not REVIEWS_PATH.exists():
        REVIEWS_PATH.write_text(
            '# Quality-finding reviews\n'
            '# Each [[reviews]] entry suppresses a specific finding from future runs.\n'
            '# Key: (ticker, period, rule)\n'
            '# status: ok | data_error\n'
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

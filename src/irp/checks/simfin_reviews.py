"""Backward-compat shim — delegates to simfin_annotations."""
from irp.checks.simfin_annotations import (
    MANUAL_RULE,
    VALID_STATUS,
    _add_flag as add_flag,
    _add_review as add_review,
    _load_flags_df as load_flags_df,
    _load_reviews as load_reviews,
    _load_reviews_df as load_reviews_df,
)

# Re-export surface (kept so F401 does not strip the aliased re-exports).
__all__ = [
    'MANUAL_RULE',
    'VALID_STATUS',
    'add_flag',
    'add_review',
    'load_flags_df',
    'load_reviews',
    'load_reviews_df',
    'period_str',
]


def period_str(fy: int, fp: str, period: str) -> str:
    return f'{fy}FY' if period == 'A' else f'{fy}{fp}'

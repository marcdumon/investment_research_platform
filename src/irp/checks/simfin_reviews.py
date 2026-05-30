"""Backward-compat shim — delegates to simfin_annotations."""
from irp.checks.simfin_annotations import (  # noqa: F401
    VALID_STATUS,
    MANUAL_RULE,
    _load_reviews as load_reviews,
    _load_reviews_df as load_reviews_df,
    _load_flags_df as load_flags_df,
    _add_review as add_review,
    _add_flag as add_flag,
)


def _period_str(fy: int, fp: str, period: str) -> str:
    return f'{fy}FY' if period == 'A' else f'{fy}{fp}'

"""Backward-compat shim — delegates to simfin_annotations."""
from irp.checks.simfin_annotations import (  # noqa: F401
    VALID_STATUS,
    MANUAL_RULE,
    load_reviews,
    load_reviews_df,
    load_flags_df,
    add_review,
    add_flag,
)


def period_str(fy: int, fp: str, period: str) -> str:
    return f'{fy}FY' if period == 'A' else f'{fy}{fp}'

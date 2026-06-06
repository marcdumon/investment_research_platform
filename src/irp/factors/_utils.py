"""Shared utilities for factor compute modules."""
import pandas as pd


def safe_div(num: pd.Series, denom: pd.Series) -> pd.Series:
    """Divide two series, replacing inf/-inf with NA."""
    return (num / denom).replace([float('inf'), float('-inf')], pd.NA)

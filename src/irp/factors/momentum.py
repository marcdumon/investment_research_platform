"""Momentum factors: trailing price returns. (phase 2)"""
import datetime

import pandas as pd


def compute_momentum(
    prices: pd.DataFrame,
    as_of_date: datetime.date,
) -> pd.DataFrame:
    raise NotImplementedError('momentum factors are phase 2')

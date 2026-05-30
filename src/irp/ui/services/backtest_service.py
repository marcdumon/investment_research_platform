"""Backtest service for the UI layer.

Wraps the three orchestrators in `irp.factors.orchestrate` and exposes them
in the form UI callbacks need: typed `BacktestResult` for single-factor +
composite runs; a tidy DataFrame for factor decay.

Universe filtering goes through `universe_service.filter_tickers`, so pages
pass market/sector/watchlist names rather than resolving themselves.
"""
import datetime
from typing import Literal

import pandas as pd

from irp.factors.compute import (
    run_backtest as _run_backtest,
    run_composite_backtest as _run_composite_backtest,
    run_factor_decay as _run_factor_decay,
)
from irp.factors.models import BacktestResult
from irp.ui.services import universe_service


def _run_factor(
    factor: str,
    horizon_days: int,
    start_date: datetime.date,
    end_date: datetime.date,
    variant: Literal['A', 'Q'] = 'A',
    freq: Literal['Q', 'A'] = 'Q',
    cost_bps: float = 0,
    market: str | None = None,
    sector: str | None = None,
    watchlist: str | None = None,
) -> BacktestResult:
    tickers = universe_service._filter_tickers(market=market, sector=sector, watchlist=watchlist)
    return _run_backtest(
        factor, horizon_days, start_date, end_date,
        variant=variant, freq=freq, cost_bps=cost_bps, tickers=tickers,
    )


def _run_composite(
    weights: dict[str, float],
    horizon_days: int,
    start_date: datetime.date,
    end_date: datetime.date,
    variant: Literal['A', 'Q'] = 'A',
    freq: Literal['Q', 'A'] = 'Q',
    normalize: str = 'zscore',
    use_sector_neutral: bool = False,
    cost_bps: float = 0,
    market: str | None = None,
    sector: str | None = None,
    watchlist: str | None = None,
) -> BacktestResult:
    tickers = universe_service._filter_tickers(market=market, sector=sector, watchlist=watchlist)
    return _run_composite_backtest(
        weights, horizon_days, start_date, end_date,
        variant=variant, freq=freq, normalize=normalize,
        use_sector_neutral=use_sector_neutral, cost_bps=cost_bps, tickers=tickers,
    )


def _run_decay(
    factor: str,
    horizons: list[int],
    start_date: datetime.date,
    end_date: datetime.date,
    variant: Literal['A', 'Q'] = 'A',
    freq: Literal['Q', 'A'] = 'Q',
    market: str | None = None,
    sector: str | None = None,
    watchlist: str | None = None,
) -> pd.DataFrame:
    tickers = universe_service._filter_tickers(market=market, sector=sector, watchlist=watchlist)
    return _run_factor_decay(
        factor, horizons, start_date, end_date,
        variant=variant, freq=freq, tickers=tickers,
    )

"""Pydantic models for factor-pipeline boundaries.

These exist primarily as typed contracts so callers don't dig into 16-key
dicts. Arbitrary types (pandas Series/DataFrame) are allowed because
serialising them to JSON is not a goal — these are in-process boundaries.

Backwards-compatible `__getitem__` / `.get()` are provided on
`BacktestResult` so existing dict-style callers (`result['mean_ic']`,
`result.get('icir', nan)`) keep working through the migration.
"""
import datetime
from typing import Any

import pandas as pd
from pydantic import BaseModel, ConfigDict, Field

_NAN = float('nan')


class BacktestResult(BaseModel):
    """Outcome of `compute_backtest`: IC series, quintile cumret + risk metrics.

    Use attribute access in new code (`result.mean_ic`); legacy `result[...]`
    / `result.get(...)` still work via `__getitem__` / `get`.
    """
    model_config = ConfigDict(arbitrary_types_allowed=True, frozen=False)

    ic_series:           pd.Series  = Field(default_factory=lambda: pd.Series(dtype=float))
    quintile_cumret:     pd.DataFrame = Field(default_factory=pd.DataFrame)
    quintile_cumret_net: pd.DataFrame | None = None
    ew_cumret:           pd.Series  = Field(default_factory=lambda: pd.Series(dtype=float))
    ls_cumret:           pd.Series  = Field(default_factory=lambda: pd.Series(dtype=float))
    mean_ic:             float = _NAN
    ic_tstat:            float = _NAN
    icir:                float = _NAN
    n_dates:             int = 0
    quintile_ann_ret:    dict[str, float] = Field(default_factory=dict)
    quintile_ann_vol:    dict[str, float] = Field(default_factory=dict)
    quintile_sharpe:     dict[str, float] = Field(default_factory=dict)
    quintile_max_dd:     dict[str, float] = Field(default_factory=dict)
    turnover_q1:         pd.Series = Field(default_factory=lambda: pd.Series(dtype=float))
    turnover_q5:         pd.Series = Field(default_factory=lambda: pd.Series(dtype=float))
    mean_turnover_q1:    float = _NAN
    mean_turnover_q5:    float = _NAN

    def __getitem__(self, key: str) -> Any:
        return getattr(self, key)

    def get(self, key: str, default: Any = None) -> Any:
        return getattr(self, key, default)

    def to_dict(self) -> dict[str, Any]:
        """Flat dict view for callers that still want one (e.g. cache writers)."""
        return {f: getattr(self, f) for f in self.model_fields}


class CrossSection(BaseModel):
    """A cross-section snapshot: factor DataFrame + metadata."""
    model_config = ConfigDict(arbitrary_types_allowed=True, frozen=True)

    as_of_date: datetime.date
    variant:    str
    factors:    pd.DataFrame

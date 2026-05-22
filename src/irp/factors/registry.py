"""Central factor registry: single source of truth for factor metadata.

Each compute module calls register() at import time. Consumers (factor_meta,
normalize) import compute modules to trigger registration, then call all_factors()
to derive their own derived data structures.
"""
from dataclasses import dataclass


@dataclass(frozen=True)
class FactorDef:
    name: str    # DataFrame column key, e.g. 'pe'
    label: str   # UI display label, e.g. 'P/E'
    pct: bool = False  # format as percentage in UI
    group: str = ''    # 'valuation' | 'profitability' | 'momentum' | 'leverage'


_REGISTRY: dict[str, FactorDef] = {}


def register(name: str, label: str, *, pct: bool = False, group: str = '') -> None:
    """Register a factor definition. Called at module level in each compute file."""
    _REGISTRY[name] = FactorDef(name=name, label=label, pct=pct, group=group)


def all_factors() -> list[FactorDef]:
    """All registered factors in registration order."""
    return list(_REGISTRY.values())

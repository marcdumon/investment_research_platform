"""Shared factor metadata — labels, formatting sets, dropdown options.

Derived from the central factor registry. Import this module anywhere that
needs factor labels or dropdown options; do NOT hardcode factor lists elsewhere.
"""
import irp.factors.valuation      # noqa: F401 — populate registry
import irp.factors.profitability  # noqa: F401
import irp.factors.momentum       # noqa: F401
import irp.factors.leverage       # noqa: F401
from irp.factors.registry import all_factors

_all = all_factors()

FACTOR_LABELS: dict[str, str]  = {f.name: f.label for f in _all}
PCT_FACTORS: frozenset[str]    = frozenset(f.name for f in _all if f.pct)
FACTOR_OPTIONS: list[dict]     = [{'label': f.label, 'value': f.name} for f in _all]

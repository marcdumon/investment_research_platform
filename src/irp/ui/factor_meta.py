"""Shared factor metadata — labels, formatting sets, dropdown options.

Derived from the central factor registry. Import this module anywhere that
needs factor labels or dropdown options; do NOT hardcode factor lists elsewhere.
"""
from irp.factors.registry import FactorDef, all_factors


def _label_with_unit(f: FactorDef) -> str:
    if f.pct:
        return f'{f.label} (%)'
    if f.name == 'piotroski_fscore':
        return f'{f.label} (0–9)'
    if f.name == 'rand':
        return f.label
    if f.group in ('size', 'fundamentals'):
        return f.label  # already contains ($B)
    return f'{f.label} (x)'


_all = all_factors()

FACTOR_LABELS: dict[str, str]  = {f.name: _label_with_unit(f) for f in _all}
PCT_FACTORS: frozenset[str]    = frozenset(f.name for f in _all if f.pct)
FACTOR_OPTIONS: list[dict]     = [{'label': _label_with_unit(f), 'value': f.name} for f in _all]

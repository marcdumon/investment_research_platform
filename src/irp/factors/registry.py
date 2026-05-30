"""Central factor registry: single source of truth for factor metadata."""

from dataclasses import dataclass

_REGISTRY: dict[str, 'FactorDef'] = {}
_populated = False


@dataclass(frozen=True)
class FactorDef:
    name: str
    label: str
    pct: bool = False
    group: str = ''
    positive_only: bool = (
        False  # hint: meaningful only when positive; callers may filter
    )


def _register(
    name: str,
    label: str,
    *,
    pct: bool = False,
    group: str = '',
    positive_only: bool = False,
) -> None:
    _REGISTRY[name] = FactorDef(
        name=name,
        label=label,
        pct=pct,
        group=group,
        positive_only=positive_only,
    )


def _populate() -> None:
    """Import all factor modules to trigger their _register() calls."""
    global _populated
    if _populated:
        return
    import irp.factors.growth  # noqa: F401
    import irp.factors.leverage  # noqa: F401
    import irp.factors.momentum  # noqa: F401
    import irp.factors.piotroski  # noqa: F401
    import irp.factors.profitability  # noqa: F401
    import irp.factors.technical  # noqa: F401
    import irp.factors.valuation  # noqa: F401

    _populated = True


def _all_factors() -> list[FactorDef]:
    """All registered factors in registration order."""
    _populate()
    return list(_REGISTRY.values())

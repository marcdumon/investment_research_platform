"""Project-wide string enums.

`StrEnum` members behave as strings, so existing `'A'`/`'Q'` literals
remain compatible while new code can use `Variant.A` / `Variant.Q` for
type-safety and self-documenting call sites.

Existing `Literal['A', 'Q']` annotations stay valid; migration is
opt-in. Pages can build dropdown options from these enums to avoid
hand-rolled label/value dict literals.
"""
from enum import StrEnum


class Variant(StrEnum):
    """SimFin fundamentals variant: annual vs quarterly."""
    A = 'A'
    Q = 'Q'


class RebalanceFreq(StrEnum):
    """Backtest rebalance schedule."""
    QUARTERLY = 'Q'
    ANNUAL = 'A'


class MarketFilter(StrEnum):
    """Backtest market filter dropdown values."""
    ALL = ''
    STOCKS = 'stocks'
    ETFS = 'etfs'


class PriceMetric(StrEnum):
    """Column selector for the price panel."""
    CLOSE = 'Close'
    VOLUME = 'Volume'


class Feed(StrEnum):
    """Ingestion feed kind."""
    BULK = 'bulk'
    UPDATE = 'update'


class Source(StrEnum):
    """Price data source on the /ticker page."""
    STOOQ = 'stooq'
    YAHOO = 'yahoo'


def _dropdown_options(enum_cls: type[StrEnum]) -> list[dict[str, str]]:
    """Build a Dash dropdown options list from a StrEnum.

    Labels are humanised member names (`SOMETHING_FOO` → `Something foo`).
    """
    return [
        {'label': member.name.replace('_', ' ').title(), 'value': member.value}
        for member in enum_cls
    ]

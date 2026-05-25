"""UI service layer.

Pages should import only from this package, not from `irp.factors`,
`irp.query`, or `irp.panel` directly. Each service module wraps an
external dependency, normalises any schema quirks, and centralises
cross-page logic (universe filtering, watchlist resolution, etc.).
"""
from irp.ui.services import (
    backtest_service,
    factors_service,
    price_service,
    universe_service,
    watchlist_service,
)

__all__ = [
    'backtest_service',
    'factors_service',
    'price_service',
    'universe_service',
    'watchlist_service',
]

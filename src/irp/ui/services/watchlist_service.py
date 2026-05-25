"""Watchlist persistence for the UI layer.

Thin passthrough to `irp.query.watchlists`. Lives in the services layer so
pages have one import surface and we can add caching/validation later
without touching callers.
"""
from irp.query.watchlists import (
    delete_watchlist,
    list_watchlists,
    load_watchlist,
    save_watchlist,
)

__all__ = [
    'delete_watchlist',
    'list_watchlists',
    'load_watchlist',
    'save_watchlist',
]

"""Panel-based analytics layer.

Wide (Date × Ticker) and long-format parquet panels materialized from DuckDB.
Pure Polars / pandas computation — no SQL in the hot path.

Public API:
    build_panels()                    # one-time materialization
    load_prices_wide(metric='Close')  # (Date × Ticker) DataFrame
    load_fundamentals(stmt, variant)  # long-format DataFrame
    forward_returns_panel(...)        # vectorized log returns
    cross_section_panel(...)          # factor cross-section at a date
"""
from irp.panel.build import build_panels
from irp.panel.cross_section import cross_section_panel
from irp.panel.load import (
    clear_cache,
    load_fundamentals,
    load_prices_wide,
    panel_root,
)
from irp.panel.returns import forward_returns_panel

__all__ = [
    'build_panels',
    'load_prices_wide',
    'load_fundamentals',
    'panel_root',
    'clear_cache',
    'forward_returns_panel',
    'cross_section_panel',
]

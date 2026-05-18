"""DuckDB-backed row accessors per source.

Split by source for clarity:
    from irp.data.simfin   import fundamentals, statement, companies, universe
    from irp.data.stooq    import prices, markets
    from irp.data.yahoo    import prices, dividends, splits
    from irp.data.catalog  import catalog

`catalog` — per-ticker coverage table across all sources. Rebuild via `uv run irp-catalog`.

Shared helpers (connection, ticker filter) live in `_common`.
"""

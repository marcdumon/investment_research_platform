"""DuckDB-backed row accessors per source.

Split by source for clarity:
    from irp.query.simfin   import fundamentals, statement, companies, universe
    from irp.query.stooq    import prices
    from irp.query.universe import universe
    from irp.query.yahoo    import prices, dividends, splits
    from irp.query.catalog  import catalog

`catalog` — per-ticker coverage table across all sources. Rebuild via `uv run irp-catalog`.

Shared helpers (connection, ticker filter) live in `_common`.
"""

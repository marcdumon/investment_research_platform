"""DuckDB-backed row accessors per source.

Split by source for clarity:
    from irp.data.simfin import fundamentals, statement, companies, universe
    from irp.data.stooq  import prices, markets

Shared helpers (connection, ticker filter, date parsing) live in `_common`.
"""

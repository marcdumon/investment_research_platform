"""Small DuckDB SQL helpers shared by the loader and providers.

Internal: paths are project-controlled, so file paths are interpolated into SQL
(matching DuckDB's idiom for `read_*`/`COPY` table functions, which don't accept
bind parameters for file names).
"""
from pathlib import Path

import duckdb


def reader(path: Path) -> str:
    """A DuckDB table-function call to read `path`, chosen by file extension."""
    return f"read_parquet('{path}')" if path.suffix == '.parquet' else f"read_csv_auto('{path}')"


def copy_to_parquet(con: duckdb.DuckDBPyConnection, select_sql: str, out: Path) -> Path:
    """Materialize `select_sql` to a Parquet file at `out`; return `out`."""
    con.execute(f"COPY ({select_sql}) TO '{out}' (FORMAT PARQUET)")
    return out

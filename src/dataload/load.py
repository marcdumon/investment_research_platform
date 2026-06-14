"""Generic, schema-driven loader. One implementation for every table.

`load_dataset` is the only path provider output reaches the database. It
validates the parquet against the table's `TableSchema` (so a provider cannot
emit drifted columns), creates the table with canonical types, then either
MERGE-upserts (idempotent) or full-replaces per the schema's `mode`.
"""
from pathlib import Path

import duckdb

from dataload.schemas import SCHEMAS, TableSchema


def _parquet_columns(con: duckdb.DuckDBPyConnection, parquet: Path) -> list[str]:
    rows = con.execute('DESCRIBE SELECT * FROM read_parquet(?)', [str(parquet)]).fetchall()
    return [r[0] for r in rows]


def _validate_columns(con: duckdb.DuckDBPyConnection, parquet: Path, schema: TableSchema) -> None:
    """Merge tables must match their schema exactly — the drift guard.

    Replace tables are source-defined (wide fundamentals), so the parquet is
    accepted as-is.
    """
    if schema.mode != 'merge':
        return
    got = set(_parquet_columns(con, parquet))
    expected = set(schema.columns)
    if got != expected:
        missing = sorted(expected - got)
        unexpected = sorted(got - expected)
        raise ValueError(
            f'{schema.name}: parquet columns do not match schema. '
            f'missing={missing} unexpected={unexpected}'
        )


def _ensure_table(con: duckdb.DuckDBPyConnection, schema: TableSchema) -> None:
    """Create the merge table with canonical column types if it does not exist."""
    cols_sql = ', '.join(f'{c} {t}' for c, t in schema.columns.items())
    con.execute(f'CREATE TABLE IF NOT EXISTS {schema.name} ({cols_sql})')


def _merge_into(con: duckdb.DuckDBPyConnection, schema: TableSchema, parquet: Path) -> None:
    reader = f"read_parquet('{parquet}')"
    insert_cols = schema.key + schema.values + schema.extra
    on_clause = ' AND '.join(f't.{c} = s.{c}' for c in schema.key)
    update_set = ', '.join(f'{c} = s.{c}' for c in schema.values)
    insert_cols_sql = ', '.join(insert_cols)
    insert_vals_sql = ', '.join(f's.{c}' for c in insert_cols)
    distinct_key = ', '.join(schema.key)
    con.execute(f"""
        MERGE INTO {schema.name} t
        USING (SELECT DISTINCT ON ({distinct_key}) * FROM {reader}) s
        ON {on_clause}
        WHEN MATCHED THEN UPDATE SET {update_set}
        WHEN NOT MATCHED THEN INSERT ({insert_cols_sql}) VALUES ({insert_vals_sql})
    """)


def _replace_with(con: duckdb.DuckDBPyConnection, schema: TableSchema, parquet: Path) -> None:
    con.execute(f'DROP TABLE IF EXISTS {schema.name}')
    con.execute(f"CREATE TABLE {schema.name} AS SELECT * FROM read_parquet('{parquet}')")


def load_dataset(con: duckdb.DuckDBPyConnection, dataset: str, parquet: Path) -> int:
    """Load one dataset's parquet into the DB via its schema. Returns row count.

    Takes an open connection so the caller controls the write session (one
    session can load several datasets for a provider).
    """
    schema = SCHEMAS[dataset]
    _validate_columns(con, parquet, schema)
    if schema.mode == 'merge':
        _ensure_table(con, schema)
        _merge_into(con, schema, parquet)
    else:
        _replace_with(con, schema, parquet)
    count: int = con.execute(f'SELECT COUNT(*) FROM {schema.name}').fetchone()[0]  # type: ignore[index]
    return count

"""Generic CSV-into-DuckDB MERGE helper.

Idempotent upsert: CREATE TABLE from CSV header if needed, then MERGE on
`key_cols` (DISTINCT to dedupe source rows). Insert columns default to
`key_cols + value_cols + extra_insert_cols`.
"""
from pathlib import Path

import duckdb


def _reader_sql(src: Path) -> str:
    return f"read_parquet('{src}')" if src.suffix == '.parquet' else f"read_csv_auto('{src}')"


def _merge_csv(
    con: duckdb.DuckDBPyConnection,
    table: str,
    src: Path,
    key_cols: list[str],
    value_cols: list[str],
    extra_insert_cols: list[str] | None = None,
) -> None:
    """MERGE rows from `src` (CSV or Parquet, detected by extension) into
    `table`. Creates the table from the source header (no rows) if it does
    not exist.

    Args:
        con: DuckDB connection.
        table: target table name.
        src: source file path (`.csv` or `.parquet`).
        key_cols: columns used in the ON clause (also dedupe key on source).
        value_cols: columns updated on match.
        extra_insert_cols: columns inserted on no-match but not updated on
            match (e.g. `SrcId`, `Src` that are immutable per key).
    """
    extra = extra_insert_cols or []
    insert_cols = key_cols + value_cols + extra
    on_clause = ' AND '.join(f't.{c} = s.{c}' for c in key_cols)
    update_set = ', '.join(f'{c} = s.{c}' for c in value_cols)
    insert_cols_sql = ', '.join(insert_cols)
    insert_vals_sql = ', '.join(f's.{c}' for c in insert_cols)
    distinct_key_sql = ', '.join(key_cols)
    reader = _reader_sql(src)

    con.execute(f"CREATE TABLE IF NOT EXISTS {table} AS SELECT * FROM {reader} LIMIT 0")
    con.execute(f"""
        MERGE INTO {table} t
        USING (
            SELECT DISTINCT ON ({distinct_key_sql}) *
            FROM {reader}
        ) s
        ON {on_clause}
        WHEN MATCHED THEN UPDATE SET {update_set}
        WHEN NOT MATCHED THEN INSERT ({insert_cols_sql})
        VALUES ({insert_vals_sql})
    """)

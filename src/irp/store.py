
import json
from datetime import datetime, timezone
from pathlib import Path

import duckdb
import pandas as pd

import irp.config as _config
from irp.datasets.dataset import Dataset

_META_TABLE = "_irp_datasets"
_INSERTED_AT_COL = "inserted_at"


class Store:
    """
    Persist and retrieve Datasets using DuckDB.

    save()   — full overwrite (CREATE OR REPLACE)
    upsert() — insert, updating existing rows on PK conflict
    Metadata lives in _irp_datasets. Every row gets an inserted_at TIMESTAMP (UTC).
    """

    def __init__(self, db_path: str | Path | None = None) -> None:
        path = db_path or _config.load()["store"]["db_path"]
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        self._db_path = str(path)
        self._init_meta()

    def _conn(self) -> duckdb.DuckDBPyConnection:
        return duckdb.connect(self._db_path)

    def _init_meta(self) -> None:
        with self._conn() as con:
            con.execute(f"""
                CREATE TABLE IF NOT EXISTS "{_META_TABLE}" (
                    name             VARCHAR PRIMARY KEY,
                    source           VARCHAR,
                    schema_json      VARCHAR,
                    captured_at      TIMESTAMP,
                    primary_key_json VARCHAR
                )
            """)
            existing_cols = {r[0] for r in con.execute(f'DESCRIBE "{_META_TABLE}"').fetchall()}
            if "primary_key_json" not in existing_cols:
                con.execute(f'ALTER TABLE "{_META_TABLE}" ADD COLUMN primary_key_json VARCHAR')

    def _stamp(self, df: pd.DataFrame) -> pd.DataFrame:
        df = df.copy()
        df[_INSERTED_AT_COL] = datetime.now(timezone.utc)
        return df

    def _table_pk(self, con: duckdb.DuckDBPyConnection, tbl: str) -> list[str]:
        rows = con.execute(
            "SELECT constraint_column_names FROM duckdb_constraints() "
            "WHERE table_name = ? AND constraint_type = 'PRIMARY KEY'",
            [tbl],
        ).fetchall()
        return rows[0][0] if rows else []

    def _ensure_table(
        self,
        con: duckdb.DuckDBPyConnection,
        tbl: str,
        primary_key: list[str],
    ) -> None:
        existing = {r[0] for r in con.execute("SHOW TABLES").fetchall()}
        if tbl not in existing:
            cols = con.execute("DESCRIBE SELECT * FROM _df").fetchall()
            col_defs = ", ".join(f'"{c[0]}" {c[1]}' for c in cols)
            pk = ", ".join(f'"{c}"' for c in primary_key)
            con.execute(f'CREATE TABLE "{tbl}" ({col_defs}, PRIMARY KEY ({pk}))')
        elif self._table_pk(con, tbl) != primary_key:
            pk = ", ".join(f'"{c}"' for c in primary_key)
            con.execute(f'ALTER TABLE "{tbl}" ADD PRIMARY KEY ({pk})')

    def _upsert_meta(
        self,
        con: duckdb.DuckDBPyConnection,
        tbl: str,
        dataset: Dataset,
        extra_schema: dict[str, str] | None = None,
        primary_key: list[str] | None = None,
    ) -> None:
        schema = {**dataset.schema, **(extra_schema or {})}
        con.execute(
            f'INSERT OR REPLACE INTO "{_META_TABLE}" VALUES (?, ?, ?, ?, ?)',
            [tbl, dataset.source, json.dumps(schema), dataset.captured_at, json.dumps(primary_key)],
        )

    def save(self, dataset: Dataset, *, table: str | None = None) -> None:
        """Full overwrite — CREATE OR REPLACE."""
        tbl = table or dataset.name
        df = self._stamp(dataset.data)
        with self._conn() as con:
            con.register("_df", df)
            con.execute(f'CREATE OR REPLACE TABLE "{tbl}" AS SELECT * FROM _df')
            self._upsert_meta(con, tbl, dataset, {_INSERTED_AT_COL: str(df[_INSERTED_AT_COL].dtype)})

    def upsert(
        self,
        dataset: Dataset,
        *,
        table: str | None = None,
        primary_key: list[str],
    ) -> None:
        """Insert rows, updating existing ones on PK conflict."""
        tbl = table or dataset.name
        df = self._stamp(dataset.data)
        with self._conn() as con:
            con.register("_df", df)
            self._ensure_table(con, tbl, primary_key)
            update_cols = [c for c in df.columns if c not in primary_key]
            set_clause = ", ".join(f'"{c}" = excluded."{c}"' for c in update_cols)
            pk_sql = ", ".join(f'"{c}"' for c in primary_key)
            con.execute(f"""
                INSERT INTO "{tbl}" SELECT * FROM _df
                ON CONFLICT ({pk_sql}) DO UPDATE SET {set_clause}
            """)
            self._upsert_meta(con, tbl, dataset, {_INSERTED_AT_COL: str(df[_INSERTED_AT_COL].dtype)}, primary_key)

    def load(self, name: str) -> Dataset:
        """Read a dataset by name."""
        with self._conn() as con:
            row = con.execute(
                f'SELECT source, schema_json, captured_at FROM "{_META_TABLE}" WHERE name = ?',
                [name],
            ).fetchone()
            if row is None:
                raise KeyError(f"Dataset '{name}' not found in store")
            source, schema_json, captured_at = row
            df: pd.DataFrame = con.execute(f'SELECT * FROM "{name}"').df()
        return Dataset(
            name=name,
            data=df,
            schema=json.loads(schema_json),
            source=source,
            captured_at=captured_at if isinstance(captured_at, datetime)
                        else datetime.fromisoformat(str(captured_at)),
        )

    def delete(self, name: str) -> None:
        """Drop a dataset and its metadata."""
        with self._conn() as con:
            con.execute(f'DROP TABLE IF EXISTS "{name}"')
            con.execute(f'DELETE FROM "{_META_TABLE}" WHERE name = ?', [name])

    def exists(self, name: str) -> bool:
        with self._conn() as con:
            return con.execute(
                f'SELECT 1 FROM "{_META_TABLE}" WHERE name = ?', [name]
            ).fetchone() is not None

    def max_date(
        self,
        table: str,
        date_col: str,
        *,
        filter_col: str | None = None,
        filter_val: str | None = None,
    ) -> str | None:
        """Return MAX(date_col) from table, optionally filtered. None if missing or empty."""
        if not self.exists(table):
            return None
        with self._conn() as con:
            if filter_col is not None:
                row = con.execute(
                    f'SELECT MAX("{date_col}") FROM "{table}" WHERE "{filter_col}" = ?',
                    [filter_val],
                ).fetchone()
            else:
                row = con.execute(
                    f'SELECT MAX("{date_col}") FROM "{table}"',
                ).fetchone()
            result = row[0] if row is not None else None
        return str(result)[:10] if result is not None else None

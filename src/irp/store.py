import json
from datetime import datetime, timezone
from pathlib import Path

import duckdb
import pandas as pd

import irp.config as _config
from irp.datasets.dataset import Dataset

_META_TABLE = "_irp_datasets"


class Store:
    """
    Persist and retrieve Datasets using DuckDB.

    Each dataset is stored as a DuckDB table named after dataset.name.
    Shared tables (e.g. "prices") are supported via partition_col.
    Metadata (source, schema, captured_at) lives in _irp_datasets.
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
                    name        VARCHAR PRIMARY KEY,
                    source      VARCHAR,
                    schema_json VARCHAR,
                    captured_at TIMESTAMP
                )
            """)

    def save(
        self,
        dataset: Dataset,
        *,
        table: str | None = None,
        partition_col: str | None = None,
    ) -> None:
        """Write dataset to DuckDB.

        table: override table name (default: dataset.name)
        partition_col: column to partition on; existing rows for that value
                       are deleted before insert, enabling upsert per ticker.
        """
        tbl = table or dataset.name
        df = dataset.data
        with self._conn() as con:
            con.register("_df", df)
            if partition_col is not None:
                partition_val = df[partition_col].iloc[0]
                existing = [r[0] for r in con.execute("SHOW TABLES").fetchall()]
                if tbl not in existing:
                    con.execute(f'CREATE TABLE "{tbl}" AS SELECT * FROM _df WHERE false')
                con.execute(
                    f'DELETE FROM "{tbl}" WHERE "{partition_col}" = ?',
                    [partition_val],
                )
                con.execute(f'INSERT INTO "{tbl}" SELECT * FROM _df')
            else:
                con.execute(f'CREATE OR REPLACE TABLE "{tbl}" AS SELECT * FROM _df')
            con.execute(f"""
                INSERT OR REPLACE INTO "{_META_TABLE}" VALUES (?, ?, ?, ?)
            """, [
                tbl,
                dataset.source,
                json.dumps(dataset.schema),
                dataset.captured_at,
            ])

    def merge(
        self,
        dataset: Dataset,
        *,
        table: str | None = None,
        delete_col: str,
        delete_values: list,
    ) -> None:
        """Delete rows where delete_col IN delete_values, then insert dataset rows.

        Used to upsert a slice of a shared table (e.g. quarterly periods)
        without touching other slices (e.g. annual rows).
        """
        tbl = table or dataset.name
        df = dataset.data
        placeholders = ",".join(["?" for _ in delete_values])
        with self._conn() as con:
            con.register("_df", df)
            existing = [r[0] for r in con.execute("SHOW TABLES").fetchall()]
            if tbl not in existing:
                con.execute(f'CREATE TABLE "{tbl}" AS SELECT * FROM _df WHERE false')
            con.execute(
                f'DELETE FROM "{tbl}" WHERE "{delete_col}" IN ({placeholders})',
                delete_values,
            )
            con.execute(f'INSERT INTO "{tbl}" SELECT * FROM _df')
            con.execute(f"""
                INSERT OR REPLACE INTO "{_META_TABLE}" VALUES (?, ?, ?, ?)
            """, [
                tbl,
                dataset.source,
                json.dumps(dataset.schema),
                dataset.captured_at,
            ])

    def load(self, name: str) -> Dataset:
        """Read a dataset by name (full table)."""
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

    def load_partition(self, table: str, partition_col: str, partition_val: str) -> Dataset:
        """Read rows for a single partition value from a shared table."""
        with self._conn() as con:
            row = con.execute(
                f'SELECT source, schema_json, captured_at FROM "{_META_TABLE}" WHERE name = ?',
                [table],
            ).fetchone()
            if row is None:
                raise KeyError(f"Table '{table}' not found in store")
            source, schema_json, captured_at = row
            df: pd.DataFrame = con.execute(
                f'SELECT * FROM "{table}" WHERE "{partition_col}" = ?',
                [partition_val],
            ).df()

        return Dataset(
            name=table,
            data=df,
            schema=json.loads(schema_json),
            source=source,
            captured_at=captured_at if isinstance(captured_at, datetime)
                        else datetime.fromisoformat(str(captured_at)),
        )

    def list(self) -> list[str]:
        """Return names of all saved datasets."""
        with self._conn() as con:
            rows = con.execute(f'SELECT name FROM "{_META_TABLE}" ORDER BY name').fetchall()
        return [r[0] for r in rows]

    def delete(self, name: str) -> None:
        """Drop a dataset and its metadata."""
        with self._conn() as con:
            con.execute(f'DROP TABLE IF EXISTS "{name}"')
            con.execute(f'DELETE FROM "{_META_TABLE}" WHERE name = ?', [name])

    def exists(self, name: str) -> bool:
        return name in self.list()

    def exists_partition(self, table: str, partition_col: str, partition_val: str) -> bool:
        """Check if rows exist for a partition value in a shared table."""
        if not self.exists(table):
            return False
        with self._conn() as con:
            count = con.execute(
                f'SELECT COUNT(*) FROM "{table}" WHERE "{partition_col}" = ?',
                [partition_val],
            ).fetchone()[0]
        return count > 0

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
    Metadata (source, schema, captured_at) lives in _irp_datasets.
    """

    def __init__(self, db_path: str | Path | None = None) -> None:
        path = db_path or _config.load()["store"]["db_path"]
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        self._db_path = str(path)
        self._init_meta()

    # -- connection (short-lived per operation) --

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

    # -- public API --

    def save(self, dataset: Dataset) -> None:
        """Write dataset to DuckDB. Overwrites if name exists."""
        df = dataset.data
        with self._conn() as con:
            con.register("_df", df)
            con.execute(f'CREATE OR REPLACE TABLE "{dataset.name}" AS SELECT * FROM _df')
            con.execute(f"""
                INSERT OR REPLACE INTO "{_META_TABLE}" VALUES (?, ?, ?, ?)
            """, [
                dataset.name,
                dataset.source,
                json.dumps(dataset.schema),
                dataset.captured_at,
            ])

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

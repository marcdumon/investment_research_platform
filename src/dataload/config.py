"""Optional TOML config loader → IngestContext, for standalone CLI use.

This is a *convenience* for running `dataload` on its own; the library core never
depends on it. A host application (e.g. irp) builds its own IngestContext and
ignores this module entirely.

Expected TOML shape::

    [database]
    path = "/path/to/db.duckdb"
    [data]
    root_dir = "/path/to/data"
    [providers.stooq]
    raw_dir = "stooq/raw"
    processed_dir = "stooq/processed"
    bulk_files = ["d_us_txt.zip"]
    update_file = "data_d.txt"
    # ... providers.yahoo, providers.simfin similarly
"""
import tomllib
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path

import duckdb

from dataload.context import IngestContext


def load_context(config_path: str | Path) -> IngestContext:
    """Build an IngestContext from a TOML file with a plain DuckDB connect factory."""
    with open(config_path, 'rb') as f:
        raw = tomllib.load(f)
    db_path = Path(raw['database']['path'])

    @contextmanager
    def connect() -> Iterator[duckdb.DuckDBPyConnection]:
        con = duckdb.connect(str(db_path))
        try:
            yield con
        finally:
            con.close()

    return IngestContext(
        data_root=Path(raw['data']['root_dir']),
        provider_cfg=raw.get('providers', {}),
        connect=connect,
    )

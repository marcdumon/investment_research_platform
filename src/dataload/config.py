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
import os
import tomllib
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path

import duckdb
from dotenv import load_dotenv

from dataload.context import IngestContext


def load_context(config_path: str | Path) -> IngestContext:
    """Build an IngestContext from a TOML file with a plain DuckDB connect factory.

    Secrets (e.g. the SimFin API key) come from the environment / a `.env` file, never
    the TOML — matching how the host app handles them. Set `SIMFIN_API_KEY` to use SimFin.
    """
    load_dotenv()
    with open(config_path, 'rb') as f:
        raw = tomllib.load(f)
    db_path = Path(raw['database']['path'])

    providers = raw.get('providers', {})
    if 'simfin' in providers and 'SIMFIN_API_KEY' in os.environ:
        providers['simfin']['api_key'] = os.environ['SIMFIN_API_KEY']

    @contextmanager
    def connect() -> Iterator[duckdb.DuckDBPyConnection]:
        con = duckdb.connect(str(db_path))
        try:
            yield con
        finally:
            con.close()

    return IngestContext(
        data_root=Path(raw['data']['root_dir']),
        provider_cfg=providers,
        connect=connect,
    )

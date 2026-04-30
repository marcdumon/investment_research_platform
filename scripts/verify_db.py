"""Print row counts for all tables in the store."""
from pathlib import Path

import duckdb

import irp.config as _config

db_path = _config.load()["store"]["db_path"]

with duckdb.connect(db_path, read_only=True) as con:
    tables = [r[0] for r in con.execute("SELECT name FROM _irp_datasets ORDER BY name").fetchall()]
    for t in tables:
        n = con.execute(f'SELECT COUNT(*) FROM "{t}"').fetchone()
        print(f"  {t:<20} {(n[0] if n else 0):>14,}")

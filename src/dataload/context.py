"""IngestContext — the injected configuration handle.

Replaces the old import-time global `config` singleton. A host application
builds one of these and passes it to `dataload.run`; a standalone consumer
supplies its own paths, provider settings, and DB connection factory.
"""
from collections.abc import Callable
from contextlib import AbstractContextManager
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import duckdb


def _never_cancelled() -> bool:
    return False


@dataclass(frozen=True, slots=True)
class IngestContext:
    """Everything providers + the loader need, with zero global state.

    Args:
        data_root: base directory under which provider raw/processed dirs live.
        provider_cfg: per-provider settings (paths, rate limits, keys, etc.).
        connect: factory yielding a writable DuckDB connection as a context
            manager, so the host controls transaction + locking policy.
        is_cancelled: cooperative-cancellation probe for long fetches; defaults
            to never-cancelled. A host (e.g. a UI) injects its own.
        threads: cap DuckDB worker threads on every connection; None (default)
            leaves DuckDB's own default (one per core).
    """
    data_root: Path
    provider_cfg: dict[str, dict[str, Any]]
    connect: Callable[[], AbstractContextManager[duckdb.DuckDBPyConnection]]
    is_cancelled: Callable[[], bool] = _never_cancelled
    threads: int | None = None

    def cfg(self, provider: str) -> dict[str, Any]:
        return self.provider_cfg[provider]

    def configure(self, con: duckdb.DuckDBPyConnection) -> None:
        """Apply context-wide DuckDB settings (currently the optional thread cap)."""
        if self.threads is not None:
            con.execute(f'SET threads = {self.threads}')

    def duck(self) -> duckdb.DuckDBPyConnection:
        """A fresh in-memory DuckDB connection with context settings applied."""
        con = duckdb.connect()
        self.configure(con)
        return con

    def raw_dir(self, provider: str) -> Path:
        return self.data_root / self.provider_cfg[provider]['raw_dir']

    def processed_dir(self, provider: str) -> Path:
        return self.data_root / self.provider_cfg[provider]['processed_dir']

"""IngestContext: injected config + DB connect factory (no global singleton)."""
from contextlib import contextmanager

import duckdb

from dataload.context import IngestContext


def _ctx(tmp_path) -> IngestContext:
    @contextmanager
    def connect():
        con = duckdb.connect(str(tmp_path / 'db.duckdb'))
        try:
            yield con
        finally:
            con.close()

    return IngestContext(
        data_root=tmp_path,
        provider_cfg={'yahoo': {'raw_dir': 'yahoo/raw', 'processed_dir': 'yahoo/processed'}},
        connect=connect,
    )


def test_raw_and_processed_dirs(tmp_path) -> None:
    ctx = _ctx(tmp_path)
    assert ctx.raw_dir('yahoo') == tmp_path / 'yahoo' / 'raw'
    assert ctx.processed_dir('yahoo') == tmp_path / 'yahoo' / 'processed'


def test_cfg_accessor(tmp_path) -> None:
    assert _ctx(tmp_path).cfg('yahoo')['raw_dir'] == 'yahoo/raw'


def test_connect_yields_usable_connection(tmp_path) -> None:
    with _ctx(tmp_path).connect() as con:
        assert con.execute('SELECT 42').fetchone()[0] == 42


def test_default_is_not_cancelled(tmp_path) -> None:
    """Standalone use needs no cancellation; the host injects its own check."""
    assert _ctx(tmp_path).is_cancelled() is False

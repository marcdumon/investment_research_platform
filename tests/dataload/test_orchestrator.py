"""Orchestrator: capability gating + generic load wiring."""
from contextlib import contextmanager
from pathlib import Path

import duckdb
import pandas as pd
import pytest

from dataload.context import IngestContext
from dataload.orchestrator import run
from dataload.providers import PROVIDERS, make_provider
from dataload.providers.base import Capability


def _ctx(tmp_path) -> IngestContext:
    @contextmanager
    def connect():
        con = duckdb.connect(str(tmp_path / 'db.duckdb'))
        try:
            yield con
        finally:
            con.close()

    return IngestContext(tmp_path, {}, connect)


class FakeProvider:
    """Records the datasets it was asked to produce; loads nothing."""

    def __init__(self, name: str, caps: dict[str, Capability]) -> None:
        self.name = name
        self._caps = caps
        self.produce_calls: list[tuple[list[str], bool]] = []

    def capabilities(self) -> dict[str, Capability]:
        return self._caps

    def produce(self, ctx, datasets, *, incremental) -> dict[str, Path]:
        self.produce_calls.append((list(datasets), incremental))
        return {}

    def cleanup(self, ctx) -> None:
        pass


_MIXED = {'prices': Capability(incremental=True), 'companies': Capability(incremental=False)}


def test_incremental_skips_non_incremental_datasets(tmp_path) -> None:
    fp = FakeProvider('x', _MIXED)
    run(_ctx(tmp_path), [fp], datasets=['prices', 'companies'], incremental=True)
    assert fp.produce_calls == [(['prices'], True)]


def test_full_run_includes_all_requested(tmp_path) -> None:
    fp = FakeProvider('x', _MIXED)
    run(_ctx(tmp_path), [fp], datasets=['prices', 'companies'], incremental=False)
    assert fp.produce_calls == [(['prices', 'companies'], False)]


def test_datasets_none_defaults_to_all_capabilities(tmp_path) -> None:
    fp = FakeProvider('x', _MIXED)
    run(_ctx(tmp_path), [fp])
    assert sorted(fp.produce_calls[0][0]) == ['companies', 'prices']


def test_unknown_requested_dataset_is_ignored(tmp_path) -> None:
    fp = FakeProvider('x', {'prices': Capability(incremental=True)})
    run(_ctx(tmp_path), [fp], datasets=['prices', 'bogus'])
    assert fp.produce_calls == [(['prices'], False)]


def test_provider_with_nothing_to_do_is_not_produced(tmp_path) -> None:
    fp = FakeProvider('x', {'companies': Capability(incremental=False)})
    summary = run(_ctx(tmp_path), [fp], datasets=['companies'], incremental=True)
    assert fp.produce_calls == []
    assert summary['x'] == {}


def test_run_loads_produced_parquet_into_db(tmp_path) -> None:
    parquet = tmp_path / 'prices.parquet'
    pd.DataFrame([['AAPL', '2024-01-02', 1.0, 2.0, 0.5, 1.5, 100, 'stooq', 'AAPL.US']],
                 columns=['Ticker', 'Date', 'Open', 'High', 'Low', 'Close', 'Volume', 'Src', 'SrcId']
                 ).to_parquet(parquet, index=False)

    class RealishProvider:
        name = 'p'

        def capabilities(self):
            return {'prices': Capability(incremental=False)}

        def produce(self, ctx, datasets, *, incremental):
            return {'prices': parquet}

        def cleanup(self, ctx):
            pass

    ctx = _ctx(tmp_path)
    summary = run(ctx, [RealishProvider()], datasets=['prices'])
    assert summary['p']['prices'] == 1
    with ctx.connect() as con:
        assert con.execute('SELECT COUNT(*) FROM prices').fetchone()[0] == 1


def test_make_provider_resolves_registry() -> None:
    from dataload.providers.stooq import StooqProvider
    assert isinstance(make_provider('stooq'), StooqProvider)
    assert set(PROVIDERS) == {'stooq', 'yahoo', 'simfin'}


def test_make_provider_unknown_raises() -> None:
    with pytest.raises(ValueError, match='nope'):
        make_provider('nope')

"""dataload — standalone, schema-enforced market-data ingestion.

Zero dependencies on any host application: all configuration (paths, DB
connection, provider settings) is injected via an `IngestContext`. A central
schema registry (`schemas.SCHEMAS`) is the single source of truth for every
target table, and a generic loader writes provider output through it, so the
column/key drift that plagued the previous per-provider design cannot recur.

Typical use::

    from dataload import IngestContext, make_provider, run
    ctx = IngestContext(data_root, provider_cfg, connect)
    run(ctx, [make_provider('stooq'), make_provider('yahoo')], datasets=['prices'])
"""
from dataload import universe
from dataload.context import IngestContext
from dataload.orchestrator import run
from dataload.providers import PROVIDERS, Capability, Provider, make_provider
from dataload.schemas import SCHEMAS, TableSchema

__all__ = [
    'PROVIDERS',
    'SCHEMAS',
    'Capability',
    'IngestContext',
    'Provider',
    'TableSchema',
    'make_provider',
    'run',
    'universe',
]

"""Provider interface: a cohesive produce() + a capability map.

A provider owns acquire + normalize internally (with whatever per-source state
it needs) and returns, per dataset, a canonical parquet that conforms to the
schema registry. The generic loader (`dataload.load`) does the rest. There is
no forced bulk/update method that some providers cannot honour — incremental is
a declared capability, passed as a flag.
"""
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Protocol, runtime_checkable

from dataload.context import IngestContext


@dataclass(frozen=True, slots=True)
class Capability:
    """What a provider can do for one dataset."""
    incremental: bool = False


@runtime_checkable
class Provider(Protocol):
    name: str

    def capabilities(self) -> dict[str, Capability]:
        """Map of dataset name -> Capability this provider can produce."""
        ...

    def produce(self, ctx: IngestContext, datasets: Sequence[str], *, incremental: bool) -> dict[str, Path]:
        """Acquire + normalize the requested datasets. Returns {dataset: parquet_path}."""
        ...

    def cleanup(self, ctx: IngestContext) -> None:
        """Delete intermediate artifacts, keeping raw inputs + markers + DB."""
        ...

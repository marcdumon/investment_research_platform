from dataclasses import dataclass, field

from irp.datasets.dataset import Dataset
from irp.store import Store
from irp.transforms.base import Transformer


@dataclass
class _Step:
    transformer: Transformer
    name: str
    force: bool = False
    save: bool = True
    table: str | None = None
    partition_col: str | None = None


class Pipeline:
    """
    Chain transforms with automatic caching in the Store.

    Each step is saved under its name. On re-run, cached steps are
    skipped unless force=True or Pipeline(force=True).

    Use table + partition_col to write into a shared table keyed by a
    column value (e.g. table="prices", partition_col="ticker").
    """

    def __init__(self, store: Store, force: bool = False) -> None:
        self._store = store
        self._force = force
        self._steps: list[_Step] = []

    def step(
        self,
        transformer: Transformer,
        name: str,
        force: bool = False,
        save: bool = True,
        table: str | None = None,
        partition_col: str | None = None,
    ) -> "Pipeline":
        self._steps.append(
            _Step(transformer, name, force or self._force, save, table, partition_col)
        )
        return self

    def run(self, dataset: Dataset) -> Dataset:
        current = dataset
        for step in self._steps:
            tbl = step.table or step.name

            if step.save and not step.force:
                if step.partition_col is not None:
                    partition_val = current.data[step.partition_col].iloc[0]
                    cached = self._store.exists_partition(tbl, step.partition_col, partition_val)
                else:
                    cached = self._store.exists(tbl)
            else:
                cached = False

            if cached:
                if step.partition_col is not None:
                    current = self._store.load_partition(tbl, step.partition_col, partition_val)
                else:
                    current = self._store.load(tbl)
                continue

            current = step.transformer.transform(current)
            current = Dataset(
                name=tbl,
                data=current.data,
                schema=current.schema,
                source=current.source,
                captured_at=current.captured_at,
            )
            if step.save:
                self._store.save(current, table=tbl, partition_col=step.partition_col)

        return current

    def reset(self, name: str | None = None) -> None:
        """Delete one cached step (or all if name=None)."""
        if name:
            self._store.delete(name)
        else:
            for step in self._steps:
                self._store.delete(step.table or step.name)

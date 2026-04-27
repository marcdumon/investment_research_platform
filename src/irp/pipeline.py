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


class Pipeline:
    """
    Chain transforms with automatic caching in the Store.

    Each step is saved under its name. On re-run, cached steps are
    skipped unless force=True or Pipeline(force=True).
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
    ) -> "Pipeline":
        self._steps.append(_Step(transformer, name, force or self._force, save))
        return self

    def run(self, dataset: Dataset) -> Dataset:
        current = dataset
        for step in self._steps:
            if not step.force and step.save and self._store.exists(step.name):
                current = self._store.load(step.name)
                continue
            current = step.transformer.transform(current)
            current = Dataset(
                name=step.name,
                data=current.data,
                schema=current.schema,
                source=current.source,
                captured_at=current.captured_at,
            )
            if step.save:
                self._store.save(current)
        return current

    def reset(self, name: str | None = None) -> None:
        """Delete one cached step (or all if name=None)."""
        if name:
            self._store.delete(name)
        else:
            for step in self._steps:
                self._store.delete(step.name)

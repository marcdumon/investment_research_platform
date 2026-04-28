from irp.datasets.dataset import Dataset
from irp.store import Store
from irp.transforms.base import Transformer


class CachingTransformer(Transformer):
    """
    Wraps any Transformer. On transform():
    - if name exists in store and force=False: return cached Dataset
    - otherwise: run transformer, save result under name, return it
    """

    def __init__(
        self,
        transformer: Transformer,
        store: Store,
        name: str,
        force: bool = False,
    ) -> None:
        self.transformer = transformer
        self.store = store
        self.name = name
        self.force = force

    def transform(self, dataset: Dataset) -> Dataset:
        if not self.force and self.store.exists(self.name):
            return self.store.load(self.name)

        result = self.transformer.transform(dataset)
        named = result.evolve(name=self.name)
        self.store.save(named)
        return named

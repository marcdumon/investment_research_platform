
from abc import ABC, abstractmethod

from irp.datasets.dataset import Dataset


class Transformer(ABC):
    @abstractmethod
    def transform(self, dataset: Dataset) -> Dataset: ...

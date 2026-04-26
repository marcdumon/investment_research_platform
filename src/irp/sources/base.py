
from abc import ABC, abstractmethod

from irp.datasets.dataset import Dataset


class BaseSource(ABC):
    @abstractmethod
    def fetch(self, **kwargs) -> Dataset: ...

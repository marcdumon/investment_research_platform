from __future__ import annotations

from abc import ABC, abstractmethod

import pandas as pd

from irp.datasets.dataset import Dataset


class Feature(ABC):
    name: str
    required_datasets: list[str]

    @abstractmethod
    def compute(self, datasets: dict[str, Dataset]) -> pd.Series | pd.DataFrame: ...

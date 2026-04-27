import pandas as pd

from irp.datasets.dataset import Dataset
from irp.transforms.base import Transformer


class DateParser(Transformer):
    def __init__(self, col: str = "date") -> None:
        self.col = col

    def transform(self, dataset: Dataset) -> Dataset:
        df = dataset.data.copy()
        df[self.col] = pd.to_datetime(df[self.col])
        schema = {**dataset.schema, self.col: "datetime64[ns]"}
        return Dataset(
            name=dataset.name,
            data=df,
            schema=schema,
            source=dataset.source,
        )

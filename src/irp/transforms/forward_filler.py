
import pandas as pd

from irp.datasets.dataset import Dataset
from irp.transforms.base import Transformer


class ForwardFiller(Transformer):
    def __init__(
        self,
        date_col: str = "date",
        ticker_col: str = "ticker",
        target_dates: pd.DatetimeIndex | None = None,
        limit: int | None = None,
    ) -> None:
        self.date_col = date_col
        self.ticker_col = ticker_col
        self.target_dates = target_dates
        self.limit = limit

    def transform(self, dataset: Dataset) -> Dataset:
        df = dataset.data.copy()
        value_cols = [c for c in df.columns if c not in (self.date_col, self.ticker_col)]

        wide = df.pivot(index=self.date_col, columns=self.ticker_col, values=value_cols)

        target = self.target_dates
        if target is None:
            target = pd.date_range(wide.index.min(), wide.index.max(), freq="D")

        wide = wide.reindex(target).ffill(limit=self.limit)

        long = wide.stack(level=self.ticker_col, future_stack=True).reset_index()
        long = long.rename(columns={"level_0": self.date_col})

        return dataset.evolve(
            data=long,
            schema={c: str(long[c].dtype) for c in long.columns},
        )

from typing import Literal

import pandas as pd

from irp.datasets.dataset import Dataset
from irp.transforms.base import Transformer


class DateAligner(Transformer):
    def __init__(
        self,
        target_dates: pd.DatetimeIndex,
        date_col: str = "date",
        ticker_col: str | None = "ticker",
        method: Literal["ffill", "bfill", "nearest", "none"] = "ffill",
    ) -> None:
        self.target_dates = target_dates
        self.date_col = date_col
        self.ticker_col = ticker_col
        self.method = method

    def transform(self, dataset: Dataset) -> Dataset:
        df = dataset.data.copy()

        if self.ticker_col and self.ticker_col in df.columns:
            aligned = (
                df.groupby(self.ticker_col, group_keys=False)
                .apply(self._align_group, include_groups=False)
            )
        else:
            aligned = self._align_group(df)

        return Dataset(
            name=dataset.name,
            data=aligned.reset_index(drop=True),
            schema=dataset.schema,
            source=dataset.source,
        )

    def _align_group(self, group: pd.DataFrame) -> pd.DataFrame:
        group = group.set_index(self.date_col) if self.date_col in group.columns else group
        group = group.reindex(self.target_dates)
        if self.method != "none":
            group = getattr(group, self.method)()
        group.index.name = self.date_col
        return group.reset_index()

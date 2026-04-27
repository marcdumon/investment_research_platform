import numpy as np
import pandas as pd

from irp.datasets.dataset import Dataset
from irp.transforms.base import Transformer


class Cleaner(Transformer):
    def __init__(
        self,
        drop_all_nan: bool = True,
        clip_percentile: float | None = None,
    ) -> None:
        self.drop_all_nan = drop_all_nan
        self.clip_percentile = clip_percentile

    def transform(self, dataset: Dataset) -> Dataset:
        df = dataset.data.copy()
        numeric_cols = df.select_dtypes(include="number").columns.tolist()

        df[numeric_cols] = df[numeric_cols].replace([np.inf, -np.inf], np.nan)

        if self.drop_all_nan and numeric_cols:
            df = df.dropna(subset=numeric_cols, how="all")

        if self.clip_percentile is not None:
            p = self.clip_percentile
            lo = df[numeric_cols].quantile(p)
            hi = df[numeric_cols].quantile(1 - p)
            df[numeric_cols] = df[numeric_cols].clip(lower=lo, upper=hi, axis=1)

        return Dataset(
            name=dataset.name,
            data=df.reset_index(drop=True),
            schema=dataset.schema,
            source=dataset.source,
        )

from typing import Literal

from irp.datasets.dataset import Dataset
from irp.transforms.base import Transformer


class Joiner(Transformer):
    def __init__(
        self,
        right: Dataset,
        on: list[str] | None = None,
        how: Literal["left", "inner", "outer"] = "left",
        suffixes: tuple[str, str] = ("_left", "_right"),
    ) -> None:
        self.right = right
        self.on = on or ["ticker", "date"]
        self.how = how
        self.suffixes = suffixes

    def transform(self, dataset: Dataset) -> Dataset:
        merged = dataset.data.merge(
            self.right.data,
            on=self.on,
            how=self.how,
            suffixes=self.suffixes,
        )
        schema = {c: str(merged[c].dtype) for c in merged.columns}
        return Dataset(
            name=f"{dataset.name}+{self.right.name}",
            data=merged,
            schema=schema,
            source="join",
        )

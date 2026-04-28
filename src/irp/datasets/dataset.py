
from dataclasses import dataclass, field, replace
from datetime import datetime, timezone
from functools import partial
from typing import Self

import pandas as pd


@dataclass(frozen=True)
class Dataset:
    name: str
    data: pd.DataFrame
    schema: dict[str, str]
    source: str
    captured_at: datetime = field(default_factory=partial(datetime.now, timezone.utc))

    def __post_init__(self) -> None:
        object.__setattr__(self, "data", self.data.copy())

    @classmethod
    def from_df(cls, df: pd.DataFrame, *, name: str, source: str) -> Self:
        """Create a Dataset from a DataFrame, inferring schema from column dtypes."""
        return cls(
            name=name,
            data=df,
            schema={c: str(df[c].dtype) for c in df.columns},
            source=source,
        )

    def evolve(self, **overrides) -> Self:
        """Return a copy with fields replaced. Preserves captured_at unless overridden."""
        return replace(self, **overrides)

    def validate(self) -> None:
        missing = set(self.schema) - set(self.data.columns)
        if missing:
            raise ValueError(f"Dataset '{self.name}' missing columns: {missing}")
        mismatched = {
            col: (expected, str(self.data[col].dtype))
            for col, expected in self.schema.items()
            if col in self.data.columns and str(self.data[col].dtype) != expected
        }
        if mismatched:
            details = ", ".join(
                f"{c}: expected {e} got {a}" for c, (e, a) in mismatched.items()
            )
            raise ValueError(f"Dataset '{self.name}' dtype mismatch: {details}")

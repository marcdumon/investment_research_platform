
import os
from contextlib import contextmanager
from typing import Literal

import pandas as pd
import simfin

import irp.config as _config
from irp.datasets.dataset import Dataset
from irp.sources.base import BaseSource

_LOADERS = {
    "income": simfin.load_income,
    "balance": simfin.load_balance,
    "cashflow": simfin.load_cashflow,
    "industries": simfin.load_industries,
    "companies": simfin.load_companies,
}

SimFinDatasetType = Literal["income", "balance", "cashflow", "industries", "companies"] 


@contextmanager
def _pandas_compat():
    """Patch pd.read_csv to drop kwargs removed in pandas 2.0 that simfin still passes."""
    _orig = pd.read_csv

    def _patched(*args, **kwargs):
        kwargs.pop("date_parser", None)
        return _orig(*args, **kwargs)

    pd.read_csv = _patched
    try:
        yield
    finally:
        pd.read_csv = _orig


class SimFinFundamentalsSource(BaseSource):
    def __init__(
        self,
        statement: SimFinDatasetType,
        variant: str = "annual",
        market: str = "us",
        data_dir: str | None = None,
    ) -> None:
        """
        statement: one of "income", "balance", "cashflow", "industries", "companies"
        Reads SIMFIN_API_KEY from environment.
        data_dir defaults to config [simfin] data_dir.
        """
        self.statement = statement
        self.variant = variant
        self.market = market
        self._api_key = os.environ["SIMFIN_API_KEY"]
        self._data_dir = data_dir or _config.load()["simfin"]["data_dir"]

    def fetch(
        self,
        start_date: str | None = None,
        end_date: str | None = None,
        **kwargs,
    ) -> Dataset:
        simfin.set_api_key(self._api_key)
        simfin.set_data_dir(self._data_dir)

        is_reference_table = self.statement in ("industries", "companies")
        with _pandas_compat():
            if is_reference_table:
                df = _LOADERS[self.statement]()
            else:
                df = _LOADERS[self.statement](
                    variant=self.variant,
                    market=self.market,
                    start_date=start_date,
                    end_date=end_date,
                )
        df = df.reset_index()
        if "SimFinId" in df.columns:
            df = df.rename(columns={"SimFinId": "source_id"})
            df.insert(df.columns.get_loc("source_id") + 1, "source", "simfin")
        else:
            df.insert(0, "source", "simfin")
        if not is_reference_table:
            df.insert(0, "variant", "A" if self.variant == "annual" else "Q")

        return Dataset.from_df(df, name=self.statement, source="simfin")

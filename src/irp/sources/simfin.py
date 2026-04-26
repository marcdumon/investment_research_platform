
import os
from typing import Literal

import simfin

import irp.config as _config
from irp.datasets.dataset import Dataset
from irp.sources.base import BaseSource

_LOADERS = {
    "income": simfin.load_income,
    "balance": simfin.load_balance,
    "cashflow": simfin.load_cashflow,
}

Statement = Literal["income", "balance", "cashflow"]


class SimFinFundamentalsSource(BaseSource):
    def __init__(
        self,
        statement: Statement,
        variant: str = "annual",
        market: str = "us",
        data_dir: str | None = None,
    ) -> None:
        """
        statement: one of "income", "balance", "cashflow"
        Reads SIMFIN_API_KEY from environment.
        data_dir defaults to SIMFIN_DATA_DIR env var or "data/simfin".
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
    ) -> Dataset:
        simfin.set_api_key(self._api_key)
        simfin.set_data_dir(self._data_dir)

        df = _LOADERS[self.statement](
            variant=self.variant,
            market=self.market,
            start_date=start_date,
            end_date=end_date,
        )
        df = df.reset_index()
        schema = {c: str(df[c].dtype) for c in df.columns}

        return Dataset(
            name=f"simfin_{self.statement}_{self.variant}",
            data=df,
            schema=schema,
            source="simfin",
        )

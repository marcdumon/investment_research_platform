
import io
import os
import urllib.request
from datetime import datetime

import pandas as pd

from irp.datasets.dataset import Dataset
from irp.sources.base import BaseSource

PRICE_SCHEMA = {
    "ticker": "object",
    "source_id": "object",
    "source": "object",
    "date": "object",
    "open": "float64",
    "high": "float64",
    "low": "float64",
    "close": "float64",
    "volume": "float64",
}

_BASE_URL = "https://stooq.com/q/d/l/"


def normalize_ticker(ticker: str) -> str:
    """Strip country suffix and uppercase: 'msft.us' -> 'MSFT', '^spx' -> '^SPX'."""
    return ticker.split(".")[0].upper()


class StooqPriceSource(BaseSource):
    def __init__(self, ticker: str, start: str, end: str) -> None:
        """
        ticker: stooq symbol, e.g. "msft.us"
        start/end: "YYYY-MM-DD"
        Reads STOOQ_API_KEY from environment.
        """
        self.ticker = ticker
        self.start = start
        self.end = end
        self._api_key = os.environ["STOOQ_API_KEY"]

    def fetch(self, **kwargs) -> Dataset:
        d1 = datetime.strptime(self.start, "%Y-%m-%d").strftime("%Y%m%d")
        d2 = datetime.strptime(self.end, "%Y-%m-%d").strftime("%Y%m%d")
        url = (
            f"{_BASE_URL}?s={self.ticker}&d1={d1}&d2={d2}"
            f"&i=d&apikey={self._api_key}"
        )
        with urllib.request.urlopen(url) as resp:
            raw = resp.read().decode("utf-8")

        df = pd.read_csv(io.StringIO(raw))
        df.columns = [c.lower() for c in df.columns]
        for col in ("open", "high", "low", "close", "volume"):
            df[col] = pd.to_numeric(df[col], errors="coerce")
        df.insert(0, "ticker", normalize_ticker(self.ticker))
        df.insert(1, "source_id", self.ticker)
        df.insert(2, "source", "stooq")

        return Dataset(
            name=self.ticker,
            data=df,
            schema=PRICE_SCHEMA,
            source="stooq",
        )

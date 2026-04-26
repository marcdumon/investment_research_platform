import zipfile
from pathlib import Path

import pandas as pd

import irp.config as _config
from irp.datasets.dataset import Dataset
from irp.sources.base import BaseSource
from irp.sources.stooq import PRICE_SCHEMA


class StooqBulkSource(BaseSource):
    """
    Reads price data from Stooq bulk zips (d_us_txt.zip, d_world_txt.zip, etc.).

    Drop any Stooq bulk zip into config.toml [stooq] download_dir.
    Each zip is extracted once into its own subdirectory of data_dir.
    Ticker search covers all extracted zips.
    """

    def __init__(
        self,
        ticker: str,
        start: str | None = None,
        end: str | None = None,
    ) -> None:
        """
        ticker: stooq symbol e.g. "msft.us" or "^spx" (index)
        start/end: "YYYY-MM-DD" optional date filter
        """
        self.ticker = ticker.lower()
        self.start = start
        self.end = end
        cfg = _config.load()["stooq"]
        self._download_dir = Path(cfg["download_dir"])
        self._data_dir = Path(cfg["data_dir"])

    def _zips(self) -> list[Path]:
        zips = sorted(self._download_dir.glob("*.zip"))
        if not zips:
            raise FileNotFoundError(
                f"No zip files found in {self._download_dir}\n"
                "Download bulk zips from https://stooq.com/db/h/"
            )
        return zips

    def _ensure_extracted(self) -> None:
        self._data_dir.mkdir(parents=True, exist_ok=True)
        for zip_path in self._zips():
            marker = self._data_dir / f".extracted_{zip_path.stem}"
            if marker.exists():
                continue
            dest = self._data_dir / zip_path.stem
            dest.mkdir(parents=True, exist_ok=True)
            with zipfile.ZipFile(zip_path) as zf:
                zf.extractall(dest)
            marker.touch()

    def _find_file(self) -> Path:
        matches = list(self._data_dir.rglob(f"{self.ticker}.txt"))
        if not matches:
            raise FileNotFoundError(
                f"Ticker '{self.ticker}' not found across all bulk zips in {self._data_dir}"
            )
        return matches[0]

    def fetch(self, **kwargs) -> Dataset:
        self._ensure_extracted()
        path = self._find_file()

        df = pd.read_csv(path, header=0)
        df.columns = [c.strip().lower() for c in df.columns]

        df["date"] = pd.to_datetime(df["date"].astype(str), format="%Y%m%d").dt.strftime("%Y-%m-%d")

        for col in ("open", "high", "low", "close", "volume"):
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors="coerce")

        if self.start:
            df = df[df["date"] >= self.start]
        if self.end:
            df = df[df["date"] <= self.end]

        return Dataset(
            name=self.ticker,
            data=df.reset_index(drop=True),
            schema=PRICE_SCHEMA,
            source="stooq_bulk",
        )


import zipfile
from pathlib import Path

import pandas as pd

import irp.config as _config
from irp.datasets.dataset import Dataset
from irp.sources.base import BaseSource
from irp.sources.stooq import PRICE_SCHEMA, normalize_ticker


def ensure_zips_extracted(download_dir: Path, data_dir: Path) -> None:
    """Extract any not-yet-extracted Stooq bulk zips from download_dir into data_dir."""
    zips = sorted(download_dir.glob("*.zip"))
    if not zips:
        raise FileNotFoundError(
            f"No zip files found in {download_dir}\n"
            "Download bulk zips from https://stooq.com/db/h/"
        )
    data_dir.mkdir(parents=True, exist_ok=True)
    for zip_path in zips:
        marker = data_dir / f".extracted_{zip_path.stem}"
        if marker.exists() and marker.stat().st_mtime >= zip_path.stat().st_mtime:
            continue
        dest = data_dir / zip_path.stem
        dest.mkdir(parents=True, exist_ok=True)
        with zipfile.ZipFile(zip_path) as zf:
            zf.extractall(dest)
        marker.touch()


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

    def _find_file(self) -> Path:
        matches = list(self._data_dir.rglob(f"{self.ticker}.txt"))
        if not matches:
            raise FileNotFoundError(
                f"Ticker '{self.ticker}' not found across all bulk zips in {self._data_dir}"
            )
        return matches[0]

    def fetch(self, **kwargs) -> Dataset:
        ensure_zips_extracted(self._download_dir, self._data_dir)
        path = kwargs.get("file_path") or self._find_file()

        df = pd.read_csv(path, header=0)
        df.columns = [c.strip().strip("<>").lower() for c in df.columns]
        df = df.rename(columns={"vol": "volume"})
        df = df[["date", "open", "high", "low", "close", "volume"]]
        df.insert(0, "ticker", normalize_ticker(self.ticker))
        df.insert(1, "source_id", self.ticker)
        df.insert(2, "source", "stooq")

        df["date"] = pd.to_datetime(df["date"].astype(str), format="%Y%m%d").dt.strftime("%Y-%m-%d")

        for col in ("open", "high", "low", "close", "volume"):
            df[col] = pd.to_numeric(df[col], errors="coerce").astype("float64")

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

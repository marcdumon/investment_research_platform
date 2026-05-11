import csv
import logging
import shutil
import zipfile
from pathlib import Path
from collections.abc import Iterator

from irp.core.config import config
from irp.core.logging import configure

configure()

root_dir = config.data.root_dir
stooq_cfg = config.providers.stooq
raw_dir = root_dir / stooq_cfg.raw_dir


def _ensure_bulk_files_available() -> None:
    """Checks if the bulk files file is available in the raw data directory. If not, raises an error."""
    filenames = [p.name for p in raw_dir.glob("*.zip")]
    if sorted(filenames) == sorted(stooq_cfg.bulk_files):
        return
    logging.error(f"Bulk files not available in {stooq_cfg.raw_dir}.")
    message = stooq_cfg.bulk_download_instruction
    print(message)
    raise FileNotFoundError()


def _unzip_bulk_files() -> None:
    """Unzips the bulk files in the raw data directory. Uses marker files to avoid re-unzipping files that have already been extracted."""
    logging.debug("Unzipping bulk files...")
    for fname in stooq_cfg.bulk_files:
        zip_path = raw_dir / fname
        marker = raw_dir / f".extracted_{zip_path.stem}"
        if marker.exists() and marker.stat().st_mtime >= zip_path.stat().st_mtime:
            logging.debug(f"{fname} already extracted, skipping.")
            continue
        with zipfile.ZipFile(zip_path) as zf:
            zf.extractall(raw_dir)
        marker.touch()


def _iter_file_dirs(path: Path) -> Iterator[Path]:
    """Yield dirs that contain ticker files.
    Assumes file-containing dirs are terminal leaves (no mixed file+subdir dirs)."""
    entries = list(path.iterdir())
    if any(p.is_file() for p in entries):
        yield path
    else:
        for child in entries:
            if child.is_dir():
                yield from _iter_file_dirs(child)


def _category_of(leaf: Path, data_root: Path) -> Path:
    """Return the category dir (region/category structure: parts[1] below data_root)."""
    rel = leaf.relative_to(data_root)
    return data_root / rel.parts[1]


def _flatten_bulk_files() -> None:
    """Move all ticker files into raw_dir/ticker_prices/, write raw_dir/ticker_markets.csv, delete data/."""
    logging.debug("Flattening bulk files...")
    tickers_dir = raw_dir / "ticker_prices"
    tickers_dir.mkdir(exist_ok=True)
    rows: list[dict[str, str]] = []
    data_root = raw_dir / "data"
    for freq_dir in data_root.iterdir():
        if not freq_dir.is_dir():
            continue
        for leaf in _iter_file_dirs(freq_dir):
            market = _category_of(leaf, freq_dir).name
            for f in leaf.iterdir():
                if f.is_file():
                    target = tickers_dir / f.name
                    if target.exists():
                        raise FileExistsError(target)
                    shutil.move(str(f), str(target))
                    rows.append({"ticker": f.stem, "market": market})
    csv_path = raw_dir / "ticker_markets.csv"
    with open(csv_path, "w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=["ticker", "market"])
        writer.writeheader()
        writer.writerows(rows)
    shutil.rmtree(data_root)
    logging.debug(f"Flattened {len(rows)} tickers, markets written to {csv_path}.")




class StooqProvider:
    def fetch(self):
        logging.info("Fetching Stooq price data...")
        _ensure_bulk_files_available()
        _unzip_bulk_files()
        _flatten_bulk_files()
        logging.info("Stooq price data fetched successfully.")

    def update(self): ...

    def transform(self, raw): ...

    def store(self, data): ...


def main():
    provider = StooqProvider()
    provider.fetch()
    # _flatten_bulk_files()


if __name__ == "__main__":
    main()

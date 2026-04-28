
import tomllib
from pathlib import Path
from typing import TypedDict

from dotenv import load_dotenv

_ROOT = Path(__file__).parents[2]

load_dotenv(_ROOT / ".env")


class _StooqConfig(TypedDict):
    download_dir: str
    data_dir: str


class _SimFinConfig(TypedDict):
    data_dir: str


class _StoreConfig(TypedDict):
    db_path: str


class Config(TypedDict):
    stooq: _StooqConfig
    simfin: _SimFinConfig
    store: _StoreConfig


def load() -> Config:
    with open(_ROOT / "config.toml", "rb") as f:
        return tomllib.load(f)  # type: ignore[return-value]

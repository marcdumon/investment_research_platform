import tomllib
from pathlib import Path

from dotenv import load_dotenv

_ROOT = Path(__file__).parents[2]

load_dotenv(_ROOT / ".env")


def load() -> dict:
    with open(_ROOT / "config.toml", "rb") as f:
        return tomllib.load(f)

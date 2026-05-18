import os
import tomllib
from pathlib import Path

from dotenv import load_dotenv
from pydantic import BaseModel, Field

load_dotenv()


class DatabaseConfig(BaseModel):
    path: Path


class DataConfig(BaseModel):
    root_dir: Path


class ProviderConfig(BaseModel):
    raw_dir: Path
    processed_dir: Path


class SimfinConfig(ProviderConfig):
    api_key: str = Field(default_factory=lambda: os.environ['SIMFIN_API_KEY'])
    refresh_days_fundamentals: int
    refresh_days_meta: int



class YahooConfig(ProviderConfig):
    batch_sleep: float
    prices_batch_size: int = 50
    markets_exclude: list[str]


class StooqConfig(ProviderConfig):
    bulk_url: str
    bulk_files: list[str]
    bulk_instruction_template: str

    update_url: str
    update_file: str
    update_instruction_template: str

    @property
    def bulk_download_instruction(self) -> str:
        """Resolves the download instruction by formatting the template with the config values."""
        return self.bulk_instruction_template.format(**self.model_dump())

    @property
    def update_download_instruction(self) -> str:
        """Resolves the download instruction by formatting the template with the config values."""
        return self.update_instruction_template.format(**self.model_dump())


class ProvidersConfig(BaseModel):
    simfin: SimfinConfig
    stooq: StooqConfig
    yahoo: YahooConfig


class Config(BaseModel):
    database: DatabaseConfig
    data: DataConfig
    providers: ProvidersConfig

    @classmethod
    def load(cls, path: str | Path | None = None) -> 'Config':
        if path is None:
            path = Path(__file__).parents[3] / 'config.toml'
        with open(path, 'rb') as f:
            raw = tomllib.load(f)
        return cls.model_validate(raw)


config = Config.load()

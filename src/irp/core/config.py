from pathlib import Path
from pydantic import BaseModel, Field
import tomllib


class DatabaseConfig(BaseModel):
    path: Path


class DataConfig(BaseModel):
    root_dir: Path


class ProviderConfig(BaseModel):
    raw_dir: Path
    processed_dir: Path


class SimfinConfig(ProviderConfig):
    api_key: str | None = None


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


class Config(BaseModel):
    database: DatabaseConfig
    data: DataConfig
    providers: ProvidersConfig

    @classmethod
    def load(cls, path: str | Path = 'config.toml') -> 'Config':
        with open(path, 'rb') as f:
            raw = tomllib.load(f)
        return cls.model_validate(raw)


config = Config.load()

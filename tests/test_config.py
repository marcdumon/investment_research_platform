from pathlib import Path
from irp.core.config import Config

CONFIG_PATH = Path(__file__).parents[1] / "config.toml"


def test_load_real_config():
    cfg = Config.load(CONFIG_PATH)
    assert isinstance(cfg, Config)
    assert cfg.database.path.suffix == ".duckdb"
    assert cfg.providers.stooq.bulk_url
    assert cfg.providers.stooq.bulk_files
    assert "http" in cfg.providers.stooq.bulk_download_instruction

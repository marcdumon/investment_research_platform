"""Glue between irp's global config and the standalone `dataload` package.

`build_ingest_context()` adapts the pydantic `config` singleton + irp's
write-session/cancel machinery into a `dataload.IngestContext`. This is the only
place irp reaches into dataload's configuration shape; the package itself stays
free of any irp import.
"""
from dataload import IngestContext
from dataload.state import MarkerSet
from irp.core import cancel
from irp.core.config import config


def yahoo_datasets(content: list[str]) -> list[str] | None:
    """Map the UI's Yahoo content checkboxes to dataset names (None = provider default)."""
    datasets: list[str] = []
    if 'prices' in content:
        datasets.append('prices')
    if 'actions' in content:
        datasets += ['dividends', 'splits']
    return datasets or None


def build_ingest_context(yahoo_overrides: dict | None = None) -> IngestContext:
    """Construct a dataload IngestContext from the global irp config.

    `yahoo_overrides` (e.g. from the UI's batch/sleep inputs) replace the
    corresponding Yahoo settings; None values in it are ignored.
    """
    s = config.providers.stooq
    y = config.providers.yahoo
    f = config.providers.simfin

    yahoo_cfg = {
        'raw_dir': y.raw_dir, 'processed_dir': y.processed_dir,
        'prices_batch_size': y.prices_batch_size, 'batch_sleep': y.batch_sleep,
        'actions_sleep': y.actions_sleep, 'markets_exclude': y.markets_exclude,
    }
    if yahoo_overrides:
        yahoo_cfg.update({k: v for k, v in yahoo_overrides.items() if v is not None})

    provider_cfg = {
        'stooq': {
            'raw_dir': s.raw_dir, 'processed_dir': s.processed_dir,
            'bulk_files': s.bulk_files, 'update_file': s.update_file,
        },
        'yahoo': yahoo_cfg,
        'simfin': {
            'raw_dir': f.raw_dir, 'processed_dir': f.processed_dir,
            'api_key': f.api_key,
            'refresh_days_fundamentals': f.refresh_days_fundamentals,
            'refresh_days_meta': f.refresh_days_meta,
        },
    }
    return IngestContext(
        data_root=config.data.root_dir,
        provider_cfg=provider_cfg,
        connect=config_write_session,
        is_cancelled=cancel.is_cancelled,
    )


def config_write_session():
    """Indirection so the context's connect factory is picklable-by-reference and testable."""
    from irp.core.db import write_session
    return write_session()


def reset_state(ctx: IngestContext, name: str) -> None:
    """Force a clean re-run by clearing the provider's resume/freshness state."""
    raw = ctx.raw_dir(name)
    if name == 'stooq':
        MarkerSet(raw).clear()
    elif name == 'yahoo':
        for fname in ('queried_prices.json', 'queried_actions.json', 'error_tickers.json'):
            (raw / fname).unlink(missing_ok=True)
    # simfin is file-age based: delete the download dir to force a re-pull
    elif name == 'simfin':
        download = raw / 'download'
        if download.exists():
            for p in download.glob('*.zip'):
                p.unlink()

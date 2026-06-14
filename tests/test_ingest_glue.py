"""irp -> dataload glue: build an IngestContext from the global config."""
from irp.core.config import config
from irp.core.ingest_context import build_ingest_context, reset_state, yahoo_datasets


def test_yahoo_datasets_mapping() -> None:
    assert set(yahoo_datasets(['prices'])) == {'prices'}
    assert set(yahoo_datasets(['actions'])) == {'dividends', 'splits'}
    assert set(yahoo_datasets(['actions', 'prices'])) == {'prices', 'dividends', 'splits'}
    assert yahoo_datasets([]) is None


def test_build_context_maps_config() -> None:
    ctx = build_ingest_context()
    assert ctx.data_root == config.data.root_dir
    assert ctx.cfg('yahoo')['markets_exclude'] == config.providers.yahoo.markets_exclude
    assert ctx.raw_dir('stooq') == config.data.root_dir / config.providers.stooq.raw_dir
    assert ctx.cfg('stooq')['bulk_files'] == config.providers.stooq.bulk_files
    assert ctx.cfg('simfin')['api_key'] == config.providers.simfin.api_key


def test_build_context_applies_yahoo_overrides() -> None:
    ctx = build_ingest_context(yahoo_overrides={'prices_batch_size': 7, 'batch_sleep': None})
    assert ctx.cfg('yahoo')['prices_batch_size'] == 7
    # None override is ignored — keeps the config value
    assert ctx.cfg('yahoo')['batch_sleep'] == config.providers.yahoo.batch_sleep


def test_reset_state_clears_yahoo_resume_files(tmp_path) -> None:
    from dataload import IngestContext

    raw = tmp_path / 'yahoo' / 'raw'
    raw.mkdir(parents=True)
    for f in ('queried_prices.json', 'queried_actions.json', 'error_tickers.json'):
        (raw / f).write_text('[]')
    ctx = IngestContext(tmp_path, {'yahoo': {'raw_dir': 'yahoo/raw', 'processed_dir': 'yahoo/processed'}},
                        connect=lambda: None)
    reset_state(ctx, 'yahoo')
    assert not (raw / 'queried_prices.json').exists()
    assert not (raw / 'error_tickers.json').exists()

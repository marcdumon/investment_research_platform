import questionary
from questionary import Choice, Style

from irp.core.config import config
from irp.core.logging import configure_logging
from irp.runner import DataProvider

STYLE = Style([
    ('question',    'bold'),
    ('instruction', 'fg:gray'),
    ('pointer',     'fg:cyan bold'),
    ('highlighted', 'fg:cyan bold'),
    ('selected',    'fg:green bold'),
    ('answer',      'fg:green bold'),
])


def main() -> None:
    configure_logging()
    providers = questionary.checkbox(
        'Providers:',
        choices=[Choice('simfin', checked=True), Choice('stooq', checked=True), Choice('yahoo', checked=False)],
        style=STYLE,
    ).ask()
    if not providers:
        return

    yahoo_content = ['actions', 'prices']
    if 'yahoo' in providers:
        yahoo_content = questionary.checkbox(
            'Yahoo content:',
            choices=[
                Choice('actions  (dividends + splits)', value='actions', checked=True),
                Choice('prices   (OHLCV)', value='prices', checked=True),
            ],
            style=STYLE,
        ).ask() or yahoo_content

    feed = questionary.select(
        'Feed:',
        choices=['bulk', 'update'],
        style=STYLE,
    ).ask()
    if not feed:
        return

    steps = questionary.checkbox(
        'Steps:',
        choices=[
            Choice('fetch', checked=True),
            Choice('transform', checked=True),
            Choice('store', checked=True),
            Choice('cleanup', checked=False),
            Choice('seed-universe', checked=False),
            Choice('universe', checked=False),
            Choice('catalog', checked=False),
        ],
        style=STYLE,
    ).ask()
    if not steps:
        return

    force = questionary.confirm(
        'Force re-run (delete markers)?', default=False, style=STYLE
    ).ask()

    print('\nWill execute:')
    for name in providers:
        forced = '  [FORCED]' if force else ''
        print(f'  {name}  [{feed}]  →  {", ".join(steps)}{forced}')
    print()

    if force:
        for name in providers:
            delete_markers(name, feed)
        print()

    for name in providers:
        src = make_source(name, yahoo_content=yahoo_content)
        print(f'── {name} ──')
        if feed not in src.SUPPORTED_FEEDS:
            print(f'  feed {feed!r} not supported by {name}, skipping')
            continue
        if 'fetch' in steps:
            src.fetch_bulk() if feed == 'bulk' else src.update()
        if 'transform' in steps:
            src.transform(feed)
        if 'store' in steps:
            src.store(feed)
        if 'cleanup' in steps:
            src.cleanup()

    if 'seed-universe' in steps:
        from irp.query.universe import seed as _seed_universe
        print('── seed-universe ──')
        n = _seed_universe()
        print(f'  {n:,} tickers written to universe.csv')

    if 'universe' in steps:
        from irp.query.universe import refresh as _refresh_universe
        print('── universe ──')
        n = _refresh_universe()
        print(f'  {n:,} tickers')

    if 'catalog' in steps:
        from irp.query.catalog import refresh as _refresh_catalog
        print('── catalog ──')
        n = _refresh_catalog()
        print(f'  {n:,} tickers')


def delete_markers(name: str, feed: str) -> None:
    from irp.core.markers import MarkerSet
    cfg = getattr(config.providers, name)
    raw_dir = config.data.root_dir / cfg.raw_dir
    n = MarkerSet(raw_dir).clear_feed(feed)
    if n:
        print(f'  deleted {n} {name}/{feed} marker{"s" if n != 1 else ""}')


def make_source(
    name: str,
    yahoo_content: list[str] | None = None,
    yahoo_batch_size: int | None = None,
    yahoo_batch_sleep: float | None = None,
    yahoo_actions_sleep: float | None = None,
) -> DataProvider:
    if name == 'simfin':
        from irp.ingest.sim_fin import SimFinSource
        return SimFinSource()
    if name == 'stooq':
        from irp.ingest.stooq import StooqSource
        return StooqSource()
    if name == 'yahoo':
        from irp.ingest.yahoo import YahooSource
        content = yahoo_content or ['actions', 'prices']
        return YahooSource(
            fetch_actions='actions' in content,
            fetch_prices='prices' in content,
            prices_batch_size=yahoo_batch_size,
            batch_sleep=yahoo_batch_sleep,
            actions_sleep=yahoo_actions_sleep,
        )
    raise ValueError(f'unknown provider: {name!r}')


if __name__ == '__main__':
    main()

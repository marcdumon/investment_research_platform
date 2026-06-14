import questionary
from questionary import Choice, Style

from irp.core.logging import configure_logging

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
        choices=[Choice('bulk    (full history)', value='bulk'), Choice('update  (incremental)', value='update')],
        style=STYLE,
    ).ask()
    if not feed:
        return

    steps = questionary.checkbox(
        'Steps:',
        choices=[
            Choice('ingest   (fetch + normalize + load)', value='ingest', checked=True),
            Choice('cleanup', value='cleanup'),
            Choice('seed-universe', value='seed-universe'),
            Choice('universe', value='universe'),
            Choice('catalog', value='catalog'),
        ],
        style=STYLE,
    ).ask()
    if not steps:
        return

    force = questionary.confirm('Force re-run (reset fetch state)?', default=False, style=STYLE).ask()

    from dataload import make_provider, run
    from dataload import universe as ul
    from irp.core.ingest_context import build_ingest_context, reset_state, yahoo_datasets

    ctx = build_ingest_context()
    incremental = feed == 'update'

    print('\nWill execute:')
    mode = 'incremental' if incremental else 'full'
    for name in providers:
        print(f'  {name}  [{mode}]  →  {", ".join(steps)}')
    print()

    if force:
        for name in providers:
            reset_state(ctx, name)

    for name in providers:
        provider = make_provider(name)
        datasets = yahoo_datasets(yahoo_content) if name == 'yahoo' else None
        print(f'── {name} ──')
        if 'ingest' in steps:
            summary = run(ctx, [provider], datasets=datasets, incremental=incremental, cleanup='cleanup' in steps)
            for dataset, n in summary.get(name, {}).items():
                print(f'  {dataset}: {n:,} rows')
        elif 'cleanup' in steps:
            provider.cleanup(ctx)
            print('  cleaned')

    if 'seed-universe' in steps:
        print('── seed-universe ──')
        print(f'  {ul.seed(ctx):,} tickers written to universe.csv')

    if 'universe' in steps:
        print('── universe ──')
        print(f'  {ul.refresh(ctx):,} tickers')

    if 'catalog' in steps:
        from irp.query.catalog import refresh as _refresh_catalog
        print('── catalog ──')
        print(f'  {_refresh_catalog():,} tickers')


if __name__ == '__main__':
    main()

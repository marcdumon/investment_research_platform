import questionary
from questionary import Choice, Style

from irp.core.config import config
from irp.core.logging import configure_logging
import logging

configure_logging(level=logging.DEBUG)

STYLE = Style([
    ('question',    'bold'),
    ('instruction', 'fg:gray'),
    ('pointer',     'fg:cyan bold'),
    ('highlighted', 'fg:cyan bold'),
    ('selected',    'fg:green bold'),
    ('answer',      'fg:green bold'),
])


def main() -> None:
    providers = questionary.checkbox(
        'Providers:',
        choices=[Choice('simfin', checked=True), Choice('stooq', checked=True), Choice('yahoo', checked=False)],
        style=STYLE,
    ).ask()
    if not providers:
        return

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
        ],
        style=STYLE,
    ).ask()
    if not steps:
        return

    force = questionary.confirm(
        'Force re-run (delete markers)?', default=False, style=STYLE
    ).ask()

    print(f'\nWill execute:')
    for name in providers:
        forced = '  [FORCED]' if force else ''
        print(f'  {name}  [{feed}]  →  {", ".join(steps)}{forced}')
    print()

    if force:
        for name in providers:
            _delete_markers(name, feed)
        print()

    for name in providers:
        src = _make_source(name)
        print(f'── {name} ──')
        if 'fetch' in steps:
            src.fetch_bulk() if feed == 'bulk' else src.update()
        if 'transform' in steps:
            src.transform(feed)
        if 'store' in steps:
            src.store(feed)
        if 'cleanup' in steps:
            src.cleanup()


def _delete_markers(name: str, feed: str) -> None:
    cfg = getattr(config.providers, name)
    raw_dir = config.data.root_dir / cfg.raw_dir
    fetch_marker = raw_dir / ('.fetched_update' if feed == 'update' else '.fetched')
    markers = [fetch_marker, raw_dir / f'.transformed_{feed}', raw_dir / f'.stored_{feed}']
    for m in markers:
        if m.exists():
            m.unlink()
            print(f'  deleted {name}/{m.name}')


def _make_source(name: str):
    if name == 'simfin':
        from irp.sources.sim_fin import SimFinSource
        return SimFinSource()
    if name == 'yahoo':
        from irp.sources.yahoo import YahooSource
        return YahooSource()
    from irp.sources.stooq import StooqSource
    return StooqSource()


if __name__ == '__main__':
    main()

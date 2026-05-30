import logging
from typing import Literal, Protocol

logger = logging.getLogger(__name__)

Feed = Literal['bulk', 'update']


class DataProvider(Protocol):
    SUPPORTED_FEEDS: frozenset[Feed]

    def fetch_bulk(self) -> None: ...
    def update(self) -> None: ...
    def transform(self, feed: Feed) -> None: ...
    def store(self, feed: Feed) -> None: ...
    def cleanup(self) -> None: ...


def _load_data(provider: DataProvider, feed: Feed) -> None:
    name = provider.__class__.__name__
    if feed not in provider.SUPPORTED_FEEDS:
        logger.info(f'{name}: feed {feed!r} not supported, skipping')
        return
    logger.info(f'Loading {feed} data from {name}')
    provider.fetch_bulk() if feed == 'bulk' else provider.update()
    provider.transform(feed)
    provider.store(feed)
    logger.info(f'Finished loading {feed} data from {name}')


if __name__ == '__main__':
    from irp.cli import main
    main()

    # from irp.ingest.sim_fin import SimFinSource
    # a:DataProvider=SimFinSource

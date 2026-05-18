import logging
from typing import Literal, Protocol

logger = logging.getLogger(__name__)


class DataProvider(Protocol):
    def fetch_bulk(self) -> None: ...
    def update(self) -> None: ...
    def transform(self, feed: Literal['bulk', 'update']) -> None: ...
    def store(self, feed: Literal['bulk', 'update']) -> None: ...
    def cleanup(self) -> None: ...


def load_data(provider: DataProvider, feed: Literal['bulk', 'update']) -> None:
    logger.info(f'Loading {feed} data from {provider.__class__.__name__}')
    provider.fetch_bulk() if feed == 'bulk' else provider.update()
    provider.transform(feed)
    provider.store(feed)
    logger.info(f'Finished loading {feed} data from {provider.__class__.__name__}')


if __name__ == '__main__':
    from irp.cli import main
    main()

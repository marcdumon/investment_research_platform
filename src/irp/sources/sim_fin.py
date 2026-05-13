import logging
from dataclasses import dataclass
from typing import Literal

import simfin as sf

from irp.core.config import config
from irp.core.logging import configure_logging

logger = logging.getLogger(__name__)
configure_logging(level=logging.DEBUG)

root_dir = config.data.root_dir
simfin_cfg = config.providers.simfin
raw_dir = root_dir / simfin_cfg.raw_dir
processed_dir = root_dir / simfin_cfg.processed_dir


FundamentalsVariant = Literal['annual', 'quarterly', 'ttm']
SharepricesVariant = Literal['daily', 'latest']
Market = Literal['us', 'de', 'cn']


@dataclass(frozen=True)
class FundamentalsDataset:
    name: Literal['income', 'balance', 'cashflow']
    variant: FundamentalsVariant
    market: Market
    refresh_days: int = simfin_cfg.refresh_days_fundamentals


@dataclass(frozen=True)
class SharepricesDataset:
    variant: SharepricesVariant
    market: Market
    name: Literal['shareprices'] = 'shareprices'
    refresh_days: int = simfin_cfg.refresh_days_shareprices


@dataclass(frozen=True)
class MetaDataset:
    name: Literal['markets', 'sectors', 'industries']
    variant: None = None
    market: None = None
    refresh_days: int = simfin_cfg.refresh_days_meta


SimFinDataset = FundamentalsDataset | SharepricesDataset | MetaDataset


_FUNDAMENTALS_NAMES: list[Literal['income', 'balance', 'cashflow']] = [
    'income',
    'balance',
    'cashflow',
]
_FUNDAMENTALS_VARIANTS: list[FundamentalsVariant] = ['annual', 'quarterly', 'ttm']
_SHAREPRICES_VARIANTS: list[SharepricesVariant] = ['daily', 'latest']
_MARKETS: list[Market] = ['us', 'de']
_META_NAMES: list[Literal['markets', 'industries']] = [
    'markets',
    'industries',
]

BULK_DATASETS: list[SimFinDataset] = (
    [
        FundamentalsDataset(name, variant, market)
        for name in _FUNDAMENTALS_NAMES
        for variant in _FUNDAMENTALS_VARIANTS
        for market in _MARKETS
    ]
    + [
        SharepricesDataset(variant, market)
        for variant in _SHAREPRICES_VARIANTS
        for market in _MARKETS
    ]
    + [MetaDataset(name) for name in _META_NAMES]
)


class SimFinSource:
    def fetch_bulk(self) -> None:
        print(simfin_cfg.api_key)
        sf.set_api_key(simfin_cfg.api_key)
        sf.set_data_dir(str(raw_dir))

        for dataset in BULK_DATASETS:

            logger.debug(
                'Fetching %s/%s/%s', dataset.name, dataset.variant, dataset.market
            )
            sf.load(
                dataset=dataset.name,
                variant=dataset.variant,
                market=dataset.market,
                refresh_days=dataset.refresh_days,
            )

    def update(self): ...

    def transform(self, raw): ...

    def store(self, data): ...

    def cleanup(self): ...


def main():
    source = SimFinSource()
    source.fetch_bulk()


if __name__ == '__main__':
    main()

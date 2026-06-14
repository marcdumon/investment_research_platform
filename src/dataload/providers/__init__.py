"""Concrete data providers. Each implements the `base.Provider` Protocol."""
from dataload.providers.base import Capability, Provider
from dataload.providers.simfin import SimFinProvider
from dataload.providers.stooq import StooqProvider
from dataload.providers.yahoo import YahooProvider

PROVIDERS: dict[str, type[Provider]] = {
    'stooq': StooqProvider,
    'yahoo': YahooProvider,
    'simfin': SimFinProvider,
}

__all__ = ['PROVIDERS', 'Capability', 'Provider', 'SimFinProvider', 'StooqProvider', 'YahooProvider', 'make_provider']


def make_provider(name: str) -> Provider:
    """Instantiate a provider by name."""
    try:
        return PROVIDERS[name]()
    except KeyError:
        raise ValueError(f'unknown provider: {name!r}') from None

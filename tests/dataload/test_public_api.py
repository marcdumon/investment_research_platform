"""The package's public surface — what other projects import."""


def test_public_api_is_importable() -> None:
    from dataload import (
        SCHEMAS,
        Capability,
        IngestContext,
        TableSchema,
        make_provider,
        run,
        universe,
    )

    assert callable(run)
    assert callable(make_provider)
    assert 'prices' in SCHEMAS
    assert isinstance(SCHEMAS['prices'], TableSchema)
    assert Capability(incremental=True).incremental is True
    assert hasattr(universe, 'seed_from_markets')
    assert IngestContext is not None


def test_importing_dataload_does_not_require_yfinance() -> None:
    """yfinance is heavy + only needed at Yahoo fetch time; import must not pull it."""
    import sys

    sys.modules.pop('yfinance', None)
    import dataload  # noqa: F401

    assert 'yfinance' not in sys.modules

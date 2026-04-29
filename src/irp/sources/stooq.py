
PRICE_SCHEMA = {
    "ticker": "str",
    "source_id": "str",
    "source": "str",
    "date": "str",
    "open": "float64",
    "high": "float64",
    "low": "float64",
    "close": "float64",
    "volume": "float64",
}


def normalize_ticker(ticker: str) -> str:
    """Strip country suffix and uppercase: 'msft.us' -> 'MSFT', '^spx' -> '^SPX'."""
    return ticker.split(".")[0].upper()

"""Unified OHLCV price accessor.

Stooq and Yahoo prices live in one `prices` table, distinguished by the `Src`
column (canonical schema: Ticker, Date, Open, High, Low, Close, Volume, Src,
SrcId). `irp.query.stooq` and `irp.query.yahoo` are thin `src`-filtered views
over this accessor.
"""
import pandas as pd

from irp.query._common import build_where, db


def prices(
    tickers: str | list[str] | None = None,
    start: str | None = None,
    end: str | None = None,
    src: str | None = None,
) -> pd.DataFrame:
    """OHLCV prices. `src` restricts to one source ('stooq'/'yahoo'); None = all.

    `start` / `end` as 'YYYY-MM-DD'.
    """
    where, params = build_where(tickers, start, end)
    if src is not None:
        where = f'{where} AND Src = ?' if where else 'WHERE Src = ?'
        params.append(src)
    return db().execute(f'SELECT * FROM prices {where} ORDER BY Ticker, Date', params).df()

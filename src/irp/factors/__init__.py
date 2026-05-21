"""Quant factor computation for investment research.

Usage:
    from irp.factors import cross_section
    import datetime
    df = cross_section(datetime.date(2024, 12, 31), variant='A')
"""
from irp.factors.compute import cross_section, ticker_factor_history

__all__ = ['cross_section', 'ticker_factor_history']

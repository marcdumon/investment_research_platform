"""Shared factor metadata — labels, formatting sets, dropdown options.

Imported by both factors.py and backtest.py to avoid page-module duplication.
"""

FACTOR_LABELS: dict[str, str] = {
    'pe':           'P/E',
    'pb':           'P/B',
    'ps':           'P/S',
    'ev_ebitda':    'EV/EBITDA',
    'ev_ebit':      'EV/EBIT',
    'ev_sales':     'EV/Sales',
    'fcf_yield':    'FCF Yield',
    'gross_margin': 'Gross Margin',
    'op_margin':    'Op. Margin',
    'net_margin':   'Net Margin',
    'roe':          'ROE',
    'roa':          'ROA',
    'roic':         'ROIC',
    'fcf_margin':   'FCF Margin',
    'mom_12_1':    '12-1m Mom',
    'mom_6_1':     '6-1m Mom',
    'vol_21d':     'Vol 21d',
    'ma200_ratio': 'MA200 Ratio',
}

PCT_FACTORS: frozenset[str] = frozenset({
    'fcf_yield', 'gross_margin', 'op_margin', 'net_margin',
    'roe', 'roa', 'roic', 'fcf_margin',
    'mom_12_1', 'mom_6_1', 'vol_21d',
})

FACTOR_OPTIONS: list[dict] = [{'label': v, 'value': k} for k, v in FACTOR_LABELS.items()]

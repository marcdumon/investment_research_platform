"""Shared Dash table helpers — column formatters + statement-table renderer.

Centralises numeric column formatting (currency, percent, fixed) used by
both `/factors` and `/screener`, plus the html.Table boilerplate that
`/ticker` uses for income/balance/cashflow statements.
"""
from math import isfinite
from typing import Sequence

import pandas as pd
from dash import html
from dash.dash_table.Format import Format, Scheme, Symbol

from irp.ui.factor_meta import FACTOR_LABELS, PCT_FACTORS


def column_format(col: str) -> dict:
    """Dash DataTable column-format dict for a known factor column.

    `mktcap` is dollar-prefixed; percentage factors use the `percentage`
    scheme; remaining registered factors use 2-decimal fixed. Unknown
    columns return an empty dict (no special formatting).
    """
    if col == 'mktcap':
        return {
            'type': 'numeric',
            'format': Format(precision=1, scheme=Scheme.fixed, symbol=Symbol.yes, symbol_prefix='$'),
        }
    if col in PCT_FACTORS:
        return {
            'type': 'numeric',
            'format': Format(precision=1, scheme=Scheme.percentage),
        }
    if col in FACTOR_LABELS:
        return {
            'type': 'numeric',
            'format': Format(precision=2, scheme=Scheme.fixed),
        }
    return {}


def format_factor_value(val: object, factor: str) -> str:
    """String form of a factor value for hover labels + plain-text cells."""
    try:
        v = float(val)  # type: ignore[arg-type]
    except (TypeError, ValueError):
        return '—'
    if pd.isna(v) or not isfinite(v):
        return '—'
    if factor == 'mktcap':
        return f'${v / 1e9:.1f}B'
    if factor in PCT_FACTORS:
        return f'{v * 100:.1f}%'
    return f'{v:.1f}x'


def render_statement_table(
    df: pd.DataFrame,
    columns: Sequence[tuple[str, str]],
    class_name: str = 'stmt-table',
) -> html.Table:
    """Render a (date × line-item) statement as an html.Table.

    `columns` is `[(label, df_column_name), ...]` so the caller controls
    which fields appear and in what order.
    """
    headers = [
        html.Th(label, className='stmt-th' + (' stmt-th--item' if i == 0 else ''))
        for i, (label, _) in enumerate(columns)
    ]
    rows = [
        html.Tr([
            html.Td(
                str(row.get(field, ''))[:10] if i == 0 else str(row.get(field, '')),
                className='stmt-td' + (' stmt-td--item' if i == 0 else ''),
            )
            for i, (_, field) in enumerate(columns)
        ])
        for _, row in df.iterrows()
    ]
    return html.Table(
        className=class_name,
        children=[html.Thead(html.Tr(headers)), html.Tbody(rows)],
    )

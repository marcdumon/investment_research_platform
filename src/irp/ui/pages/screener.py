"""Stock screener page: progressive filter stack + scatter/histogram/price/correlation charts
and inline per-ticker detail panel (prices, fundamentals, factor history).
"""

import contextlib
import datetime
import logging
from math import isfinite
from typing import Any, Literal

import dash
import numpy as np
import pandas as pd
import plotly.colors as pc
import plotly.graph_objects as go
from dash import ALL, Input, Output, State, callback, ctx, dash_table as _dt, dcc, html
from dash.exceptions import PreventUpdate
from plotly.subplots import make_subplots

from irp.factors._cols import PRICE_CLOSE, PRICE_DATE, PRICE_TICKER
from irp.factors.registry import all_factors
from irp.ui.charts import (
    base_chart_layout as _chart_layout,
    corr_heatmap_figure as _corr_heatmap,
    empty_figure,
    scatter_chart_layout as _base_layout,
)
from irp.ui.factor_meta import FACTOR_LABELS, FACTOR_OPTIONS, PCT_FACTORS
from irp.ui.services import (
    factors_service,
    price_service,
    universe_service,
    watchlist_service,
)
from irp.ui.tables import column_format as _col_fmt
from irp.ui.theme import (
    ACCENT,
    DIV_COLOR,
    GRID,
    HOVER_LABEL,
    MUTED,
    SPLIT_COLOR,
    TABLE_STYLE,
)
from irp.ui.ticker_fmt import date_range_for_preset, fmt_statement

dash.register_page(__name__, path='/screener', name='Screener')

logger = logging.getLogger(__name__)

_PCT_FACTORS = PCT_FACTORS
_ALL_FACTOR_COLS = list(FACTOR_LABELS.keys())
_DEFAULT_DATE = (datetime.date.today() - datetime.timedelta(days=90)).isoformat()
_VARIANT_OPTIONS = [
    {'label': ' Annual', 'value': 'A'},
    {'label': ' Quarterly', 'value': 'Q'},
]
_MAX_PRICE_TICKERS = 50
_MAX_CORR_TICKERS = 150
_SECTOR_PALETTE = pc.qualitative.Set3
_RANGE_PRESETS = ['1M', '6M', '1Y', '3Y', '5Y', 'Max']
_CORR_WINDOW_OPTIONS = [
    {'label': '1 year', 'value': 252},
    {'label': '2 years', 'value': 504},
    {'label': '5 years', 'value': 1260},
]


# ── Helpers ───────────────────────────────────────────────────────────


def _apply_step(df: pd.DataFrame, step: dict) -> pd.DataFrame:
    if step['type'] == 'range':
        col, lo, hi = step['col'], step.get('min'), step.get('max')
        if col in df.columns:
            if lo is not None:
                df = df[df[col] >= lo]
            if hi is not None:
                df = df[df[col] <= hi]
    elif step['type'] == 'keep':
        df = df[df['Ticker'].isin(step['tickers'])]
    elif step['type'] == 'remove':
        df = df[~df['Ticker'].isin(step['tickers'])]
    return df


def _apply_steps(df: pd.DataFrame, steps: list[dict]) -> pd.DataFrame:
    for step in steps:
        df = _apply_step(df, step)
    return df


def _auto_name(steps: list[dict], as_of_date: str | None) -> str:
    range_parts = [
        s['label'].replace(' ', '').replace('≥', 'ge').replace('≤', 'le')
        for s in (steps or [])
        if s.get('type') == 'range'
    ]
    suffix = (as_of_date or '')[:10]
    parts = range_parts + ([suffix] if suffix else [])
    return '_'.join(parts) if parts else f'screener_{suffix}'


# ── Layout ────────────────────────────────────────────────────────────

layout = html.Div(
    className='screener-page',
    children=[
        dcc.Store(id='screener-init', data=1),
        dcc.Store(id='screener-base-store'),
        dcc.Store(id='screener-steps-store', data=[]),
        dcc.Store(id='screener-result-store'),
        dcc.Store(id='screener-selection-store', data=[]),
        dcc.Store(id='screener-wl-trigger', data=0),
        dcc.Store(id='screener-wl-pending-delete', data=None),
        dcc.Store(id='screener-detail-ticker', data=None),
        html.H2('Stock Screener', className='page-title'),
        html.P(
            'Build a filter stack to narrow the universe. '
            'Lasso-select points in the scatter plot, then Keep or Remove.',
            className='page-subtitle',
        ),
        # ── Universe controls ──────────────────────────────────────────
        html.Div(
            className='screener-universe-row control-row',
            children=[
                dcc.DatePickerSingle(
                    id='screener-date',
                    date=_DEFAULT_DATE,
                    display_format='YYYY-MM-DD',
                    style={'fontSize': '13px'},
                ),
                dcc.RadioItems(
                    id='screener-variant',
                    options=_VARIANT_OPTIONS,
                    value='A',
                    inline=True,
                    labelClassName='check-item',
                ),
                dcc.Dropdown(
                    id='screener-market',
                    placeholder='All markets',
                    clearable=True,
                    className='filter-dropdown',
                    style={'minWidth': '130px'},
                ),
                dcc.Dropdown(
                    id='screener-sector',
                    placeholder='All sectors',
                    clearable=True,
                    className='filter-dropdown',
                    style={'minWidth': '160px'},
                ),
                html.Button(
                    'Run', id='screener-run-btn', className='run-btn', n_clicks=0
                ),
                html.Span(
                    id='screener-universe-count',
                    style={'color': MUTED, 'fontSize': '12px'},
                ),
            ],
        ),
        # ── Filter stack ───────────────────────────────────────────────
        html.Div(
            className='screener-filter-panel',
            style={'margin': '12px 0'},
            children=[
                html.Div(id='screener-step-stack', style={'marginBottom': '8px'}),
                # Add-filter row
                html.Div(
                    className='control-row',
                    style={'gap': '8px', 'alignItems': 'flex-end'},
                    children=[
                        dcc.Dropdown(
                            id='screener-add-factor',
                            options=FACTOR_OPTIONS,
                            value='pe',
                            clearable=False,
                            className='filter-dropdown',
                            style={'minWidth': '140px'},
                        ),
                        dcc.Input(
                            id='screener-add-min',
                            type='number',
                            placeholder='Min',
                            debounce=False,
                            style={'width': '80px', 'fontSize': '12px'},
                        ),
                        dcc.Input(
                            id='screener-add-max',
                            type='number',
                            placeholder='Max',
                            debounce=False,
                            style={'width': '80px', 'fontSize': '12px'},
                        ),
                        html.Button(
                            '+ Add filter',
                            id='screener-add-filter-btn',
                            className='run-btn',
                            n_clicks=0,
                            style={'fontSize': '11px', 'padding': '4px 10px'},
                        ),
                        html.Span(
                            '(% factors: enter as %, e.g. 15 for 15%)',
                            style={'color': MUTED, 'fontSize': '11px'},
                        ),
                    ],
                ),
            ],
        ),
        # ── Chart selection action bar (hidden until selection exists) ──
        html.Div(
            id='screener-selection-bar',
            style={
                'display': 'none',
                'gap': '12px',
                'alignItems': 'center',
                'padding': '8px 12px',
                'background': 'var(--surface-2)',
                'borderRadius': '4px',
                'marginBottom': '12px',
            },
            children=[
                html.Span(
                    id='screener-selection-count',
                    style={'color': MUTED, 'fontSize': '12px'},
                ),
                html.Button(
                    'Keep selection',
                    id='screener-keep-btn',
                    className='run-btn',
                    style={
                        'fontSize': '11px',
                        'padding': '4px 10px',
                        'background': '#2ea043',
                    },
                ),
                html.Button(
                    'Remove selection',
                    id='screener-remove-btn',
                    className='run-btn',
                    style={
                        'fontSize': '11px',
                        'padding': '4px 10px',
                        'background': '#e05252',
                    },
                ),
                html.Button(
                    'Clear',
                    id='screener-clear-selection-btn',
                    className='run-btn',
                    style={'fontSize': '11px', 'padding': '4px 10px'},
                ),
            ],
        ),
        # ── Charts ────────────────────────────────────────────────────
        dcc.Tabs(
            id='screener-chart-tabs',
            value='scatter',
            className='ticker-tabs',
            children=[
                # Scatter tab
                dcc.Tab(
                    label='Scatter',
                    value='scatter',
                    className='ticker-tab',
                    selected_className='ticker-tab--active',
                    children=[
                        html.Div(
                            className='control-row',
                            style={'padding': '8px 0', 'gap': '12px'},
                            children=[
                                html.Div([
                                    html.Label('X axis', className='control-label'),
                                    dcc.Dropdown(
                                        id='screener-x-factor',
                                        options=FACTOR_OPTIONS,
                                        value='pe',
                                        clearable=False,
                                        className='filter-dropdown',
                                        style={'minWidth': '130px'},
                                    ),
                                ]),
                                html.Div([
                                    html.Label('Y axis', className='control-label'),
                                    dcc.Dropdown(
                                        id='screener-y-factor',
                                        options=FACTOR_OPTIONS,
                                        value='roe',
                                        clearable=False,
                                        className='filter-dropdown',
                                        style={'minWidth': '130px'},
                                    ),
                                ]),
                                html.Div([
                                    html.Label('Color', className='control-label'),
                                    dcc.Dropdown(
                                        id='screener-color-by',
                                        options=(
                                            [
                                                {'label': 'Sector', 'value': 'sector'},
                                                {'label': 'Market', 'value': 'market'},
                                                {'label': 'None', 'value': 'none'},
                                            ]
                                            + [
                                                {'label': f.label, 'value': f.name}
                                                for f in all_factors()
                                            ]
                                        ),
                                        value='sector',
                                        clearable=False,
                                        className='filter-dropdown',
                                        style={'minWidth': '140px'},
                                    ),
                                ]),
                                html.Div([
                                    html.Label('X scale', className='control-label'),
                                    dcc.RadioItems(
                                        id='screener-x-scale',
                                        options=[
                                            {'label': ' Lin', 'value': 'linear'},
                                            {'label': ' Log', 'value': 'log'},
                                        ],
                                        value='linear',
                                        inline=True,
                                        labelClassName='check-item',
                                    ),
                                ]),
                                html.Div([
                                    html.Label('Y scale', className='control-label'),
                                    dcc.RadioItems(
                                        id='screener-y-scale',
                                        options=[
                                            {'label': ' Lin', 'value': 'linear'},
                                            {'label': ' Log', 'value': 'log'},
                                        ],
                                        value='linear',
                                        inline=True,
                                        labelClassName='check-item',
                                    ),
                                ]),
                                html.Div([
                                    html.Label(
                                        'Color scale', className='control-label'
                                    ),
                                    dcc.RadioItems(
                                        id='screener-color-scale',
                                        options=[
                                            {'label': ' Lin', 'value': 'linear'},
                                            {'label': ' Log', 'value': 'log'},
                                        ],
                                        value='log',
                                        inline=True,
                                        labelClassName='check-item',
                                    ),
                                ]),
                            ],
                        ),
                        dcc.Loading(
                            dcc.Graph(
                                id='screener-scatter',
                                config={
                                    'displayModeBar': True,
                                    'modeBarButtonsToAdd': ['select2d', 'lasso2d'],
                                    'modeBarButtonsToRemove': ['toImage'],
                                },
                                style={'height': '460px'},
                            )
                        ),
                    ],
                ),
                # Histogram tab
                dcc.Tab(
                    label='Histogram',
                    value='histogram',
                    className='ticker-tab',
                    selected_className='ticker-tab--active',
                    children=[
                        html.Div(
                            className='control-row',
                            style={'padding': '8px 0'},
                            children=[
                                html.Label('Factor', className='control-label'),
                                dcc.Dropdown(
                                    id='screener-hist-factor',
                                    options=FACTOR_OPTIONS,
                                    value='pe',
                                    clearable=False,
                                    className='filter-dropdown',
                                    style={'minWidth': '140px'},
                                ),
                            ],
                        ),
                        dcc.Loading(
                            dcc.Graph(
                                id='screener-histogram',
                                config={'displayModeBar': False},
                                style={'height': '400px'},
                            )
                        ),
                    ],
                ),
                # Prices tab
                dcc.Tab(
                    label='Prices',
                    value='prices',
                    className='ticker-tab',
                    selected_className='ticker-tab--active',
                    children=[
                        html.Div(
                            className='control-row',
                            style={'padding': '8px 0', 'gap': '12px'},
                            children=[
                                html.Span(
                                    id='screener-prices-info',
                                    style={'color': MUTED, 'fontSize': '12px'},
                                ),
                                html.Button(
                                    id='screener-load-prices-btn',
                                    children='Load Price History',
                                    className='run-btn',
                                    n_clicks=0,
                                    style={'fontSize': '11px', 'padding': '4px 10px'},
                                ),
                            ],
                        ),
                        dcc.Loading(
                            dcc.Graph(
                                id='screener-prices-chart',
                                config={'displayModeBar': False},
                                style={'height': '400px'},
                            )
                        ),
                    ],
                ),
                # Correlation tab
                dcc.Tab(
                    label='Correlation',
                    value='correlation',
                    className='ticker-tab',
                    selected_className='ticker-tab--active',
                    children=[
                        html.Div(
                            className='control-row',
                            style={'padding': '8px 0', 'gap': '12px'},
                            children=[
                                html.Label('Return window', className='control-label'),
                                dcc.Dropdown(
                                    id='screener-corr-window',
                                    options=_CORR_WINDOW_OPTIONS,
                                    value=252,
                                    clearable=False,
                                    className='filter-dropdown',
                                    style={'minWidth': '110px'},
                                ),
                                html.Button(
                                    'Run Correlation',
                                    id='screener-corr-run-btn',
                                    className='run-btn',
                                    n_clicks=0,
                                    style={'fontSize': '11px', 'padding': '4px 10px'},
                                ),
                                html.Span(
                                    id='screener-corr-info',
                                    style={'color': MUTED, 'fontSize': '12px'},
                                ),
                            ],
                        ),
                        dcc.Loading(
                            dcc.Graph(
                                id='screener-corr-chart',
                                figure=empty_figure(
                                    'Select filters and click Run Correlation'
                                ),
                                config={'displayModeBar': False},
                                style={'minHeight': '400px'},
                            )
                        ),
                    ],
                ),
            ],
        ),
        # ── Results table ─────────────────────────────────────────────
        dcc.Loading(html.Div(id='screener-table-container')),
        # ── Ticker detail panel ───────────────────────────────────────
        html.Div(
            id='screener-detail-panel',
            style={'display': 'none'},
            children=[
                html.Div(
                    style={
                        'display': 'flex',
                        'justifyContent': 'space-between',
                        'alignItems': 'center',
                        'padding': '12px 0 8px 0',
                        'borderTop': '1px solid var(--border)',
                        'marginTop': '16px',
                    },
                    children=[
                        html.H3(
                            id='screener-detail-title',
                            style={
                                'color': 'var(--text)',
                                'fontSize': '14px',
                                'margin': '0',
                            },
                        ),
                        html.Div(
                            style={
                                'display': 'flex',
                                'gap': '8px',
                                'alignItems': 'center',
                            },
                            children=[
                                html.Button(
                                    '− Remove from list',
                                    id='screener-detail-remove-btn',
                                    n_clicks=0,
                                    style={
                                        'fontSize': '11px',
                                        'padding': '3px 10px',
                                        'background': '#e05252',
                                        'border': '1px solid #e05252',
                                        'color': '#fff',
                                        'cursor': 'pointer',
                                        'borderRadius': '4px',
                                    },
                                ),
                                html.Button(
                                    '✕ Close',
                                    id='screener-detail-close-btn',
                                    n_clicks=0,
                                    style={
                                        'background': 'none',
                                        'border': 'none',
                                        'color': MUTED,
                                        'cursor': 'pointer',
                                        'fontSize': '12px',
                                    },
                                ),
                            ],
                        ),
                    ],
                ),
                dcc.Tabs(
                    id='screener-detail-tabs',
                    value='detail-prices',
                    className='ticker-tabs',
                    children=[
                        # ── Prices + Volume tab ────────────────────────
                        dcc.Tab(
                            label='Prices',
                            value='detail-prices',
                            className='ticker-tab',
                            selected_className='ticker-tab--active',
                            children=[
                                html.Div(
                                    className='control-row',
                                    style={
                                        'padding': '8px 0',
                                        'gap': '8px',
                                        'flexWrap': 'wrap',
                                    },
                                    children=[
                                        *[
                                            html.Button(
                                                p,
                                                id={
                                                    'type': 'detail-preset-btn',
                                                    'index': p,
                                                },
                                                n_clicks=0,
                                                style={
                                                    'fontSize': '11px',
                                                    'padding': '2px 8px',
                                                    'background': 'var(--surface-2)',
                                                    'border': '1px solid var(--border)',
                                                    'color': MUTED,
                                                    'cursor': 'pointer',
                                                    'borderRadius': '3px',
                                                },
                                            )
                                            for p in _RANGE_PRESETS
                                        ],
                                        html.Span(
                                            'MA:',
                                            style={
                                                'color': MUTED,
                                                'fontSize': '11px',
                                                'marginLeft': '8px',
                                            },
                                        ),
                                        dcc.Checklist(
                                            id='screener-detail-ma',
                                            options=[
                                                {'label': ' MA50', 'value': 'MA50'},
                                                {'label': ' MA200', 'value': 'MA200'},
                                            ],
                                            value=[],
                                            inline=True,
                                            labelClassName='check-item',
                                        ),
                                    ],
                                ),
                                dcc.Store(id='screener-detail-preset', data='1Y'),
                                dcc.Loading(
                                    dcc.Graph(
                                        id='screener-detail-price-chart',
                                        config={'displayModeBar': False},
                                        style={'height': '420px'},
                                    )
                                ),
                            ],
                        ),
                        # ── Fundamentals tab ──────────────────────────
                        dcc.Tab(
                            label='Fundamentals',
                            value='detail-fundamentals',
                            className='ticker-tab',
                            selected_className='ticker-tab--active',
                            children=[
                                html.Div(
                                    className='control-row',
                                    style={'padding': '8px 0', 'gap': '12px'},
                                    children=[
                                        dcc.RadioItems(
                                            id='screener-detail-stmt-variant',
                                            options=[
                                                {'label': ' Annual', 'value': 'A'},
                                                {'label': ' Quarterly', 'value': 'Q'},
                                            ],
                                            value='A',
                                            inline=True,
                                            labelClassName='check-item',
                                        ),
                                    ],
                                ),
                                dcc.Tabs(
                                    id='screener-detail-stmt-tabs',
                                    value='detail-income',
                                    className='ticker-tabs',
                                    children=[
                                        dcc.Tab(
                                            label='Income',
                                            value='detail-income',
                                            className='ticker-tab',
                                            selected_className='ticker-tab--active',
                                            children=[
                                                dcc.Loading(
                                                    html.Div(
                                                        id='screener-detail-income'
                                                    )
                                                ),
                                                html.Small(
                                                    id='screener-detail-income-note',
                                                    style={'color': MUTED},
                                                ),
                                            ],
                                        ),
                                        dcc.Tab(
                                            label='Balance Sheet',
                                            value='detail-balance',
                                            className='ticker-tab',
                                            selected_className='ticker-tab--active',
                                            children=[
                                                dcc.Loading(
                                                    html.Div(
                                                        id='screener-detail-balance'
                                                    )
                                                ),
                                                html.Small(
                                                    id='screener-detail-balance-note',
                                                    style={'color': MUTED},
                                                ),
                                            ],
                                        ),
                                        dcc.Tab(
                                            label='Cash Flow',
                                            value='detail-cashflow',
                                            className='ticker-tab',
                                            selected_className='ticker-tab--active',
                                            children=[
                                                dcc.Loading(
                                                    html.Div(
                                                        id='screener-detail-cashflow'
                                                    )
                                                ),
                                                html.Small(
                                                    id='screener-detail-cashflow-note',
                                                    style={'color': MUTED},
                                                ),
                                            ],
                                        ),
                                    ],
                                ),
                            ],
                        ),
                        # ── Factor History tab ────────────────────────
                        dcc.Tab(
                            label='Factor History',
                            value='detail-factors',
                            className='ticker-tab',
                            selected_className='ticker-tab--active',
                            children=[
                                html.Div(
                                    className='control-row',
                                    style={'padding': '8px 0', 'gap': '12px'},
                                    children=[
                                        dcc.RadioItems(
                                            id='screener-detail-factor-variant',
                                            options=[
                                                {'label': ' Annual', 'value': 'A'},
                                                {'label': ' Quarterly', 'value': 'Q'},
                                            ],
                                            value='A',
                                            inline=True,
                                            labelClassName='check-item',
                                        ),
                                    ],
                                ),
                                dcc.Loading(html.Div(id='screener-detail-factors')),
                            ],
                        ),
                    ],
                ),
            ],
        ),
        # ── Save panel ────────────────────────────────────────────────
        html.Div(
            style={
                'marginTop': '24px',
                'borderTop': '1px solid var(--border)',
                'paddingTop': '16px',
            },
            children=[
                html.H3(
                    'Save as Watchlist',
                    style={'color': MUTED, 'fontSize': '13px', 'marginBottom': '8px'},
                ),
                html.Div(
                    className='control-row',
                    style={'gap': '10px'},
                    children=[
                        dcc.Input(
                            id='screener-watchlist-name',
                            type='text',
                            placeholder='watchlist name…',
                            debounce=False,
                            style={'fontSize': '12px', 'minWidth': '300px'},
                        ),
                        html.Button(
                            'Save Watchlist',
                            id='screener-save-btn',
                            className='run-btn',
                            n_clicks=0,
                        ),
                        html.Span(
                            id='screener-save-status',
                            style={'color': MUTED, 'fontSize': '12px'},
                        ),
                    ],
                ),
                html.Div(
                    id='screener-wl-description',
                    style={
                        'color': MUTED,
                        'fontSize': '11px',
                        'marginTop': '4px',
                        'fontStyle': 'italic',
                    },
                ),
                html.Div(
                    id='screener-watchlists-container', style={'marginTop': '16px'}
                ),
            ],
        ),
    ],
)


# ── Callbacks ─────────────────────────────────────────────────────────


@callback(
    Output('screener-market', 'options'),
    Output('screener-sector', 'options'),
    Input('screener-init', 'data'),
)
def load_options(_: Any) -> tuple[list, list]:
    try:
        df = universe_service._get_companies()
    except Exception:
        return [], []
    if df.empty:
        return [], []
    markets = sorted(df['Market'].dropna().unique())
    sectors = sorted(df['Sector'].dropna().unique())
    return (
        [{'label': m, 'value': m} for m in markets],
        [{'label': s, 'value': s} for s in sectors],
    )


@callback(
    Output('screener-base-store', 'data'),
    Output('screener-steps-store', 'data'),
    Input('screener-run-btn', 'n_clicks'),
    State('screener-date', 'date'),
    State('screener-variant', 'value'),
    State('screener-market', 'value'),
    State('screener-sector', 'value'),
    running=[
        (Output('screener-run-btn', 'disabled'), True, False),
        (Output('screener-run-btn', 'children'), 'Running…', 'Run'),
    ],
    prevent_initial_call=True,
)
def run_screener(
    n_clicks: int,
    date_str: str | None,
    variant: Literal['A', 'Q'],
    market: str | None,
    sector: str | None,
) -> tuple[Any, list]:
    if not n_clicks or not date_str:
        raise PreventUpdate
    as_of = datetime.date.fromisoformat(date_str[:10])
    df = factors_service._load_cross_section(
        as_of, variant, enrich_company_columns=False
    )
    if df.empty:
        return {}, []
    df = df.reset_index()
    try:
        comp = universe_service._get_companies()[
            ['Ticker', 'Company Name', 'Sector', 'Market']
        ].fillna('')
        df = df.merge(comp, on='Ticker', how='left')
    except Exception:
        logger.warning('run_screener: could not merge company metadata')
    if market:
        df = df[df['Market'] == market]
    if sector:
        df = df[df['Sector'] == sector]
    meta = {
        'n': len(df),
        'date': date_str[:10],
        'variant': variant,
        'market': market or '',
        'sector': sector or '',
    }
    return {'records': df.to_dict('records'), 'meta': meta}, []


@callback(
    Output('screener-result-store', 'data'),
    Output('screener-universe-count', 'children'),
    Input('screener-base-store', 'data'),
    Input('screener-steps-store', 'data'),
)
def apply_steps_callback(
    base: dict | None,
    steps: list | None,
) -> tuple[Any, str]:
    if not base or not base.get('records'):
        return {}, ''
    df = pd.DataFrame(base['records'])
    base_n = len(df)
    df = _apply_steps(df, steps or [])
    result_n = len(df)
    meta = base.get('meta', {})
    parts = [f'Base: {base_n:,} stocks']
    if meta.get('market'):
        parts[0] += f' ({meta["market"]}'
        if meta.get('sector'):
            parts[0] += f' / {meta["sector"]}'
        parts[0] += ')'
    count_label = f'{" → ".join(parts)}  |  Filtered: {result_n:,}'
    return {'records': df.to_dict('records'), 'n': result_n}, count_label


@callback(
    Output('screener-step-stack', 'children'),
    Input('screener-base-store', 'data'),
    Input('screener-steps-store', 'data'),
)
def render_step_stack(base: dict | None, steps: list | None) -> Any:
    if not steps:
        return html.Span(
            'No additional filters.', style={'color': MUTED, 'fontSize': '12px'}
        )
    if not base or not base.get('records'):
        return []

    df = pd.DataFrame(base['records'])
    current_n = len(df)
    rows = []
    for i, step in enumerate(steps):
        before_n = current_n
        df = _apply_step(df, step)
        after_n = len(df)
        current_n = after_n
        rows.append(
            html.Div(
                style={
                    'display': 'flex',
                    'alignItems': 'center',
                    'gap': '12px',
                    'padding': '4px 8px',
                    'background': 'var(--surface-2)',
                    'borderRadius': '4px',
                    'marginBottom': '4px',
                    'fontSize': '12px',
                },
                children=[
                    html.Span(f'Step {i + 1}:', style={'color': MUTED}),
                    html.Span(
                        step.get('label', ''),
                        style={'color': 'var(--text)', 'flex': '1'},
                    ),
                    html.Span(
                        f'{after_n:,} / {before_n:,}',
                        style={
                            'color': ACCENT,
                            'minWidth': '90px',
                            'textAlign': 'right',
                        },
                    ),
                    html.Button(
                        '×',
                        id={'type': 'delete-step-btn', 'index': i},
                        n_clicks=0,
                        style={
                            'background': 'none',
                            'border': 'none',
                            'color': MUTED,
                            'cursor': 'pointer',
                            'fontSize': '14px',
                            'padding': '0 4px',
                        },
                    ),
                ],
            )
        )
    return rows


@callback(
    Output('screener-steps-store', 'data', allow_duplicate=True),
    Input('screener-add-filter-btn', 'n_clicks'),
    Input('screener-keep-btn', 'n_clicks'),
    Input('screener-remove-btn', 'n_clicks'),
    Input('screener-detail-remove-btn', 'n_clicks'),
    Input({'type': 'delete-step-btn', 'index': ALL}, 'n_clicks'),
    Input({'type': 'wl-load-btn', 'index': ALL}, 'n_clicks'),
    State('screener-add-factor', 'value'),
    State('screener-add-min', 'value'),
    State('screener-add-max', 'value'),
    State('screener-selection-store', 'data'),
    State('screener-detail-ticker', 'data'),
    State('screener-steps-store', 'data'),
    prevent_initial_call=True,
)
def mutate_steps(
    add_n: int,
    keep_n: int,
    remove_n: int,
    detail_remove_n: int,
    delete_clicks: list,
    load_clicks: list,
    add_factor: str | None,
    add_min: float | None,
    add_max: float | None,
    selection: list | None,
    detail_ticker: str | None,
    steps: list | None,
) -> list:
    steps = list(steps or [])
    triggered = ctx.triggered_id

    if triggered == 'screener-add-filter-btn':
        if not add_factor:
            raise PreventUpdate
        if add_min is None and add_max is None:
            raise PreventUpdate
        is_pct = add_factor in _PCT_FACTORS
        stored_min = add_min / 100 if (is_pct and add_min is not None) else add_min
        stored_max = add_max / 100 if (is_pct and add_max is not None) else add_max
        def _fmt(v):
            return f'{v:g}%' if is_pct else f'{v:g}'
        label = FACTOR_LABELS.get(add_factor, add_factor)
        if add_min is not None and add_max is not None:
            label += f' {_fmt(add_min)}–{_fmt(add_max)}'
        elif add_min is not None:
            label += f' ≥ {_fmt(add_min)}'
        else:
            label += f' ≤ {_fmt(add_max)}'
        steps.append({
            'type': 'range',
            'col': add_factor,
            'min': stored_min,
            'max': stored_max,
            'label': label,
        })
        return steps

    if triggered in ('screener-keep-btn', 'screener-remove-btn'):
        tickers = selection or []
        if not tickers:
            raise PreventUpdate
        kind = 'keep' if triggered == 'screener-keep-btn' else 'remove'
        n = len(tickers)
        steps.append({
            'type': kind,
            'tickers': tickers,
            'label': f'chart-{kind} {n:,} stocks',
        })
        return steps

    if triggered == 'screener-detail-remove-btn':
        if not detail_ticker:
            raise PreventUpdate
        steps.append({
            'type': 'remove',
            'tickers': [detail_ticker],
            'label': detail_ticker,
        })
        return steps

    if isinstance(triggered, dict) and triggered.get('type') == 'delete-step-btn':
        if not any(delete_clicks):
            raise PreventUpdate
        idx = triggered['index']
        if 0 <= idx < len(steps):
            steps.pop(idx)
        return steps

    if isinstance(triggered, dict) and triggered.get('type') == 'wl-load-btn':
        if not any(load_clicks):
            raise PreventUpdate
        name = triggered['index']
        try:
            tickers = watchlist_service.load_watchlist(name)
        except KeyError:
            raise PreventUpdate from None
        return [
            {
                'type': 'keep',
                'tickers': tickers,
                'label': f'watchlist "{name}" ({len(tickers):,} stocks)',
            }
        ]

    raise PreventUpdate


@callback(
    Output('screener-add-min', 'value'),
    Output('screener-add-max', 'value'),
    Input('screener-add-filter-btn', 'n_clicks'),
    prevent_initial_call=True,
)
def clear_filter_inputs(_: int) -> tuple[None, None]:
    return None, None


@callback(
    Output('screener-selection-store', 'data'),
    Input('screener-scatter', 'selectedData'),
    Input('screener-clear-selection-btn', 'n_clicks'),
    prevent_initial_call=True,
)
def capture_selection(selected: dict | None, clear_n: int) -> list:
    if ctx.triggered_id == 'screener-clear-selection-btn':
        return []
    if not selected or not selected.get('points'):
        return []
    tickers = []
    for pt in selected['points']:
        cd = pt.get('customdata')
        if cd and len(cd) >= 1:
            first = cd[0]
            ticker = first[0] if isinstance(first, (list, tuple)) else first
            tickers.append(ticker)
    return list(dict.fromkeys(tickers))


@callback(
    Output('screener-selection-bar', 'style'),
    Output('screener-selection-count', 'children'),
    Input('screener-selection-store', 'data'),
)
def show_selection_bar(selection: list | None) -> tuple[dict, str]:
    n = len(selection or [])
    if n == 0:
        return {'display': 'none'}, ''
    return (
        {
            'display': 'flex',
            'gap': '12px',
            'alignItems': 'center',
            'padding': '8px 12px',
            'background': 'var(--surface-2)',
            'borderRadius': '4px',
            'marginBottom': '12px',
        },
        f'{n:,} stock{"s" if n != 1 else ""} selected in chart',
    )


@callback(
    Output('screener-scatter', 'figure'),
    Input('screener-result-store', 'data'),
    Input('screener-x-factor', 'value'),
    Input('screener-y-factor', 'value'),
    Input('screener-color-by', 'value'),
    Input('screener-x-scale', 'value'),
    Input('screener-y-scale', 'value'),
    Input('screener-color-scale', 'value'),
)
def render_scatter(
    result: dict | None,
    x_factor: str | None,
    y_factor: str | None,
    color_by: str | None,
    x_scale: str,
    y_scale: str,
    color_scale: str,
) -> go.Figure:
    if not result or not result.get('records'):
        return empty_figure('Run screener to see data.')
    if not x_factor or not y_factor:
        return empty_figure()

    df = pd.DataFrame(result['records'])
    if x_factor not in df.columns or y_factor not in df.columns:
        return empty_figure(f'Factor "{x_factor}" or "{y_factor}" not in data.')

    df = df.dropna(subset=[x_factor, y_factor])
    df = df[df[x_factor].apply(lambda v: isfinite(float(v)) if pd.notna(v) else False)]
    df = df[df[y_factor].apply(lambda v: isfinite(float(v)) if pd.notna(v) else False)]

    if df.empty:
        return empty_figure('No valid data for selected factors.')

    x_label = FACTOR_LABELS.get(x_factor, x_factor)
    y_label = FACTOR_LABELS.get(y_factor, y_factor)

    cd = (
        df[['Ticker', 'Company Name']].fillna('').values.tolist()
        if 'Company Name' in df.columns
        else df[['Ticker']].values.tolist()
    )

    # Categorical color (sector / market) → one trace per group
    cat_col = None
    if color_by == 'sector' and 'Sector' in df.columns:
        cat_col = 'Sector'
    elif color_by == 'market' and 'Market' in df.columns:
        cat_col = 'Market'

    # Continuous color (any registered factor)
    factor_color = (
        color_by
        if (color_by and color_by in FACTOR_LABELS and color_by in df.columns)
        else None
    )

    traces = []
    if cat_col:
        cat_series = df[cat_col].fillna('').replace('', 'Unknown')
        groups = sorted(cat_series.unique())
        for i, grp in enumerate(groups):
            mask = cat_series == grp
            sub = df[mask]
            if sub.empty:
                continue
            sub_cd = (
                sub[['Ticker', 'Company Name']].fillna('').values.tolist()
                if 'Company Name' in sub.columns
                else sub[['Ticker']].values.tolist()
            )
            traces.append(
                go.Scatter(
                    x=sub[x_factor].tolist(),
                    y=sub[y_factor].tolist(),
                    mode='markers',
                    name=grp,
                    customdata=sub_cd,
                    hovertemplate=(
                        f'<b>%{{customdata[0]}}</b> %{{customdata[1]}}<br>'
                        f'{x_label}: %{{x:.3f}}<br>{y_label}: %{{y:.3f}}<extra></extra>'
                    ),
                    hoverlabel=dict(**HOVER_LABEL),
                    marker=dict(
                        size=6,
                        color=_SECTOR_PALETTE[i % len(_SECTOR_PALETTE)],
                        opacity=0.75,
                    ),
                )
            )
    elif factor_color:
        color_label = FACTOR_LABELS.get(factor_color, factor_color)
        color_vals = pd.to_numeric(df[factor_color], errors='coerce')
        is_pct = factor_color in _PCT_FACTORS
        fmt = '.1%' if is_pct else '.2f'
        if color_scale == 'log':
            plot_color = color_vals.apply(
                lambda v: (
                    np.sign(v) * np.log10(abs(v))
                    if pd.notna(v) and v != 0
                    else (0.0 if pd.notna(v) else np.nan)
                )
            )
            color_label_full = f'{color_label} (log)'
        else:
            plot_color = color_vals
            color_label_full = color_label
        valid_plot = plot_color.dropna()
        cmin = (
            float(valid_plot.quantile(0.02))
            if len(valid_plot) >= 10
            else float(valid_plot.min())
        )
        cmax = (
            float(valid_plot.quantile(0.98))
            if len(valid_plot) >= 10
            else float(valid_plot.max())
        )
        if cmin == cmax:
            cmin -= 0.5
            cmax += 0.5
        traces.append(
            go.Scatter(
                x=df[x_factor].tolist(),
                y=df[y_factor].tolist(),
                mode='markers',
                name='',
                showlegend=False,
                customdata=list(zip(cd, color_vals.tolist(), strict=False)),
                hovertemplate=(
                    f'<b>%{{customdata[0][0]}}</b> %{{customdata[0][1]}}<br>'
                    f'{x_label}: %{{x:.3f}}<br>{y_label}: %{{y:.3f}}<br>'
                    f'{color_label}: %{{customdata[1]:{fmt}}}<extra></extra>'
                ),
                hoverlabel=dict(**HOVER_LABEL),
                marker=dict(
                    size=6,
                    color=plot_color.tolist(),
                    colorscale='RdYlGn',
                    cmin=cmin,
                    cmax=cmax,
                    showscale=True,
                    colorbar=dict(
                        title=dict(
                            text=color_label_full, font=dict(color=MUTED, size=11)
                        ),
                        thickness=12,
                        len=0.8,
                        tickfont=dict(color=MUTED, size=10),
                        outlinewidth=0,
                    ),
                    opacity=0.8,
                ),
            )
        )
    else:
        traces.append(
            go.Scatter(
                x=df[x_factor].tolist(),
                y=df[y_factor].tolist(),
                mode='markers',
                name='',
                customdata=cd,
                hovertemplate=(
                    f'<b>%{{customdata[0]}}</b> %{{customdata[1]}}<br>'
                    f'{x_label}: %{{x:.3f}}<br>{y_label}: %{{y:.3f}}<extra></extra>'
                ),
                hoverlabel=dict(**HOVER_LABEL),
                marker=dict(size=6, color=ACCENT, opacity=0.7),
            )
        )

    n = len(df)
    fig = go.Figure(
        data=traces,
        layout=_base_layout(
            title=dict(
                text=f'{x_label} vs {y_label}  ({n:,} stocks)',
                font=dict(color=MUTED, size=12),
                x=0,
                xref='paper',
            ),
            xaxis_title=x_label,
            yaxis_title=y_label,
            xaxis_type=x_scale,
            yaxis_type=y_scale,
        ),
    )
    return fig


@callback(
    Output('screener-histogram', 'figure'),
    Input('screener-result-store', 'data'),
    Input('screener-hist-factor', 'value'),
)
def render_histogram(result: dict | None, factor: str | None) -> go.Figure:
    if not result or not result.get('records') or not factor:
        return empty_figure('Run screener to see data.')

    df = pd.DataFrame(result['records'])
    if factor not in df.columns:
        return empty_figure(f'Factor "{factor}" not in data.')

    vals = df[factor].dropna()
    vals = vals[vals.apply(lambda v: isfinite(float(v)))]
    if vals.empty:
        return empty_figure('No valid data.')

    label = FACTOR_LABELS.get(factor, factor)
    fig = go.Figure(
        data=go.Histogram(
            x=vals.tolist(),
            nbinsx=60,
            marker_color=ACCENT,
            opacity=0.8,
            hovertemplate=f'{label}: %{{x:.3f}}<br>Count: %{{y}}<extra></extra>',
        ),
        layout=_base_layout(
            title=dict(
                text=f'{label} distribution  ({len(vals):,} stocks)',
                font=dict(color=MUTED, size=12),
                x=0,
                xref='paper',
            ),
            xaxis_title=label,
            yaxis_title='Count',
            bargap=0.05,
        ),
    )
    return fig


@callback(
    Output('screener-prices-info', 'children'),
    Output('screener-load-prices-btn', 'disabled'),
    Input('screener-result-store', 'data'),
)
def update_prices_controls(result: dict | None) -> tuple[str, bool]:
    if not result or not result.get('records'):
        return 'Run screener first.', True
    n = result.get('n', len(result['records']))
    if n > _MAX_PRICE_TICKERS:
        return (
            f'{n:,} stocks — narrow to ≤ {_MAX_PRICE_TICKERS} before loading prices.',
            True,
        )
    return f'{n} stocks selected — 1-year price history.', False


@callback(
    Output('screener-prices-chart', 'figure'),
    Input('screener-load-prices-btn', 'n_clicks'),
    State('screener-result-store', 'data'),
    running=[
        (
            Output('screener-load-prices-btn', 'children'),
            'Loading…',
            'Load Price History',
        )
    ],
    prevent_initial_call=True,
)
def load_prices(n_clicks: int, result: dict | None) -> go.Figure:
    if not n_clicks or not result or not result.get('records'):
        raise PreventUpdate

    tickers = [r['Ticker'] for r in result['records'] if r.get('Ticker')]
    if not tickers or len(tickers) > _MAX_PRICE_TICKERS:
        return empty_figure(f'Narrow to ≤ {_MAX_PRICE_TICKERS} stocks first.')

    from irp.query.yahoo import prices as yahoo_prices

    start = (datetime.date.today() - datetime.timedelta(days=365)).isoformat()
    px_df = yahoo_prices(tickers, start=start)
    if px_df.empty:
        return empty_figure('No price data available.')

    px_df = px_df.sort_values([PRICE_TICKER, PRICE_DATE])
    px_df['norm'] = px_df.groupby(PRICE_TICKER)[PRICE_CLOSE].transform(
        lambda s: s / s.iloc[0] * 100
    )

    traces = []
    for tk, grp in px_df.groupby(PRICE_TICKER):
        traces.append(
            go.Scatter(
                x=grp[PRICE_DATE].tolist(),
                y=grp['norm'].tolist(),
                mode='lines',
                name=tk,
                line=dict(width=1.2),
                hovertemplate=f'<b>{tk}</b><br>%{{x}}<br>Index: %{{y:.1f}}<extra></extra>',
            )
        )

    fig = go.Figure(
        data=traces,
        layout=_base_layout(
            title=dict(
                text='1-year price index (100 = start)',
                font=dict(color=MUTED, size=12),
                x=0,
                xref='paper',
            ),
            xaxis_title='Date',
            yaxis_title='Price index',
            hovermode='x unified',
        ),
    )
    return fig


@callback(
    Output('screener-table-container', 'children'),
    Input('screener-result-store', 'data'),
)
def render_results_table(result: dict | None) -> Any:
    if not result or not result.get('records'):
        return html.P('No data — select a date and click Run.', className='no-data')

    df = pd.DataFrame(result['records'])
    if df.empty:
        return html.P('No stocks match current filters.', className='no-data')

    display_cols = ['Ticker', 'Company Name', 'Sector'] + _ALL_FACTOR_COLS
    display_cols = [c for c in display_cols if c in df.columns]
    numeric_cols = {c for c in display_cols if c in FACTOR_LABELS}

    rows = []
    for r in df[display_cols].to_dict('records'):
        row: dict = {}
        for c in display_cols:
            if c in numeric_cols:
                v = r.get(c)
                try:
                    fv = float(v)  # type: ignore[arg-type]
                    row[c] = round(fv, 4) if isfinite(fv) else None
                except (TypeError, ValueError):
                    row[c] = None
            else:
                row[c] = r.get(c, '')
        rows.append(row)

    columns: list[Any] = [
        {
            'name': FACTOR_LABELS.get(c, c),
            'id': c,
            **_col_fmt(c),
        }
        for c in display_cols
    ]
    left_cols = [
        {'if': {'column_id': c}, 'textAlign': 'left'}
        for c in ('Ticker', 'Company Name', 'Sector')
        if c in display_cols
    ]

    return _dt.DataTable(
        id='screener-results-table',
        data=rows,
        columns=columns,
        sort_action='native',
        page_size=50,
        **{
            **TABLE_STYLE,
            'style_cell_conditional': TABLE_STYLE.get('style_cell_conditional', [])
            + left_cols,
        },
    )


# ── Correlation tab callback ──────────────────────────────────────────


@callback(
    Output('screener-corr-chart', 'figure'),
    Output('screener-corr-info', 'children'),
    Input('screener-corr-run-btn', 'n_clicks'),
    State('screener-result-store', 'data'),
    State('screener-corr-window', 'value'),
    running=[(Output('screener-corr-run-btn', 'disabled'), True, False)],
    prevent_initial_call=True,
)
def run_correlation(
    n_clicks: int, result: dict | None, window: int
) -> tuple[go.Figure, str]:
    if not n_clicks or not result or not result.get('records'):
        raise PreventUpdate

    tickers = [r['Ticker'] for r in result['records'] if r.get('Ticker')]
    n = len(tickers)

    if n > _MAX_CORR_TICKERS:
        return (
            empty_figure(
                f'{n:,} tickers — narrow to ≤{_MAX_CORR_TICKERS} before running correlation'
            ),
            f'{n:,} tickers (too many)',
        )

    window = window or 252
    try:
        corr, labels, warning = factors_service._compute_return_corr(
            tickers, window, datetime.date.today()
        )
    except Exception as exc:
        logger.exception('screener correlation error')
        return empty_figure(f'Error: {exc}'), ''

    if corr.empty:
        return empty_figure(warning or 'No data'), warning or ''

    w_label = next(
        (o['label'] for o in _CORR_WINDOW_OPTIONS if o['value'] == window), str(window)
    )
    title = f'Return correlation — {w_label}  ({len(labels)} tickers)'
    info = warning or f'{len(labels)} tickers'
    return _corr_heatmap(corr, labels, title, warning=warning), info


# ── Detail panel — open/close ─────────────────────────────────────────


@callback(
    Output('screener-detail-ticker', 'data'),
    Input('screener-results-table', 'active_cell'),
    Input('screener-detail-close-btn', 'n_clicks'),
    Input('screener-detail-remove-btn', 'n_clicks'),
    State('screener-results-table', 'data'),
    State('screener-detail-ticker', 'data'),
    prevent_initial_call=True,
)
def set_detail_ticker(
    active_cell: dict | None,
    close_n: int,
    remove_n: int,
    table_data: list | None,
    current_ticker: str | None,
) -> str | None:
    tid = ctx.triggered_id
    if tid in ('screener-detail-close-btn', 'screener-detail-remove-btn'):
        return None
    if not active_cell or not table_data:
        raise PreventUpdate
    row = active_cell.get('row', 0)
    if row >= len(table_data):
        raise PreventUpdate
    ticker = table_data[row].get('Ticker')
    if ticker == current_ticker:
        return None  # toggle off
    return ticker


@callback(
    Output('screener-detail-panel', 'style'),
    Output('screener-detail-title', 'children'),
    Input('screener-detail-ticker', 'data'),
)
def show_detail_panel(ticker: str | None) -> tuple[dict, str]:
    if not ticker:
        return {'display': 'none'}, ''
    return {'display': 'block'}, f'Detail: {ticker}'


# ── Detail panel — preset store ───────────────────────────────────────


@callback(
    Output('screener-detail-preset', 'data'),
    Input({'type': 'detail-preset-btn', 'index': ALL}, 'n_clicks'),
    prevent_initial_call=True,
)
def update_preset(clicks: list) -> str:
    if not any(clicks):
        raise PreventUpdate
    triggered = ctx.triggered_id
    if isinstance(triggered, dict):
        return triggered['index']
    return '1Y'


# ── Detail panel — price + volume chart ───────────────────────────────


@callback(
    Output('screener-detail-price-chart', 'figure'),
    Input('screener-detail-ticker', 'data'),
    Input('screener-detail-preset', 'data'),
    Input('screener-detail-ma', 'value'),
)
def render_detail_prices(
    ticker: str | None,
    preset: str | None,
    ma_overlays: list | None,
) -> go.Figure:
    if not ticker:
        return empty_figure('Click a row to load price chart')

    preset = preset or '1Y'
    start, end = date_range_for_preset(preset)

    try:
        df = price_service._get_prices(ticker, source='yahoo', start=start, end=end)
    except Exception:
        df = pd.DataFrame()

    if df.empty or 'Close' not in df.columns:
        try:
            df = price_service._get_prices(ticker, source='stooq', start=start, end=end)
            close_col = 'C' if 'C' in df.columns else 'Close'
        except Exception:
            return empty_figure(f'No price data for {ticker}')
    else:
        close_col = 'Close'

    if df.empty or close_col not in df.columns:
        return empty_figure(f'No price data for {ticker}')

    df = df.sort_values('Date').copy()
    df['_date_str'] = pd.to_datetime(df['Date']).dt.strftime('%Y-%m-%d')
    vol_col = 'Volume' if 'Volume' in df.columns else None

    fig = make_subplots(
        rows=2 if vol_col else 1,
        cols=1,
        shared_xaxes=True,
        row_heights=[0.75, 0.25] if vol_col else [1.0],
        vertical_spacing=0.03,
    )

    # Price line
    fig.add_trace(
        go.Scatter(
            x=df['_date_str'],
            y=df[close_col],
            name='Close',
            line=dict(color=ACCENT, width=1.5),
            hovertemplate='<b>%{x}</b><br>Close: %{y:,.2f}<extra></extra>',
            hoverlabel=dict(**HOVER_LABEL, bordercolor=ACCENT),
        ),
        row=1,
        col=1,
    )

    # MA overlays
    for ma_label in ma_overlays or []:
        n = 50 if ma_label == 'MA50' else 200
        ma = df[close_col].rolling(n).mean()
        fig.add_trace(
            go.Scatter(
                x=df['_date_str'],
                y=ma,
                name=ma_label,
                line=dict(width=1, dash='dash'),
                hovertemplate=f'{ma_label}: %{{y:,.2f}}<extra></extra>',
            ),
            row=1,
            col=1,
        )

    # Dividend markers
    try:
        divs = price_service._get_dividends(ticker)
        if not divs.empty:
            div_dates = pd.to_datetime(divs['Date']).dt.strftime('%Y-%m-%d')
            mask = (div_dates >= df['_date_str'].min()) & (
                div_dates <= df['_date_str'].max()
            )
            divs_in_range = divs[mask.values]
            div_dates_in_range = div_dates[mask.values]
            if not divs_in_range.empty:
                price_at_div = (
                    df
                    .set_index('_date_str')[close_col]
                    .reindex(div_dates_in_range)
                    .values
                )
                fig.add_trace(
                    go.Scatter(
                        x=div_dates_in_range.values,
                        y=price_at_div,
                        mode='markers',
                        marker=dict(symbol='triangle-up', size=9, color=DIV_COLOR),
                        name='Dividend',
                        hovertemplate='<b>%{x}</b><br>Div: $%{customdata:.4f}<extra></extra>',
                        customdata=divs_in_range['Amount'].values,
                        hoverlabel=dict(**HOVER_LABEL, bordercolor=DIV_COLOR),
                    ),
                    row=1,
                    col=1,
                )
    except Exception:
        pass

    # Split lines
    shapes = []
    try:
        spls = price_service._get_splits(ticker)
        if not spls.empty:
            for _, row in spls.iterrows():
                d = pd.to_datetime(row['Date']).strftime('%Y-%m-%d')
                if df['_date_str'].min() <= d <= df['_date_str'].max():
                    shapes.append(
                        dict(
                            type='line',
                            x0=d,
                            x1=d,
                            y0=0,
                            y1=1,
                            yref='paper',
                            xref='x',
                            line=dict(color=SPLIT_COLOR, width=1, dash='dash'),
                        )
                    )
    except Exception:
        pass

    # Volume bars
    if vol_col:
        fig.add_trace(
            go.Bar(
                x=df['_date_str'],
                y=df[vol_col],
                name='Volume',
                marker_color=MUTED,
                opacity=0.5,
                hovertemplate='Vol: %{y:,}<extra></extra>',
                showlegend=False,
            ),
            row=2,
            col=1,
        )

    common_axis = dict(
        gridcolor=GRID,
        linecolor=GRID,
        tickfont=dict(color=MUTED, size=10),
        zeroline=False,
    )
    fig.update_layout(
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(128,128,128,0.05)',
        font=dict(color=MUTED, size=11),
        hovermode='x unified',
        margin=dict(l=0, r=0, t=8, b=0),
        legend=dict(
            bgcolor='rgba(0,0,0,0)',
            font=dict(color=MUTED, size=11),
            orientation='h',
            y=1.02,
            x=1,
            xanchor='right',
        ),
        shapes=shapes,
        xaxis=dict(**common_axis, showticklabels=not vol_col),
        yaxis=dict(**common_axis, title='Price'),
    )
    if vol_col:
        fig.update_layout(
            xaxis2=dict(**common_axis),
            yaxis2=dict(**common_axis, title='Volume', tickformat='.2s'),
        )
    return fig


# ── Detail panel — statements ─────────────────────────────────────────


def _render_detail_statement(
    ticker: str | None, kind: str, variant: str
) -> tuple[Any, str]:
    if not ticker:
        return html.P('Click a row to load', className='no-data'), ''
    try:
        df = universe_service._get_statement(ticker, kind)  # type: ignore[arg-type]
    except Exception as exc:
        return html.P(f'Failed: {exc}', className='no-data'), ''
    if df.empty:
        return html.P('No data available.', className='no-data'), ''

    def _period_key(p: str) -> tuple:
        p = str(p)
        try:
            year = int(p[:4])
            q = int(p[5]) if len(p) > 5 and p[4] == 'Q' else 0
        except (ValueError, IndexError):
            return (0, 0)
        return (year, q)

    if variant == 'A':
        cols = sorted(
            [c for c in df.columns if str(c).endswith('FY')],
            key=_period_key,
            reverse=True,
        )
    else:
        cols = sorted(
            [c for c in df.columns if not str(c).endswith('FY')],
            key=_period_key,
            reverse=True,
        )

    df = df[cols] if cols else df
    if df.empty or df.columns.empty:
        return html.P('No data available.', className='no-data'), ''

    fmt_df, note = fmt_statement(df)
    table_df = fmt_df.reset_index(names='Metric')
    period_cols = list(fmt_df.columns)
    columns: list = [{'name': 'Metric', 'id': 'Metric'}] + [
        {'name': str(c), 'id': str(c)} for c in period_cols
    ]
    table_df.columns = [c['id'] for c in columns]

    style = dict(TABLE_STYLE)
    style['style_cell'] = {
        **TABLE_STYLE.get('style_cell', {}),
        'fontSize': '11px',
        'padding': '4px 8px',
    }
    data: list = table_df.to_dict('records')
    return _dt.DataTable(data=data, columns=columns, **style), note


@callback(
    Output('screener-detail-income', 'children'),
    Output('screener-detail-income-note', 'children'),
    Input('screener-detail-ticker', 'data'),
    Input('screener-detail-stmt-variant', 'value'),
)
def render_detail_income(ticker: str | None, variant: str) -> tuple[Any, str]:
    return _render_detail_statement(ticker, 'income', variant)


@callback(
    Output('screener-detail-balance', 'children'),
    Output('screener-detail-balance-note', 'children'),
    Input('screener-detail-ticker', 'data'),
    Input('screener-detail-stmt-variant', 'value'),
)
def render_detail_balance(ticker: str | None, variant: str) -> tuple[Any, str]:
    return _render_detail_statement(ticker, 'balance', variant)


@callback(
    Output('screener-detail-cashflow', 'children'),
    Output('screener-detail-cashflow-note', 'children'),
    Input('screener-detail-ticker', 'data'),
    Input('screener-detail-stmt-variant', 'value'),
)
def render_detail_cashflow(ticker: str | None, variant: str) -> tuple[Any, str]:
    return _render_detail_statement(ticker, 'cashflow', variant)


# ── Detail panel — factor history ────────────────────────────────────


@callback(
    Output('screener-detail-factors', 'children'),
    Input('screener-detail-ticker', 'data'),
    Input('screener-detail-factor-variant', 'value'),
)
def render_detail_factors(ticker: str | None, variant: Literal['A', 'Q']) -> Any:
    if not ticker:
        return html.P('Click a row to load factor history', className='no-data')

    try:
        hist = factors_service._load_ticker_history(ticker, variant)
    except Exception as exc:
        return html.P(f'Failed: {exc}', className='no-data')

    if hist.empty:
        return html.P('No factor history available.', className='no-data')

    factor_defs = all_factors()
    groups: dict[str, list] = {}
    for f in factor_defs:
        if f.name in hist.columns and f.group:
            groups.setdefault(f.group, []).append(f)

    date_col = 'Report Date' if 'Report Date' in hist.columns else hist.columns[0]
    charts_out = []
    for group_name, group_factors in groups.items():
        valid = [f for f in group_factors if hist[f.name].notna().any()]
        if not valid:
            continue

        pct_factors = [f for f in valid if f.pct]
        ratio_factors = [f for f in valid if not f.pct]

        fig = go.Figure()
        for f in ratio_factors:
            fig.add_trace(
                go.Scatter(
                    x=hist[date_col],
                    y=hist[f.name],
                    mode='lines+markers',
                    name=f.label,
                    marker=dict(size=5),
                    hovertemplate=f'{f.label}: %{{y:.2f}}<br>%{{x}}<extra></extra>',
                )
            )
        for f in pct_factors:
            fig.add_trace(
                go.Scatter(
                    x=hist[date_col],
                    y=hist[f.name] * 100,
                    mode='lines+markers',
                    name=f'{f.label} (%)',
                    marker=dict(size=5),
                    yaxis='y2',
                    hovertemplate=f'{f.label}: %{{y:.1f}}%<br>%{{x}}<extra></extra>',
                )
            )

        layout_kwargs: dict[str, Any] = dict(
            title=dict(text=group_name.title(), font=dict(size=12, color=MUTED), x=0),
            height=220,
            margin=dict(l=0, r=0, t=28, b=0),
            legend=dict(
                bgcolor='rgba(0,0,0,0)',
                font=dict(color=MUTED, size=10),
                orientation='h',
            ),
        )
        if pct_factors and ratio_factors:
            layout_kwargs['yaxis2'] = dict(
                overlaying='y',
                side='right',
                gridcolor='rgba(0,0,0,0)',
                tickfont=dict(color=MUTED, size=10),
                ticksuffix='%',
            )
        fig.update_layout(_chart_layout(**layout_kwargs))
        charts_out.append(dcc.Graph(figure=fig, config={'displayModeBar': False}))

    if not charts_out:
        return html.P('No factor data available.', className='no-data')
    return html.Div(charts_out)


def _build_summary(steps: list) -> str:
    parts = []
    for s in steps:
        label = s.get('label', '')
        if not label:
            continue
        t = s.get('type', '')
        if t == 'range':
            parts.append(label)
        elif t == 'keep':
            parts.append(f'+{label}')
        elif t == 'remove':
            parts.append(f'-{label}')
    return '; '.join(parts)


@callback(
    Output('screener-watchlist-name', 'value'),
    Input('screener-steps-store', 'data'),
    State('screener-date', 'date'),
)
def auto_name(steps: list | None, date_str: str | None) -> str:
    return _auto_name(steps or [], date_str)


@callback(
    Output('screener-wl-description', 'children'),
    Input('screener-steps-store', 'data'),
)
def auto_description(steps: list | None) -> str:
    return _build_summary(steps or [])


@callback(
    Output('screener-save-status', 'children'),
    Output('screener-wl-trigger', 'data'),
    Input('screener-save-btn', 'n_clicks'),
    State('screener-watchlist-name', 'value'),
    State('screener-result-store', 'data'),
    State('screener-steps-store', 'data'),
    State('screener-wl-trigger', 'data'),
    prevent_initial_call=True,
)
def save_watchlist_action(
    n_clicks: int,
    name: str | None,
    result: dict | None,
    steps: list | None,
    trigger: int,
) -> tuple[str, int]:
    if not n_clicks or not name or not result or not result.get('records'):
        raise PreventUpdate
    name = name.strip()
    if not name:
        return 'Enter a name first.', trigger or 0
    tickers = [r['Ticker'] for r in result['records'] if r.get('Ticker')]
    summary = _build_summary(steps or [])
    try:
        watchlist_service.save_watchlist(name, tickers, summary)
    except Exception as exc:
        return f'Error: {exc}', trigger or 0
    return f'Saved "{name}" ({len(tickers):,} stocks).', (trigger or 0) + 1


@callback(
    Output('screener-wl-pending-delete', 'data'),
    Output('screener-wl-trigger', 'data', allow_duplicate=True),
    Input({'type': 'wl-delete-btn', 'index': ALL}, 'n_clicks'),
    Input({'type': 'wl-confirm-btn', 'index': ALL}, 'n_clicks'),
    State('screener-wl-pending-delete', 'data'),
    State('screener-wl-trigger', 'data'),
    prevent_initial_call=True,
)
def delete_watchlist_action(
    delete_clicks: list,
    confirm_clicks: list,
    pending: str | None,
    trigger: int,
) -> tuple:
    triggered = ctx.triggered_id
    if not isinstance(triggered, dict):
        raise PreventUpdate

    btn_type = triggered.get('type')
    name = triggered.get('index')

    if btn_type == 'wl-delete-btn' and any(delete_clicks):
        # Cancel button clears pending state
        if isinstance(name, str) and name.startswith('__cancel__'):
            return None, trigger or 0
        # First click: enter pending state
        return name, trigger or 0

    if btn_type == 'wl-confirm-btn' and any(confirm_clicks) and pending:
        with contextlib.suppress(Exception):
            watchlist_service.delete_watchlist(pending)
        return None, (trigger or 0) + 1

    raise PreventUpdate


@callback(
    Output('screener-watchlists-container', 'children'),
    Input('screener-init', 'data'),
    Input('screener-wl-trigger', 'data'),
    Input('screener-wl-pending-delete', 'data'),
)
def render_watchlists_table(_init: Any, _trigger: Any, pending: str | None) -> Any:
    wl = watchlist_service.list_watchlists()
    if wl.empty:
        return html.P(
            'No saved watchlists.', style={'color': MUTED, 'fontSize': '12px'}
        )

    rows = []
    for _, r in wl.iterrows():
        rows.append(
            html.Tr([
                html.Td(
                    r['name'], style={'color': 'var(--text)', 'padding': '4px 10px'}
                ),
                html.Td(
                    str(r['n']),
                    style={'color': MUTED, 'padding': '4px 10px', 'textAlign': 'right'},
                ),
                html.Td(
                    str(r['created']), style={'color': MUTED, 'padding': '4px 10px'}
                ),
                html.Td(
                    str(r['summary'] or ''),
                    style={
                        'color': MUTED,
                        'padding': '4px 10px',
                        'fontSize': '11px',
                        'maxWidth': '300px',
                        'overflow': 'hidden',
                        'textOverflow': 'ellipsis',
                    },
                ),
                html.Td(
                    [
                        html.Button(
                            'Load',
                            id={'type': 'wl-load-btn', 'index': r['name']},
                            n_clicks=0,
                            style={
                                'fontSize': '11px',
                                'padding': '2px 8px',
                                'marginRight': '6px',
                                'cursor': 'pointer',
                            },
                        ),
                        *(
                            [
                                html.Button(
                                    'Confirm delete',
                                    id={'type': 'wl-confirm-btn', 'index': r['name']},
                                    n_clicks=0,
                                    style={
                                        'fontSize': '11px',
                                        'padding': '2px 8px',
                                        'color': '#fff',
                                        'background': '#e05252',
                                        'border': '1px solid #e05252',
                                        'cursor': 'pointer',
                                        'borderRadius': '4px',
                                        'marginRight': '4px',
                                    },
                                ),
                                html.Button(
                                    'Cancel',
                                    id={
                                        'type': 'wl-delete-btn',
                                        'index': f'__cancel__{r["name"]}',
                                    },
                                    n_clicks=0,
                                    style={
                                        'fontSize': '11px',
                                        'padding': '2px 8px',
                                        'cursor': 'pointer',
                                    },
                                ),
                            ]
                            if pending == r['name']
                            else [
                                html.Button(
                                    'Delete',
                                    id={'type': 'wl-delete-btn', 'index': r['name']},
                                    n_clicks=0,
                                    style={
                                        'fontSize': '11px',
                                        'padding': '2px 8px',
                                        'color': '#e05252',
                                        'cursor': 'pointer',
                                    },
                                ),
                            ]
                        ),
                    ],
                    style={'padding': '4px 10px'},
                ),
            ])
        )

    return html.Table(
        style={'width': '100%', 'fontSize': '12px', 'borderCollapse': 'collapse'},
        children=[
            html.Thead(
                html.Tr([
                    html.Th(
                        h,
                        style={
                            'color': MUTED,
                            'textAlign': 'left',
                            'padding': '4px 10px',
                            'fontWeight': '600',
                            'fontSize': '11px',
                            'textTransform': 'uppercase',
                            'borderBottom': '1px solid var(--border)',
                        },
                    )
                    for h in ['Name', 'Tickers', 'Created', 'Summary', 'Actions']
                ])
            ),
            html.Tbody(rows),
        ],
    )

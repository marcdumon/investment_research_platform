"""Stock screener page: progressive filter stack + scatter/histogram/price charts."""

import datetime
import logging
from math import isfinite
from typing import Any

import dash
import pandas as pd
import plotly.colors as pc
import plotly.graph_objects as go
from dash import ALL, Input, Output, State, callback, ctx, dcc, html
from dash import dash_table as _dt
from dash.exceptions import PreventUpdate

from irp.factors._cols import PRICE_CLOSE, PRICE_DATE, PRICE_TICKER
from irp.ui.charts import empty_figure as _empty_figure
from irp.ui.charts import scatter_chart_layout as _base_layout
from irp.ui.factor_meta import FACTOR_LABELS, FACTOR_OPTIONS, PCT_FACTORS
from irp.ui.services import factors_service, universe_service, watchlist_service
from irp.ui.tables import column_format as _col_fmt
from irp.ui.theme import ACCENT, GRID, HOVER_LABEL, MUTED, TABLE_STYLE

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
_SECTOR_PALETTE = pc.qualitative.Set3


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
                            '(% factors: decimal, e.g. 0.15 for 15%)',
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
                                        options=[
                                            {'label': 'Sector', 'value': 'sector'},
                                            {'label': 'Market', 'value': 'market'},
                                            {'label': 'None', 'value': 'none'},
                                        ],
                                        value='sector',
                                        clearable=False,
                                        className='filter-dropdown',
                                        style={'minWidth': '110px'},
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
            ],
        ),
        # ── Results table ─────────────────────────────────────────────
        dcc.Loading(html.Div(id='screener-table-container')),
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
                    style={'color': MUTED, 'fontSize': '11px', 'marginTop': '4px', 'fontStyle': 'italic'},
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
        df = universe_service.get_companies()
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
    variant: str,
    market: str | None,
    sector: str | None,
) -> tuple[Any, list]:
    if not n_clicks or not date_str:
        raise PreventUpdate
    as_of = datetime.date.fromisoformat(date_str[:10])
    df = factors_service.load_cross_section(
        as_of, variant, enrich_company_columns=False
    )
    if df.empty:
        return {}, []
    df = df.reset_index()
    try:
        comp = universe_service.get_companies()[
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
    Input({'type': 'delete-step-btn', 'index': ALL}, 'n_clicks'),
    Input({'type': 'wl-load-btn', 'index': ALL}, 'n_clicks'),
    State('screener-add-factor', 'value'),
    State('screener-add-min', 'value'),
    State('screener-add-max', 'value'),
    State('screener-selection-store', 'data'),
    State('screener-steps-store', 'data'),
    prevent_initial_call=True,
)
def mutate_steps(
    add_n: int,
    keep_n: int,
    remove_n: int,
    delete_clicks: list,
    load_clicks: list,
    add_factor: str | None,
    add_min: float | None,
    add_max: float | None,
    selection: list | None,
    steps: list | None,
) -> list:
    steps = list(steps or [])
    triggered = ctx.triggered_id

    if triggered == 'screener-add-filter-btn':
        if not add_factor:
            raise PreventUpdate
        if add_min is None and add_max is None:
            raise PreventUpdate
        label = FACTOR_LABELS.get(add_factor, add_factor)
        if add_min is not None and add_max is not None:
            label += f' {add_min}–{add_max}'
        elif add_min is not None:
            label += f' ≥ {add_min}'
        else:
            label += f' ≤ {add_max}'
        steps.append({
            'type': 'range',
            'col': add_factor,
            'min': add_min,
            'max': add_max,
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
            raise PreventUpdate
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
            tickers.append(cd[0])
    return list(dict.fromkeys(tickers))  # deduplicate, preserve order


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
)
def render_scatter(
    result: dict | None,
    x_factor: str | None,
    y_factor: str | None,
    color_by: str | None,
) -> go.Figure:
    if not result or not result.get('records'):
        return _empty_figure('Run screener to see data.')
    if not x_factor or not y_factor:
        return _empty_figure()

    df = pd.DataFrame(result['records'])
    if x_factor not in df.columns or y_factor not in df.columns:
        return _empty_figure(f'Factor "{x_factor}" or "{y_factor}" not in data.')

    df = df.dropna(subset=[x_factor, y_factor])
    df = df[df[x_factor].apply(lambda v: isfinite(float(v)) if pd.notna(v) else False)]
    df = df[df[y_factor].apply(lambda v: isfinite(float(v)) if pd.notna(v) else False)]

    if df.empty:
        return _empty_figure('No valid data for selected factors.')

    x_label = FACTOR_LABELS.get(x_factor, x_factor)
    y_label = FACTOR_LABELS.get(y_factor, y_factor)

    color_col = None
    if color_by == 'sector' and 'Sector' in df.columns:
        color_col = 'Sector'
    elif color_by == 'market' and 'Market' in df.columns:
        color_col = 'Market'

    traces = []
    if color_col:
        groups = sorted(df[color_col].fillna('Unknown').unique())
        for i, grp in enumerate(groups):
            mask = df[color_col].fillna('Unknown') == grp
            sub = df[mask]
            if sub.empty:
                continue
            hover_parts = [
                f'<b>%{{customdata[0]}}</b> %{{customdata[1]}}<br>',
                f'{x_label}: %{{x:.3f}}<br>',
                f'{y_label}: %{{y:.3f}}<extra></extra>',
            ]
            traces.append(
                go.Scatter(
                    x=sub[x_factor].tolist(),
                    y=sub[y_factor].tolist(),
                    mode='markers',
                    name=grp,
                    customdata=sub[['Ticker', 'Company Name']]
                    .fillna('')
                    .values.tolist()
                    if 'Company Name' in sub.columns
                    else sub[['Ticker']].values.tolist(),
                    hovertemplate=''.join(hover_parts),
                    hoverlabel=dict(**HOVER_LABEL),
                    marker=dict(
                        size=6,
                        color=_SECTOR_PALETTE[i % len(_SECTOR_PALETTE)],
                        opacity=0.75,
                    ),
                )
            )
    else:
        cd = (
            df[['Ticker', 'Company Name']].fillna('').values.tolist()
            if 'Company Name' in df.columns
            else df[['Ticker']].values.tolist()
        )
        traces.append(
            go.Scatter(
                x=df[x_factor].tolist(),
                y=df[y_factor].tolist(),
                mode='markers',
                name='',
                customdata=cd,
                hovertemplate=f'<b>%{{customdata[0]}}</b> %{{customdata[1]}}<br>'
                f'{x_label}: %{{x:.3f}}<br>{y_label}: %{{y:.3f}}'
                '<extra></extra>',
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
        return _empty_figure('Run screener to see data.')

    df = pd.DataFrame(result['records'])
    if factor not in df.columns:
        return _empty_figure(f'Factor "{factor}" not in data.')

    vals = df[factor].dropna()
    vals = vals[vals.apply(lambda v: isfinite(float(v)))]
    if vals.empty:
        return _empty_figure('No valid data.')

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
        return _empty_figure(f'Narrow to ≤ {_MAX_PRICE_TICKERS} stocks first.')

    from irp.query.yahoo import prices as yahoo_prices

    start = (datetime.date.today() - datetime.timedelta(days=365)).isoformat()
    px_df = yahoo_prices(tickers, start=start)
    if px_df.empty:
        return _empty_figure('No price data available.')

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
        try:
            watchlist_service.delete_watchlist(pending)
        except Exception:
            pass
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
                                    id={'type': 'wl-delete-btn', 'index': f'__cancel__{r["name"]}'},
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

"""Factors page: cross-section screening + per-ticker factor history."""

import datetime
import logging
from math import isfinite
from typing import Any

import dash
import pandas as pd
import plotly.graph_objects as go
from dash import Input, Output, State, callback, dcc, html
from dash import dash_table as _dt
from dash.exceptions import PreventUpdate

from irp.factors import cross_section, ticker_factor_history
from irp.query.simfin import companies as _companies
from irp.query.watchlists import list_watchlists, load_watchlist
from irp.ui.factor_meta import FACTOR_LABELS, FACTOR_OPTIONS, PCT_FACTORS
from dash.dash_table.Format import Format, Scheme, Symbol

from irp.ui.theme import ACCENT, GRID, HOVER_LABEL, MUTED, TABLE_STYLE

dash.register_page(__name__, path='/factors', name='Factors')

logger = logging.getLogger(__name__)

_PCT_FACTORS = PCT_FACTORS
_ALL_FACTOR_COLS = list(FACTOR_LABELS.keys())
_VARIANT_OPTIONS = [
    {'label': ' Annual', 'value': 'A'},
    {'label': ' Quarterly', 'value': 'Q'},
]
_TOP_OPTIONS = [
    {'label': 'Top 50', 'value': 50},
    {'label': 'Top 100', 'value': 100},
    {'label': 'All', 'value': 0},
]

_DEFAULT_DATE = (datetime.date.today() - datetime.timedelta(days=90)).isoformat()


def _col_fmt(c: str) -> dict:
    if c == 'mktcap':
        return {'type': 'numeric', 'format': Format(precision=1, scheme=Scheme.fixed, symbol=Symbol.yes, symbol_prefix='$')}
    if c in _PCT_FACTORS:
        return {'type': 'numeric', 'format': Format(precision=1, scheme=Scheme.percentage)}
    if c in FACTOR_LABELS:
        return {'type': 'numeric', 'format': Format(precision=2, scheme=Scheme.fixed)}
    return {}



def _fmt(val: object, factor: str) -> str:
    """Format a factor value for table cells and hover text."""
    try:
        v = float(val)  # type: ignore[arg-type]
    except (TypeError, ValueError):
        return '—'
    if pd.isna(v) or not isfinite(v):
        return '—'
    if factor == 'mktcap':
        return f'${v / 1e9:.1f}B'
    if factor in _PCT_FACTORS:
        return f'{v * 100:.1f}%'
    return f'{v:.1f}x'


def _empty_figure(message: str = 'No data') -> go.Figure:
    fig = go.Figure()
    fig.update_layout(
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)',
        annotations=[
            dict(
                text=message,
                x=0.5,
                y=0.5,
                xref='paper',
                yref='paper',
                showarrow=False,
                font=dict(color=MUTED, size=13),
            )
        ],
        xaxis=dict(visible=False),
        yaxis=dict(visible=False),
        margin=dict(l=0, r=0, t=8, b=0),
    )
    return fig


def _chart_layout(**extra: Any) -> go.Layout:
    return go.Layout(
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(128,128,128,0.05)',
        font=dict(color=MUTED, size=11),
        margin=dict(l=0, r=0, t=24, b=0),
        hovermode='closest',
        xaxis=dict(
            gridcolor=GRID,
            linecolor=GRID,
            tickfont=dict(color=MUTED, size=11),
            zeroline=False,
            showline=True,
        ),
        yaxis=dict(
            gridcolor=GRID,
            linecolor=GRID,
            tickfont=dict(color=MUTED, size=11),
            zeroline=True,
            showline=False,
        ),
        **extra,
    )


# ── Layout ────────────────────────────────────────────────────────────

layout = html.Div(
    className='factors-page',
    children=[
        dcc.Store(id='factors-init', data=1),
        dcc.Store(id='companies-store'),
        dcc.Store(id='xsection-store'),
        dcc.Store(id='factor-history-store'),
        dcc.Store(id='factors-wl-trigger', data=0),
        dcc.Tabs(
            id='factors-tabs',
            value='tab-screening',
            className='ticker-tabs',
            children=[
                # ── Tab 1: Screening ──────────────────────────────────
                dcc.Tab(
                    label='Screening',
                    value='tab-screening',
                    className='ticker-tab',
                    selected_className='ticker-tab--active',
                    children=[
                        html.Div(
                            className='factors-controls',
                            children=[
                                # Row 1: Snapshot params
                                html.Div(
                                    className='control-row',
                                    children=[
                                        dcc.DatePickerSingle(
                                            id='xsection-date',
                                            date=_DEFAULT_DATE,
                                            display_format='YYYY-MM-DD',
                                            style={'fontSize': '13px'},
                                        ),
                                        dcc.RadioItems(
                                            id='xsection-variant',
                                            options=_VARIANT_OPTIONS,
                                            value='A',
                                            inline=True,
                                            labelClassName='check-item',
                                        ),
                                        dcc.Dropdown(
                                            id='xsection-sector',
                                            placeholder='All sectors',
                                            clearable=True,
                                            className='filter-dropdown',
                                            style={'minWidth': '160px'},
                                        ),
                                        dcc.Dropdown(
                                            id='xsection-market',
                                            placeholder='All markets',
                                            clearable=True,
                                            className='filter-dropdown',
                                            style={'minWidth': '130px'},
                                        ),
                                        dcc.Dropdown(
                                            id='xsection-watchlist',
                                            placeholder='Watchlist…',
                                            clearable=True,
                                            className='filter-dropdown',
                                            style={'minWidth': '160px'},
                                        ),
                                    ],
                                ),
                                # Row 2: Display params
                                html.Div(
                                    className='control-row',
                                    children=[
                                        dcc.Dropdown(
                                            id='ranking-factor',
                                            options=FACTOR_OPTIONS,
                                            value='pe',
                                            clearable=False,
                                            className='filter-dropdown',
                                            style={'minWidth': '130px'},
                                        ),
                                        dcc.RadioItems(
                                            id='ranking-top',
                                            options=_TOP_OPTIONS,
                                            value=50,
                                            inline=True,
                                            labelClassName='check-item',
                                        ),
                                        html.Button(
                                            'Run',
                                            id='xsection-run-btn',
                                            className='run-btn',
                                            n_clicks=0,
                                        ),
                                    ],
                                ),
                            ],
                        ),
                        html.Div(
                            className='tab-content',
                            children=[
                                dcc.Loading(
                                    dcc.Graph(
                                        id='ranking-chart',
                                        config={'displayModeBar': False},
                                        style={'height': '420px'},
                                    )
                                ),
                                dcc.Loading(html.Div(id='xsection-table-container')),
                            ],
                        ),
                    ],
                ),
                # ── Tab 2: Factor History ─────────────────────────────
                dcc.Tab(
                    label='Factor History',
                    value='tab-history',
                    className='ticker-tab',
                    selected_className='ticker-tab--active',
                    children=[
                        html.Div(
                            className='factors-controls',
                            children=[
                                dcc.Dropdown(
                                    id='ts-ticker',
                                    placeholder='Search ticker or company name…',
                                    searchable=True,
                                    clearable=True,
                                    className='ticker-dropdown',
                                    style={'minWidth': '240px'},
                                ),
                                dcc.RadioItems(
                                    id='ts-variant',
                                    options=_VARIANT_OPTIONS,
                                    value='A',
                                    inline=True,
                                    labelClassName='check-item',
                                ),
                                dcc.Dropdown(
                                    id='ts-factor',
                                    options=FACTOR_OPTIONS,
                                    value='pe',
                                    clearable=False,
                                    className='filter-dropdown',
                                    style={'minWidth': '130px'},
                                ),
                                html.Button(
                                    'Run',
                                    id='ts-run-btn',
                                    className='run-btn',
                                    n_clicks=0,
                                ),
                            ],
                        ),
                        html.Div(
                            className='tab-content',
                            children=[
                                dcc.Loading(
                                    dcc.Graph(
                                        id='ts-chart',
                                        config={'displayModeBar': False},
                                        style={'height': '380px'},
                                    )
                                ),
                                dcc.Loading(html.Div(id='ts-table-container')),
                            ],
                        ),
                    ],
                ),
            ],
        ),
    ],
)


# ── Callbacks ─────────────────────────────────────────────────────────


@callback(
    Output('companies-store', 'data'),
    Output('xsection-sector', 'options'),
    Output('xsection-market', 'options'),
    Output('ts-ticker', 'options'),
    Output('xsection-watchlist', 'options'),
    Input('factors-init', 'data'),
    Input('factors-wl-trigger', 'data'),
)
def load_options(_: Any, _wl: Any) -> tuple[Any, list, list, list, list]:
    """Populate filter dropdowns from SimFin companies table on page load."""
    try:
        df = _companies()
    except Exception:
        logger.exception('load_options: _companies() failed')
        return None, [], [], []
    if df.empty:
        logger.warning('load_options: companies() returned empty DataFrame')
        return None, [], [], [], []

    sectors = sorted(df['Sector'].dropna().unique())
    markets = sorted(df['Market'].dropna().unique())

    sector_opts = [{'label': s, 'value': s} for s in sectors]
    market_opts = [{'label': m, 'value': m} for m in markets]
    ticker_opts = [
        {'label': f'{r["Ticker"]} – {r["Company Name"]}', 'value': r['Ticker']}
        for _, r in df.sort_values('Ticker').iterrows()
    ]
    wl_df = list_watchlists()
    wl_opts = (
        [{'label': f'{r["name"]} ({r["n"]})', 'value': r['name']}
         for _, r in wl_df.iterrows()]
        if not wl_df.empty else []
    )
    slim = df[['Ticker', 'Company Name', 'Sector', 'Market']].fillna('')
    logger.info(
        f'load_options: {len(sector_opts)} sectors, {len(market_opts)} markets, '
        f'{len(ticker_opts)} tickers, {len(wl_opts)} watchlists'
    )
    return slim.to_dict('records'), sector_opts, market_opts, ticker_opts, wl_opts


@callback(
    Output('xsection-sector', 'disabled'),
    Output('xsection-market', 'disabled'),
    Input('xsection-watchlist', 'value'),
)
def toggle_xsection_filters(watchlist: str | None) -> tuple[bool, bool]:
    active = bool(watchlist)
    return active, active


@callback(
    Output('xsection-store', 'data'),
    Input('xsection-run-btn', 'n_clicks'),
    Input('companies-store', 'data'),
    State('xsection-date', 'date'),
    State('xsection-variant', 'value'),
    State('xsection-sector', 'value'),
    State('xsection-market', 'value'),
    State('xsection-watchlist', 'value'),
    State('ranking-factor', 'value'),
    State('ranking-top', 'value'),
    running=[
        (Output('xsection-run-btn', 'disabled'), True, False),
        (Output('xsection-run-btn', 'children'), 'Running...', 'Run'),
    ],
)
def compute_xsection(
    n_clicks: int,
    companies_data: list | None,
    as_of_str: str | None,
    variant: str,
    sector: str | None,
    market: str | None,
    watchlist: str | None,
    factor: str | None,
    top_n: int | None,
) -> Any:
    """Compute full cross-section, join metadata, apply filters."""
    if not as_of_str or not companies_data:
        raise PreventUpdate
    as_of = datetime.date.fromisoformat(as_of_str[:10])

    tickers = None
    if watchlist:
        try:
            tickers = load_watchlist(watchlist)
        except KeyError:
            logger.warning(f'compute_xsection: watchlist "{watchlist}" not found')

    df = cross_section(as_of, variant, tickers=tickers)  # type: ignore[arg-type]
    if df.empty:
        return {}
    df = df.reset_index()
    comp = pd.DataFrame(companies_data)[['Ticker', 'Company Name', 'Sector', 'Market']]
    df = df.merge(comp, on='Ticker', how='left')
    if not watchlist:
        if sector:
            df = df[df['Sector'] == sector]
        if market:
            df = df[df['Market'] == market]
    return {'records': df.to_dict('records'), 'factor': factor or 'pe', 'top_n': top_n or 50}


@callback(
    Output('ranking-chart', 'figure'),
    Input('xsection-store', 'data'),
)
def render_ranking_chart(store: dict | None) -> go.Figure:
    """Horizontal bar chart of tickers ranked by selected factor."""
    if not store or not store.get('records'):
        return _empty_figure()

    data = store['records']
    factor: str = store.get('factor', 'pe')
    top_n: int = store.get('top_n', 50)

    df = pd.DataFrame(data)
    if factor not in df.columns:
        return _empty_figure()

    df = df.dropna(subset=[factor])
    df = df[df[factor].apply(lambda v: isfinite(float(v)) if v is not None else False)]
    if df.empty:
        return _empty_figure()

    df = df.sort_values(factor, ascending=False)
    if top_n:
        df = df.head(top_n)

    label = FACTOR_LABELS.get(factor, factor)
    hover = [
        f'<b>{r.get("Ticker", "")}</b> {r.get("Company Name", "")}<br>'
        f'{label}: {_fmt(r.get(factor), factor)}'
        for _, r in df.iterrows()
    ]

    fig = go.Figure(
        data=go.Bar(
            x=df[factor].tolist(),
            y=df['Ticker'].tolist(),
            orientation='h',
            marker_color=ACCENT,
            hovertext=hover,
            hoverinfo='text',
            hoverlabel=dict(**HOVER_LABEL, bordercolor=ACCENT),
        ),
        layout=_chart_layout(
            title=dict(
                text=f'{label} — cross-section',
                font=dict(color=MUTED, size=12),
                x=0,
                xref='paper',
            ),
            yaxis_autorange='reversed',
            yaxis_tickfont=dict(color=MUTED, size=10),
            xaxis_title=label,
        ),
    )
    return fig


@callback(
    Output('xsection-table-container', 'children'),
    Input('xsection-store', 'data'),
)
def render_xsection_table(store: dict | None) -> Any:
    """DataTable of all tickers with all factors, formatted."""
    if not store or not store.get('records'):
        return html.P(
            'No data — select a date and click Run.', className='no-data'
        )

    df = pd.DataFrame(store['records'])
    if df.empty:
        return html.P('No data for selected filters.', className='no-data')

    display_cols = ['Ticker', 'Company Name', 'Sector', 'mktcap'] + _ALL_FACTOR_COLS
    display_cols = [c for c in display_cols if c in df.columns]

    numeric_cols = {c for c in display_cols if c in FACTOR_LABELS or c == 'mktcap'}

    rows = []
    for _, r in df.iterrows():
        row: dict = {}
        for c in display_cols:
            if c in numeric_cols:
                v = r.get(c)
                try:
                    fv = float(v)  # type: ignore[arg-type]
                    row[c] = round(fv / 1e9, 2) if c == 'mktcap' else (round(fv, 4) if isfinite(fv) else None)
                except (TypeError, ValueError):
                    row[c] = None
            else:
                row[c] = r.get(c, '')
        rows.append(row)

    mktcap_label = 'Mkt Cap ($B)'
    columns: list[Any] = [
        {'name': mktcap_label if c == 'mktcap' else FACTOR_LABELS.get(c, c), 'id': c, **_col_fmt(c)}
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


@callback(
    Output('factor-history-store', 'data'),
    Input('ts-run-btn', 'n_clicks'),
    State('ts-ticker', 'value'),
    State('ts-variant', 'value'),
    State('ts-factor', 'value'),
    running=[
        (Output('ts-run-btn', 'disabled'), True, False),
        (Output('ts-run-btn', 'children'), 'Running...', 'Run'),
    ],
    prevent_initial_call=True,
)
def compute_history(n_clicks: int, ticker: str | None, variant: str, factor: str | None) -> Any:
    """Compute per-ticker factor history on Run click."""
    if not ticker:
        raise PreventUpdate
    df = ticker_factor_history(ticker, variant)  # type: ignore[arg-type]
    if df.empty:
        return {}
    df = df.copy()
    from irp.factors._cols import REPORT_DATE

    if REPORT_DATE in df.columns:
        df[REPORT_DATE] = pd.to_datetime(df[REPORT_DATE]).dt.strftime('%Y-%m-%d')
    return {'records': df.to_dict('records'), 'factor': factor or 'pe'}


@callback(
    Output('ts-chart', 'figure'),
    Input('factor-history-store', 'data'),
)
def render_history_chart(store: dict | None) -> go.Figure:
    """Line chart of selected factor over filing dates."""
    if not store or not store.get('records'):
        return _empty_figure('Select a ticker and click Run.')

    factor: str = store.get('factor', 'pe')
    df = pd.DataFrame(store['records'])
    from irp.factors._cols import REPORT_DATE

    if REPORT_DATE not in df.columns or factor not in df.columns:
        return _empty_figure()

    df = df.dropna(subset=[factor]).sort_values(REPORT_DATE)
    if df.empty:
        return _empty_figure('No data for this factor.')

    label = FACTOR_LABELS.get(factor, factor)
    hover = [
        f'<b>{r[REPORT_DATE]}</b><br>{label}: {_fmt(r.get(factor), factor)}'
        for _, r in df.iterrows()
    ]

    fig = go.Figure(
        data=go.Scatter(
            x=df[REPORT_DATE].tolist(),
            y=df[factor].tolist(),
            mode='lines+markers',
            line=dict(color=ACCENT, width=1.5),
            marker=dict(color=ACCENT, size=6),
            hovertext=hover,
            hoverinfo='text',
            hoverlabel=dict(**HOVER_LABEL, bordercolor=ACCENT),
        ),
        layout=_chart_layout(
            yaxis_title=label,
            xaxis_title='Filing Date',
        ),
    )
    return fig


@callback(
    Output('ts-table-container', 'children'),
    Input('factor-history-store', 'data'),
)
def render_history_table(store: dict | None) -> Any:
    """Table of all factors by filing date for the selected ticker."""
    if not store or not store.get('records'):
        return html.P('Select a ticker and click Run.', className='no-data')

    df = pd.DataFrame(store['records'])
    from irp.factors._cols import REPORT_DATE

    if REPORT_DATE in df.columns:
        df = df.sort_values(REPORT_DATE, ascending=False)

    display_cols = [REPORT_DATE, 'mktcap'] + _ALL_FACTOR_COLS
    display_cols = [c for c in display_cols if c in df.columns]

    numeric_cols = {c for c in display_cols if c in FACTOR_LABELS or c == 'mktcap'}

    rows = []
    for _, r in df.iterrows():
        row: dict = {}
        for c in display_cols:
            if c in numeric_cols:
                v = r.get(c)
                try:
                    fv = float(v)  # type: ignore[arg-type]
                    row[c] = round(fv / 1e9, 2) if c == 'mktcap' else (round(fv, 4) if isfinite(fv) else None)
                except (TypeError, ValueError):
                    row[c] = None
            else:
                row[c] = str(r.get(c, ''))
        rows.append(row)

    mktcap_label = 'Mkt Cap ($B)'
    columns: list[Any] = [
        {
            'name': mktcap_label if c == 'mktcap' else FACTOR_LABELS.get(c, 'Filing Date' if c == REPORT_DATE else c),
            'id': c,
            **_col_fmt(c),
        }
        for c in display_cols
    ]

    left_cols = [{'if': {'column_id': REPORT_DATE}, 'textAlign': 'left'}]

    return _dt.DataTable(
        data=rows,
        columns=columns,
        sort_action='native',
        page_size=20,
        **{
            **TABLE_STYLE,
            'style_cell_conditional': TABLE_STYLE.get('style_cell_conditional', [])
            + left_cols,
        },
    )

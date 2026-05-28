import logging
from datetime import date, timedelta
from typing import Any

import dash
import pandas as pd
import plotly.graph_objects as go
from dash import ALL, Input, Output, State, callback, dcc, html
from dash import dash_table as _dt
from dash.exceptions import PreventUpdate
from plotly.basedatatypes import BaseTraceType

from irp.ui.services import price_service, universe_service
from irp.ui.theme import (
    ACCENT,
    DIV_COLOR,
    GRID,
    HOVER_LABEL,
    MUTED,
    SPLIT_COLOR,
    TABLE_STYLE,
)
from irp.ui.ticker_fmt import (
    date_range_for_preset,
    detect_scale,
    fmt_cell,
    fmt_price_table,
    fmt_statement,
    is_per_share,
    is_pct_item,
)

logger = logging.getLogger(__name__)

dash.register_page(__name__, path='/ticker', name='Ticker')

_HIDE = {'display': 'none'}
_SHOW = {'display': 'block'}
_RANGE_PRESETS = ['1M', '6M', '1Y', '3Y', '5Y', 'Max']

_today = date.today().isoformat()
_1y_ago = (date.today() - timedelta(days=365)).isoformat()


def _period_radio(prefix: str) -> html.Div:
    return html.Div(
        className='stmt-controls',
        children=[
            dcc.RadioItems(
                id=f'{prefix}-period',
                options=[
                    {'label': ' Annual', 'value': 'A'},
                    {'label': ' Quarterly', 'value': 'Q'},
                ],
                value='A',
                inline=True,
                labelClassName='check-item',
            ),
        ],
    )


layout = html.Div(
    className='ticker-page',
    children=[
        html.Div(
            className='ticker-search-bar',
            children=[
                dcc.Dropdown(
                    id='ticker-select',
                    placeholder='Search ticker or company...',
                    searchable=True,
                    clearable=True,
                    className='ticker-dropdown',
                ),
                dcc.Dropdown(
                    id='filter-market',
                    placeholder='Market',
                    clearable=True,
                    className='filter-dropdown',
                ),
                dcc.Dropdown(
                    id='filter-sector',
                    placeholder='Sector',
                    clearable=True,
                    className='filter-dropdown',
                ),
                dcc.Dropdown(
                    id='filter-industry',
                    placeholder='Industry',
                    clearable=True,
                    className='filter-dropdown',
                ),
            ],
        ),
        dcc.Store(id='ticker-init', data=1),
        dcc.Store(id='all-tickers-store'),
        dcc.Store(id='ticker-store'),
        dcc.Store(id='price-data-store'),
        html.Div(
            id='ticker-content',
            style=_HIDE,
            children=[
                html.Div(id='ticker-header', className='ticker-header'),
                dcc.Tabs(
                    id='ticker-tabs',
                    value='prices',
                    className='ticker-tabs',
                    children=[
                        dcc.Tab(
                            label='Prices',
                            value='prices',
                            className='ticker-tab',
                            selected_className='ticker-tab--active',
                            children=[
                                html.Div(
                                    className='price-controls',
                                    children=[
                                        html.Div(
                                            className='price-presets',
                                            children=[
                                                html.Button(
                                                    p,
                                                    id=f'preset-{p}',
                                                    n_clicks=0,
                                                    className='btn price-preset-btn',
                                                )
                                                for p in _RANGE_PRESETS
                                            ],
                                        ),
                                        dcc.DatePickerRange(
                                            id='price-dates',
                                            display_format='YYYY-MM-DD',
                                            start_date=_1y_ago,
                                            end_date=_today,
                                            className='price-date-picker',
                                        ),
                                        dcc.RadioItems(
                                            id='price-source',
                                            options=[
                                                {'label': ' Yahoo', 'value': 'yahoo'},
                                                {'label': ' Stooq', 'value': 'stooq'},
                                            ],
                                            value='yahoo',
                                            inline=True,
                                            className='price-source-radio',
                                            labelClassName='check-item',
                                        ),
                                    ],
                                ),
                                dcc.Loading(
                                    dcc.Graph(
                                        id='price-chart',
                                        config={'displayModeBar': False},
                                    )
                                ),
                                dcc.Loading(html.Div(id='price-table-container')),
                            ],
                        ),
                        dcc.Tab(
                            label='Income',
                            value='income',
                            className='ticker-tab',
                            selected_className='ticker-tab--active',
                            children=[
                                _period_radio('income'),
                                dcc.Loading(
                                    html.Div(
                                        id='income-table', style={'overflowX': 'auto'}
                                    )
                                ),
                                html.P(id='income-note', className='stmt-note'),
                            ],
                        ),
                        dcc.Tab(
                            label='Balance',
                            value='balance',
                            className='ticker-tab',
                            selected_className='ticker-tab--active',
                            children=[
                                _period_radio('balance'),
                                dcc.Loading(
                                    html.Div(
                                        id='balance-table', style={'overflowX': 'auto'}
                                    )
                                ),
                                html.P(id='balance-note', className='stmt-note'),
                            ],
                        ),
                        dcc.Tab(
                            label='Cash Flow',
                            value='cashflow',
                            className='ticker-tab',
                            selected_className='ticker-tab--active',
                            children=[
                                _period_radio('cashflow'),
                                dcc.Loading(
                                    html.Div(
                                        id='cashflow-table', style={'overflowX': 'auto'}
                                    )
                                ),
                                html.P(id='cashflow-note', className='stmt-note'),
                            ],
                        ),
                        dcc.Tab(
                            label='Dividends & Splits',
                            value='actions',
                            className='ticker-tab',
                            selected_className='ticker-tab--active',
                            children=[
                                dcc.Loading(html.Div(id='div-split-content')),
                            ],
                        ),
                        dcc.Tab(
                            label='Restatements',
                            value='restatements',
                            className='ticker-tab',
                            selected_className='ticker-tab--active',
                            children=[
                                html.Div(
                                    className='stmt-controls',
                                    style={'display': 'flex', 'gap': '16px',
                                           'alignItems': 'center', 'flexWrap': 'wrap'},
                                    children=[
                                        dcc.RadioItems(
                                            id='restatements-variant',
                                            options=[
                                                {'label': ' Annual', 'value': 'A'},
                                                {'label': ' Quarterly', 'value': 'Q'},
                                            ],
                                            value='A',
                                            inline=True,
                                            labelClassName='check-item',
                                        ),
                                        dcc.RadioItems(
                                            id='restatements-stmt',
                                            options=[
                                                {'label': ' Income', 'value': 'income'},
                                                {'label': ' Balance', 'value': 'balance'},
                                                {'label': ' Cash Flow', 'value': 'cashflow'},
                                            ],
                                            value='income',
                                            inline=True,
                                            labelClassName='check-item',
                                        ),
                                    ],
                                ),
                                dcc.Loading(html.Div(id='restatements-list')),
                                html.Div(id='restatements-detail', style={'marginTop': '16px'}),
                            ],
                        ),
                    ],
                ),
            ],
        ),
    ],
)


# ---------------------------------------------------------------------------
# Callbacks
# ---------------------------------------------------------------------------


@callback(
    Output('all-tickers-store', 'data'),
    Output('filter-market', 'options'),
    Output('filter-sector', 'options'),
    Output('filter-industry', 'options'),
    Input('ticker-init', 'data'),
)
def load_ticker_data(_: Any) -> tuple[Any, ...]:

    try:
        uni = universe_service.get_universe()
    except Exception:
        return [], [], [], []

    try:
        comp = universe_service.get_companies()[
            ['Ticker', 'Company Name', 'Sector', 'Industry']
        ]
    except Exception:
        comp = pd.DataFrame(columns=['Ticker', 'Company Name', 'Sector', 'Industry'])

    merged = uni.merge(comp, on='Ticker', how='left')

    markets = [
        {'label': m, 'value': m} for m in sorted(uni['Market'].dropna().unique())
    ]
    sectors = [
        {'label': s, 'value': s} for s in sorted(comp['Sector'].dropna().unique())
    ]
    industries = [
        {'label': i, 'value': i} for i in sorted(comp['Industry'].dropna().unique())
    ]

    return merged.to_dict('records'), markets, sectors, industries


def _ticker_sort_key(opt: dict, q: str) -> tuple:
    ticker = opt['value'].upper()
    name = opt['label'].upper()
    if ticker == q:
        return (0, ticker)
    if ticker.startswith(q):
        return (1, ticker)
    if q in ticker:
        return (2, ticker)
    if name.startswith(q):
        return (3, name)
    return (4, name)


@callback(
    Output('ticker-select', 'options'),
    Input('all-tickers-store', 'data'),
    Input('filter-market', 'value'),
    Input('filter-sector', 'value'),
    Input('filter-industry', 'value'),
    Input('ticker-select', 'search_value'),
)
def filter_tickers(
    all_data: list[dict] | None,
    market: str | None,
    sector: str | None,
    industry: str | None,
    search: str | None,
) -> list[dict]:
    if not all_data:
        return []
    df = pd.DataFrame(all_data)
    if market:
        df = df[df['Market'] == market]
    if sector and 'Sector' in df.columns:
        df = df[df['Sector'] == sector]
    if industry and 'Industry' in df.columns:
        df = df[df['Industry'] == industry]

    options = []
    for _, row in df.iterrows():
        ticker = row['Ticker']
        if not isinstance(ticker, str):
            continue
        name = row.get('Company Name')
        label = f'{ticker}  {name}' if name and pd.notna(name) else ticker
        options.append({'label': label, 'value': ticker})

    if search:
        q = search.upper()
        options = [
            o for o in options if q in o['value'].upper() or q in o['label'].upper()
        ]
        options.sort(key=lambda o: _ticker_sort_key(o, q))

    return options


@callback(
    Output('ticker-store', 'data'),
    Output('ticker-content', 'style'),
    Input('ticker-select', 'value'),
)
def update_ticker_store(ticker: str | None) -> tuple[Any, dict]:
    if not ticker:
        return None, _HIDE
    return ticker, _SHOW


@callback(
    Output('ticker-header', 'children'),
    Input('ticker-store', 'data'),
)
def render_header(ticker: str | None) -> list:
    if not ticker:
        raise PreventUpdate

    try:
        uni_row = universe_service.get_universe(ticker)
    except Exception:
        uni_row = pd.DataFrame()

    try:
        comp_row = universe_service.get_companies(ticker)
    except Exception:
        comp_row = pd.DataFrame()

    comp = comp_row.iloc[0].to_dict() if not comp_row.empty else {}
    uni = uni_row.iloc[0].to_dict() if not uni_row.empty else {}

    name = comp.get('Company Name', '')
    sector = comp.get('Sector', '')
    industry = comp.get('Industry', '')
    isin = comp.get('ISIN', '')
    cik = comp.get('CIK', '')
    currency = comp.get('Main Currency', '')
    employees = comp.get('Number of Employees') or comp.get('Number Employees')
    market = uni.get('Market', '')

    badges = [b for b in [market, sector, industry] if b]
    meta_items = [
        ('ISIN', isin),
        ('CIK', cik),
        ('Currency', currency),
        (
            'Employees',
            f'{int(employees):,}' if employees and pd.notna(employees) else '',
        ),
    ]

    return [
        html.Div(
            className='ticker-title-row',
            children=[
                html.Span(ticker, className='ticker-symbol'),
                html.Span(name, className='ticker-name') if name else None,
            ],
        ),
        html.Div(
            className='ticker-badges',
            children=[html.Span(b, className='ticker-badge') for b in badges],
        )
        if badges
        else None,
        html.Div(
            className='ticker-meta',
            children=[
                html.Span(
                    className='ticker-meta-item',
                    children=[
                        html.Span(f'{label}: ', className='ticker-meta-label'),
                        html.Span(val, className='ticker-meta-value'),
                    ],
                )
                for label, val in meta_items
                if val
            ],
        )
        if any(v for _, v in meta_items)
        else None,
    ]


@callback(
    Output('price-dates', 'start_date'),
    Output('price-dates', 'end_date'),
    [Input(f'preset-{p}', 'n_clicks') for p in _RANGE_PRESETS],
    prevent_initial_call=True,
)
def handle_preset(*args: int) -> tuple[str, str]:
    ctx = dash.callback_context
    if not ctx.triggered:
        raise PreventUpdate
    triggered_id = ctx.triggered[0]['prop_id'].split('.')[0]
    preset = triggered_id.replace('preset-', '')
    return date_range_for_preset(preset)


@callback(
    Output('price-chart', 'figure'),
    Output('price-data-store', 'data'),
    Input('ticker-store', 'data'),
    Input('price-dates', 'start_date'),
    Input('price-dates', 'end_date'),
    Input('price-source', 'value'),
)
def render_prices(
    ticker: str | None,
    start: str | None,
    end: str | None,
    source: str,
) -> tuple[Any, Any]:
    empty_fig = go.Figure(
        layout=go.Layout(
            paper_bgcolor='rgba(0,0,0,0)',
            plot_bgcolor='rgba(0,0,0,0)',
            margin=dict(l=0, r=0, t=8, b=0),
            xaxis=dict(visible=False),
            yaxis=dict(visible=False),
        )
    )
    no_data: list = []

    if not ticker:
        raise PreventUpdate

    try:
        df = price_service.get_prices(ticker, source=source, start=start, end=end)
        close_col = 'C' if source == 'stooq' else 'Close'
    except Exception as exc:
        logger.warning(f'Price query failed for {ticker}: {exc}')
        return empty_fig, no_data

    if df.empty or close_col not in df.columns:
        return empty_fig, no_data  # type: ignore[return-value]

    df = df.sort_values('Date').copy()
    df['_date_str'] = pd.to_datetime(df['Date']).dt.strftime('%Y-%m-%d')

    # Dividend + split markers
    div_traces: list[BaseTraceType] = []
    shapes: list[dict] = []

    try:
        divs = price_service.get_dividends(ticker)
        spls = price_service.get_splits(ticker)

        # Restrict actions to chart date range so x-axis doesn't span full history.
        df_min = df['_date_str'].min()
        df_max = df['_date_str'].max()
        if not divs.empty:
            div_dates_all = pd.to_datetime(divs['Date'])
            divs = divs.loc[(div_dates_all >= df_min) & (div_dates_all <= df_max)]
        if not spls.empty:
            spl_dates_all = pd.to_datetime(spls['Date'])
            spls = spls.loc[(spl_dates_all >= df_min) & (spl_dates_all <= df_max)]

        if not divs.empty:
            div_dates = pd.to_datetime(divs['Date']).dt.strftime('%Y-%m-%d')
            price_at_div = (
                df.set_index('_date_str')[close_col].reindex(div_dates).values
            )
            div_traces.append(
                go.Scatter(
                    x=div_dates,
                    y=price_at_div,
                    mode='markers',
                    marker=dict(symbol='triangle-up', size=10, color=DIV_COLOR),
                    name='Dividend',
                    hovertemplate='<b>%{x}</b><br>Div: $%{customdata:.4f}<extra></extra>',
                    customdata=divs['Amount'],
                    hoverlabel=dict(**HOVER_LABEL, bordercolor=DIV_COLOR),
                )
            )
        if not spls.empty:
            for _, row in spls.iterrows():
                d = pd.to_datetime(row['Date']).strftime('%Y-%m-%d')
                shapes.append(
                    dict(
                        type='line',
                        x0=d,
                        x1=d,
                        y0=0,
                        y1=1,
                        yref='paper',
                        line=dict(color=SPLIT_COLOR, width=1, dash='dash'),
                    )
                )
    except Exception:
        pass

    fig = go.Figure(
        data=[
            go.Scatter(
                x=df['_date_str'],
                y=df[close_col],
                name='Close',
                line=dict(color=ACCENT, width=1.5),
                hovertemplate='<b>%{x}</b><br>Close: %{y:,.2f}<extra></extra>',
                hoverlabel=dict(**HOVER_LABEL, bordercolor=ACCENT),
            ),
            *div_traces,
        ],
        layout=go.Layout(
            paper_bgcolor='rgba(0,0,0,0)',
            plot_bgcolor='rgba(128,128,128,0.05)',
            margin=dict(l=0, r=0, t=8, b=0),
            hovermode='x unified',
            font=dict(color=MUTED, size=11),
            legend=dict(
                orientation='h',
                y=1.02,
                x=1,
                xanchor='right',
                yanchor='bottom',
                font=dict(color=MUTED, size=11),
                bgcolor='rgba(0,0,0,0)',
                bordercolor='rgba(0,0,0,0)',
            ),
            xaxis=dict(
                rangeslider_visible=False,
                gridcolor=GRID,
                linecolor=GRID,
                tickfont=dict(color=MUTED, size=11),
                tickcolor=GRID,
                zeroline=False,
                showline=True,
                showspikes=True,
                spikecolor=GRID,
                spikethickness=1,
                spikedash='dot',
                spikemode='across',
            ),
            yaxis=dict(
                gridcolor=GRID,
                linecolor=GRID,
                tickfont=dict(color=MUTED, size=11),
                tickcolor=GRID,
                zeroline=True,
                showline=False,
                showspikes=False,
            ),
            shapes=shapes,
        ),
    )

    # Serialize price data for the table callback
    ohlcv_cols = (
        ['O', 'H', 'L', 'C', 'V']
        if source == 'stooq'
        else ['Open', 'High', 'Low', 'Close', 'Volume']
    )
    display_cols = ['Date'] + [c for c in ohlcv_cols if c in df.columns]
    table_df = fmt_price_table(df[display_cols].copy())
    store_data = table_df.to_dict('records')
    return fig, store_data


@callback(
    Output('price-table-container', 'children'),
    Input('price-data-store', 'data'),
    Input('price-chart', 'relayoutData'),
)
def update_price_table(price_data: list | None, relayout_data: dict | None) -> Any:
    if not price_data:
        raise PreventUpdate

    ctx = dash.callback_context
    triggered = ctx.triggered[0]['prop_id'].split('.')[0] if ctx.triggered else ''

    ############# MD> This should go in a service #######################################
    rows = price_data

    # Only filter by zoom when chart interaction triggered this (not a data reload)
    if (
        triggered == 'price-chart'
        and relayout_data
        and 'xaxis.range[0]' in relayout_data
    ):
        x0 = str(relayout_data['xaxis.range[0]'])[:10]
        x1 = str(relayout_data['xaxis.range[1]'])[:10]
        rows = [r for r in rows if x0 <= r['Date'] <= x1]

    rows = sorted(rows, key=lambda r: r['Date'], reverse=True)

    if not rows:
        return html.P('No data in selected range.', className='no-data')

    cols = [c for c in rows[0] if c != 'Ticker']
    table_cols = [{'name': c, 'id': c} for c in cols] + [{'name': '', 'id': '_spacer'}]
    rows_sp = [{**r, '_spacer': ''} for r in rows]
    _nw = {'whiteSpace': 'nowrap', 'overflow': 'hidden', 'textOverflow': 'ellipsis'}
    style = dict(TABLE_STYLE)
    style['style_cell_conditional'] = [
        {'if': {'column_id': 'Date'},   'textAlign': 'left', 'width': '100px', **_nw},
        {'if': {'column_id': 'Open'},   'width': '90px',  **_nw},
        {'if': {'column_id': 'High'},   'width': '90px',  **_nw},
        {'if': {'column_id': 'Low'},    'width': '90px',  **_nw},
        {'if': {'column_id': 'Close'},  'width': '90px',  **_nw},
        {'if': {'column_id': 'Volume'}, 'width': '120px', **_nw},
    ]
    return _dt.DataTable(
        data=rows_sp,
        columns=table_cols,
        page_size=50,
        sort_action='native',
        **{k: v for k, v in style.items()},
    )


@callback(
    Output('income-table', 'children'),
    Output('income-note', 'children'),
    Input('ticker-store', 'data'),
    Input('income-period', 'value'),
)
def render_income(ticker: str | None, period: str) -> tuple[Any, str]:
    return _render_statement(ticker, 'income', period)


@callback(
    Output('balance-table', 'children'),
    Output('balance-note', 'children'),
    Input('ticker-store', 'data'),
    Input('balance-period', 'value'),
)
def render_balance(ticker: str | None, period: str) -> tuple[Any, str]:
    return _render_statement(ticker, 'balance', period)


@callback(
    Output('cashflow-table', 'children'),
    Output('cashflow-note', 'children'),
    Input('ticker-store', 'data'),
    Input('cashflow-period', 'value'),
)
def render_cashflow(ticker: str | None, period: str) -> tuple[Any, str]:
    return _render_statement(ticker, 'cashflow', period)


def _edgar_period_strip(ticker: str, name: str, period_cols: list) -> html.Div:
    """Row of EDGAR filing links, one per displayed period column."""
    from irp.checks.edgar import filing_url
    try:
        cik = universe_service.get_ticker_cik(ticker)
        report_dates = universe_service.get_period_report_dates(ticker, name)  # type: ignore[arg-type]
    except Exception:
        return html.Div()

    chips = []
    for p in period_cols:
        p_str = str(p)
        rd = report_dates.get(p_str)
        variant = 'A' if (p_str.endswith('FY') or p_str.endswith('Q4')) else 'Q'
        url = None
        if cik and rd:
            try:
                url = filing_url(cik, rd, variant)
            except Exception:
                pass
        chips.append(
            html.A(p_str, href=url or '#', target='_blank',
                   style={'pointerEvents': 'none' if not url else 'auto',
                          'opacity': '0.35' if not url else '1',
                          'fontSize': '11px', 'padding': '2px 7px',
                          'border': f'1px solid {"rgba(255,255,255,0.18)" if url else "rgba(255,255,255,0.07)"}',
                          'borderRadius': '3px', 'color': MUTED,
                          'textDecoration': 'none', 'whiteSpace': 'nowrap'})
        )
    if not chips:
        return html.Div()
    return html.Div(
        chips,
        style={'display': 'flex', 'flexWrap': 'wrap', 'gap': '4px',
               'marginBottom': '8px', 'alignItems': 'center'},
    )


def _render_statement(ticker: str | None, name: str, period: str) -> tuple[Any, str]:
    if not ticker:
        raise PreventUpdate
    try:
        df = universe_service.get_statement(ticker, name)  # type: ignore[arg-type]
    except Exception as exc:
        logger.warning(f'Statement query failed {ticker}/{name}: {exc}')
        return html.P('Failed to load data.', className='no-data'), ''

    if df.empty:
        return html.P('No data available.', className='no-data'), ''

    # Filter columns to requested variant, sorted newest first
    def _period_key(p: str) -> tuple:
        p = str(p)
        try:
            year = int(p[:4])
            q = int(p[5]) if len(p) > 5 and p[4] == 'Q' else 0
        except (ValueError, IndexError):
            return (0, 0)
        return (year, q)

    if period == 'A':
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
    period_cols = [c for c in fmt_df.columns]
    columns: list = [{'name': 'Metric', 'id': 'Metric'}] + [
        {'name': str(c), 'id': str(c)} for c in period_cols
    ]
    # Rename any columns that duplicate to avoid DataTable id collisions
    for col in columns:
        col['id'] = col['id']
    table_df.columns = [c['id'] for c in columns]

    style = dict(TABLE_STYLE)
    style['style_cell_conditional'] = [
        {'if': {'column_id': 'Metric'}, 'textAlign': 'left', 'fontWeight': '500'},
    ]
    data: list = table_df.to_dict('records')
    table = _dt.DataTable(
        data=data,
        columns=columns,
        sort_action='native',
        page_size=40,
        **{k: v for k, v in style.items()},
    )

    edgar_strip = _edgar_period_strip(ticker, name, period_cols)
    return html.Div([edgar_strip, table]), note


# ── restatements helpers ──────────────────────────────────────────────────────

def _is_null_date(v: object) -> bool:
    if v is None:
        return True
    try:
        if pd.isna(v):
            return True
    except (TypeError, ValueError):
        pass
    return str(v).strip() in ('', 'None', 'nan', 'NaT', 'NaN')


_REST_META = {
    'Ticker', 'SrcId', 'Src', 'Currency', 'Fiscal Year', 'Fiscal Period',
    'Report Date', 'Publish Date', 'Restated Date', 'Market', 'Period', '_period',
}


def _raw_filings(ticker: str, stmt_kind: str, variant: str) -> pd.DataFrame:
    from irp.query.simfin import fundamentals as _fund
    df = _fund(ticker, stmt_kind, variant)
    if df.empty:
        return df
    df = df.copy()
    df.loc[df['Period'] == 'A', 'Fiscal Period'] = 'FY'
    df['_period'] = [
        f'{int(fy)}FY' if p == 'A' else f'{int(fy)}{fp}'
        for fy, fp, p in zip(df['Fiscal Year'], df['Fiscal Period'], df['Period'])
    ]
    return df


def _period_sort_key(p: str) -> tuple:
    try:
        year = int(p[:4])
        q = int(p[5]) if len(p) > 5 and p[4] == 'Q' else 0
    except (ValueError, IndexError):
        return (0, 0)
    return (year, q)


def _restatement_detail_html(
    ticker: str, stmt_kind: str, period: str, variant: str
) -> html.Div:
    from irp.query.simfin import statement as _sf_stmt

    df = _raw_filings(ticker, stmt_kind, variant)
    if df.empty:
        return html.Div('No data.', className='no-data')
    df_p = df[df['_period'] == period]
    if df_p.empty:
        return html.Div('No data for this period.', className='no-data')
    row = df_p.iloc[0]
    report_date   = str(row.get('Report Date',   ''))[:10]
    restated_date = str(row.get('Restated Date', ''))[:10]

    filed_df = _sf_stmt(ticker, stmt_kind,                    periods=[period])  # type: ignore[arg-type]
    rest_df  = _sf_stmt(ticker, f'{stmt_kind}_restated',      periods=[period])  # type: ignore[arg-type]

    filed_col = filed_df[period] if (not filed_df.empty and period in filed_df.columns) else pd.Series(dtype=float)
    rest_col  = rest_df[period]  if (not rest_df.empty  and period in rest_df.columns)  else pd.Series(dtype=float)

    if filed_col.empty and rest_col.empty:
        return html.Div('No data for this period.', className='no-data')

    scale_df = pd.concat([filed_col.rename('f'), rest_col.rename('r')], axis=1)
    divisor, scale_label = detect_scale(scale_df)

    records = []
    for item in dict.fromkeys(list(filed_col.index) + list(rest_col.index)):
        ps  = is_per_share(str(item))
        pct = is_pct_item(str(item), filed_col)
        v_f = filed_col.get(item)
        v_r = rest_col.get(item)
        filed_fmt = fmt_cell(v_f, per_share=ps, pct=pct, divisor=divisor) if pd.notna(v_f) else '—'
        rest_fmt  = fmt_cell(v_r, per_share=ps, pct=pct, divisor=divisor) if pd.notna(v_r) else '—'
        try:
            diff     = float(v_r) - float(v_f)
            diff_fmt = '—' if diff == 0 else fmt_cell(diff, per_share=ps, pct=pct, divisor=divisor)
            diff_pct = diff / abs(float(v_f)) * 100 if float(v_f) != 0 else float('nan')
            diff_pct_fmt = '—' if diff == 0 or pd.isna(diff_pct) else f'{diff_pct:+.1f}%'
        except (TypeError, ValueError):
            diff_fmt = diff_pct_fmt = '—'
        records.append({
            'Report Date': report_date, 'Restated Date': restated_date,
            'Item': str(item), 'As Filed': filed_fmt,
            'Restated': rest_fmt, 'Diff': diff_fmt, 'Diff%': diff_pct_fmt,
            '_': '',
        })

    _W = {'width': '93px', 'minWidth': '93px', 'maxWidth': '93px'}
    _N = {'width': '91px', 'minWidth': '91px', 'maxWidth': '91px'}
    style = dict(TABLE_STYLE)
    style['style_cell_conditional'] = [
        {'if': {'column_id': 'Report Date'},   'textAlign': 'center', **_W},
        {'if': {'column_id': 'Restated Date'}, 'textAlign': 'center', **_W},
        {'if': {'column_id': 'Item'},          'textAlign': 'right', 'fontWeight': '500',
         'width': '260px', 'minWidth': '180px', 'maxWidth': '260px'},
        {'if': {'column_id': 'As Filed'},      'textAlign': 'right', **_N},
        {'if': {'column_id': 'Restated'},      'textAlign': 'right', **_N},
        {'if': {'column_id': 'Diff'},          'textAlign': 'right',
         'width': '91px', 'minWidth': '91px', 'maxWidth': '91px'},
        {'if': {'column_id': 'Diff%'},         'textAlign': 'right',
         'width': '60px', 'minWidth': '60px', 'maxWidth': '60px'},
    ]
    table = _dt.DataTable(
        data=records,
        columns=[
            {'name': 'Report Date',   'id': 'Report Date'},
            {'name': 'Restated Date', 'id': 'Restated Date'},
            {'name': 'Item',          'id': 'Item'},
            {'name': 'As Filed',      'id': 'As Filed'},
            {'name': 'Restated',      'id': 'Restated'},
            {'name': 'Diff',          'id': 'Diff'},
            {'name': 'Diff%',         'id': 'Diff%'},
            {'name': '',              'id': '_'},
        ],
        page_size=80,
        **{k: v for k, v in style.items()},
    )

    note = (f'Amounts in {scale_label} except per-share data.'
            if divisor != 1.0 else f'Amounts in {scale_label}.')
    return html.Div([
        html.Div(
            f'{period} — {stmt_kind.title()} Statement',
            style={'fontSize': '13px', 'fontWeight': '600', 'color': MUTED,
                   'marginBottom': '4px', 'marginTop': '8px'},
        ),
        html.P(note, className='stmt-note'),
        table,
    ])


# ── restatements callbacks ────────────────────────────────────────────────────

@callback(
    Output('restatements-list', 'children'),
    Input('ticker-store', 'data'),
    Input('restatements-variant', 'value'),
    Input('restatements-stmt', 'value'),
)
def load_restatements(ticker: str | None, variant: str, stmt_kind: str) -> Any:
    if not ticker:
        raise PreventUpdate

    df = _raw_filings(ticker, stmt_kind, variant)
    if df.empty:
        return html.P('No filings available.', className='no-data')

    _THRESHOLD = 180  # days gap between Report Date and Restated Date

    periods: list[tuple[str, bool]] = []
    for period, grp in df.groupby('_period'):
        row = grp.iloc[0]
        has_rest = False
        if not _is_null_date(row.get('Restated Date')) and not _is_null_date(row.get('Report Date')):
            try:
                gap = (pd.to_datetime(row['Restated Date']) - pd.to_datetime(row['Report Date'])).days
                has_rest = gap > _THRESHOLD
            except Exception:
                pass
        periods.append((str(period), has_rest))

    periods = [(p, r) for p, r in periods if r]
    periods.sort(key=lambda x: _period_sort_key(x[0]), reverse=True)

    if not periods:
        return html.P('No restated filings found.', className='no-data')

    chips = []
    for period, has_rest in periods:
        chips.append(html.Button(
            period,
            id={'type': 'rst-chip', 'index': period},
            n_clicks=0,
            style={
                'fontSize': '11px', 'padding': '3px 9px', 'cursor': 'pointer',
                'border': '1px solid rgba(255,165,0,0.55)',
                'borderRadius': '3px',
                'background': 'rgba(255,165,0,0.08)',
                'color': MUTED,
            },
        ))

    return html.Div(chips, style={'display': 'flex', 'flexWrap': 'wrap',
                                  'gap': '5px', 'marginTop': '8px'})


@callback(
    Output('restatements-detail', 'children'),
    Input({'type': 'rst-chip', 'index': ALL}, 'n_clicks'),
    State('ticker-store', 'data'),
    State('restatements-variant', 'value'),
    State('restatements-stmt', 'value'),
    prevent_initial_call=True,
)
def show_restatement_detail(
    _n: list,
    ticker: str | None,
    variant: str,
    stmt_kind: str,
) -> Any:
    import json as _json
    ctx = dash.callback_context
    if not ctx.triggered or not ticker:
        raise PreventUpdate
    triggered_id = ctx.triggered[0]['prop_id'].split('.')[0]
    period = _json.loads(triggered_id)['index']
    return _restatement_detail_html(ticker, stmt_kind, period, variant)


@callback(
    Output('div-split-content', 'children'),
    Input('ticker-store', 'data'),
)
def render_actions(ticker: str | None) -> Any:
    if not ticker:
        raise PreventUpdate

    try:
        divs = price_service.get_dividends(ticker)
        spls = price_service.get_splits(ticker)
    except Exception as exc:
        logger.warning(f'Actions query failed {ticker}: {exc}')
        return html.P('Failed to load data.', className='no-data')

    children = []

    if divs.empty:
        div_block = html.P('No dividend history.', className='no-data')
    else:
        divs = divs.sort_values('Date', ascending=False).reset_index(drop=True)
        rows = [
            html.Tr([
                html.Td(str(row['Date'])[:10], className='stmt-td stmt-td--item'),
                html.Td(
                    f'{float(row["Amount"]):,.4f}'
                    if row['Amount'] is not None
                    and str(row['Amount']) not in ('', 'nan')
                    else '—',
                    className='stmt-td',
                ),
            ])
            for _, row in divs.iterrows()
        ]
        div_block = html.Div(
            children=[
                html.H3('Dividends', className='actions-title'),
                html.Table(
                    className='stmt-table',
                    children=[
                        html.Thead(
                            html.Tr([
                                html.Th('Date', className='stmt-th stmt-th--item'),
                                html.Th('Amount', className='stmt-th'),
                            ])
                        ),
                        html.Tbody(rows),
                    ],
                ),
            ]
        )
    children.append(html.Div(className='actions-col', children=[div_block]))

    if not spls.empty:
        spls = spls.sort_values('Date', ascending=False).reset_index(drop=True)
        ratio_col = (
            'Stock Splits' if 'Stock Splits' in spls.columns else spls.columns[-1]
        )
        spl_rows = [
            html.Tr([
                html.Td(str(row['Date'])[:10], className='stmt-td stmt-td--item'),
                html.Td(str(row[ratio_col]), className='stmt-td'),
            ])
            for _, row in spls.iterrows()
        ]
        split_block = html.Div(
            children=[
                html.H3('Splits', className='actions-title'),
                html.Table(
                    className='stmt-table',
                    children=[
                        html.Thead(
                            html.Tr([
                                html.Th('Date', className='stmt-th stmt-th--item'),
                                html.Th('Ratio', className='stmt-th'),
                            ])
                        ),
                        html.Tbody(spl_rows),
                    ],
                ),
            ]
        )
        children.append(html.Div(className='actions-col', children=[split_block]))

    return html.Div(className='actions-grid', children=children)

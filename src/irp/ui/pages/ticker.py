import logging
from datetime import date, timedelta
from typing import Any

import dash
import pandas as pd
import plotly.graph_objects as go
from dash import Input, Output, callback, dcc, html
from dash import dash_table as _dt
from dash.exceptions import PreventUpdate
from plotly.basedatatypes import BaseTraceType

from irp.query.simfin import companies as _companies
from irp.query.simfin import statement as _stmt
from irp.query.stooq import prices as _stooq_prices
from irp.query.universe import universe as _universe
from irp.query.yahoo import dividends as _divs
from irp.query.yahoo import prices as _yahoo_prices
from irp.query.yahoo import splits as _splits
from irp.ui.theme import ACCENT, DIV_COLOR, GRID, HOVER_LABEL, MUTED, SPLIT_COLOR, TABLE_STYLE
from irp.ui.ticker_fmt import date_range_for_preset, fmt_price_table, fmt_statement

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
                                dcc.Loading(html.Div(id='income-table')),
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
                                dcc.Loading(html.Div(id='balance-table')),
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
                                dcc.Loading(html.Div(id='cashflow-table')),
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
        uni = _universe()
    except Exception:
        return [], [], [], []

    try:
        comp = _companies()[['Ticker', 'Company Name', 'Sector', 'Industry']]
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


@callback(
    Output('ticker-select', 'options'),
    Input('all-tickers-store', 'data'),
    Input('filter-market', 'value'),
    Input('filter-sector', 'value'),
    Input('filter-industry', 'value'),
)
def filter_tickers(
    all_data: list[dict] | None,
    market: str | None,
    sector: str | None,
    industry: str | None,
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
        name = row.get('Company Name')
        label = f'{row["Ticker"]}  {name}' if name and pd.notna(name) else row['Ticker']
        options.append({'label': label, 'value': row['Ticker']})
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
        uni_row = _universe(ticker)
    except Exception:
        uni_row = pd.DataFrame()

    try:
        comp_row = _companies(ticker)
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
        if source == 'stooq':
            df = _stooq_prices(ticker, start=start, end=end)
            close_col = 'C'
        else:
            df = _yahoo_prices(ticker, start=start, end=end)
            close_col = 'Close'
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
        divs = _divs(ticker, start=start, end=end)
        spls = _splits(ticker, start=start, end=end)

        if not divs.empty:
            div_dates = pd.to_datetime(divs['Date']).dt.strftime('%Y-%m-%d')
            price_at_div = (
                df.set_index('_date_str')[close_col].reindex(div_dates).values
            )
            div_col = 'Dividends' if 'Dividends' in divs.columns else divs.columns[-1]
            div_traces.append(
                go.Scatter(
                    x=div_dates,
                    y=price_at_div,
                    mode='markers',
                    marker=dict(symbol='triangle-up', size=10, color=DIV_COLOR),
                    name='Dividend',
                    hovertemplate='<b>%{x}</b><br>Div: $%{customdata:.4f}<extra></extra>',
                    customdata=divs[div_col],
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
            hovermode='x',
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
            ),
            yaxis=dict(
                gridcolor=GRID,
                linecolor=GRID,
                tickfont=dict(color=MUTED, size=11),
                tickcolor=GRID,
                zeroline=False,
                showline=True,
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
    ###################################################################################
    return _dt.DataTable(
        data=rows,
        columns=[{'name': c, 'id': c} for c in cols],
        page_size=50,
        sort_action='native',
        **TABLE_STYLE,
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


def _render_statement(ticker: str | None, name: str, period: str) -> tuple[Any, str]:
    if not ticker:
        raise PreventUpdate
    try:
        df = _stmt(ticker, name)  # type: ignore[arg-type]
    except Exception as exc:
        logger.warning(f'Statement query failed {ticker}/{name}: {exc}')
        return html.P('Failed to load data.', className='no-data'), ''

    if df.empty:
        return html.P('No data available.', className='no-data'), ''

    # Filter columns to requested variant
    if period == 'A':
        cols = [c for c in df.columns if str(c).endswith('FY')]
    else:
        cols = [c for c in df.columns if not str(c).endswith('FY')]

    df = df[cols] if cols else df
    if df.empty or df.columns.empty:
        return html.P('No data available.', className='no-data'), ''

    fmt_df, note = fmt_statement(df)

    header = html.Tr(
        [
            html.Th('', className='stmt-th stmt-th--item'),
            *[html.Th(str(c), className='stmt-th') for c in fmt_df.columns],
        ]
    )
    rows = [
        html.Tr(
            [
                html.Td(str(item), className='stmt-td stmt-td--item'),
                *[
                    html.Td(str(fmt_df.loc[item, c]), className='stmt-td')
                    for c in fmt_df.columns
                ],
            ]
        )
        for item in fmt_df.index
    ]
    table = html.Table(
        className='stmt-table',
        children=[
            html.Thead(header),
            html.Tbody(rows),
        ],
    )
    return table, note


@callback(
    Output('div-split-content', 'children'),
    Input('ticker-store', 'data'),
)
def render_actions(ticker: str | None) -> Any:
    if not ticker:
        raise PreventUpdate

    try:
        divs = _divs(ticker)
        spls = _splits(ticker)
    except Exception as exc:
        logger.warning(f'Actions query failed {ticker}: {exc}')
        return html.P('Failed to load data.', className='no-data')

    children = []

    if divs.empty:
        div_block = html.P('No dividend history.', className='no-data')
    else:
        divs = divs.sort_values('Date', ascending=False).reset_index(drop=True)
        div_col = 'Dividends' if 'Dividends' in divs.columns else divs.columns[-1]
        rows = [
            html.Tr(
                [
                    html.Td(str(row['Date'])[:10], className='stmt-td stmt-td--item'),
                    html.Td(f'{float(row[div_col]):,.4f}', className='stmt-td'),
                ]
            )
            for _, row in divs.iterrows()
        ]
        div_block = html.Div(
            children=[
                html.H3('Dividends', className='actions-title'),
                html.Table(
                    className='stmt-table',
                    children=[
                        html.Thead(
                            html.Tr(
                                [
                                    html.Th('Date', className='stmt-th stmt-th--item'),
                                    html.Th('Amount', className='stmt-th'),
                                ]
                            )
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
            html.Tr(
                [
                    html.Td(str(row['Date'])[:10], className='stmt-td stmt-td--item'),
                    html.Td(str(row[ratio_col]), className='stmt-td'),
                ]
            )
            for _, row in spls.iterrows()
        ]
        split_block = html.Div(
            children=[
                html.H3('Splits', className='actions-title'),
                html.Table(
                    className='stmt-table',
                    children=[
                        html.Thead(
                            html.Tr(
                                [
                                    html.Th('Date', className='stmt-th stmt-th--item'),
                                    html.Th('Ratio', className='stmt-th'),
                                ]
                            )
                        ),
                        html.Tbody(spl_rows),
                    ],
                ),
            ]
        )
        children.append(html.Div(className='actions-col', children=[split_block]))

    return html.Div(className='actions-grid', children=children)

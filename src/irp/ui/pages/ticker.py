import contextlib
import logging
from collections.abc import Callable
from dataclasses import dataclass
from datetime import date, timedelta
from typing import Any

import dash
import numpy as np
import pandas as pd
import plotly.graph_objects as go
from dash import ALL, Input, Output, State, callback, dash_table as _dt, dcc, html
from dash.exceptions import PreventUpdate
from plotly.subplots import make_subplots

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
    is_pct_item,
    is_per_share,
)

logger = logging.getLogger(__name__)

dash.register_page(__name__, path='/ticker', name='Ticker')

_HIDE = {'display': 'none'}
_SHOW = {'display': 'block'}
_RANGE_PRESETS = ['1M', '6M', '1Y', '3Y', '5Y', 'Max']
_COMPARE_COLOR = '#f4a261'

_today = date.today().isoformat()
_1y_ago = (date.today() - timedelta(days=365)).isoformat()


# ---------------------------------------------------------------------------
# UI technical indicator specs
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class UiTaSpec:
    """One entry = one selectable indicator on the price chart.
    Adding a new indicator = add one UiTaSpec to _UI_TA_SPECS. No other changes.

    fn(close, dates) → list of Plotly traces to add to the target row.
    """
    name: str
    label: str
    is_overlay: bool                            # True = price row; False = own subplot
    fn: Callable[[pd.Series, list], list]
    row_height: float = 0.15                    # subplot height (ignored if overlay)
    y_range: tuple | None = None
    h_lines: list[dict] | None = None


def _ma_fn(window: int, color: str, dash: str) -> Callable:
    def _fn(c: pd.Series, d: list) -> list:
        ma = c.rolling(window).mean()
        return [go.Scatter(
            x=d, y=ma, name=f'MA{window}',
            line=dict(color=color, width=1, dash=dash),
            hovertemplate=f'MA{window}: %{{y:,.2f}}<extra></extra>',
        )]
    return _fn


def _bb_fn(c: pd.Series, d: list) -> list:
    sma = c.rolling(20).mean()
    std = c.rolling(20).std()
    upper = sma + 2 * std
    lower = sma - 2 * std
    return [
        go.Scatter(x=d, y=upper, name='BB Upper',
            line=dict(color='rgba(120,120,220,0.6)', width=1),
            showlegend=False,
            hovertemplate='BB Upper: %{y:,.2f}<extra></extra>'),
        go.Scatter(x=d, y=lower, name='BB Lower',
            line=dict(color='rgba(120,120,220,0.6)', width=1),
            fill='tonexty', fillcolor='rgba(120,120,220,0.07)',
            showlegend=False,
            hovertemplate='BB Lower: %{y:,.2f}<extra></extra>'),
        go.Scatter(x=d, y=sma, name='BB Mid',
            line=dict(color='rgba(120,120,220,0.4)', width=1, dash='dot'),
            showlegend=False,
            hovertemplate='BB Mid: %{y:,.2f}<extra></extra>'),
    ]


def _rsi_fn(c: pd.Series, d: list) -> list:
    delta = c.diff()
    gain = delta.clip(lower=0).ewm(com=13, adjust=False).mean()
    loss = (-delta.clip(upper=0)).ewm(com=13, adjust=False).mean()
    with np.errstate(divide='ignore', invalid='ignore'):
        rsi = 100 - 100 / (1 + gain / loss.replace(0, np.nan))
    return [go.Scatter(
        x=d, y=rsi, name='RSI(14)',
        line=dict(color='#a8dadc', width=1.2),
        hovertemplate='RSI: %{y:.1f}<extra></extra>',
    )]


def _macd_fn(c: pd.Series, d: list) -> list:
    ema12 = c.ewm(span=12, adjust=False).mean()
    ema26 = c.ewm(span=26, adjust=False).mean()
    macd_line = ema12 - ema26
    signal = macd_line.ewm(span=9, adjust=False).mean()
    hist = macd_line - signal
    hist_colors = ['#4ec94e' if (v is not None and not np.isnan(v) and v >= 0) else '#e05252'
                   for v in hist.fillna(0)]
    return [
        go.Bar(x=d, y=hist, name='MACD Hist', marker_color=hist_colors,
               showlegend=False, hovertemplate='Hist: %{y:.4f}<extra></extra>'),
        go.Scatter(x=d, y=macd_line, name='MACD',
            line=dict(color='#58a6ff', width=1),
            hovertemplate='MACD: %{y:.4f}<extra></extra>'),
        go.Scatter(x=d, y=signal, name='Signal',
            line=dict(color=_COMPARE_COLOR, width=1),
            hovertemplate='Signal: %{y:.4f}<extra></extra>'),
    ]


_UI_TA_SPECS: list[UiTaSpec] = [
    UiTaSpec('MA50',  'MA 50',     is_overlay=True,
             fn=_ma_fn(50,  '#f4a261', 'dot')),
    UiTaSpec('MA200', 'MA 200',    is_overlay=True,
             fn=_ma_fn(200, '#e76f51', 'dash')),
    UiTaSpec('BB',    'Bollinger', is_overlay=True,
             fn=_bb_fn),
    UiTaSpec('RSI',   'RSI',       is_overlay=False, row_height=0.15,
             fn=_rsi_fn, y_range=(0, 100),
             h_lines=[{'y': 70, 'color': 'rgba(220,80,80,0.45)'},
                      {'y': 30, 'color': 'rgba(80,200,80,0.45)'}]),
    UiTaSpec('MACD',  'MACD',      is_overlay=False, row_height=0.20,
             fn=_macd_fn),
]


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
                                            style={'display': 'flex', 'flexWrap': 'wrap',
                                                   'gap': '8px', 'alignItems': 'center'},
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
                                        html.Div(
                                            style={'display': 'flex', 'flexWrap': 'wrap',
                                                   'gap': '12px', 'alignItems': 'center',
                                                   'marginTop': '8px'},
                                            children=[
                                                dcc.Dropdown(
                                                    id='price-compare',
                                                    placeholder='Compare ticker...',
                                                    searchable=True,
                                                    clearable=True,
                                                    style={'minWidth': '200px',
                                                           'maxWidth': '280px'},
                                                ),
                                                dcc.Checklist(
                                                    id='price-indicators',
                                                    options=(
                                                        [{'label': ' Dividends', 'value': 'DIVS'}]
                                                        + [{'label': f' {s.label}', 'value': s.name}
                                                           for s in _UI_TA_SPECS]
                                                    ),
                                                    value=[],
                                                    inline=True,
                                                    labelClassName='check-item',
                                                ),
                                            ],
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
                                        dcc.RadioItems(
                                            id='restatements-scale',
                                            options=[
                                                {'label': ' Auto', 'value': 'auto'},
                                                {'label': ' K',    'value': 'K'},
                                                {'label': ' M',    'value': 'M'},
                                                {'label': ' B',    'value': 'B'},
                                                {'label': ' Raw',  'value': 'raw'},
                                            ],
                                            value='auto',
                                            inline=True,
                                            labelClassName='check-item',
                                        ),
                                    ],
                                ),
                                dcc.Store(id='restatements-period-store'),
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
        uni = universe_service._get_universe()
    except Exception:
        return [], [], [], []

    try:
        comp = universe_service._get_companies()[
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
    Output('price-compare', 'options'),
    Input('all-tickers-store', 'data'),
    Input('price-compare', 'search_value'),
    State('price-compare', 'value'),
)
def filter_compare_tickers(
    all_data: list[dict] | None,
    search: str | None,
    current_value: str | None,
) -> list[dict]:
    if not all_data:
        return []
    options = []
    for row in all_data:
        ticker = row.get('Ticker', '')
        if not isinstance(ticker, str):
            continue
        name = row.get('Company Name')
        label = f'{ticker}  {name}' if name and pd.notna(name) else ticker
        options.append({'label': label, 'value': ticker})
    if search:
        q = search.upper()
        options = [o for o in options if q in o['value'].upper() or q in o['label'].upper()]
        options.sort(key=lambda o: _ticker_sort_key(o, q))
    result = options[:150]
    # Always include currently selected value so Dash doesn't reset it to None
    if current_value and not any(o['value'] == current_value for o in result):
        pinned = next((o for o in options if o['value'] == current_value), None)
        if pinned:
            result = [pinned] + result[:149]
    return result


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
        uni_row = universe_service._get_universe(ticker)
    except Exception:
        uni_row = pd.DataFrame()

    try:
        comp_row = universe_service._get_companies(ticker)
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
    Input('price-compare', 'value'),
    Input('price-indicators', 'value'),
)
def render_prices(
    ticker: str | None,
    start: str | None,
    end: str | None,
    source: str,
    compare_ticker: str | None,
    indicators: list[str] | None,
) -> tuple[Any, Any]:
    def _empty_fig() -> go.Figure:
        return go.Figure(layout=go.Layout(
            paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)',
            margin=dict(l=0, r=0, t=8, b=0),
            xaxis=dict(visible=False), yaxis=dict(visible=False),
        ))

    if not ticker:
        raise PreventUpdate

    close_col = 'C' if source == 'stooq' else 'Close'
    vol_col   = 'V' if source == 'stooq' else 'Volume'

    try:
        df = price_service._get_prices(ticker, source=source, start=start, end=end)
    except Exception as exc:
        logger.warning(f'Price query failed for {ticker}: {exc}')
        return _empty_fig(), []

    if df.empty or close_col not in df.columns:
        return _empty_fig(), []

    df = df.sort_values('Date').copy()
    df['_date_str'] = pd.to_datetime(df['Date']).dt.strftime('%Y-%m-%d')
    dates = df['_date_str'].tolist()
    close = df[close_col].reset_index(drop=True).astype(float)

    # ---- comparison ticker ----
    compare_df: pd.DataFrame | None = None
    if compare_ticker and compare_ticker != ticker:
        try:
            cdf = price_service._get_prices(compare_ticker, source=source, start=start, end=end)
            if not cdf.empty and close_col in cdf.columns:
                cdf = cdf.sort_values('Date').copy()
                cdf['_date_str'] = pd.to_datetime(cdf['Date']).dt.strftime('%Y-%m-%d')
                compare_df = cdf
        except Exception as exc:
            logger.warning(f'compare fetch failed for {compare_ticker!r}: {exc}')

    # ---- dividends + splits ----
    div_traces: list = []
    split_dates: list[str] = []
    try:
        divs = price_service._get_dividends(ticker)
        spls = price_service._get_splits(ticker)
        df_min, df_max = df['_date_str'].min(), df['_date_str'].max()
        if not divs.empty:
            d_dates = pd.to_datetime(divs['Date'])
            divs = divs.loc[(d_dates >= df_min) & (d_dates <= df_max)]
        if not spls.empty:
            s_dates = pd.to_datetime(spls['Date'])
            spls = spls.loc[(s_dates >= df_min) & (s_dates <= df_max)]

        if not divs.empty:
            div_date_strs = pd.to_datetime(divs['Date']).dt.strftime('%Y-%m-%d')
            price_at_div = df.set_index('_date_str')[close_col].reindex(div_date_strs).values
            div_traces.append(go.Scatter(
                x=div_date_strs, y=price_at_div, mode='markers',
                marker=dict(symbol='triangle-up', size=10, color=DIV_COLOR),
                name='Dividend',
                hovertemplate='<b>%{x}</b><br>Div: $%{customdata:.4f}<extra></extra>',
                customdata=divs['Amount'],
                hoverlabel=dict(**HOVER_LABEL, bordercolor=DIV_COLOR),
            ))
        if not spls.empty:
            split_dates = [pd.to_datetime(r['Date']).strftime('%Y-%m-%d')
                           for _, r in spls.iterrows()]
    except Exception:
        # Dividend/split markers are optional chart decoration — never let bad
        # actions data break the price chart.
        logger.debug('dividend/split markers skipped', exc_info=True)

    # ---- subplot layout ----
    ind = set(indicators or [])
    selected_subplots = [s for s in _UI_TA_SPECS if not s.is_overlay and s.name in ind]

    PRICE_H  = 0.55
    VOLUME_H = 0.12
    sub_heights = [s.row_height for s in selected_subplots]
    total_h = PRICE_H + sum(sub_heights) + VOLUME_H
    n_rows = 2 + len(selected_subplots)  # price + subplots + volume
    row_heights = [PRICE_H / total_h] + [h / total_h for h in sub_heights] + [VOLUME_H / total_h]

    price_row = 1
    vol_row   = n_rows
    subplot_row = {s.name: i + 2 for i, s in enumerate(selected_subplots)}

    fig = make_subplots(
        rows=n_rows, cols=1,
        shared_xaxes=True,
        vertical_spacing=0.02,
        row_heights=row_heights,
    )

    # ---- price row traces ----
    if compare_df is not None:
        # Rebase both to 100 at first common date
        primary_s = df.set_index('_date_str')[close_col].astype(float)
        compare_s = compare_df.set_index('_date_str')[close_col].astype(float)
        common_start = max(primary_s.index[0], compare_s.index[0])
        pb = primary_s.loc[primary_s.index >= common_start].iloc[0]
        cb = compare_s.loc[compare_s.index >= common_start].iloc[0]
        primary_rb = primary_s / pb * 100
        compare_rb = compare_s / cb * 100

        fig.add_trace(go.Scatter(
            x=primary_rb.index.tolist(), y=primary_rb.values,
            name=ticker, line=dict(color=ACCENT, width=1.5),
            hovertemplate='<b>%{x}</b><br>' + ticker + ': %{y:.2f}<extra></extra>',
            hoverlabel=dict(**HOVER_LABEL, bordercolor=ACCENT),
        ), row=price_row, col=1)
        fig.add_trace(go.Scatter(
            x=compare_rb.index.tolist(), y=compare_rb.values,
            name=compare_ticker, line=dict(color=_COMPARE_COLOR, width=1.5),
            hovertemplate='<b>%{x}</b><br>' + compare_ticker + ': %{y:.2f}<extra></extra>',
            hoverlabel=dict(**HOVER_LABEL, bordercolor=_COMPARE_COLOR),
        ), row=price_row, col=1)

        # In compare mode, pass rebased primary to overlay indicators so MAs align
        close_for_ta = primary_rb.reindex(dates).reset_index(drop=True)
    else:
        fig.add_trace(go.Scatter(
            x=dates, y=close,
            name='Close', line=dict(color=ACCENT, width=1.5),
            hovertemplate='<b>%{x}</b><br>Close: %{y:,.2f}<extra></extra>',
            hoverlabel=dict(**HOVER_LABEL, bordercolor=ACCENT),
        ), row=price_row, col=1)
        close_for_ta = close

    # Overlay indicators (MA50, MA200, Bollinger)
    for spec in _UI_TA_SPECS:
        if spec.is_overlay and spec.name in ind:
            for trace in spec.fn(close_for_ta, dates):
                fig.add_trace(trace, row=price_row, col=1)

    # Dividend markers (off by default; skip in compare mode — y values are absolute prices, wrong scale)
    if 'DIVS' in ind and compare_df is None:
        for trace in div_traces:
            fig.add_trace(trace, row=price_row, col=1)

    # Split lines (spans full chart height)
    for sd in split_dates:
        fig.add_vline(x=sd, line=dict(color=SPLIT_COLOR, width=1, dash='dash'))

    # ---- subplot indicator rows ----
    for spec in selected_subplots:
        row = subplot_row[spec.name]
        for trace in spec.fn(close, dates):   # always computed on raw close
            fig.add_trace(trace, row=row, col=1)
        if spec.y_range:
            fig.update_yaxes(range=list(spec.y_range), row=row, col=1)
        for hline in (spec.h_lines or []):
            fig.add_hline(
                y=hline['y'],
                line=dict(color=hline['color'], width=1, dash='dash'),
                row=row, col=1,
            )

    # ---- volume row ----
    if vol_col in df.columns:
        prev_close = df[close_col].shift(1)
        up = df[close_col] >= prev_close
        bar_colors = ['#4ec94e' if u else '#e05252' for u in up]
        fig.add_trace(go.Bar(
            x=dates, y=df[vol_col], name='Volume',
            marker_color=bar_colors, showlegend=False,
            hovertemplate='<b>%{x}</b><br>Vol: %{y:,.0f}<extra></extra>',
        ), row=vol_row, col=1)

    # ---- shared axis styling ----
    _ax = dict(
        gridcolor=GRID, linecolor=GRID,
        tickfont=dict(color=MUTED, size=11),
        tickcolor=GRID, zeroline=False, showline=True,
    )
    fig.update_xaxes(**_ax, showspikes=True, spikecolor=GRID,
                     spikethickness=1, spikedash='dot', spikemode='across',
                     rangeslider_visible=False)
    fig.update_yaxes(**_ax, showspikes=False)
    fig.update_yaxes(zeroline=True, row=price_row, col=1)

    fig.update_layout(
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(128,128,128,0.05)',
        margin=dict(l=0, r=0, t=8, b=0),
        hovermode='x unified',
        font=dict(color=MUTED, size=11),
        barmode='overlay',
        legend=dict(
            orientation='h', y=1.02, x=1, xanchor='right', yanchor='bottom',
            font=dict(color=MUTED, size=11),
            bgcolor='rgba(0,0,0,0)', bordercolor='rgba(0,0,0,0)',
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
        cik = universe_service._get_ticker_cik(ticker)
        report_dates = universe_service._get_period_report_dates(ticker, name)  # type: ignore[arg-type]
    except Exception:
        return html.Div()

    chips = []
    for p in period_cols:
        p_str = str(p)
        rd = report_dates.get(p_str)
        variant = 'A' if (p_str.endswith('FY') or p_str.endswith('Q4')) else 'Q'
        url = None
        if cik and rd:
            with contextlib.suppress(Exception):
                url = filing_url(cik, rd, variant)
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
        df = universe_service._get_statement(ticker, name)  # type: ignore[arg-type]
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
        for fy, fp, p in zip(df['Fiscal Year'], df['Fiscal Period'], df['Period'], strict=False)
    ]
    return df


def _period_sort_key(p: str) -> tuple:
    try:
        year = int(p[:4])
        q = int(p[5]) if len(p) > 5 and p[4] == 'Q' else 0
    except (ValueError, IndexError):
        return (0, 0)
    return (year, q)


_SCALE_MAP = {
    'K':   (1e3,  'USD thousands'),
    'M':   (1e6,  'USD millions'),
    'B':   (1e9,  'USD billions'),
    'raw': (1.0,  'USD'),
}
_SCALE_WIDTH = {'K': '105px', 'M': '91px', 'B': '75px', 'raw': '130px', 'auto': '91px'}


def _restatement_detail_html(
    ticker: str, stmt_kind: str, period: str, variant: str, scale: str = 'auto'
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

    if scale in _SCALE_MAP:
        divisor, scale_label = _SCALE_MAP[scale]
    else:
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

    _W  = {'width': '93px',  'minWidth': '93px',  'maxWidth': '93px'}
    _nw = _SCALE_WIDTH.get(scale, '91px')
    _N  = {'width': _nw, 'minWidth': _nw, 'maxWidth': _nw}
    style = dict(TABLE_STYLE)
    style['style_cell_conditional'] = [
        {'if': {'column_id': 'Report Date'},   'textAlign': 'center', **_W},
        {'if': {'column_id': 'Restated Date'}, 'textAlign': 'center', **_W},
        {'if': {'column_id': 'Item'},          'textAlign': 'right', 'fontWeight': '500',
         'width': '260px', 'minWidth': '180px', 'maxWidth': '260px'},
        {'if': {'column_id': 'As Filed'},      'textAlign': 'right', **_N},
        {'if': {'column_id': 'Restated'},      'textAlign': 'right', **_N},
        {'if': {'column_id': 'Diff'},          'textAlign': 'right', **_N},
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
    from irp.query.simfin import statement as _sf_stmt

    if not ticker:
        raise PreventUpdate

    filed_df = _sf_stmt(ticker, stmt_kind,                periods=None)   # type: ignore[arg-type]
    rest_df  = _sf_stmt(ticker, f'{stmt_kind}_restated',  periods=None)   # type: ignore[arg-type]

    if filed_df.empty and rest_df.empty:
        return html.P('No filings available.', className='no-data')

    # Filter to annual (FY) or quarterly (Q1-Q4) based on variant
    suffix = 'FY' if variant == 'A' else ('Q1', 'Q2', 'Q3', 'Q4')
    _match = (lambda c: c.endswith(suffix)) if isinstance(suffix, str) else (lambda c: any(c.endswith(s) for s in suffix))
    filed_cols = [c for c in filed_df.columns if _match(c)] if not filed_df.empty else []
    rest_cols  = [c for c in rest_df.columns  if _match(c)] if not rest_df.empty  else []
    common     = [c for c in filed_cols if c in rest_cols]

    if not common:
        return html.P('No restated filings found.', className='no-data')

    restated_periods = []
    for period in common:
        filed_col = pd.to_numeric(filed_df[period], errors='coerce').fillna(0)
        rest_col  = pd.to_numeric(rest_df[period],  errors='coerce').fillna(0)
        if abs(filed_col.sum() - rest_col.sum()) > 0:
            restated_periods.append(period)

    restated_periods.sort(key=_period_sort_key, reverse=True)

    if not restated_periods:
        return html.P('No restated filings found.', className='no-data')

    chips = []
    for period in restated_periods:
        chips.append(html.Button(
            period,
            id={'type': 'rst-chip', 'index': period},
            n_clicks=0,
            style={
                'fontSize': '11px', 'padding': '3px 9px', 'cursor': 'pointer',
                'border': '1px solid rgba(255,165,0,0.55)',
                'borderRadius': '3px', 'background': 'rgba(255,165,0,0.08)',
                'color': MUTED,
            },
        ))

    return html.Div(chips, style={'display': 'flex', 'flexWrap': 'wrap',
                                  'gap': '5px', 'marginTop': '8px'})


@callback(
    Output('restatements-period-store', 'data'),
    Input({'type': 'rst-chip', 'index': ALL}, 'n_clicks'),
    prevent_initial_call=True,
)
def store_restatement_period(_n: list) -> str | None:
    import json as _json
    ctx = dash.callback_context
    if not ctx.triggered:
        raise PreventUpdate
    triggered_id = ctx.triggered[0]['prop_id'].split('.')[0]
    try:
        return _json.loads(triggered_id)['index']
    except (ValueError, KeyError):
        raise PreventUpdate from None


@callback(
    Output('restatements-detail', 'children'),
    Input('restatements-period-store', 'data'),
    Input('restatements-scale', 'value'),
    State('ticker-store', 'data'),
    State('restatements-variant', 'value'),
    State('restatements-stmt', 'value'),
    prevent_initial_call=True,
)
def show_restatement_detail(
    period: str | None,
    scale: str,
    ticker: str | None,
    variant: str,
    stmt_kind: str,
) -> Any:
    if not period or not ticker:
        raise PreventUpdate
    return _restatement_detail_html(ticker, stmt_kind, period, variant, scale)


@callback(
    Output('div-split-content', 'children'),
    Input('ticker-store', 'data'),
)
def render_actions(ticker: str | None) -> Any:
    if not ticker:
        raise PreventUpdate

    try:
        divs = price_service._get_dividends(ticker)
        spls = price_service._get_splits(ticker)
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

"""Data quality review page: fundamental + price anomaly triage."""
import logging

import dash
import pandas as pd
import plotly.graph_objects as go
from dash import Input, Output, State, callback, dcc, html
from dash.exceptions import PreventUpdate

import irp.checks.simfin_rules as _simfin_rules
import irp.checks.stooq_rules as _stooq_rules
from irp.ui.charts import base_chart_layout as _chart_layout, empty_figure as _empty_figure
from irp.ui.services import data_quality_service as dq
from irp.ui.services import universe_service as _uv
from irp.ui.theme import ACCENT, MUTED

dash.register_page(__name__, path='/data-quality', name='Data Quality')
logger = logging.getLogger(__name__)

_SIMFIN_RULE_OPTIONS = [{'label': r.name, 'value': r.name} for r in _simfin_rules.REGISTRY]
_SIMFIN_RULE_NAMES = [r.name for r in _simfin_rules.REGISTRY]
_STOOQ_RULE_OPTIONS = [{'label': r.name, 'value': r.name} for r in _stooq_rules.REGISTRY]
_STOOQ_RULE_NAMES = [r.name for r in _stooq_rules.REGISTRY]

_SF_RULE_INFO: dict[str, str] = {
    r.name: f'{r.lhs} == {r.rhs}'
    for r in _simfin_rules.REGISTRY
}
_STQ_RULE_INFO: dict[str, str] = {
    r.name: r.description
    for r in _stooq_rules.REGISTRY
}

# Dynamic options loaded once at startup
def _market_opts(markets: list[str]) -> list[dict]:
    return [{'label': 'All', 'value': ''}] + [{'label': m, 'value': m} for m in sorted(markets)]

_companies = _uv.get_companies()
_universe = _uv.get_universe()
_SF_MARKET_OPTS = _market_opts(_companies['Market'].dropna().unique().tolist())
_STQ_MARKET_OPTS = _market_opts(_universe['Market'].dropna().unique().tolist())
_STOCK_MARKETS = {'nasdaq stocks', 'nyse stocks', 'nysemkt stocks', 'stooq stocks indices'}
_stock_tickers = _universe[_universe['Market'].isin(_STOCK_MARKETS)]['Ticker'].dropna().unique().tolist()
_TICKER_OPTS = [{'label': t, 'value': t} for t in sorted(_stock_tickers)]
_STATUS_OPTIONS = [
    {'label': 'OK — data is correct', 'value': 'ok'},
    {'label': 'Data error', 'value': 'data_error'},
    {'label': 'To check', 'value': 'to_check'},
]
_TABLE_STYLE = dict(
    style_header=dict(backgroundColor='rgba(255,255,255,0.06)', color=MUTED, fontWeight='600', fontSize='12px'),
    style_cell=dict(backgroundColor='rgba(0,0,0,0)', color=MUTED, fontSize='12px',
                    padding='6px 10px', border='1px solid rgba(255,255,255,0.07)'),
    style_data=dict(backgroundColor='rgba(0,0,0,0)'),
)
_STATUS_ROW_STYLES = [
    {'if': {'filter_query': '{_status} = "ok"'},         'backgroundColor': 'rgba(46,204,113,0.12)'},
    {'if': {'filter_query': '{_status} = "data_error"'}, 'backgroundColor': 'rgba(231,76,60,0.12)'},
    {'if': {'filter_query': '{_status} = "to_check"'},   'backgroundColor': 'rgba(243,156,18,0.12)'},
]


def _fmt_date(d: int) -> str:
    s = str(int(d))
    return f'{s[:4]}-{s[4:6]}-{s[6:8]}' if len(s) == 8 else s


def _fmt_dates(dates: list) -> str:
    return ', '.join(_fmt_date(d) for d in dates[:5])


def _chip(label: str, value: str | int, color: str = MUTED) -> html.Div:
    return html.Div(className='metric-chip', children=[
        html.Span(str(value), style={'color': color, 'fontWeight': '700', 'fontSize': '18px'}),
        html.Span(label, style={'color': MUTED, 'fontSize': '11px', 'marginLeft': '4px'}),
    ])


_TH = {'padding': '4px 10px', 'fontSize': '11px', 'color': MUTED,
        'border': '1px solid rgba(255,255,255,0.08)',
        'backgroundColor': 'rgba(255,255,255,0.05)', 'whiteSpace': 'nowrap'}
_TD = {'padding': '3px 10px', 'fontSize': '12px', 'color': MUTED,
        'border': '1px solid rgba(255,255,255,0.08)'}


def _rules_reference(rule_info: dict[str, str], col2_header: str) -> html.Details:
    """Collapsible table listing every rule with its definition."""
    header = html.Tr([html.Th('Rule', style=_TH), html.Th(col2_header, style={**_TH, 'width': '100%'})])
    rows = [
        html.Tr([
            html.Td(name, style={**_TD, 'whiteSpace': 'nowrap', 'fontFamily': 'monospace'}),
            html.Td(desc, style=_TD),
        ])
        for name, desc in rule_info.items()
    ]
    return html.Details(style={'marginBottom': '12px'}, children=[
        html.Summary('Rules reference', style={
            'cursor': 'pointer', 'color': MUTED, 'fontSize': '13px', 'userSelect': 'none',
        }),
        html.Div(style={'marginTop': '8px', 'overflowX': 'auto'}, children=[
            html.Table([html.Thead(header), html.Tbody(rows)],
                       style={'borderCollapse': 'collapse', 'width': '100%'}),
        ]),
    ])


def _review_form(id_prefix: str) -> html.Div:
    return html.Div(style={'marginTop': '12px', 'padding': '12px',
                           'border': '1px solid rgba(255,255,255,0.1)', 'borderRadius': '6px'}, children=[
        html.H5('Review', style={'color': MUTED, 'marginBottom': '8px', 'fontSize': '13px'}),
        dcc.RadioItems(
            id=f'{id_prefix}-status',
            options=_STATUS_OPTIONS,
            value='to_check',
            labelClassName='check-item',
            style={'display': 'flex', 'gap': '16px', 'flexWrap': 'wrap', 'marginBottom': '8px'},
        ),
        dcc.Textarea(
            id=f'{id_prefix}-note',
            placeholder='Note (optional)',
            style={'width': '100%', 'minHeight': '56px', 'fontSize': '12px',
                   'backgroundColor': 'rgba(255,255,255,0.04)', 'color': MUTED,
                   'border': '1px solid rgba(255,255,255,0.12)', 'borderRadius': '4px', 'padding': '6px'},
        ),
        html.Div(style={'display': 'flex', 'gap': '8px', 'marginTop': '8px', 'alignItems': 'center'}, children=[
            html.Button('Save review', id=f'{id_prefix}-submit', className='run-btn', n_clicks=0),
            html.Span(id=f'{id_prefix}-msg', style={'color': ACCENT, 'fontSize': '12px'}),
        ]),
    ])


# ── Fundamentals tab ──────────────────────────────────────────────────

_fund_tab = html.Div(children=[
    # Run controls
    html.Div(className='control-row sticky-controls', children=[
        dcc.RadioItems(
            id='dq-sf-variant', options=[{'label': 'Annual', 'value': 'A'}, {'label': 'Quarterly', 'value': 'Q'}],
            value='A', inline=True, labelClassName='check-item',
        ),
        dcc.Checklist(
            id='dq-sf-include-reviewed',
            options=[{'label': 'Include reviewed', 'value': 'yes'}],
            value=[],
            inline=True,
            labelClassName='check-item',
        ),
        html.Button('Run checks', id='dq-sf-run', className='run-btn', n_clicks=0),
    ]),
    # Filter controls (applied instantly, no re-run)
    html.Div(className='control-row', style={'marginTop': '8px'}, children=[
        dcc.Dropdown(
            id='dq-sf-filter-ticker',
            options=_TICKER_OPTS,
            multi=True, searchable=True, clearable=True, placeholder='Ticker…',
            className='filter-dropdown', style={'minWidth': '160px', 'fontSize': '12px'},
        ),
        dcc.Dropdown(
            id='dq-sf-filter-market',
            options=_SF_MARKET_OPTS,
            value='', clearable=False, placeholder='Market',
            className='filter-dropdown', style={'minWidth': '100px'},
        ),
        dcc.Dropdown(
            id='dq-sf-rules',
            options=_SIMFIN_RULE_OPTIONS,
            value=_SIMFIN_RULE_NAMES,
            multi=True, placeholder='Rules…',
            className='filter-dropdown', style={'minWidth': '220px', 'fontSize': '12px'},
        ),
    ]),

    # Rules reference
    _rules_reference(_SF_RULE_INFO, 'Checks (LHS == RHS)'),

    # Table
    dcc.Loading(type='circle', color=ACCENT, children=html.Div(id='dq-sf-table-wrap')),

    # Detail panel
    html.Div(id='dq-sf-detail', style={'display': 'none'}, children=[
        html.Div(style={'display': 'flex', 'justifyContent': 'space-between', 'alignItems': 'center',
                        'margin': '12px 0 8px'}, children=[
            html.H4(id='dq-sf-detail-title', style={'color': MUTED, 'margin': 0, 'fontSize': '14px'}),
            html.Div(style={'display': 'flex', 'gap': '8px'}, children=[
                html.Button('Open EDGAR ↗', id='dq-sf-edgar-btn', className='run-btn', n_clicks=0),
                html.Span(id='dq-sf-edgar-msg', style={'color': MUTED, 'fontSize': '12px'}),
                html.Button('✕ Close', id='dq-sf-close', className='run-btn', n_clicks=0,
                            style={'background': 'rgba(255,255,255,0.06)'}),
            ]),
        ]),
        html.Div(id='dq-sf-violations-summary', style={'marginBottom': '8px'}),
        html.Div(style={'display': 'flex', 'alignItems': 'center', 'gap': '8px', 'margin': '8px 0'}, children=[
            html.Span('Inspect rule:', style={'color': MUTED, 'fontSize': '13px', 'whiteSpace': 'nowrap'}),
            dcc.Dropdown(id='dq-sf-rule-select', options=[], value=None, clearable=False,
                         className='filter-dropdown', style={'minWidth': '240px', 'fontSize': '12px'}),
        ]),
        html.Div(id='dq-sf-inspect-table'),
        # SimFin statement section for EDGAR comparison
        html.Div(style={'marginTop': '16px', 'borderTop': '1px solid rgba(255,255,255,0.08)',
                        'paddingTop': '12px'}, children=[
            html.Div(style={'display': 'flex', 'alignItems': 'center', 'gap': '12px',
                            'marginBottom': '8px'}, children=[
                html.Span('SimFin Statement', style={'color': MUTED, 'fontSize': '13px',
                                                     'fontWeight': '600', 'whiteSpace': 'nowrap'}),
                dcc.RadioItems(
                    id='dq-sf-stmt-kind',
                    options=[
                        {'label': 'Income', 'value': 'income'},
                        {'label': 'Balance', 'value': 'balance'},
                        {'label': 'Cash Flow', 'value': 'cashflow'},
                    ],
                    value='income', inline=True, labelClassName='check-item',
                ),
            ]),
            dcc.Loading(type='circle', color=ACCENT,
                        children=html.Div(id='dq-sf-statement-wrap')),
        ]),
        _review_form('dq-sf'),
    ]),

    # Manual flag
    html.Details(style={'marginTop': '24px'}, children=[
        html.Summary('Add manual flag', style={'color': MUTED, 'cursor': 'pointer', 'fontSize': '13px'}),
        html.Div(style={'display': 'flex', 'gap': '8px', 'flexWrap': 'wrap', 'marginTop': '10px',
                        'alignItems': 'flex-end'}, children=[
            dcc.Input(id='dq-flag-ticker', placeholder='Ticker', debounce=False,
                      style={'width': '90px'}, className='filter-input'),
            dcc.Input(id='dq-flag-period', placeholder='Period (e.g. 2023FY)', debounce=False,
                      style={'width': '130px'}, className='filter-input'),
            dcc.Input(id='dq-flag-subject', placeholder='Subject (e.g. Revenue)', debounce=False,
                      style={'width': '160px'}, className='filter-input'),
            dcc.RadioItems(id='dq-flag-status', options=_STATUS_OPTIONS, value='to_check',
                           inline=True, labelClassName='check-item'),
            dcc.Textarea(id='dq-flag-note', placeholder='Note',
                         style={'width': '220px', 'minHeight': '36px', 'fontSize': '12px',
                                'backgroundColor': 'rgba(255,255,255,0.04)', 'color': MUTED,
                                'border': '1px solid rgba(255,255,255,0.12)', 'borderRadius': '4px',
                                'padding': '6px'}),
            html.Button('Add flag', id='dq-flag-submit', className='run-btn', n_clicks=0),
            html.Span(id='dq-flag-msg', style={'color': ACCENT, 'fontSize': '12px'}),
        ]),
    ]),
])

# ── Prices tab ────────────────────────────────────────────────────────

_prices_tab = html.Div(children=[
    html.Div(className='control-row sticky-controls', children=[
        dcc.Checklist(
            id='dq-stq-include-reviewed',
            options=[{'label': 'Include reviewed', 'value': 'yes'}],
            value=[],
            inline=True,
            labelClassName='check-item',
        ),
        html.Button('Run checks', id='dq-stq-run', className='run-btn', n_clicks=0),
    ]),
    html.Div(className='control-row', style={'marginTop': '8px'}, children=[
        dcc.Dropdown(
            id='dq-stq-filter-ticker',
            options=_TICKER_OPTS,
            multi=True, searchable=True, clearable=True, placeholder='Ticker…',
            className='filter-dropdown', style={'minWidth': '160px', 'fontSize': '12px'},
        ),
        dcc.Dropdown(
            id='dq-stq-filter-market',
            options=_STQ_MARKET_OPTS,
            value='', clearable=False, placeholder='Market',
            className='filter-dropdown', style={'minWidth': '150px'},
        ),
        dcc.Dropdown(
            id='dq-stq-rules',
            options=_STOOQ_RULE_OPTIONS,
            value=_STOOQ_RULE_NAMES,
            multi=True, placeholder='Rules…',
            className='filter-dropdown', style={'minWidth': '200px', 'fontSize': '12px'},
        ),
    ]),
    # Rules reference
    _rules_reference(_STQ_RULE_INFO, 'Description'),

    dcc.Loading(type='circle', color=ACCENT, children=html.Div(id='dq-stq-table-wrap')),
    html.Div(id='dq-stq-detail', style={'display': 'none'}, children=[
        html.Div(style={'display': 'flex', 'justifyContent': 'space-between', 'alignItems': 'center',
                        'margin': '12px 0 8px'}, children=[
            html.H4(id='dq-stq-detail-title', style={'color': MUTED, 'margin': 0, 'fontSize': '14px'}),
            html.Div(style={'display': 'flex', 'gap': '8px'}, children=[
                html.A(id='dq-stq-yahoo-link', target='_blank', rel='noopener',
                       className='run-btn', style={'textDecoration': 'none'},
                       children='Yahoo Finance ↗'),
                html.Button('✕ Close', id='dq-stq-close', className='run-btn', n_clicks=0,
                            style={'background': 'rgba(255,255,255,0.06)'}),
            ]),
        ]),
        dcc.Graph(id='dq-stq-chart', config={'displayModeBar': False},
                  figure=_empty_figure('Select a row'), style={'minHeight': '300px'}),
        html.Div(id='dq-stq-bars-table'),
        _review_form('dq-stq'),
    ]),
])

# ── Dashboard tab ─────────────────────────────────────────────────────

_dashboard_tab = html.Div(id='dq-dashboard', style={'padding': '16px 0'})

# ── Full layout ───────────────────────────────────────────────────────

layout = html.Div(className='page-content', children=[
    dcc.Store(id='dq-sf-results'),
    dcc.Store(id='dq-stq-results'),
    dcc.Store(id='dq-sf-sel'),
    dcc.Store(id='dq-stq-sel'),
    dcc.Store(id='dq-sf-edgar-url'),
    html.H2('Data Quality', className='page-title'),
    html.P('Triage fundamental and price anomalies; mark reviewed; add manual flags.',
           className='page-subtitle'),
    dcc.Tabs(
        id='dq-tabs',
        value='fundamentals',
        className='page-tabs',
        children=[
            dcc.Tab(label='Fundamentals', value='fundamentals', className='page-tab',
                    selected_className='page-tab-selected', children=[_fund_tab]),
            dcc.Tab(label='Prices', value='prices', className='page-tab',
                    selected_className='page-tab-selected', children=[_prices_tab]),
            dcc.Tab(label='Dashboard', value='dashboard', className='page-tab',
                    selected_className='page-tab-selected', children=[_dashboard_tab]),
        ],
    ),
])


# ── Helpers ───────────────────────────────────────────────────────────

def _make_simfin_table(records: list[dict]) -> dash.dash_table.DataTable:
    Fmt = dash.dash_table.Format
    filings: dict[tuple, dict] = {}
    for r in records:
        key = (r['Ticker'], r['Period_str'])
        if key not in filings:
            filings[key] = {
                'Ticker': r['Ticker'],
                'Company Name': r.get('Company Name', ''),
                'Market': r.get('Market', ''),
                'Period_str': r['Period_str'],
                'CIK': r.get('CIK'),
                'Report Date': r.get('Report Date', ''),
                'Period': r.get('Period', ''),
                '_rules': [],
                '_max_diff': 0.0,
            }
        filings[key]['_rules'].append(r['Rule'])
        filings[key]['_max_diff'] = max(
            filings[key]['_max_diff'], abs(r.get('rel_diff_pct') or 0)
        )

    rows = []
    for f in filings.values():
        rows.append({
            'Ticker': f['Ticker'],
            'Company Name': f['Company Name'],
            'Market': f['Market'],
            'Period_str': f['Period_str'],
            'n_rules': len(f['_rules']),
            'rules_str': ', '.join(f['_rules']),
            'max_diff_pct': round(f['_max_diff'], 2),
            'CIK': f['CIK'],
            'Report Date': f['Report Date'],
            'Period': f['Period'],
        })

    cols = [
        {'name': 'Ticker',   'id': 'Ticker'},
        {'name': 'Company',  'id': 'Company Name'},
        {'name': 'Market',   'id': 'Market'},
        {'name': 'Period',   'id': 'Period_str'},
        {'name': 'Rules',    'id': 'rules_str'},
        {'name': 'Max Δ%',   'id': 'max_diff_pct', 'type': 'numeric',
         'format': Fmt.Format(precision=2, scheme=Fmt.Scheme.fixed)},
    ]
    return dash.dash_table.DataTable(
        id='dq-sf-table',
        data=rows,
        columns=cols,
        hidden_columns=['CIK', 'Report Date', 'Period'],
        row_selectable=False,
        active_cell=None,
        page_size=25,
        sort_action='native',
        filter_action='native',
        style_table={'overflowX': 'auto'},
        **_TABLE_STYLE,
    )


def _make_stooq_table(records: list[dict]) -> dash.dash_table.DataTable:
    display = []
    for r in records:
        d = dict(r)
        d['_dates_str'] = _fmt_dates(r.get('sample_dates') or [])
        display.append(d)
    cols = [
        {'name': 'Rule',   'id': 'Rule'},
        {'name': 'Ticker', 'id': 'Ticker'},
        {'name': 'Market', 'id': 'Market'},
        {'name': 'Year',   'id': 'Period_str'},
        {'name': 'Count',  'id': 'count', 'type': 'numeric'},
        {'name': 'Sample dates', 'id': '_dates_str'},
    ]
    return dash.dash_table.DataTable(
        id='dq-stq-table',
        data=display,
        columns=cols,
        row_selectable=False,
        active_cell=None,
        page_size=25,
        sort_action='native',
        filter_action='native',
        style_table={'overflowX': 'auto'},
        **_TABLE_STYLE,
    )


def _inspect_html(df: pd.DataFrame) -> html.Div:
    if df.empty:
        return html.Div()
    num_cols = df.select_dtypes('number').columns.tolist()
    rows = []
    for _, row in df.iterrows():
        cells = [html.Td(str(row[c]) if c not in num_cols else f'{row[c]:,.2f}',
                         style={'padding': '4px 8px', 'fontSize': '12px', 'color': MUTED,
                                'border': '1px solid rgba(255,255,255,0.08)'})
                 for c in df.columns]
        rows.append(html.Tr(cells))
    header = html.Tr([html.Th(c, style={'padding': '4px 8px', 'fontSize': '11px', 'color': MUTED,
                                         'border': '1px solid rgba(255,255,255,0.08)',
                                         'backgroundColor': 'rgba(255,255,255,0.05)'})
                      for c in df.columns])
    return html.Div(style={'overflowX': 'auto', 'marginBottom': '12px'}, children=[
        html.Table([html.Thead(header), html.Tbody(rows)],
                   style={'borderCollapse': 'collapse', 'width': '100%'}),
    ])


def _fmt_num(x) -> str:
    try:
        if pd.isna(x):
            return '—'
        return f'{int(round(float(x))):,}'
    except (TypeError, ValueError):
        return str(x) if x is not None else '—'


def _render_statement(ticker: str, period_str: str, kind: str) -> html.Div:
    """SimFin statement table focused on the selected period + 3 older."""
    try:
        df = _uv.get_statement(ticker, kind)
    except Exception as exc:
        logger.debug(f'statement load error: {exc}')
        return html.Div('No data.', style={'color': MUTED, 'fontSize': '12px'})
    if df is None or df.empty:
        return html.Div('No data.', style={'color': MUTED, 'fontSize': '12px'})

    variant = 'A' if period_str.endswith('FY') else 'Q'
    all_cols = list(df.columns)
    matching = [c for c in all_cols if c.endswith('FY')] if variant == 'A' \
        else [c for c in all_cols if not c.endswith('FY')]
    if not matching:
        return html.Div('No matching periods.', style={'color': MUTED, 'fontSize': '12px'})

    start = matching.index(period_str) if period_str in matching else 0
    show_cols = matching[start:start + 4]

    sub = df[show_cols].copy()
    for col in show_cols:
        sub[col] = sub[col].apply(_fmt_num)
    sub = sub.reset_index()
    sub.rename(columns={sub.columns[0]: 'Line Item'}, inplace=True)
    return _inspect_html(sub)


# ── Callbacks: Fundamentals ───────────────────────────────────────────

def _apply_filters(records: list[dict], tickers: list | None, market: str | None,
                   rules: list | None, ticker_col: str = 'Ticker',
                   market_col: str = 'Market', rule_col: str = 'Rule') -> list[dict]:
    out = records
    if tickers:
        ticker_set = {t.upper() for t in tickers}
        out = [r for r in out if r.get(ticker_col, '').upper() in ticker_set]
    if market:
        out = [r for r in out if (r.get(market_col) or '').lower() == market.lower()]
    if rules:
        rule_set = set(rules)
        out = [r for r in out if r.get(rule_col) in rule_set]
    return out


@callback(
    Output('dq-sf-results', 'data'),
    Input('dq-sf-run', 'n_clicks'),
    State('dq-sf-variant', 'value'),
    State('dq-sf-include-reviewed', 'value'),
    prevent_initial_call=True,
)
def run_simfin(n: int, variant: str, inc_reviewed: list):
    skip = 'yes' not in (inc_reviewed or [])
    try:
        df = dq.run_simfin(skip_reviewed=skip, variant=variant or 'A')
    except Exception as exc:
        logger.exception('simfin run error')
        return None
    return [] if df.empty else df.to_dict('records')


@callback(
    Output('dq-sf-table-wrap', 'children'),
    Input('dq-sf-results', 'data'),
    Input('dq-sf-filter-ticker', 'value'),
    Input('dq-sf-filter-market', 'value'),
    Input('dq-sf-rules', 'value'),
)
def filter_simfin_table(results, ticker, market, rules):
    if results is None:
        return html.P('Click Run checks to load findings.', style={'color': MUTED, 'padding': '16px 0'})
    filtered = _apply_filters(results, ticker, market, rules)
    if not filtered:
        return html.P('No findings match current filters.', style={'color': MUTED, 'padding': '16px 0'})
    return _make_simfin_table(filtered)


def _violations_summary(violations: list[dict]) -> html.Div:
    header = html.Tr([
        html.Th(c, style={'padding': '4px 10px', 'fontSize': '11px', 'color': MUTED,
                          'border': '1px solid rgba(255,255,255,0.08)',
                          'backgroundColor': 'rgba(255,255,255,0.05)'})
        for c in ('Rule', 'Δ%', 'LHS', 'RHS')
    ])
    rows = []
    for v in violations:
        diff = v.get('rel_diff_pct') or 0
        rows.append(html.Tr([
            html.Td(v['Rule'],
                    title=_SF_RULE_INFO.get(v['Rule'], ''),
                    style={'padding': '3px 10px', 'fontSize': '12px', 'color': MUTED,
                           'border': '1px solid rgba(255,255,255,0.08)',
                           'cursor': 'help', 'fontFamily': 'monospace'}),
            html.Td(f'{diff:+.2f}%',
                    style={'padding': '3px 10px', 'fontSize': '12px',
                           'color': 'rgba(231,76,60,0.9)' if abs(diff) >= 1 else MUTED,
                           'border': '1px solid rgba(255,255,255,0.08)'}),
            html.Td(f'{v.get("LHS_value") or "":,.2f}' if isinstance(v.get('LHS_value'), (int, float)) else '',
                    style={'padding': '3px 10px', 'fontSize': '12px', 'color': MUTED,
                           'border': '1px solid rgba(255,255,255,0.08)'}),
            html.Td(f'{v.get("RHS_value") or "":,.2f}' if isinstance(v.get('RHS_value'), (int, float)) else '',
                    style={'padding': '3px 10px', 'fontSize': '12px', 'color': MUTED,
                           'border': '1px solid rgba(255,255,255,0.08)'}),
        ]))
    return html.Div(style={'overflowX': 'auto', 'marginBottom': '4px'}, children=[
        html.Table([html.Thead(header), html.Tbody(rows)],
                   style={'borderCollapse': 'collapse'}),
    ])


@callback(
    Output('dq-sf-detail', 'style'),
    Output('dq-sf-detail-title', 'children'),
    Output('dq-sf-edgar-msg', 'children'),
    Output('dq-sf-violations-summary', 'children'),
    Output('dq-sf-rule-select', 'options'),
    Output('dq-sf-rule-select', 'value'),
    Output('dq-sf-sel', 'data'),
    Output('dq-sf-note', 'value', allow_duplicate=True),
    Input('dq-sf-table', 'active_cell'),
    Input('dq-sf-close', 'n_clicks'),
    State('dq-sf-table', 'data'),
    State('dq-sf-results', 'data'),
    prevent_initial_call=True,
)
def show_simfin_detail(cell, close_n, table_data, results):
    from dash import ctx
    _hide = {'display': 'none'}
    _show = {'display': 'block'}
    _empty = (_hide, '', '', html.Div(), [], None, None, '')

    if ctx.triggered_id == 'dq-sf-close' or not cell or not table_data:
        return _empty

    filing = table_data[cell['row']]
    ticker, period_str = filing['Ticker'], filing['Period_str']
    violations = [r for r in (results or [])
                  if r['Ticker'] == ticker and r['Period_str'] == period_str]
    if not violations:
        return _empty

    title = f'{ticker} — {period_str} — {len(violations)} rule{"s" if len(violations) != 1 else ""}'
    summary = _violations_summary(violations)
    rule_opts = [
        {'label': f'{v["Rule"]}  Δ{v.get("rel_diff_pct", 0):+.1f}%', 'value': v['Rule']}
        for v in violations
    ]
    sel = {
        'Ticker': ticker,
        'Period_str': period_str,
        'CIK': filing.get('CIK'),
        'Report Date': filing.get('Report Date', ''),
        'Period': filing.get('Period', ''),
    }
    note_lines = [
        f'{v["Rule"]}: SimFin {_fmt_num(v.get("LHS_value"))} vs {_fmt_num(v.get("RHS_value"))}'
        f' ({v.get("rel_diff_pct", 0):+.1f}%)'
        for v in violations
    ]
    note_template = '\n'.join(note_lines) + '\n\nEDGAR: '
    return _show, title, '', summary, rule_opts, violations[0]['Rule'], sel, note_template


@callback(
    Output('dq-sf-inspect-table', 'children'),
    Output('dq-sf-status', 'value'),
    Input('dq-sf-rule-select', 'value'),
    State('dq-sf-sel', 'data'),
    State('dq-sf-results', 'data'),
    prevent_initial_call=True,
)
def update_simfin_inspect(rule, sel, results):
    if not rule or not sel:
        raise PreventUpdate
    violation = next(
        (r for r in (results or [])
         if r['Ticker'] == sel['Ticker'] and r['Period_str'] == sel['Period_str']
         and r['Rule'] == rule),
        None,
    )
    if not violation:
        return html.Div(), 'to_check'
    inspect_div = html.Div()
    try:
        inspect_df = dq.inspect_simfin(
            rule, sel['Ticker'],
            int(violation['Fiscal Year']), str(violation['Fiscal Period']), str(violation['Period']),
        )
        inspect_div = _inspect_html(inspect_df)
    except Exception as exc:
        logger.debug(f'inspect error: {exc}')
    return inspect_div, dq.auto_suggest_status(violation.get('rel_diff', 0.0))


@callback(
    Output('dq-sf-statement-wrap', 'children'),
    Input('dq-sf-sel', 'data'),
    Input('dq-sf-stmt-kind', 'value'),
    prevent_initial_call=True,
)
def update_sf_statement(sel, kind):
    if not sel:
        raise PreventUpdate
    return _render_statement(sel['Ticker'], sel['Period_str'], kind or 'income')


@callback(
    Output('dq-sf-edgar-url', 'data'),
    Output('dq-sf-edgar-msg', 'children', allow_duplicate=True),
    Input('dq-sf-edgar-btn', 'n_clicks'),
    State('dq-sf-sel', 'data'),
    prevent_initial_call=True,
)
def fetch_edgar(n, sel):
    if not sel:
        raise PreventUpdate
    cik = sel.get('CIK')
    if not cik or cik != cik:  # None or NaN (NaN != NaN)
        return None, 'No CIK for this ticker'
    try:
        rd = sel.get('Report Date')
        rd_str = str(rd)[:10] if rd else None
        url = dq.fetch_edgar_url(int(cik), rd_str, str(sel.get('Period', '')))
    except Exception as exc:
        logger.warning(f'EDGAR fetch error: {exc}')
        return None, f'Error: {exc}'
    if not url:
        return None, 'No matching filing on SEC EDGAR'
    return url, ''


dash.clientside_callback(
    "function(url) { if (url) { window.open(url, '_blank', 'noopener'); } return null; }",
    Output('dq-sf-edgar-url', 'data', allow_duplicate=True),
    Input('dq-sf-edgar-url', 'data'),
    prevent_initial_call=True,
)


@callback(
    Output('dq-sf-results', 'data', allow_duplicate=True),
    Output('dq-sf-note', 'value'),
    Output('dq-sf-msg', 'children'),
    Input('dq-sf-submit', 'n_clicks'),
    State('dq-sf-sel', 'data'),
    State('dq-sf-rule-select', 'value'),
    State('dq-sf-status', 'value'),
    State('dq-sf-note', 'value'),
    State('dq-sf-results', 'data'),
    prevent_initial_call=True,
)
def submit_simfin_review(n, sel, rule, status, note, results):
    if not sel or not rule or not status:
        raise PreventUpdate
    try:
        dq.add_simfin_review(sel['Ticker'], sel['Period_str'], rule, status, note or '')
    except Exception as exc:
        return results, note, f'Error: {exc}'
    updated = [r for r in (results or [])
               if not (r['Ticker'] == sel['Ticker']
                       and r['Period_str'] == sel['Period_str']
                       and r['Rule'] == rule)]
    return updated, '', f'{rule} → {status}'


@callback(
    Output('dq-flag-msg', 'children'),
    Output('dq-flag-ticker', 'value'),
    Output('dq-flag-period', 'value'),
    Output('dq-flag-subject', 'value'),
    Output('dq-flag-note', 'value'),
    Input('dq-flag-submit', 'n_clicks'),
    State('dq-flag-ticker', 'value'),
    State('dq-flag-period', 'value'),
    State('dq-flag-subject', 'value'),
    State('dq-flag-status', 'value'),
    State('dq-flag-note', 'value'),
    prevent_initial_call=True,
)
def add_flag(n, ticker, period, subject, status, note):
    if not ticker or not period:
        return 'Ticker and period are required.', ticker, period, subject, note
    try:
        dq.add_manual_flag(ticker.strip().upper(), period.strip(), subject or '', status or 'to_check', note or '')
    except Exception as exc:
        return f'Error: {exc}', ticker, period, subject, note
    return 'Flag saved.', '', '', '', ''


# ── Callbacks: Prices ─────────────────────────────────────────────────

@callback(
    Output('dq-stq-results', 'data'),
    Input('dq-stq-run', 'n_clicks'),
    State('dq-stq-include-reviewed', 'value'),
    prevent_initial_call=True,
)
def run_stooq(n: int, inc_reviewed: list):
    skip = 'yes' not in (inc_reviewed or [])
    try:
        df = dq.run_stooq(skip_reviewed=skip)
    except Exception as exc:
        logger.exception('stooq run error')
        return None
    return [] if df.empty else df.to_dict('records')


@callback(
    Output('dq-stq-table-wrap', 'children'),
    Input('dq-stq-results', 'data'),
    Input('dq-stq-filter-ticker', 'value'),
    Input('dq-stq-filter-market', 'value'),
    Input('dq-stq-rules', 'value'),
)
def filter_stooq_table(results, ticker, market, rules):
    if results is None:
        return html.P('Click Run checks to load findings.', style={'color': MUTED, 'padding': '16px 0'})
    filtered = _apply_filters(results, ticker, market, rules)
    if not filtered:
        return html.P('No findings match current filters.', style={'color': MUTED, 'padding': '16px 0'})
    return _make_stooq_table(filtered)


@callback(
    Output('dq-stq-detail', 'style'),
    Output('dq-stq-detail-title', 'children'),
    Output('dq-stq-yahoo-link', 'href'),
    Output('dq-stq-chart', 'figure'),
    Output('dq-stq-bars-table', 'children'),
    Output('dq-stq-status', 'value'),
    Output('dq-stq-sel', 'data'),
    Input('dq-stq-table', 'active_cell'),
    Input('dq-stq-close', 'n_clicks'),
    State('dq-stq-results', 'data'),
    prevent_initial_call=True,
)
def show_stooq_detail(cell, close_n, results):
    from dash import ctx
    _hide = {'display': 'none'}
    _show = {'display': 'block'}

    if ctx.triggered_id == 'dq-stq-close' or not cell or not results:
        return _hide, '', '#', _empty_figure('Select a row'), html.Div(), 'to_check', None

    row = results[cell['row']]
    title = f'{row["Ticker"]} — {row["Period_str"]} — {row["Rule"]}'
    yahoo = dq.yahoo_url(row['Ticker'], row.get('Period_str'))

    sample_dates = row.get('sample_dates') or []
    fig = _empty_figure('No price data')
    try:
        raw_fig = dq.stooq_figure(row['Rule'], row['Ticker'], row['Period_str'], sample_dates)
        if raw_fig is not None:
            fig = raw_fig
    except Exception as exc:
        logger.debug(f'stooq inspect error: {exc}')

    bars_div = html.Div()
    try:
        bars = dq.flagged_bars(row['Ticker'], sample_dates)
        if not bars.empty:
            bars_div = _inspect_html(bars)
    except Exception as exc:
        logger.debug(f'flagged_bars error: {exc}')

    return _show, title, yahoo, fig, bars_div, 'to_check', row


@callback(
    Output('dq-stq-results', 'data', allow_duplicate=True),
    Output('dq-stq-note', 'value'),
    Output('dq-stq-msg', 'children'),
    Output('dq-stq-detail', 'style', allow_duplicate=True),
    Input('dq-stq-submit', 'n_clicks'),
    State('dq-stq-sel', 'data'),
    State('dq-stq-status', 'value'),
    State('dq-stq-note', 'value'),
    State('dq-stq-results', 'data'),
    prevent_initial_call=True,
)
def submit_stooq_review(n, sel, status, note, results):
    if not sel or not status:
        raise PreventUpdate
    try:
        dq.add_stooq_review(sel['Ticker'], sel['Period_str'], sel['Rule'], status, note or '')
    except Exception as exc:
        return results, note, f'Error: {exc}', dash.no_update
    updated = [r for r in (results or [])
               if not (r['Ticker'] == sel['Ticker']
                       and r['Period_str'] == sel['Period_str']
                       and r['Rule'] == sel['Rule'])]
    return updated, '', f'Saved as {status}', {'display': 'none'}


# ── Callbacks: Dashboard ──────────────────────────────────────────────

@callback(
    Output('dq-dashboard', 'children'),
    Input('dq-tabs', 'value'),
    State('dq-sf-results', 'data'),
    State('dq-stq-results', 'data'),
)
def update_dashboard(tab, sf_results, stq_results):
    if tab != 'dashboard':
        raise PreventUpdate

    def _section(label: str, results: list | None, reviews_fn) -> html.Div:
        n_unreviewed = len(results) if results else 0
        try:
            rev = reviews_fn()
            n_ok = int((rev['status'] == 'ok').sum()) if not rev.empty else 0
            n_err = int((rev['status'] == 'data_error').sum()) if not rev.empty else 0
            n_chk = int((rev['status'] == 'to_check').sum()) if not rev.empty else 0
            n_total = len(rev) + n_unreviewed
        except Exception:
            n_ok = n_err = n_chk = n_total = 0

        chips = html.Div(style={'display': 'flex', 'gap': '16px', 'flexWrap': 'wrap', 'marginBottom': '16px'},
                         children=[
                             _chip('total', n_total),
                             _chip('unreviewed', n_unreviewed, '#f39c12'),
                             _chip('ok', n_ok, '#2ecc71'),
                             _chip('data errors', n_err, '#e74c3c'),
                             _chip('to check', n_chk, '#f39c12'),
                         ])

        bar_fig = _empty_figure('No data — run checks first')
        if results:
            try:
                df = pd.DataFrame(results)
                counts = df.groupby('Rule').size().reset_index(name='n').sort_values('n', ascending=True)
                bar_trace = go.Bar(x=counts['n'], y=counts['Rule'], orientation='h',
                                   marker_color=ACCENT)
                bar_fig = go.Figure(data=[bar_trace], layout=_chart_layout(
                    height=max(200, len(counts) * 28 + 60),
                    margin=dict(l=0, r=0, t=8, b=0),
                    xaxis=dict(title='Unreviewed findings'),
                ))
            except Exception:
                pass

        return html.Div(style={'marginBottom': '32px'}, children=[
            html.H4(label, style={'color': MUTED, 'marginBottom': '12px'}),
            chips,
            dcc.Graph(figure=bar_fig, config={'displayModeBar': False}, style={'minHeight': '160px'}),
        ])

    return html.Div([
        _section('Fundamentals (SimFin)', sf_results, dq.get_all_simfin_reviews),
        _section('Prices (Stooq)', stq_results, dq.get_all_stooq_reviews),
    ])

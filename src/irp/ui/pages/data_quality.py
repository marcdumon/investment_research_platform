"""Data quality review page: fundamental + price anomaly triage."""
import logging
import re

import dash
import pandas as pd
import plotly.graph_objects as go
from dash import ALL, Input, Output, State, callback, dcc, html
from dash.exceptions import PreventUpdate

import irp.checks.simfin_rules as _simfin_rules
import irp.checks.stooq_rules as _stooq_rules
from irp.ui.charts import _base_chart_layout as _chart_layout, _empty_figure
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
_SF_RULE_LABELS: dict[str, tuple[str, str]] = {
    r.name: (r.lhs, r.rhs)
    for r in _simfin_rules.REGISTRY
}
_STQ_RULE_INFO: dict[str, str] = {
    r.name: r.description
    for r in _stooq_rules.REGISTRY
}

# Dynamic options loaded once at startup
def _market_opts(markets: list[str]) -> list[dict]:
    return [{'label': 'All', 'value': ''}] + [{'label': m, 'value': m} for m in sorted(markets)]

_companies = _uv._get_companies()
_universe = _uv._get_universe()
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
        # EDGAR annotation table
        html.Div(style={'marginTop': '16px', 'borderTop': '1px solid rgba(255,255,255,0.08)',
                        'paddingTop': '12px'}, children=[
            html.Div(style={'display': 'flex', 'alignItems': 'center', 'gap': '12px',
                            'marginBottom': '8px', 'flexWrap': 'wrap'}, children=[
                html.Span('EDGAR Annotation', style={'color': MUTED, 'fontSize': '13px',
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
                dcc.RadioItems(
                    id='dq-sf-unit-radio',
                    options=[
                        {'label': 'Auto', 'value': 'auto'},
                        {'label': 'B',    'value': 'B'},
                        {'label': 'M',    'value': 'M'},
                        {'label': 'K',    'value': 'K'},
                        {'label': 'Raw',  'value': 'raw'},
                    ],
                    value='auto', inline=True, labelClassName='check-item',
                ),
            ]),
            dcc.Loading(type='circle', color=ACCENT,
                        children=html.Div(id='dq-sf-statement-wrap')),
            # Review + save row (combined)
            html.Div(style={'display': 'flex', 'alignItems': 'center', 'gap': '12px',
                            'marginTop': '10px', 'flexWrap': 'wrap'}, children=[
                dcc.RadioItems(
                    id='dq-sf-status',
                    options=_STATUS_OPTIONS,
                    value='to_check',
                    labelClassName='check-item',
                    style={'display': 'flex', 'gap': '12px'},
                ),
                dcc.Textarea(
                    id='dq-sf-note',
                    placeholder='Note (optional)',
                    style={'flex': '1', 'minWidth': '200px', 'minHeight': '36px',
                           'fontSize': '12px', 'backgroundColor': 'rgba(255,255,255,0.04)',
                           'color': MUTED, 'border': '1px solid rgba(255,255,255,0.12)',
                           'borderRadius': '4px', 'padding': '4px 6px'},
                ),
                html.Button('Save', id='dq-sf-annotation-save',
                            className='run-btn', n_clicks=0),
                html.Button('Show hidden', id='dq-sf-annotation-show-all',
                            className='run-btn', n_clicks=0,
                            style={'fontSize': '11px', 'opacity': '0.7'}),
                html.Span(id='dq-sf-annotation-msg',
                          style={'color': ACCENT, 'fontSize': '12px'}),
            ]),
        ]),
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
    dcc.Store(id='dq-sf-corrections'),
    dcc.Store(id='dq-manual-corrections'),
    dcc.Store(id='dq-manual-sel'),
    dcc.Store(id='dq-manual-periods-data'),
    html.H2('Data Quality', className='page-title'),
    html.P('Rule-based and manual data quality checks for fundamentals and prices.',
           className='page-subtitle'),
    dcc.Tabs(
        id='dq-tabs',
        value='fundamentals',
        className='page-tabs',
        children=[
            dcc.Tab(label='Rule Check', value='fundamentals', className='page-tab',
                    selected_className='page-tab-selected', children=[_fund_tab]),
            dcc.Tab(label='Manual Check', value='edgar-corrections', className='page-tab',
                    selected_className='page-tab-selected',
                    children=[html.Div(id='dq-edgar-corrections-tab',
                                       style={'padding': '16px 0'})]),
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
            filings[key]['_max_diff'], abs(r.get('diff_M') or 0)
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
            'max_diff_M': round(f['_max_diff'], 1),
            'CIK': f['CIK'],
            'Report Date': f['Report Date'],
            'Period': f['Period'],
        })

    cols = [
        {'name': 'Ticker',    'id': 'Ticker'},
        {'name': 'Period',    'id': 'Period_str'},
        {'name': 'Max Δ (M)', 'id': 'max_diff_M', 'type': 'numeric',
         'format': Fmt.Format(precision=1, scheme=Fmt.Scheme.fixed)},
        {'name': 'Rules',     'id': 'rules_str'},
        {'name': '',          'id': '_sf_spacer'},
    ]
    for r in rows:
        r['_sf_spacer'] = ''
    _nw = {'whiteSpace': 'nowrap', 'overflow': 'hidden', 'textOverflow': 'ellipsis'}
    return dash.dash_table.DataTable(
        id='dq-sf-table',
        data=rows,
        columns=cols,
        hidden_columns=['CIK', 'Report Date', 'Period', 'Company Name', 'Market'],
        row_selectable=False,
        active_cell=None,
        page_size=25,
        sort_action='native',
        filter_action='native',
        style_table={'overflowX': 'auto', 'width': '100%'},
        style_cell_conditional=[
            {'if': {'column_id': 'Ticker'},      'width': '70px',  **_nw},
            {'if': {'column_id': 'Period_str'},  'width': '80px',  **_nw},
            {'if': {'column_id': 'max_diff_M'},  'width': '80px',  **_nw},
            {'if': {'column_id': 'rules_str'},   'width': '300px', **_nw},
        ],
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
        {'name': '',       'id': '_stq_spacer'},
    ]
    for d in display:
        d['_stq_spacer'] = ''
    _nw = {'whiteSpace': 'nowrap', 'overflow': 'hidden', 'textOverflow': 'ellipsis'}
    return dash.dash_table.DataTable(
        id='dq-stq-table',
        data=display,
        columns=cols,
        row_selectable=False,
        active_cell=None,
        page_size=25,
        sort_action='native',
        filter_action='native',
        style_table={'overflowX': 'auto', 'width': '100%'},
        style_cell_conditional=[
            {'if': {'column_id': 'Rule'},        'width': '160px', **_nw},
            {'if': {'column_id': 'Ticker'},      'width': '70px',  **_nw},
            {'if': {'column_id': 'Market'},      'width': '70px',  **_nw},
            {'if': {'column_id': 'Period_str'},  'width': '60px',  **_nw},
            {'if': {'column_id': 'count'},       'width': '60px',  **_nw},
            {'if': {'column_id': '_dates_str'},  'width': '280px', **_nw},
        ],
        **_TABLE_STYLE,
    )


def _inspect_html(df: pd.DataFrame) -> html.Div:
    if df.empty:
        return html.Div()
    num_cols = df.select_dtypes('number').columns.tolist()
    label_col = next((c for c in df.columns if c not in num_cols), None)
    _cell = {'padding': '4px 8px', 'fontSize': '12px', 'color': MUTED,
             'border': '1px solid rgba(255,255,255,0.08)'}
    _hdr = {'padding': '4px 8px', 'fontSize': '11px', 'color': MUTED,
            'border': '1px solid rgba(255,255,255,0.08)',
            'backgroundColor': 'rgba(255,255,255,0.05)'}
    rows = []
    for _, row in df.iterrows():
        cells = []
        for c in df.columns:
            align = 'right' if (c not in num_cols and c == label_col) or c in num_cols else 'left'
            val = str(row[c]) if c not in num_cols else f'{row[c]:,.2f}'
            cells.append(html.Td(val, style={**_cell, 'textAlign': align,
                                             'whiteSpace': 'nowrap' if c == label_col else 'normal'}))
        rows.append(html.Tr(cells))
    header = html.Tr([
        html.Th(c, style={**_hdr,
                          'textAlign': 'right' if (c not in num_cols and c == label_col) or c in num_cols else 'left'})
        for c in df.columns
    ])
    return html.Div(style={'overflowX': 'auto', 'marginBottom': '12px'}, children=[
        html.Table([html.Thead(header), html.Tbody(rows)],
                   style={'borderCollapse': 'collapse'}),
    ])


def _fmt_num(x) -> str:
    try:
        if pd.isna(x):
            return '—'
        return f'{int(round(float(x))):,}'
    except (TypeError, ValueError):
        return str(x) if x is not None else '—'


def _unit_for_values(vals: list[float]) -> tuple[float, str]:
    """Pick a consistent scale (billions/millions/thousands/units) from a list of values."""
    if not vals:
        return 1.0, ''
    # Use median of values > 1000 to avoid EPS/per-share items skewing scale
    large = sorted(abs(v) for v in vals if v is not None and abs(v) > 1000)
    if not large:
        return 1.0, ''
    med = large[len(large) // 2]
    if med >= 1e9:
        return 1e9, 'in billions'
    if med >= 1e6:
        return 1e6, 'in millions'
    if med >= 1e3:
        return 1e3, 'in thousands'
    return 1.0, ''


def _fmt_scaled(v: float, unit: float) -> str:
    """Format v / unit as a plain integer with commas. No decimals."""
    scaled = int(round(v / unit))
    return f'{scaled:,}'


def _fmt_violation_line(label: str, value) -> str:
    """'Net Income (income)' → '- Income: Net Income: -37,000,000'"""
    m = re.match(r'^(.*?)\s*\((\w+)\)\s*$', label)
    if m:
        return f'- {m.group(2).capitalize()}: {m.group(1).strip()}: {_fmt_num(value)}'
    return f'- {label}: {_fmt_num(value)}'


def _render_statement(ticker: str, period_str: str, kind: str) -> html.Div:
    """SimFin statement table focused on the selected period + 3 older."""
    try:
        df = _uv._get_statement(ticker, kind)
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


_UNIT_MAP = {'B': 1e9, 'M': 1e6, 'K': 1e3, 'raw': 1.0}


def _render_annotation_table(
    ticker: str,
    period_str: str,
    kind: str,
    existing: dict[str, float],
    input_type: str = 'edgar-input',
    existing_notes: dict[str, str] | None = None,
    unit_choice: str = 'auto',
) -> tuple[html.Div, dict]:
    """Comparison table: Line Item | SimFin value | EDGAR input | Note | flag.

    Returns (html.Div, store_payload) where store_payload holds row order + simfin values.
    input_type distinguishes violation-panel inputs ('edgar-input') from manual ones ('edgar-input-manual').
    """
    try:
        df = _uv._get_statement(ticker, kind)
    except Exception as exc:
        logger.debug(f'annotation table load error: {exc}')
        return html.Div('No data.', style={'color': MUTED, 'fontSize': '12px'}), {}
    if df is None or df.empty:
        return html.Div('No data.', style={'color': MUTED, 'fontSize': '12px'}), {}

    variant = 'A' if period_str.endswith('FY') else 'Q'
    matching = [c for c in df.columns if c.endswith('FY')] if variant == 'A' \
        else [c for c in df.columns if not c.endswith('FY')]
    if not matching or period_str not in matching:
        return html.Div('Period not found.', style={'color': MUTED, 'fontSize': '12px'}), {}

    items = list(df.index)
    sf_col = df[period_str]

    store_payload: dict = {'_items': []}
    for item in items:
        v = sf_col.get(item)
        try:
            parsed = float(v) if v is not None and str(v) not in ('nan', 'None') else None
        except (TypeError, ValueError):
            parsed = None
        store_payload[item] = parsed
        if parsed is not None:
            store_payload['_items'].append(item)

    note_type = input_type.replace('edgar-input', 'edgar-note')
    hide_type = input_type.replace('edgar-input', 'hide-row-btn')
    row_type  = input_type.replace('edgar-input', 'annotation-row')
    existing_notes = existing_notes or {}

    # Determine consistent scale
    sf_vals = [v for v in store_payload.values() if isinstance(v, float)]
    if unit_choice in _UNIT_MAP:
        unit = _UNIT_MAP[unit_choice]
        _labels = {'B': 'in billions', 'M': 'in millions', 'K': 'in thousands', 'raw': ''}
        unit_label = _labels[unit_choice]
    else:
        unit, unit_label = _unit_for_values(sf_vals)
    store_payload['_unit'] = unit

    _TH_A  = {**_TH, 'width': '240px', 'textAlign': 'right'}
    _TH_V  = {**_TH, 'width': '110px', 'textAlign': 'right'}
    _TH_I  = {**_TH, 'width': '120px'}
    _TH_N  = {**_TH, 'width': '200px'}
    _TH_F  = {**_TH, 'width': '32px', 'textAlign': 'center'}
    _TH_H  = {**_TH, 'width': '28px', 'textAlign': 'center'}
    _TH_SP = {**_TH, 'border': 'none', 'backgroundColor': 'transparent'}

    header = html.Tr([
        html.Th('Line Item', style=_TH_A),
        html.Th('SimFin',    style=_TH_V),
        html.Th('EDGAR',     style=_TH_I),
        html.Th('Note',      style=_TH_N),
        html.Th('',          style=_TH_F),
        html.Th('',          style=_TH_H),
        html.Th('',          style=_TH_SP),
    ])
    rows = []
    _btn_style = {
        'background': 'none', 'border': '1px solid rgba(255,255,255,0.15)',
        'color': 'rgba(100,200,100,0.7)', 'cursor': 'pointer',
        'borderRadius': '3px', 'fontSize': '11px', 'padding': '1px 5px',
        'lineHeight': '1.2',
    }
    for i, item in enumerate(items):
        sf_raw = store_payload.get(item)
        if sf_raw is None:
            continue
        sf_fmt = _fmt_scaled(sf_raw, unit)
        edgar_val = existing.get(item)
        prefilled = _fmt_scaled(edgar_val, unit) if edgar_val is not None else ''
        note_val = existing_notes.get(item, '')
        try:
            differs = edgar_val is not None and (
                sf_raw is None or abs(float(edgar_val) - float(sf_raw)) > 0.5
            )
        except (TypeError, ValueError):
            differs = False
        row_style = {'backgroundColor': 'rgba(88,166,255,0.08)'} if differs else {}
        _inp_style = {
            'width': '100%', 'fontSize': '12px',
            'backgroundColor': 'rgba(255,255,255,0.04)', 'color': MUTED,
            'border': '1px solid rgba(255,255,255,0.12)',
            'borderRadius': '3px', 'padding': '2px 6px',
        }
        rows.append(html.Tr(id={'type': row_type, 'index': i}, style=row_style, children=[
            html.Td(item, style={**_TD, 'whiteSpace': 'nowrap', 'textAlign': 'right'}),
            html.Td(sf_fmt, style={**_TD, 'textAlign': 'right', 'fontFamily': 'monospace'}),
            html.Td(dcc.Input(
                id={'type': input_type, 'index': i},
                value=prefilled,
                placeholder='EDGAR value',
                debounce=False,
                type='text',
                style=_inp_style,
            ), style={**_TD, 'padding': '2px 6px'}),
            html.Td(dcc.Input(
                id={'type': note_type, 'index': i},
                value=note_val,
                placeholder='note',
                debounce=False,
                type='text',
                style=_inp_style,
            ), style={**_TD, 'padding': '2px 6px'}),
            html.Td('●' if differs else '',
                    style={**_TD, 'textAlign': 'center',
                           'color': ACCENT if differs else 'transparent'}),
            html.Td(html.Button('✓', id={'type': hide_type, 'index': i},
                                n_clicks=0, style=_btn_style,
                                title='Mark as verified and hide'),
                    style={**_TD, 'padding': '2px 4px', 'textAlign': 'center'}),
            html.Td('', style={'border': 'none'}),
        ]))

    unit_note = html.Div(
        f'({unit_label})' if unit_label else '',
        style={'fontSize': '11px', 'color': MUTED, 'textAlign': 'right',
               'marginBottom': '4px', 'fontStyle': 'italic'},
    )
    table_div = html.Div(style={'overflowX': 'auto'}, children=[
        unit_note,
        html.Table([html.Thead(header), html.Tbody(rows)],
                   style={'borderCollapse': 'collapse', 'width': '100%'}),
    ])
    return table_div, store_payload


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
        df = dq._run_simfin(skip_reviewed=skip, variant=variant or 'A')
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
        for c in ('Rule', 'Δ (M)', 'LHS', 'RHS')
    ])
    rows = []
    for v in violations:
        diff = v.get('diff_M') or 0
        rows.append(html.Tr([
            html.Td(v['Rule'],
                    title=_SF_RULE_INFO.get(v['Rule'], ''),
                    style={'padding': '3px 10px', 'fontSize': '12px', 'color': MUTED,
                           'border': '1px solid rgba(255,255,255,0.08)',
                           'cursor': 'help', 'fontFamily': 'monospace'}),
            html.Td(f'{diff:+.1f}M',
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
    _empty = (_hide, '', '', html.Div(), [], None, None)

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
        {'label': f'{v["Rule"]}  Δ{v.get("diff_M", 0):+.1f}M', 'value': v['Rule']}
        for v in violations
    ]
    sel = {
        'Ticker': ticker,
        'Period_str': period_str,
        'CIK': filing.get('CIK'),
        'Report Date': filing.get('Report Date', ''),
        'Period': filing.get('Period', ''),
    }
    return _show, title, '', summary, rule_opts, violations[0]['Rule'], sel


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
        inspect_df = dq._inspect_simfin(
            rule, sel['Ticker'],
            int(violation['Fiscal Year']), str(violation['Fiscal Period']), str(violation['Period']),
        )
        inspect_div = _inspect_html(inspect_df)
    except Exception as exc:
        logger.debug(f'inspect error: {exc}')
    return inspect_div, dq._auto_suggest_status(violation.get('rel_diff', 0.0))


@callback(
    Output('dq-sf-statement-wrap', 'children'),
    Output('dq-sf-corrections', 'data'),
    Input('dq-sf-sel', 'data'),
    Input('dq-sf-stmt-kind', 'value'),
    Input('dq-sf-unit-radio', 'value'),
    prevent_initial_call=True,
)
def update_sf_annotation_table(sel, kind, unit_choice):
    if not sel:
        raise PreventUpdate
    ticker = sel['Ticker']
    period = sel['Period_str']
    stmt_kind = kind or 'income'
    existing = dq._load_edgar_corrections(ticker, period, stmt_kind)
    existing_notes = dq._load_edgar_correction_notes(ticker, period, stmt_kind)
    table_div, store_payload = _render_annotation_table(
        ticker, period, stmt_kind, existing, existing_notes=existing_notes,
        unit_choice=unit_choice or 'auto',
    )
    return table_div, store_payload


@callback(
    Output('dq-sf-annotation-msg', 'children'),
    Input('dq-sf-annotation-save', 'n_clicks'),
    State('dq-sf-sel', 'data'),
    State('dq-sf-stmt-kind', 'value'),
    State('dq-sf-rule-select', 'value'),
    State('dq-sf-status', 'value'),
    State('dq-sf-note', 'value'),
    State('dq-sf-corrections', 'data'),
    State({'type': 'edgar-input', 'index': ALL}, 'value'),
    State({'type': 'edgar-note',  'index': ALL}, 'value'),
    prevent_initial_call=True,
)
def save_edgar_annotations(n, sel, kind, rule, status, review_note,
                           corrections_store, input_values, note_values):
    if not sel:
        raise PreventUpdate
    ticker, period = sel['Ticker'], sel['Period_str']
    stmt_kind = kind or 'income'
    rule = rule or ''
    status = status or 'to_check'
    unit = float((corrections_store or {}).get('_unit') or 1)
    items = (corrections_store or {}).get('_items', [])
    edgar_values: dict[str, float] = {}
    simfin_values: dict[str, float] = {}
    line_notes: dict[str, str] = {}
    for item, raw, note_raw in zip(items, input_values or [], note_values or []):
        if raw and str(raw).strip():
            try:
                edgar_values[item] = float(str(raw).replace(',', '').strip()) * unit
            except ValueError:
                continue
        sf = (corrections_store or {}).get(item)
        if sf is not None:
            try:
                simfin_values[item] = float(sf)
            except (TypeError, ValueError):
                pass
        if note_raw and str(note_raw).strip():
            line_notes[item] = str(note_raw).strip()
    try:
        dq._save_statement(ticker, period, stmt_kind, rule, status,
                          review_note or '', edgar_values,
                          simfin_values or None, line_notes or None)
    except Exception as exc:
        logger.warning(f'annotation save error: {exc}')
        return f'Error: {exc}'
    n_corr = len(edgar_values)
    return f'Saved ({status}{", " + str(n_corr) + " correction(s)" if n_corr else ""}).'


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
        url = dq._fetch_edgar_url(int(cik), rd_str, str(sel.get('Period', '')))
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

_HIDE_ROW_JS = """
function(hideClicks, showAllClicks, currentStyles) {
    var ctx = window.dash_clientside.callback_context;
    if (!ctx.triggered || ctx.triggered.length === 0)
        return window.dash_clientside.no_update;
    var prop_id = ctx.triggered[0].prop_id;
    if (!currentStyles || currentStyles.length === 0)
        return window.dash_clientside.no_update;
    if (prop_id.includes('show-all')) {
        return currentStyles.map(function() { return {}; });
    }
    var match = prop_id.match(/"index":(\\d+)/);
    if (!match) return window.dash_clientside.no_update;
    var idx = parseInt(match[1]);
    var pos = ctx.inputs_list[0].findIndex(function(inp) { return inp.id.index === idx; });
    if (pos < 0) return window.dash_clientside.no_update;
    return currentStyles.map(function(s, i) {
        return i === pos ? {display: 'none'} : (s || {});
    });
}
"""

dash.clientside_callback(
    _HIDE_ROW_JS,
    Output({'type': 'annotation-row', 'index': ALL}, 'style'),
    Input({'type': 'hide-row-btn', 'index': ALL}, 'n_clicks'),
    Input('dq-sf-annotation-show-all', 'n_clicks'),
    State({'type': 'annotation-row', 'index': ALL}, 'style'),
    prevent_initial_call=True,
)

dash.clientside_callback(
    _HIDE_ROW_JS,
    Output({'type': 'annotation-row-manual', 'index': ALL}, 'style'),
    Input({'type': 'hide-row-btn-manual', 'index': ALL}, 'n_clicks'),
    Input('dq-manual-annotation-show-all', 'n_clicks'),
    State({'type': 'annotation-row-manual', 'index': ALL}, 'style'),
    prevent_initial_call=True,
)




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
        dq._add_manual_flag(ticker.strip().upper(), period.strip(), subject or '', status or 'to_check', note or '')
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
        df = dq._run_stooq(skip_reviewed=skip)
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
    yahoo = dq._yahoo_url(row['Ticker'], row.get('Period_str'))

    sample_dates = row.get('sample_dates') or []
    fig = _empty_figure('No price data')
    try:
        raw_fig = dq._stooq_figure(row['Rule'], row['Ticker'], row['Period_str'], sample_dates)
        if raw_fig is not None:
            fig = raw_fig
    except Exception as exc:
        logger.debug(f'stooq inspect error: {exc}')

    bars_div = html.Div()
    try:
        bars = dq._flagged_bars(row['Ticker'], sample_dates)
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
        dq._add_stooq_review(sel['Ticker'], sel['Period_str'], sel['Rule'], status, note or '')
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
        _section('Fundamentals (SimFin)', sf_results, dq._get_all_simfin_reviews),
        _section('Prices (Stooq)', stq_results, dq._get_all_stooq_reviews),
    ])


# ── EDGAR Corrections tab ──────────────────────────────────────────────

def _manual_annotation_section(ticker_options: list[dict] | None = None) -> html.Div:
    """Manual check page content: controls + two-column filing list / annotation table."""
    _ctrl_sep = {'width': '1px', 'height': '20px', 'backgroundColor': 'rgba(255,255,255,0.12)',
                 'flexShrink': '0'}
    _label = {'fontSize': '10px', 'color': MUTED, 'textTransform': 'uppercase',
              'letterSpacing': '0.06em', 'marginBottom': '4px'}
    return html.Div([
        # ── Controls bar ──────────────────────────────────────────────────
        html.Div(style={
            'display': 'flex', 'alignItems': 'center', 'gap': '16px',
            'flexWrap': 'wrap', 'padding': '10px 14px',
            'backgroundColor': 'rgba(255,255,255,0.03)',
            'borderRadius': '6px', 'border': '1px solid rgba(255,255,255,0.07)',
            'marginBottom': '16px',
        }, children=[
            html.Div([
                html.Div('Ticker', style=_label),
                dcc.Dropdown(
                    id='dq-manual-ticker',
                    options=ticker_options or [],
                    placeholder='Search…',
                    searchable=True,
                    clearable=True,
                    className='filter-dropdown',
                    style={'minWidth': '200px', 'fontSize': '12px'},
                ),
            ]),
            html.Div(style=_ctrl_sep),
            html.Div([
                html.Div('Variant', style=_label),
                dcc.RadioItems(
                    id='dq-manual-variant',
                    options=[{'label': 'Annual', 'value': 'A'},
                             {'label': 'Quarterly', 'value': 'Q'}],
                    value='A', inline=True, labelClassName='check-item',
                ),
            ]),
            html.Div(style=_ctrl_sep),
            html.Div([
                html.Div('Statement', style=_label),
                dcc.RadioItems(
                    id='dq-manual-stmt-kind',
                    options=[
                        {'label': 'Income', 'value': 'income'},
                        {'label': 'Balance', 'value': 'balance'},
                        {'label': 'Cash Flow', 'value': 'cashflow'},
                    ],
                    value='income', inline=True, labelClassName='check-item',
                ),
            ]),
        ]),

        # ── Two-column body ───────────────────────────────────────────────
        html.Div(style={'display': 'flex', 'gap': '16px', 'alignItems': 'flex-start'}, children=[

            # Left: filing list
            html.Div(style={
                'width': '160px', 'flexShrink': '0',
                'border': '1px solid rgba(255,255,255,0.07)',
                'borderRadius': '6px', 'overflow': 'hidden',
            }, children=[
                html.Div('Filings', style={
                    'padding': '8px 12px', 'fontSize': '10px', 'fontWeight': '600',
                    'textTransform': 'uppercase', 'letterSpacing': '0.06em', 'color': MUTED,
                    'backgroundColor': 'rgba(255,255,255,0.04)',
                    'borderBottom': '1px solid rgba(255,255,255,0.07)',
                }),
                html.Div(id='dq-manual-period-strip',
                         style={'display': 'flex', 'flexDirection': 'column',
                                'overflowY': 'auto', 'maxHeight': '520px'}),
            ]),

            # Right: annotation area
            html.Div(style={'flex': '1', 'minWidth': '0'}, children=[
                # Header: context + EDGAR link (populated by callback)
                html.Div(id='dq-manual-edgar-wrap'),
                # Annotation table
                dcc.Loading(type='circle', color=ACCENT,
                            children=html.Div(id='dq-manual-table-wrap')),
                # Save row
                html.Div(style={'display': 'flex', 'gap': '10px', 'marginTop': '10px',
                                'alignItems': 'center'}, children=[
                    dcc.RadioItems(
                        id='dq-manual-unit-radio',
                        options=[
                            {'label': 'Auto', 'value': 'auto'},
                            {'label': 'B',    'value': 'B'},
                            {'label': 'M',    'value': 'M'},
                            {'label': 'K',    'value': 'K'},
                            {'label': 'Raw',  'value': 'raw'},
                        ],
                        value='auto', inline=True, labelClassName='check-item',
                    ),
                    html.Button('Save annotations', id='dq-manual-save',
                                className='run-btn', n_clicks=0),
                    html.Button('Show hidden', id='dq-manual-annotation-show-all',
                                className='run-btn', n_clicks=0,
                                style={'fontSize': '11px', 'opacity': '0.7'}),
                    html.Span(id='dq-manual-msg',
                              style={'color': ACCENT, 'fontSize': '12px'}),
                ]),
            ]),
        ]),
    ])


def _period_sort_key(p: str) -> tuple:
    try:
        year = int(p[:4])
        q = 0 if p.endswith('FY') else int(p[5])
    except (ValueError, IndexError):
        return (0, 0)
    return (-year, -q)


@callback(
    Output('dq-manual-period-strip', 'children'),
    Output('dq-manual-periods-data', 'data'),
    Input('dq-manual-ticker', 'value'),
    Input('dq-manual-stmt-kind', 'value'),
    Input('dq-manual-variant', 'value'),
    Input('dq-manual-sel', 'data'),
    prevent_initial_call=True,
)
def update_manual_period_strip(ticker, kind, variant, sel):
    if not ticker or not kind:
        return [], {}
    ticker = ticker.strip().upper() if isinstance(ticker, str) else ticker
    try:
        report_dates = _uv._get_period_report_dates(ticker, kind)
    except Exception:
        return [html.Span('No data for this ticker/statement.',
                          style={'color': MUTED, 'fontSize': '12px'})], {}
    if not report_dates:
        return [html.Span('No filings found.',
                          style={'color': MUTED, 'fontSize': '12px'})], {}

    variant = variant or 'A'
    if variant == 'A':
        report_dates = {p: d for p, d in report_dates.items() if p.endswith('FY')}
    else:
        report_dates = {p: d for p, d in report_dates.items() if not p.endswith('FY')}

    if not report_dates:
        return [html.Span('No filings for selected variant.',
                          style={'color': MUTED, 'fontSize': '12px'})], {}

    periods = sorted(report_dates.keys(), key=_period_sort_key)
    periods_data = {'ticker': ticker, 'kind': kind,
                    'periods': periods, 'report_dates': report_dates}

    sel_period = (sel or {}).get('period')
    _BASE = {
        'width': '100%', 'textAlign': 'left', 'cursor': 'pointer',
        'padding': '7px 12px', 'border': 'none', 'borderBottom': '1px solid rgba(255,255,255,0.06)',
        'backgroundColor': 'transparent', 'color': MUTED, 'display': 'block',
    }
    _SEL = {**_BASE, 'backgroundColor': 'rgba(88,166,255,0.10)', 'color': ACCENT,
            'borderLeft': f'2px solid {ACCENT}', 'paddingLeft': '10px'}
    chips = []
    for i, p in enumerate(periods):
        rd = report_dates.get(p, '')
        date_label = str(rd)[:10] if rd else ''
        is_sel = p == sel_period
        chips.append(html.Button(
            html.Div([
                html.Span(p, style={'fontSize': '12px', 'fontWeight': '600' if is_sel else '400',
                                    'display': 'block'}),
                html.Span(date_label, style={'fontSize': '10px', 'opacity': '0.55',
                                             'display': 'block', 'marginTop': '1px'}),
            ]),
            id={'type': 'manual-period-btn', 'index': i},
            style=_SEL if is_sel else _BASE,
            n_clicks=0,
        ))
    return chips, periods_data


@callback(
    Output('dq-manual-sel', 'data'),
    Input({'type': 'manual-period-btn', 'index': ALL}, 'n_clicks'),
    State('dq-manual-periods-data', 'data'),
    prevent_initial_call=True,
)
def select_manual_period(n_clicks_list, periods_data):
    if not periods_data or not any(n_clicks_list):
        raise PreventUpdate
    from dash import ctx
    triggered = ctx.triggered_id
    if not triggered or not isinstance(triggered, dict):
        raise PreventUpdate
    idx = triggered['index']
    periods = periods_data.get('periods', [])
    if idx >= len(periods):
        raise PreventUpdate
    period = periods[idx]
    return {
        'ticker': periods_data['ticker'],
        'kind': periods_data['kind'],
        'period': period,
        'report_date': periods_data.get('report_dates', {}).get(period),
    }


@callback(
    Output('dq-manual-sel', 'data', allow_duplicate=True),
    Input('dq-manual-ticker', 'value'),
    Input('dq-manual-variant', 'value'),
    prevent_initial_call=True,
)
def reset_manual_sel(ticker, variant):
    return None


@callback(
    Output('dq-manual-table-wrap', 'children'),
    Output('dq-manual-corrections', 'data'),
    Input('dq-manual-sel', 'data'),
    Input('dq-manual-stmt-kind', 'value'),
    Input('dq-manual-unit-radio', 'value'),
    prevent_initial_call=True,
)
def load_manual_annotation_table(sel, kind, unit_choice):
    if not sel:
        raise PreventUpdate
    ticker = sel['ticker']
    kind = kind or sel.get('kind', 'income')
    period = sel['period']
    existing = dq._load_edgar_corrections(ticker, period, kind)
    existing_notes = dq._load_edgar_correction_notes(ticker, period, kind)
    table_div, store_payload = _render_annotation_table(
        ticker, period, kind, existing, input_type='edgar-input-manual',
        existing_notes=existing_notes,
        unit_choice=unit_choice or 'auto',
    )
    return table_div, store_payload


@callback(
    Output('dq-manual-edgar-wrap', 'children'),
    Input('dq-manual-sel', 'data'),
    Input('dq-manual-stmt-kind', 'value'),
    prevent_initial_call=True,
)
def fetch_manual_edgar(sel, kind):
    _wrap = {'display': 'flex', 'alignItems': 'center', 'justifyContent': 'space-between',
             'padding': '8px 12px', 'marginBottom': '10px',
             'backgroundColor': 'rgba(255,255,255,0.03)',
             'borderRadius': '6px', 'border': '1px solid rgba(255,255,255,0.07)'}
    if not sel:
        return html.Div('← Select a filing from the list',
                        style={**_wrap, 'color': MUTED, 'fontSize': '12px',
                               'fontStyle': 'italic'})
    ticker = sel['ticker']
    period = sel['period']
    kind = kind or sel.get('kind', 'income')
    report_date = sel.get('report_date')
    kind_label = {'income': 'Income', 'balance': 'Balance Sheet',
                  'cashflow': 'Cash Flow'}.get(kind, kind.capitalize())
    context = html.Span(f'{ticker}  ·  {period}  ·  {kind_label}',
                        style={'fontSize': '13px', 'fontWeight': '600', 'color': MUTED})
    cik = _uv._get_ticker_cik(ticker)
    if not cik or not report_date:
        edgar_el = html.Span('No EDGAR link', style={'fontSize': '12px', 'color': MUTED,
                                                      'opacity': '0.5'})
    else:
        period_code = 'A' if (period.endswith('FY') or period.endswith('Q4')) else 'Q'
        try:
            url = dq._fetch_edgar_url(cik, report_date, period_code)
        except Exception:
            url = None
        if url:
            edgar_el = html.A('Open EDGAR ↗', href=url, target='_blank',
                              rel='noopener noreferrer',
                              style={'fontSize': '12px', 'color': ACCENT,
                                     'textDecoration': 'none', 'fontWeight': '600'})
        else:
            edgar_el = html.Span('No EDGAR filing found',
                                 style={'fontSize': '12px', 'color': MUTED, 'opacity': '0.5'})
    return html.Div([context, edgar_el], style=_wrap)


@callback(
    Output('dq-manual-msg', 'children'),
    Input('dq-manual-save', 'n_clicks'),
    State('dq-manual-sel', 'data'),
    State('dq-manual-stmt-kind', 'value'),
    State('dq-manual-corrections', 'data'),
    State({'type': 'edgar-input-manual', 'index': ALL}, 'value'),
    State({'type': 'edgar-note-manual',  'index': ALL}, 'value'),
    prevent_initial_call=True,
)
def save_manual_annotations(n, sel, kind, corrections_store, input_values, note_values):
    if not sel or not corrections_store:
        raise PreventUpdate
    items = corrections_store.get('_items', [])
    if not items or not input_values:
        raise PreventUpdate
    ticker, period = sel['ticker'], sel['period']
    kind = kind or sel.get('kind', 'income')
    unit = float(corrections_store.get('_unit') or 1)
    edgar_values: dict[str, float] = {}
    simfin_values: dict[str, float] = {}
    notes: dict[str, str] = {}
    for item, raw, note_raw in zip(items, input_values, note_values or []):
        if raw and str(raw).strip():
            try:
                edgar_values[item] = float(str(raw).replace(',', '').strip()) * unit
            except ValueError:
                continue
        sf = corrections_store.get(item)
        if sf is not None:
            try:
                simfin_values[item] = float(sf)
            except (TypeError, ValueError):
                pass
        if note_raw and str(note_raw).strip():
            notes[item] = str(note_raw).strip()
    if not edgar_values:
        return 'Nothing to save (enter at least one EDGAR value).'
    try:
        dq.save_edgar_corrections(ticker, period, kind, edgar_values,
                                   simfin_values or None, notes or None)
    except Exception as exc:
        logger.warning(f'manual annotation save error: {exc}')
        return f'Error: {exc}'
    return f'Saved {len(edgar_values)} correction(s).'


@callback(
    Output('dq-edgar-corrections-tab', 'children'),
    Input('dq-tabs', 'value'),
    prevent_initial_call=True,
)
def update_edgar_corrections_tab(tab):
    if tab != 'edgar-corrections':
        raise PreventUpdate

    try:
        cos = _uv._get_companies()[['Ticker', 'Company Name']]
        ticker_options = [
            {'label': f"{row['Ticker']} — {row['Company Name']}" if row.get('Company Name') else row['Ticker'],
             'value': row['Ticker']}
            for _, row in cos.sort_values('Ticker').iterrows()
        ]
    except Exception:
        ticker_options = []
    manual_section = _manual_annotation_section(ticker_options)

    try:
        df = dq._get_all_edgar_corrections()
    except Exception as exc:
        logger.warning(f'corrections load error: {exc}')
        return html.Div([manual_section])

    if df.empty:
        return html.Div([
            manual_section,
        ])

    df = df.copy()

    def _diff_pct(row):
        try:
            sf, ed = float(row['simfin_value']), float(row['edgar_value'])
            return round((ed - sf) / abs(sf) * 100, 2) if sf != 0 else None
        except (TypeError, ValueError):
            return None

    df['diff_pct'] = df.apply(_diff_pct, axis=1)
    df['stmt_kind'] = df['stmt_kind'].str.capitalize()

    display_cols = [
        {'name': 'Ticker',    'id': 'ticker'},
        {'name': 'Period',    'id': 'period'},
        {'name': 'Statement', 'id': 'stmt_kind'},
        {'name': 'Line Item', 'id': 'line_item'},
        {'name': 'SimFin',    'id': 'simfin_value', 'type': 'numeric'},
        {'name': 'EDGAR',     'id': 'edgar_value',  'type': 'numeric'},
        {'name': 'Diff %',    'id': 'diff_pct',     'type': 'numeric'},
        {'name': 'Note',      'id': 'note'},
        {'name': 'Date',      'id': 'annotated_at'},
        {'name': '',          'id': '_spacer'},
    ]
    note_col = df['note'] if 'note' in df.columns else ''
    records = df[['ticker', 'period', 'stmt_kind', 'line_item',
                  'simfin_value', 'edgar_value', 'diff_pct', 'annotated_at']].copy()
    records['note'] = note_col
    records['_spacer'] = ''
    _nw = {'overflow': 'hidden', 'textOverflow': 'ellipsis', 'whiteSpace': 'nowrap'}
    report_table = dash.dash_table.DataTable(
        data=records.to_dict('records'),
        columns=display_cols,
        sort_action='native',
        filter_action='native',
        page_size=50,
        style_table={'overflowX': 'auto', 'width': '100%'},
        style_cell_conditional=[
            {'if': {'column_id': 'ticker'},       'width': '70px',  **_nw},
            {'if': {'column_id': 'period'},       'width': '80px',  **_nw},
            {'if': {'column_id': 'stmt_kind'},    'width': '80px',  **_nw},
            {'if': {'column_id': 'line_item'},    'width': '240px', **_nw},
            {'if': {'column_id': 'simfin_value'}, 'width': '110px', 'textAlign': 'right', **_nw},
            {'if': {'column_id': 'edgar_value'},  'width': '110px', 'textAlign': 'right', **_nw},
            {'if': {'column_id': 'diff_pct'},     'width': '70px',  'textAlign': 'right', **_nw},
            {'if': {'column_id': 'note'},         'width': '200px', **_nw},
            {'if': {'column_id': 'annotated_at'}, 'width': '90px',  **_nw},
        ],
        style_data_conditional=[
            {'if': {'filter_query': '{diff_pct} < -5 || {diff_pct} > 5'},
             'backgroundColor': 'rgba(231,76,60,0.10)'},
        ],
        **_TABLE_STYLE,
    )

    _divider = {'borderColor': 'rgba(255,255,255,0.07)', 'margin': '24px 0 16px'}
    return html.Div([
        manual_section,
        html.Hr(style=_divider),
        html.Div('Saved Corrections', style={
            'fontSize': '11px', 'fontWeight': '600', 'textTransform': 'uppercase',
            'letterSpacing': '0.06em', 'color': MUTED, 'marginBottom': '10px',
        }),
        report_table,
    ])

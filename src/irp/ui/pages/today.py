"""Today: the daily decision surface. Current regime + the playbook it implies + top names
ranked by the regime-appropriate composite + your watchlist's standing. Reads only from
`today_service` (services-only rule).
"""
import datetime
import logging

import dash
from dash import Input, Output, State, callback, dash_table, dcc, html
from dash.exceptions import PreventUpdate

from irp.ui.services import features_service, today_service
from irp.ui.theme import ACCENT, MUTED

dash.register_page(__name__, path='/today', name='Today')

logger = logging.getLogger(__name__)

_RED = '#e45756'
_GREEN = '#4ec94e'
_CACHE_CAP = 8
_CACHE: dict[str, today_service.TodayResult] = {}


def _put(key, val):
    _CACHE[key] = val
    while len(_CACHE) > _CACHE_CAP:
        _CACHE.pop(next(iter(_CACHE)))


def _chip(label, value, color=ACCENT):
    return html.Div(style={'display': 'inline-flex', 'flexDirection': 'column', 'gap': '2px',
                           'marginRight': '16px', 'marginBottom': '6px'}, children=[
        html.Span(label, style={'color': MUTED, 'fontSize': '10px', 'textTransform': 'uppercase'}),
        html.Span(value, style={'color': color, 'fontSize': '16px', 'fontWeight': '600'})])


def _field(label, control):
    return html.Div(style={'display': 'flex', 'flexDirection': 'column', 'gap': '2px'}, children=[
        html.Label(label, style={'color': MUTED, 'fontSize': '11px'}), control])


def _row(*children):
    return html.Div(style={'display': 'flex', 'gap': '16px', 'flexWrap': 'wrap',
                           'marginBottom': '8px'}, children=list(children))


def _section(title, sub=None):
    out = [html.H3(title, className='section-title',
                   style={'margin': '20px 0 4px', 'color': ACCENT,
                          'borderTop': '1px solid var(--border)', 'paddingTop': '14px'})]
    if sub:
        out.append(html.P(sub, style={'color': MUTED, 'fontSize': '12px'}))
    return out


layout = html.Div(className='page', children=[
    dcc.Store(id='today-init', data=1),
    dcc.Store(id='today-build'),

    html.H2('Today', className='page-title'),
    html.P('Your daily decision surface: the current macro regime, the playbook it implies, '
           'and the top names ranked by the regime-appropriate signal — plus how your '
           'watchlist stands. A starting point, not a trade order.',
           style={'color': MUTED, 'fontSize': '13px', 'maxWidth': '900px'}),

    html.Div(className='control-row', style={'alignItems': 'flex-end', 'gap': '12px',
                                             'flexWrap': 'wrap'}, children=[
        _field('Filing', dcc.RadioItems(id='today-variant',
                                        options=[{'label': ' Annual', 'value': 'A'},
                                                 {'label': ' Quarterly', 'value': 'Q'}],
                                        value='A', inline=True, labelClassName='check-item')),
        _field('Signal', dcc.Dropdown(id='today-preset', options=today_service.preset_options(),
                                      value='auto', clearable=False, className='filter-dropdown',
                                      style={'minWidth': '170px'})),
        _field('Sector', dcc.Dropdown(id='today-sector', options=today_service.sector_options(),
                                      placeholder='all', clearable=True,
                                      className='filter-dropdown', style={'minWidth': '170px'})),
        _field('Watchlist', dcc.Dropdown(id='today-watchlist',
                                         options=today_service.watchlist_options(),
                                         placeholder='none', clearable=True,
                                         className='filter-dropdown', style={'minWidth': '170px'})),
        _field('Min cap ($B)', dcc.Input(id='today-mincap', type='number', value=1, min=0,
                                         max=1000, step=1, style={'width': '90px'})),
        _field('Top N', dcc.Input(id='today-topn', type='number', value=20, min=5, max=100,
                                  step=5, style={'width': '80px'})),
        html.Button('Run', id='today-run', className='run-btn', n_clicks=0),
    ]),
    dcc.Loading(html.Div(id='today-warnings', style={'margin': '6px 0', 'fontSize': '12px',
                                                     'color': _RED})),
    dcc.Loading(html.Div(id='today-output'), type='default'),

    html.Details(style={'marginTop': '24px', 'borderTop': '1px solid var(--border)',
                        'paddingTop': '12px'}, children=[
        html.Summary('Rebuild factor cache', style={'cursor': 'pointer', 'color': MUTED,
                                                    'fontSize': '12px', 'userSelect': 'none'}),
        html.Div(style={'display': 'flex', 'gap': '10px', 'alignItems': 'flex-end',
                        'flexWrap': 'wrap', 'marginTop': '8px'}, children=[
            _field('Start year', dcc.Input(id='today-pre-start', type='number',
                                           value=datetime.date.today().year - 3,
                                           min=2010, max=2030, step=1, style={'width': '80px'})),
            _field('End year', dcc.Input(id='today-pre-end', type='number',
                                         value=datetime.date.today().year,
                                         min=2010, max=2030, step=1, style={'width': '80px'})),
            html.Button('Precompute', id='today-precompute-btn', className='run-btn',
                        n_clicks=0, style={'fontSize': '12px', 'padding': '4px 12px'}),
        ]),
        dcc.Loading(html.Div(id='today-pre-status',
                             style={'fontSize': '12px', 'marginTop': '6px', 'color': MUTED})),
    ]),
])


@callback(Output('today-watchlist', 'options'), Input('today-init', 'data'))
def today_populate(_init):
    return today_service.watchlist_options()


def _fmt_pct(x):
    import numpy as np
    return '—' if x is None or not np.isfinite(x) else f'{x * 100:.1f}%'


def _fmt_num(x, nd=1):
    import numpy as np
    return '—' if x is None or not np.isfinite(x) else f'{x:.{nd}f}'


def _regime_banner(r):
    reg = r.regime
    label = reg['label']
    color = today_service.regime_color(label)
    drivers = reg.get('drivers')
    top_drivers = ''
    if drivers is not None and not drivers.empty:
        pos = drivers[drivers > 0].sort_values(ascending=False).head(2).index.tolist()
        neg = drivers[drivers < 0].sort_values().head(2).index.tolist()
        top_drivers = 'up: ' + (', '.join(pos) or '—') + '  ·  down: ' + (', '.join(neg) or '—')
    chips = _row(
        _chip('Regime', label.replace('_', '-'), color),
        _chip('Risk score', _fmt_num(reg.get('score'), 0)),
        _chip('Playbook', r.playbook['stance']),
        _chip('Signal used', r.preset.title()),
        _chip('Regime as-of', str(reg.get('as_of') or '—')),
        _chip('Names as-of', str(r.as_of or '—')),
    )
    note = html.P(top_drivers, style={'color': MUTED, 'fontSize': '11px'}) if top_drivers else None
    link = dcc.Link('Open /regime for the full dashboard →', href='/regime',
                    className='card-link', style={'fontSize': '12px'})
    return [*_section('1 · Regime & playbook'), chips, note, link] if note else \
        [*_section('1 · Regime & playbook'), chips, link]


def _table(df, fmt_cols=(), table_id=None):
    cols = []
    for c in df.columns:
        col = {'name': str(c), 'id': str(c)}
        if c in fmt_cols:
            col['type'] = 'numeric'
            col['format'] = {'specifier': '.2f'}
        cols.append(col)
    kwargs = {'id': table_id} if table_id else {}
    return dash_table.DataTable(
        data=df.reset_index().rename(columns={'index': 'Ticker'}).to_dict('records'),
        columns=[{'name': 'Ticker', 'id': 'Ticker'}] + cols,
        sort_action='native', page_action='none', **kwargs,
        style_as_list_view=True,
        style_header={'backgroundColor': 'transparent', 'color': MUTED,
                      'fontWeight': '600', 'border': 'none'},
        style_cell={'backgroundColor': 'transparent', 'color': 'var(--text)',
                    'border': 'none', 'fontSize': '13px', 'padding': '6px 10px'},
        style_data={'borderBottom': '1px solid var(--border)'},
    )


def _build_output(r):
    blocks = _regime_banner(r)
    blocks += _section('2 · Top names', f'Ranked by the {r.preset.title()} composite on the '
                                        f'{r.as_of} cross-section.')
    if r.top is None or r.top.empty:
        blocks.append(html.P('No ranked names (cold cache or empty filter).', className='no-data'))
    else:
        num = [c for c in r.top.columns if c not in ('Rank', 'Company Name', 'Sector')]
        blocks.append(html.P('Click a row to open it on /analysis.',
                             style={'color': MUTED, 'fontSize': '11px'}))
        blocks.append(_table(r.top, fmt_cols=num, table_id='today-top-table'))
    if r.watchlist is not None and not r.watchlist.empty:
        blocks += _section('3 · Watchlist standing',
                           'Your watchlist names scored on the same signal, with their '
                           'percentile across the full universe.')
        num = [c for c in r.watchlist.columns if c not in ('Company Name', 'Sector')]
        blocks.append(_table(r.watchlist, fmt_cols=num))

    mp = r.model
    blocks += _section('4 · Model picks',
                       'Top names from your latest model export (notebook → '
                       'save_predictions), ranked by predicted forward return.')
    if mp is None or mp.picks is None or mp.picks.empty:
        msg = '  •  '.join(mp.warnings) if (mp and mp.warnings) else 'No model export yet.'
        blocks.append(html.P(msg + ' Run a model notebook and call '
                             'save_predictions(res.predictions, name).', className='no-data'))
    else:
        blocks.append(html.P(f'Source: {mp.source}  ·  prediction date {mp.as_of}  ·  '
                             'click a row to open it on /analysis.',
                             style={'color': MUTED, 'fontSize': '11px'}))
        num = [c for c in mp.picks.columns if c not in ('Rank', 'Company Name', 'Sector', 'Date')]
        view = mp.picks.drop(columns=[c for c in ('Date',) if c in mp.picks.columns])
        blocks.append(_table(view, fmt_cols=num, table_id='today-model-table'))
    return blocks


@callback(
    Output('today-build', 'data'), Output('today-warnings', 'children'),
    Input('today-run', 'n_clicks'),
    State('today-variant', 'value'), State('today-preset', 'value'),
    State('today-sector', 'value'), State('today-watchlist', 'value'),
    State('today-mincap', 'value'), State('today-topn', 'value'),
    running=[(Output('today-run', 'disabled'), True, False),
             (Output('today-run', 'children'), 'Running…', 'Run')],
    prevent_initial_call=True,
)
def today_run(_n, variant, preset, sector, watchlist, mincap, topn):
    preset_arg = None if preset == 'auto' else preset
    try:
        r = today_service.dashboard(variant=variant, sector=sector, watchlist=watchlist,
                                    top_n=int(topn or 20), preset=preset_arg,
                                    min_mktcap=float(mincap or 0))
    except Exception as exc:
        logger.exception('today dashboard failed')
        return None, f'Failed: {exc}'
    token = str(abs(hash((variant, preset, sector, watchlist, mincap, topn))))
    _put(token, r)
    return {'token': token}, '  •  '.join(r.warnings)


@callback(Output('today-output', 'children'), Input('today-build', 'data'))
def today_render(store):
    if not store or store.get('token') not in _CACHE:
        return html.P('Pick filters, then Run.', className='no-data')
    return _build_output(_CACHE[store['token']])


@callback(
    Output('workspace', 'data', allow_duplicate=True),
    Output('url-redirect', 'pathname', allow_duplicate=True),
    Input('today-top-table', 'active_cell'),
    State('today-top-table', 'data'), State('workspace', 'data'),
    prevent_initial_call=True,
)
def today_row_to_analysis(active, data, ws):
    """Click a top-name row → stash the ticker in the shared workspace and jump to /analysis."""
    if not active or not data:
        raise PreventUpdate
    row = data[active['row']]
    ticker = row.get('Ticker')
    if not ticker:
        raise PreventUpdate
    new_ws = dict(ws or {})
    new_ws['ticker'] = ticker
    return new_ws, '/analysis'


@callback(
    Output('workspace', 'data', allow_duplicate=True),
    Output('url-redirect', 'pathname', allow_duplicate=True),
    Input('today-model-table', 'active_cell'),
    State('today-model-table', 'data'), State('workspace', 'data'),
    prevent_initial_call=True,
)
def today_model_row_to_analysis(active, data, ws):
    """Click a model-pick row → carry the ticker to /analysis (same handoff as top names)."""
    if not active or not data:
        raise PreventUpdate
    ticker = data[active['row']].get('Ticker')
    if not ticker:
        raise PreventUpdate
    new_ws = dict(ws or {})
    new_ws['ticker'] = ticker
    return new_ws, '/analysis'


@callback(
    Output('today-pre-status', 'children'),
    Input('today-precompute-btn', 'n_clicks'),
    State('today-variant', 'value'),
    State('today-pre-start', 'value'),
    State('today-pre-end', 'value'),
    running=[(Output('today-precompute-btn', 'disabled'), True, False),
             (Output('today-precompute-btn', 'children'), 'Running…', 'Precompute')],
    prevent_initial_call=True,
)
def today_precompute(_n, variant, start_yr, end_yr):
    try:
        n = features_service.precompute(int(start_yr), int(end_yr), variant or 'A')
        return f'Done — {n} new snapshot(s) written.'
    except Exception as exc:
        logger.exception('precompute failed')
        return f'Failed: {exc}'

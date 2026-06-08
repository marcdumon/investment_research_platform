"""Regime page: cross-asset macro state + how it conditions stock decisions.

Dashboard (rule risk score + HMM states over ^SPX), factor IC conditioned on regime, a
regime-gated factor backtest, and a cross-asset tactical-allocation table. Reads only from
`regime_service` (services-only rule).
"""
import datetime
import logging

import dash
import numpy as np
import plotly.graph_objects as go
from dash import Input, Output, State, callback, dcc, html

from irp.ui.charts import base_chart_layout, empty_figure
from irp.ui.services import features_service, regime_service
from irp.ui.theme import ACCENT, MUTED

dash.register_page(__name__, path='/regime', name='Regime')

logger = logging.getLogger(__name__)

_RED = '#e45756'
_GREEN = '#4ec94e'
_GREY = '#9aa0a6'
_CACHE_CAP = 8
_REGIME_COLOR = {'risk_off': _RED, 'neutral': _GREY, 'risk_on': _GREEN, 'unknown': MUTED}
_HMM_PALETTE = [_RED, '#e0a040', _GREEN, '#5b8def', '#b072d0']

_PRESETS = [{'label': '5Y', 'value': 1825}, {'label': '10Y', 'value': 3650},
            {'label': '20Y', 'value': 7300}, {'label': 'Max', 'value': 0}]
_HORIZONS = [{'label': ' 21d', 'value': 21}, {'label': ' 63d', 'value': 63},
             {'label': ' 126d', 'value': 126}, {'label': ' 252d', 'value': 252}]
_REBAL = [{'label': ' Quarterly', 'value': 'Q'}, {'label': ' Annual', 'value': 'A'}]
_ALLOWED = [{'label': ' Risk-on', 'value': 'risk_on'}, {'label': ' Neutral', 'value': 'neutral'},
            {'label': ' Risk-off', 'value': 'risk_off'}]

_DASH_CACHE: dict[str, regime_service.RegimeState] = {}
_COND_CACHE: dict[str, regime_service.ConditionedFactors] = {}
_GATE_CACHE: dict[str, regime_service.GatedBacktest] = {}
_TAC_CACHE: dict[str, object] = {}


def _put(cache, key, val):
    cache[key] = val
    while len(cache) > _CACHE_CAP:
        cache.pop(next(iter(cache)))


def _chip(label, value, color=ACCENT):
    return html.Div(style={'display': 'inline-flex', 'flexDirection': 'column', 'gap': '2px',
                           'marginRight': '14px', 'marginBottom': '6px'}, children=[
        html.Span(label, style={'color': MUTED, 'fontSize': '10px', 'textTransform': 'uppercase'}),
        html.Span(value, style={'color': color, 'fontSize': '15px', 'fontWeight': '600'})])


def _fmt_pct(x):
    return '—' if x is None or not np.isfinite(x) else f'{x * 100:.1f}%'


def _fmt_num(x, nd=2):
    return '—' if x is None or not np.isfinite(x) else f'{x:.{nd}f}'


def _graph(fig, basis='1 1 calc(50% - 8px)'):
    return dcc.Graph(figure=fig, config={'displayModeBar': False},
                     style={'flex': basis, 'minWidth': '320px'})


def _row(*children):
    return html.Div(style={'display': 'flex', 'gap': '16px', 'flexWrap': 'wrap',
                           'marginBottom': '8px'}, children=list(children))


def _field(label, control):
    return html.Div(style={'display': 'flex', 'flexDirection': 'column', 'gap': '2px'}, children=[
        html.Label(label, style={'color': MUTED, 'fontSize': '11px'}), control])


def _section_head(title, blurb):
    return [html.H3(title, className='section-title',
                    style={'margin': '24px 0 4px', 'color': ACCENT,
                           'borderTop': '1px solid var(--border)', 'paddingTop': '16px'}),
            html.P(blurb, style={'color': MUTED, 'fontSize': '12px', 'maxWidth': '900px'})]


def _signal_dd(id_):
    return dcc.Dropdown(id=id_, options=regime_service.signal_options(), value='composite',
                        clearable=False, className='filter-dropdown', style={'minWidth': '240px'})


# ── layout ────────────────────────────────────────────────────────────

layout = html.Div(className='page', children=[
    dcc.Store(id='rg-init', data=1),
    dcc.Store(id='rg-dash-build'), dcc.Store(id='rg-cond-build'),
    dcc.Store(id='rg-gate-build'), dcc.Store(id='rg-tac-build'),
    dcc.Store(id='rg-mk-build'),

    html.H2('Regime', className='page-title'),
    html.P('Cross-asset macro state from the US yield curve, equity trend/vol, USD and '
           'commodities — classified two ways (a transparent risk score and a Gaussian HMM) — '
           'then fed back into stock decisions: which factors earn their edge in which regime, '
           'a regime-timed factor backtest, and a cross-asset tactical table.',
           style={'color': MUTED, 'fontSize': '13px', 'maxWidth': '900px'}),

    # ── 1 Dashboard ──────────────────────────────────────────────────
    html.Div(className='control-row', style={'alignItems': 'flex-end', 'gap': '12px'}, children=[
        _field('Period', dcc.RadioItems(id='rg-preset', options=_PRESETS, value=3650,
                                        inline=True, labelClassName='check-item')),
        _field('HMM states', dcc.RadioItems(id='rg-nstates',
                                            options=[{'label': f' {k}', 'value': k} for k in (2, 3, 4)],
                                            value=3, inline=True, labelClassName='check-item')),
        html.Button('Run dashboard', id='rg-dash-run', className='run-btn', n_clicks=0),
    ]),
    dcc.Loading(html.Div(id='rg-dash-warnings', style={'margin': '6px 0', 'fontSize': '12px',
                                                       'color': _RED})),
    dcc.Loading(html.Div(id='rg-dash-output'), type='default'),

    # ── 2 Regime-conditioned factors ─────────────────────────────────
    *_section_head('Regime-conditioned factors',
                   'Information coefficient (rank IC) of a factor or composite, grouped by the '
                   'rule regime in force at each rebalance date (causal labels — expanding '
                   'z-score, no look-ahead). Shows where a signal\'s predictive edge actually '
                   'lives. Needs a warm factor cache for the period.'),
    html.Div(className='control-row', style={'alignItems': 'flex-end', 'gap': '12px'}, children=[
        _field('Signal', _signal_dd('rg-cond-signal')),
        _field('Horizon', dcc.RadioItems(id='rg-cond-horizon', options=_HORIZONS, value=63,
                                         inline=True, labelClassName='check-item')),
        _field('Rebalance', dcc.RadioItems(id='rg-cond-rebal', options=_REBAL, value='Q',
                                           inline=True, labelClassName='check-item')),
        html.Button('Run conditioning', id='rg-cond-run', className='run-btn', n_clicks=0),
    ]),
    dcc.Loading(html.Div(id='rg-cond-warnings', style={'margin': '6px 0', 'fontSize': '12px',
                                                       'color': _RED})),
    dcc.Loading(html.Div(id='rg-cond-output'), type='default'),

    # ── 3 Regime-gated backtest ──────────────────────────────────────
    *_section_head('Regime-gated backtest',
                   'Take the factor/composite long-short return only in the allowed regimes '
                   '(else flat) and compare to always-on. Tests whether regime timing improves '
                   'Sharpe and drawdown. Gating uses the same causal rule labels.'),
    html.Div(className='control-row', style={'alignItems': 'flex-end', 'gap': '12px'}, children=[
        _field('Signal', _signal_dd('rg-gate-signal')),
        _field('Allowed regimes', dcc.Checklist(id='rg-gate-allowed', options=_ALLOWED,
                                                value=['risk_on', 'neutral'], inline=True,
                                                labelClassName='check-item')),
        _field('Horizon', dcc.RadioItems(id='rg-gate-horizon', options=_HORIZONS, value=63,
                                         inline=True, labelClassName='check-item')),
        _field('Rebalance', dcc.RadioItems(id='rg-gate-rebal', options=_REBAL, value='Q',
                                           inline=True, labelClassName='check-item')),
        html.Button('Run gated', id='rg-gate-run', className='run-btn', n_clicks=0),
    ]),
    dcc.Loading(html.Div(id='rg-gate-warnings', style={'margin': '6px 0', 'fontSize': '12px',
                                                       'color': _RED})),
    dcc.Loading(html.Div(id='rg-gate-output'), type='default'),

    # ── 4 Tactical allocation ────────────────────────────────────────
    *_section_head('Tactical allocation (cross-asset trend)',
                   'Trend / momentum of each asset class (equities, Treasuries, gold, '
                   'commodities, credit, crypto) over 63/126/252 trading days, ranked. A '
                   'descriptive relative-strength overlay, not a sized portfolio.'),
    html.Div(className='control-row', style={'alignItems': 'flex-end', 'gap': '12px'}, children=[
        html.Button('Run tactical', id='rg-tac-run', className='run-btn', n_clicks=0),
    ]),
    dcc.Loading(html.Div(id='rg-tac-output'), type='default'),

    # ── 5 Single-asset Markov regime (hedge-fund method) ─────────────
    *_section_head('Single-asset Markov regime (hedge-fund method)',
                   'Per-asset version of the "Markov hedge-fund method": label each day '
                   'bull / sideways / bear from the trailing-window return (your lookback + '
                   'threshold), build the bull/sideways/bear transition matrix, read its '
                   'persistence, project it forward (matrix powers) to a stationary '
                   'distribution, and turn it into a trade signal = P(bull)−P(bear). An HMM '
                   'overlay gives a second, threshold-free opinion. Overlapping daily windows '
                   'inflate persistence (adjacent days share data) — use non-overlapping for '
                   'an honest matrix. The walk-forward backtest builds the matrix from past '
                   'data only and is usually a sobering reality check vs buy-and-hold.'),
    html.Div(className='control-row', style={'alignItems': 'flex-end', 'gap': '12px',
                                             'flexWrap': 'wrap'}, children=[
        _field('Instrument', dcc.Dropdown(id='rg-mk-ticker', options=[], placeholder='e.g. AAPL',
                                          clearable=False, className='filter-dropdown',
                                          style={'minWidth': '200px'})),
        _field('Lookback (days)', dcc.Input(id='rg-mk-lookback', type='number', value=20,
                                            min=2, max=252, step=1, style={'width': '90px'})),
        _field('Threshold (%)', dcc.Input(id='rg-mk-threshold', type='number', value=5,
                                          min=0.5, max=50, step=0.5, style={'width': '90px'})),
        _field('Sampling', dcc.RadioItems(id='rg-mk-sampling',
                                          options=[{'label': ' Overlapping', 'value': 'over'},
                                                   {'label': ' Non-overlapping', 'value': 'non'}],
                                          value='non', inline=True, labelClassName='check-item')),
        _field('HMM overlay', dcc.Checklist(id='rg-mk-hmm',
                                            options=[{'label': ' on', 'value': 'on'}],
                                            value=['on'], inline=True, labelClassName='check-item')),
        _field('Backtest horizon (days)', dcc.RadioItems(id='rg-mk-horizon', options=_HORIZONS,
                                                         value=21, inline=True,
                                                         labelClassName='check-item')),
        html.Button('Run Markov', id='rg-mk-run', className='run-btn', n_clicks=0),
    ]),
    dcc.Loading(html.Div(id='rg-mk-warnings', style={'margin': '6px 0', 'fontSize': '12px',
                                                     'color': _RED})),
    dcc.Loading(html.Div(id='rg-mk-output'), type='default'),

    html.Details(style={'marginTop': '24px', 'borderTop': '1px solid var(--border)',
                        'paddingTop': '12px'}, children=[
        html.Summary('Rebuild factor cache', style={'cursor': 'pointer', 'color': MUTED,
                                                    'fontSize': '12px', 'userSelect': 'none'}),
        html.Div(style={'display': 'flex', 'gap': '10px', 'alignItems': 'flex-end',
                        'flexWrap': 'wrap', 'marginTop': '8px'}, children=[
            html.Div(style={'display': 'flex', 'flexDirection': 'column', 'gap': '2px'}, children=[
                html.Label('Start year', style={'color': MUTED, 'fontSize': '11px'}),
                dcc.Input(id='rg-pre-start', type='number',
                          value=datetime.date.today().year - 3,
                          min=2010, max=2030, step=1, style={'width': '80px'}),
            ]),
            html.Div(style={'display': 'flex', 'flexDirection': 'column', 'gap': '2px'}, children=[
                html.Label('End year', style={'color': MUTED, 'fontSize': '11px'}),
                dcc.Input(id='rg-pre-end', type='number',
                          value=datetime.date.today().year,
                          min=2010, max=2030, step=1, style={'width': '80px'}),
            ]),
            html.Div(style={'display': 'flex', 'flexDirection': 'column', 'gap': '2px'}, children=[
                html.Label('Variant', style={'color': MUTED, 'fontSize': '11px'}),
                dcc.RadioItems(id='rg-pre-variant',
                               options=[{'label': ' A', 'value': 'A'},
                                        {'label': ' Q', 'value': 'Q'}],
                               value='A', inline=True, labelClassName='check-item'),
            ]),
            html.Button('Precompute', id='rg-precompute-btn', className='run-btn', n_clicks=0,
                        style={'fontSize': '12px', 'padding': '4px 12px'}),
        ]),
        dcc.Loading(html.Div(id='rg-pre-status',
                             style={'fontSize': '12px', 'marginTop': '6px', 'color': MUTED})),
    ]),
])


@callback(Output('rg-mk-ticker', 'options'), Input('rg-init', 'data'))
def rg_populate_mk(_init):
    return regime_service.instrument_options()


# ── 1 Dashboard ───────────────────────────────────────────────────────

def _timeline_fig(st):
    eq = st.equity.dropna()
    if eq.empty:
        return empty_figure('Regime timeline: n/a')
    fig = go.Figure(go.Scatter(x=list(eq.index), y=np.log(eq.to_numpy()), mode='lines',
                               line={'color': ACCENT}, name='log ^SPX'))
    # shade contiguous risk-off spans over the price (stress periods)
    lab = st.rule['label'].reindex(eq.index, method='ffill')
    off = (lab == 'risk_off').to_numpy()
    idx = eq.index
    i = 0
    while i < len(off):
        if off[i]:
            j = i
            while j + 1 < len(off) and off[j + 1]:
                j += 1
            fig.add_vrect(x0=idx[i], x1=idx[j], fillcolor=_RED, opacity=0.12,
                          line_width=0, layer='below')
            i = j + 1
        else:
            i += 1
    fig.update_layout(base_chart_layout(title='^SPX with risk-off regime shaded (rule)',
                                        yaxis_title='log price', showlegend=False), height=320)
    return fig


def _score_fig(st):
    sc = st.rule['risk_score'].dropna()
    if sc.empty:
        return empty_figure('Risk score: n/a')
    fig = go.Figure(go.Scatter(x=list(sc.index), y=sc.to_numpy(), mode='lines',
                               line={'color': ACCENT}))
    for lvl in (40, 60):
        fig.add_hline(y=lvl, line_dash='dot', line_color=MUTED)
    fig.update_layout(base_chart_layout(title='Rule risk score (0–100)', yaxis_title='score'),
                      height=320)
    return fig


def _transition_fig(st):
    tm = st.hmm_transition
    if tm is None or tm.empty:
        return empty_figure('Transition matrix: n/a')
    z = tm.to_numpy()
    labels = [f'S{i}' for i in tm.index]
    fig = go.Figure(go.Heatmap(z=z, x=labels, y=labels, colorscale='Blues', zmin=0, zmax=1,
                               text=np.round(z, 2), texttemplate='%{text}', showscale=False))
    fig.update_layout(base_chart_layout(title='HMM state transition matrix (row→col)',
                                        xaxis_title='to', yaxis_title='from'), height=320)
    return fig


def _contrib_fig(st):
    c = st.contrib
    if c is None or c.empty:
        return empty_figure('Feature contribution: n/a')
    vals = c.to_numpy()
    colors = [_GREEN if v >= 0 else _RED for v in vals]
    fig = go.Figure(go.Bar(x=vals, y=list(c.index), orientation='h', marker_color=colors))
    fig.update_layout(base_chart_layout(title='Today\'s risk-on contribution by feature',
                                        xaxis_title='signed weight × z', showlegend=False),
                      height=320)
    return fig


def _build_dash_output(st):
    if st.rule.empty:
        msg = '  •  '.join(st.warnings) if st.warnings else 'No regime history.'
        return html.P(msg, className='no-data')
    last = st.rule.dropna().iloc[-1]
    cur_label = last['label']
    cur_score = last['risk_score']
    hmm_last = int(st.hmm_labels.dropna().iloc[-1]) if not st.hmm_labels.dropna().empty else -1
    persist = (float(st.hmm_transition.loc[hmm_last, hmm_last])
               if hmm_last >= 0 and hmm_last in st.hmm_transition.index else np.nan)
    chips = _row(
        _chip('Rule regime', cur_label.replace('_', '-'), _REGIME_COLOR.get(cur_label, ACCENT)),
        _chip('Risk score', _fmt_num(cur_score, 0)),
        _chip('HMM state', f'S{hmm_last}' if hmm_last >= 0 else '—',
              _HMM_PALETTE[hmm_last % len(_HMM_PALETTE)] if hmm_last >= 0 else MUTED),
        _chip('State persistence', _fmt_pct(persist)),
        _chip('Days', str(len(st.rule.dropna()))),
    )
    return [chips,
            _row(_graph(_timeline_fig(st), basis='1 1 100%')),
            _row(_graph(_score_fig(st)), _graph(_contrib_fig(st))),
            _row(_graph(_transition_fig(st))),
            html.P('HMM timeline is full-sample (in-sample, for visualization). Conditioning '
                   'and gating below use causal rule labels.',
                   style={'color': MUTED, 'fontSize': '11px'})]


@callback(
    Output('rg-dash-build', 'data'), Output('rg-dash-warnings', 'children'),
    Input('rg-dash-run', 'n_clicks'),
    State('rg-preset', 'value'), State('rg-nstates', 'value'),
    running=[(Output('rg-dash-run', 'disabled'), True, False),
             (Output('rg-dash-run', 'children'), 'Running…', 'Run dashboard')],
    prevent_initial_call=True,
)
def rg_run_dash(_n, preset_days, n_states):
    end = datetime.date.today()
    start = None if not preset_days else end - datetime.timedelta(days=int(preset_days))
    try:
        st = regime_service.dashboard(start, end, n_states=int(n_states))
    except Exception as exc:
        logger.exception('regime dashboard failed')
        return None, f'Dashboard failed: {exc}'
    token = str(abs(hash((preset_days, n_states, str(start), str(end)))))
    _put(_DASH_CACHE, token, st)
    return {'token': token}, '  •  '.join(st.warnings)


@callback(Output('rg-dash-output', 'children'), Input('rg-dash-build', 'data'))
def rg_render_dash(store):
    if not store or store.get('token') not in _DASH_CACHE:
        return html.P('Pick a period, then Run dashboard.', className='no-data')
    return _build_dash_output(_DASH_CACHE[store['token']])


# ── 2 Regime-conditioned factors ──────────────────────────────────────

def _cond_fig(cf):
    tab = cf.table
    if tab is None or tab.empty:
        return empty_figure('Conditioned IC: n/a')
    order = [r for r in ('risk_off', 'neutral', 'risk_on') if r in tab.index]
    tab = tab.reindex(order)
    colors = [_REGIME_COLOR.get(r, ACCENT) for r in tab.index]
    text = [f'IC={v:.3f}<br>n={int(n)}' for v, n in zip(tab['mean_ic'], tab['n'], strict=True)]
    fig = go.Figure(go.Bar(x=[r.replace('_', '-') for r in tab.index], y=tab['mean_ic'].to_numpy(),
                           marker_color=colors, text=text, textposition='outside'))
    fig.add_hline(y=0, line_dash='dot', line_color=MUTED)
    fig.update_layout(base_chart_layout(title=f'Mean IC by regime — {cf.signal}',
                                        yaxis_title='mean IC', showlegend=False), height=340)
    return fig


def _icir_fig(cf):
    tab = cf.table
    if tab is None or tab.empty:
        return empty_figure('ICIR: n/a')
    order = [r for r in ('risk_off', 'neutral', 'risk_on') if r in tab.index]
    tab = tab.reindex(order)
    colors = [_REGIME_COLOR.get(r, ACCENT) for r in tab.index]
    fig = go.Figure(go.Bar(x=[r.replace('_', '-') for r in tab.index], y=tab['icir'].to_numpy(),
                           marker_color=colors))
    fig.add_hline(y=0, line_dash='dot', line_color=MUTED)
    fig.update_layout(base_chart_layout(title=f'ICIR by regime — {cf.signal}',
                                        yaxis_title='ICIR', showlegend=False), height=340)
    return fig


def _build_cond_output(cf):
    if cf.table is None or cf.table.empty:
        msg = '  •  '.join(cf.warnings) if cf.warnings else 'No conditioned IC.'
        return html.P(msg, className='no-data')
    return [_row(_chip('Signal', cf.signal), _chip('Rebalance dates', str(cf.n_dates))),
            _row(_graph(_cond_fig(cf)), _graph(_icir_fig(cf)))]


@callback(
    Output('rg-cond-build', 'data'), Output('rg-cond-warnings', 'children'),
    Input('rg-cond-run', 'n_clicks'),
    State('rg-cond-signal', 'value'), State('rg-cond-horizon', 'value'),
    State('rg-cond-rebal', 'value'), State('rg-preset', 'value'),
    running=[(Output('rg-cond-run', 'disabled'), True, False),
             (Output('rg-cond-run', 'children'), 'Running…', 'Run conditioning')],
    prevent_initial_call=True,
)
def rg_run_cond(_n, signal, horizon, rebal, preset_days):
    if not signal:
        return None, 'Pick a signal.'
    end = datetime.date.today()
    days = int(preset_days) if preset_days else 5475
    start = end - datetime.timedelta(days=max(days, 1825))
    try:
        cf = regime_service.conditioned_factors(signal, start, end, horizon=int(horizon),
                                                freq=rebal)
    except Exception as exc:
        logger.exception('conditioning failed')
        return None, f'Conditioning failed: {exc}'
    token = str(abs(hash((signal, horizon, rebal, preset_days, str(start), str(end)))))
    _put(_COND_CACHE, token, cf)
    return {'token': token}, '  •  '.join(cf.warnings)


@callback(Output('rg-cond-output', 'children'), Input('rg-cond-build', 'data'))
def rg_render_cond(store):
    if not store or store.get('token') not in _COND_CACHE:
        return html.P('Pick a signal, then Run conditioning.', className='no-data')
    return _build_cond_output(_COND_CACHE[store['token']])


# ── 3 Regime-gated backtest ───────────────────────────────────────────

def _gate_fig(gb):
    r = gb.result
    if not r or r.get('gated_cumret') is None or r['gated_cumret'].empty:
        return empty_figure('Gated backtest: n/a')
    g, b = r['gated_cumret'], r['base_cumret']
    fig = go.Figure([
        go.Scatter(x=list(b.index), y=b.to_numpy(), mode='lines', name='always-on',
                   line={'color': MUTED}),
        go.Scatter(x=list(g.index), y=g.to_numpy(), mode='lines', name='regime-gated',
                   line={'color': ACCENT}),
    ])
    fig.update_layout(base_chart_layout(title=f'Cumulative L/S log return — {gb.signal}',
                                        yaxis_title='cum log ret'), height=360)
    return fig


def _build_gate_output(gb):
    r = gb.result
    if not r:
        msg = '  •  '.join(gb.warnings) if gb.warnings else 'No gated result.'
        return html.P(msg, className='no-data')
    gs, bs = r['gated_sharpe'], r['base_sharpe']
    chips = _row(
        _chip('Allowed', ', '.join(a.replace('_', '-') for a in gb.allowed)),
        _chip('Gated Sharpe', _fmt_num(gs), _GREEN if (np.isfinite(gs) and gs > bs) else MUTED),
        _chip('Always-on Sharpe', _fmt_num(bs)),
        _chip('Gated max DD', _fmt_pct(r['gated_maxdd']), _RED),
        _chip('Always-on max DD', _fmt_pct(r['base_maxdd']), _RED),
    )
    return [chips, _row(_graph(_gate_fig(gb), basis='1 1 100%'))]


@callback(
    Output('rg-gate-build', 'data'), Output('rg-gate-warnings', 'children'),
    Input('rg-gate-run', 'n_clicks'),
    State('rg-gate-signal', 'value'), State('rg-gate-allowed', 'value'),
    State('rg-gate-horizon', 'value'), State('rg-gate-rebal', 'value'), State('rg-preset', 'value'),
    running=[(Output('rg-gate-run', 'disabled'), True, False),
             (Output('rg-gate-run', 'children'), 'Running…', 'Run gated')],
    prevent_initial_call=True,
)
def rg_run_gate(_n, signal, allowed, horizon, rebal, preset_days):
    if not signal:
        return None, 'Pick a signal.'
    if not allowed:
        return None, 'Allow at least one regime.'
    end = datetime.date.today()
    days = int(preset_days) if preset_days else 5475
    start = end - datetime.timedelta(days=max(days, 1825))
    try:
        gb = regime_service.gated_backtest(signal, allowed, start, end, horizon=int(horizon),
                                           freq=rebal)
    except Exception as exc:
        logger.exception('gated backtest failed')
        return None, f'Gated backtest failed: {exc}'
    token = str(abs(hash((signal, tuple(sorted(allowed)), horizon, rebal, preset_days,
                          str(start), str(end)))))
    _put(_GATE_CACHE, token, gb)
    return {'token': token}, '  •  '.join(gb.warnings)


@callback(Output('rg-gate-output', 'children'), Input('rg-gate-build', 'data'))
def rg_render_gate(store):
    if not store or store.get('token') not in _GATE_CACHE:
        return html.P('Pick a signal + allowed regimes, then Run gated.', className='no-data')
    return _build_gate_output(_GATE_CACHE[store['token']])


# ── 4 Tactical allocation ─────────────────────────────────────────────

def _tac_fig(tab):
    if tab is None or tab.empty:
        return empty_figure('Tactical: n/a')
    t = tab.sort_values('score')
    colors = [_GREEN if v >= 0 else _RED for v in t['score'].to_numpy()]
    fig = go.Figure(go.Bar(x=t['score'].to_numpy(), y=list(t.index), orientation='h',
                           marker_color=colors))
    fig.update_layout(base_chart_layout(title='Cross-asset trend score (avg of 63/126/252d)',
                                        xaxis_title='mean log return', showlegend=False),
                      height=360)
    return fig


def _tac_table(tab):
    mom_cols = [c for c in tab.columns if c.startswith('mom_')]
    header = ['Asset', *[c.replace('mom_', '') + 'd' for c in mom_cols], 'Score', 'Rank']
    rows = [html.Tr([html.Td(idx)] + [html.Td(_fmt_pct(tab.loc[idx, c])) for c in mom_cols]
                    + [html.Td(_fmt_pct(tab.loc[idx, 'score'])), html.Td(str(int(tab.loc[idx, 'rank'])))])
            for idx in tab.index]
    return html.Table([html.Thead(html.Tr([html.Th(h) for h in header])), html.Tbody(rows)],
                      className='data-table', style={'width': '100%'})


@callback(
    Output('rg-tac-build', 'data'),
    Input('rg-tac-run', 'n_clicks'), State('rg-preset', 'value'),
    running=[(Output('rg-tac-run', 'disabled'), True, False),
             (Output('rg-tac-run', 'children'), 'Running…', 'Run tactical')],
    prevent_initial_call=True,
)
def rg_run_tac(_n, preset_days):
    end = datetime.date.today()
    start = None if not preset_days else end - datetime.timedelta(days=int(preset_days))
    try:
        tab = regime_service.tactical(start, end)
    except Exception as exc:
        logger.exception('tactical failed')
        return {'error': str(exc)}
    token = str(abs(hash((preset_days, str(start), str(end)))))
    _put(_TAC_CACHE, token, tab)
    return {'token': token}


@callback(Output('rg-tac-output', 'children'), Input('rg-tac-build', 'data'))
def rg_render_tac(store):
    if not store or 'error' in store:
        msg = store['error'] if store and 'error' in store else 'Run tactical to see the table.'
        return html.P(msg, className='no-data')
    if store.get('token') not in _TAC_CACHE:
        return html.P('Run tactical to see the table.', className='no-data')
    tab = _TAC_CACHE[store['token']]
    if tab is None or tab.empty:
        return html.P('No tactical data.', className='no-data')
    return [_row(_graph(_tac_fig(tab), basis='1 1 100%')), _tac_table(tab)]


# ── 5 Single-asset Markov regime ──────────────────────────────────────

_MK_CACHE: dict[str, regime_service.MarkovResult] = {}
_MK_STATE_COLOR = {'bear': _RED, 'sideways': _GREY, 'bull': _GREEN}
_MK_ORDER = ['bear', 'sideways', 'bull']


def _mk_transition_fig(r):
    P = r.transition
    if P is None or P.empty:
        return empty_figure('Transition matrix: n/a')
    order = [s for s in _MK_ORDER if s in P.index]
    P = P.reindex(index=order, columns=order)
    z = P.to_numpy()
    fig = go.Figure(go.Heatmap(z=z, x=order, y=order, colorscale='Blues', zmin=0, zmax=1,
                               text=np.round(z, 2), texttemplate='%{text}', showscale=False))
    fig.update_layout(base_chart_layout(title='Transition matrix (today→tomorrow)',
                                        xaxis_title='tomorrow', yaxis_title='today'), height=320)
    return fig


def _mk_nstep_fig(r):
    ns = r.n_step
    if ns is None or ns.empty:
        return empty_figure('Forecast: n/a')
    fig = go.Figure()
    for s in [c for c in _MK_ORDER if c in ns.columns]:
        fig.add_scatter(x=list(ns.index), y=ns[s].to_numpy(), mode='lines', name=s,
                        line={'color': _MK_STATE_COLOR.get(s, ACCENT)})
        if s in r.stationary.index:                       # stationary level (dashed)
            fig.add_hline(y=float(r.stationary[s]), line_dash='dot',
                          line_color=_MK_STATE_COLOR.get(s, MUTED))
    fig.update_layout(base_chart_layout(title=f'State forecast from "{r.current_state}" '
                                        f'(dashed = long-run / stationary)',
                                        xaxis_title='days ahead', yaxis_title='probability'),
                      height=320)
    return fig


def _mk_backtest_fig(r):
    bt = r.backtest
    if not bt or bt.get('strat_cumret') is None or bt['strat_cumret'].empty:
        return empty_figure('Walk-forward backtest: n/a')
    sc, hc = bt['strat_cumret'], bt['hold_cumret']
    fig = go.Figure([
        go.Scatter(x=list(hc.index), y=hc.to_numpy(), mode='lines', name='buy & hold',
                   line={'color': MUTED}),
        go.Scatter(x=list(sc.index), y=sc.to_numpy(), mode='lines', name='Markov signal',
                   line={'color': ACCENT}),
    ])
    fig.update_layout(base_chart_layout(title='Walk-forward: Markov signal vs buy & hold',
                                        yaxis_title='cum log ret'), height=340)
    return fig


def _build_mk_output(r):
    if r.states.empty:
        msg = '  •  '.join(r.warnings) if r.warnings else 'No Markov result.'
        return html.P(msg, className='no-data')
    bt = r.backtest
    ss, hs = bt.get('strat_sharpe', np.nan), bt.get('hold_sharpe', np.nan)
    persist = float(r.transition.loc[r.current_state, r.current_state]) \
        if r.current_state in r.transition.index else np.nan
    sig_dir = 'LONG' if (np.isfinite(r.signal) and r.signal > 0) else \
        ('SHORT' if (np.isfinite(r.signal) and r.signal < 0) else 'flat')
    chips = _row(
        _chip('Current state', r.current_state, _MK_STATE_COLOR.get(r.current_state, ACCENT)),
        _chip('Signal P(bull)−P(bear)', _fmt_num(r.signal, 2),
              _GREEN if (np.isfinite(r.signal) and r.signal > 0) else _RED),
        _chip('Direction', sig_dir),
        _chip('Persistence (stickiness)', _fmt_pct(persist)),
        _chip('HMM agreement', _fmt_pct(r.hmm_agree) if np.isfinite(r.hmm_agree) else '—'),
        _chip('Sampling', 'overlapping' if r.overlapping else 'non-overlap',
              _RED if r.overlapping else _GREEN),
    )
    bt_chips = _row(
        _chip('Strategy Sharpe', _fmt_num(ss),
              _GREEN if (np.isfinite(ss) and np.isfinite(hs) and ss > hs) else MUTED),
        _chip('Buy & hold Sharpe', _fmt_num(hs)),
    )
    note = html.P('Reminder: P(bull)−P(bear) is the raw signal; magnitude scales bet size. '
                  'If the strategy Sharpe does not beat buy & hold, the matrix is not adding '
                  'tradeable edge for this asset. Under overlapping sampling the absolute '
                  'Sharpe is inflated (overlapping forward returns understate variance) — '
                  'compare strategy-vs-hold, not the raw number.',
                  style={'color': MUTED, 'fontSize': '11px'})
    return [chips,
            _row(_graph(_mk_transition_fig(r)), _graph(_mk_nstep_fig(r))),
            bt_chips, _row(_graph(_mk_backtest_fig(r), basis='1 1 100%')), note]


@callback(
    Output('rg-mk-build', 'data'), Output('rg-mk-warnings', 'children'),
    Input('rg-mk-run', 'n_clicks'),
    State('rg-mk-ticker', 'value'), State('rg-mk-lookback', 'value'),
    State('rg-mk-threshold', 'value'), State('rg-mk-sampling', 'value'),
    State('rg-mk-hmm', 'value'), State('rg-mk-horizon', 'value'), State('rg-preset', 'value'),
    running=[(Output('rg-mk-run', 'disabled'), True, False),
             (Output('rg-mk-run', 'children'), 'Running…', 'Run Markov')],
    prevent_initial_call=True,
)
def rg_run_mk(_n, ticker, lookback, threshold, sampling, hmm, horizon, preset_days):
    if not ticker:
        return None, 'Pick an instrument.'
    end = datetime.date.today()
    start = None if not preset_days else end - datetime.timedelta(days=int(preset_days))
    lb = int(lookback or 20)
    thr = float(threshold or 5) / 100.0
    try:
        r = regime_service.markov_analysis(
            ticker, start, end, lookback=lb, threshold=thr,
            overlapping=(sampling == 'over'), hmm_overlay=bool(hmm), horizon=int(horizon))
    except Exception as exc:
        logger.exception('markov analysis failed')
        return None, f'Markov analysis failed: {exc}'
    token = str(abs(hash((ticker, lb, thr, sampling, bool(hmm), horizon, preset_days,
                          str(start), str(end)))))
    _put(_MK_CACHE, token, r)
    return {'token': token}, '  •  '.join(r.warnings)


@callback(Output('rg-mk-output', 'children'), Input('rg-mk-build', 'data'))
def rg_render_mk(store):
    if not store or store.get('token') not in _MK_CACHE:
        return html.P('Pick an instrument, set lookback/threshold, then Run Markov.',
                      className='no-data')
    return _build_mk_output(_MK_CACHE[store['token']])


@callback(
    Output('rg-pre-status', 'children'),
    Input('rg-precompute-btn', 'n_clicks'),
    State('rg-pre-variant', 'value'),
    State('rg-pre-start', 'value'),
    State('rg-pre-end', 'value'),
    running=[(Output('rg-precompute-btn', 'disabled'), True, False),
             (Output('rg-precompute-btn', 'children'), 'Running…', 'Precompute')],
    prevent_initial_call=True,
)
def rg_precompute(_n, variant, start_yr, end_yr):
    try:
        n = features_service.precompute(int(start_yr), int(end_yr), variant or 'A')
        return f'Done — {n} new snapshot(s) written.'
    except Exception as exc:
        logger.exception('precompute failed')
        return f'Failed: {exc}'

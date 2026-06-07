"""Analysis page: single-instrument return statistics + market-model relationship to a
benchmark/peers. Distribution & summary, time-series behavior, beta, residual diagnostics,
peers correlation. Reads only from `analysis_service` (services-only rule).
"""
import datetime
import logging

import dash
import numpy as np
import plotly.graph_objects as go
from dash import Input, Output, State, callback, dcc, html

from irp.ui.charts import base_chart_layout, corr_heatmap_figure, empty_figure, scatter_chart_layout
from irp.ui.services import analysis_service, risk_model_service
from irp.ui.theme import ACCENT, MUTED

dash.register_page(__name__, path='/analysis', name='Analysis')

logger = logging.getLogger(__name__)

_RESULT_CACHE: dict[str, analysis_service.AnalysisResult] = {}
_CACHE_CAP = 8

_FREQ_OPTIONS = [
    {'label': ' Daily', 'value': 'D'},
    {'label': ' Weekly', 'value': 'W'},
    {'label': ' Monthly', 'value': 'M'},
]
_PRESETS = [
    {'label': '1Y', 'value': 365}, {'label': '3Y', 'value': 1095},
    {'label': '5Y', 'value': 1825}, {'label': 'Max', 'value': 0},
]
_RED = '#e45756'
_GREEN = '#4ec94e'


def _cache_put(key, val):
    _RESULT_CACHE[key] = val
    while len(_RESULT_CACHE) > _CACHE_CAP:
        _RESULT_CACHE.pop(next(iter(_RESULT_CACHE)))


def _chip(label, value, color=ACCENT):
    return html.Div(style={'display': 'inline-flex', 'flexDirection': 'column',
                           'gap': '2px', 'marginRight': '14px', 'marginBottom': '6px'}, children=[
        html.Span(label, style={'color': MUTED, 'fontSize': '10px', 'textTransform': 'uppercase'}),
        html.Span(value, style={'color': color, 'fontSize': '15px', 'fontWeight': '600'}),
    ])


def _fmt_pct(x):
    return '—' if x is None or not np.isfinite(x) else f'{x * 100:.1f}%'


def _fmt_num(x, nd=2):
    return '—' if x is None or not np.isfinite(x) else f'{x:.{nd}f}'


def _dd(id_, multi=False, placeholder='', width='220px', clearable=False):
    return dcc.Dropdown(id=id_, options=[], placeholder=placeholder, multi=multi,
                        clearable=clearable, className='filter-dropdown', style={'minWidth': width})


def _field(label, control):
    return html.Div(style={'display': 'flex', 'flexDirection': 'column', 'gap': '2px'}, children=[
        html.Label(label, style={'color': MUTED, 'fontSize': '11px'}), control])


def _graph(fig, basis='1 1 calc(50% - 8px)'):
    return dcc.Graph(figure=fig, config={'displayModeBar': False},
                     style={'flex': basis, 'minWidth': '320px'})


def _row(*children):
    return html.Div(style={'display': 'flex', 'gap': '16px', 'flexWrap': 'wrap',
                           'marginBottom': '8px'}, children=list(children))


def _section(title):
    return html.H4(title, className='section-title', style={'margin': '18px 0 6px', 'color': ACCENT})


# ── layout ────────────────────────────────────────────────────────────

layout = html.Div(className='page', children=[
    dcc.Store(id='an-init', data=1),
    dcc.Store(id='an-build'),       # {token}
    dcc.Store(id='an-fm-build'),    # {token} for the factor model
    dcc.Store(id='an-pair-build'),  # {token} for the pair / cointegration

    html.H2('Analysis', className='page-title'),
    html.P('Return statistics for one instrument plus its relationship to a benchmark and '
           'peers: distribution / QQ / summary, drawdown & rolling vol, autocorrelation, '
           'market-model beta + residual diagnostics, and peer correlation.',
           style={'color': MUTED, 'fontSize': '13px'}),

    html.Div(className='control-row', style={'alignItems': 'flex-end', 'flexWrap': 'wrap',
                                             'gap': '12px'}, children=[
        _field('Instrument', _dd('an-ticker', placeholder='e.g. AAPL')),
        _field('Benchmark (index)', _dd('an-bench', placeholder='index', clearable=True)),
        _field('Peer sector', _dd('an-peer-sector', placeholder='all sectors',
                                  width='180px', clearable=True)),
        _field('Peers (corr)', _dd('an-peers', multi=True, placeholder='optional', width='260px')),
        _field('Frequency', dcc.RadioItems(id='an-freq', options=_FREQ_OPTIONS, value='D',
                                           inline=True, labelClassName='check-item')),
        _field('Period', dcc.RadioItems(id='an-preset', options=_PRESETS, value=1095,
                                        inline=True, labelClassName='check-item')),
        html.Button('Run', id='an-run', className='run-btn', n_clicks=0),
    ]),
    dcc.Loading(html.Div(id='an-warnings', style={'margin': '6px 0', 'fontSize': '12px',
                                                  'color': _RED})),
    dcc.Loading(html.Div(id='an-output'), type='default'),

    # ── Factor model (own Run; the universe factor-return build is slow) ──
    html.H3('Multi-factor risk model', className='section-title',
            style={'margin': '24px 0 4px', 'color': ACCENT,
                   'borderTop': '1px solid var(--border)', 'paddingTop': '16px'}),
    html.P('Decompose the instrument\'s return into exposures to systematic style factors '
           '(value, quality, momentum) built from the backtest long-short portfolios, plus a '
           'market factor = the Benchmark index you picked above (default ^SPX). Style factor '
           'returns are quarterly and universe-level — the build is slow the first time (then '
           'cached). Uses the Instrument above; pick a long period (Max) for a usable fit.',
           style={'color': MUTED, 'fontSize': '12px', 'maxWidth': '900px'}),
    html.Div(className='control-row', style={'alignItems': 'flex-end', 'gap': '12px'}, children=[
        _field('Rebalance', dcc.RadioItems(id='an-fm-rebalance',
                                           options=[{'label': ' Quarterly', 'value': 'Q'},
                                                    {'label': ' Annual', 'value': 'A'}],
                                           value='Q', inline=True, labelClassName='check-item')),
        html.Button('Run factor model', id='an-fm-run', className='run-btn', n_clicks=0),
    ]),
    dcc.Loading(html.Div(id='an-fm-warnings', style={'margin': '6px 0', 'fontSize': '12px',
                                                     'color': _RED})),
    dcc.Loading(html.Div(id='an-fm-output'), type='default'),

    # ── Pair / cointegration ─────────────────────────────────────────
    html.H3('Pair (cointegration)', className='section-title',
            style={'margin': '24px 0 4px', 'color': ACCENT,
                   'borderTop': '1px solid var(--border)', 'paddingTop': '16px'}),
    html.P('Statistical-arbitrage diagnostics for the Instrument above (A) versus a second '
           'instrument (B) on log prices: cointegration p-value (Engle-Granger), hedge ratio, '
           'spread z-score, mean-reversion half-life, and lead-lag. Low p (<0.05) + a short '
           'half-life = a tradeable mean-reverting pair (half-life is greyed out when p≥0.05). '
           'Descriptive only: hedge ratio + z-score are full-sample fits (look-ahead, not a PIT '
           'entry signal), and Engle-Granger is direction-dependent (A vs B order matters).',
           style={'color': MUTED, 'fontSize': '12px', 'maxWidth': '900px'}),
    html.Div(className='control-row', style={'alignItems': 'flex-end', 'gap': '12px'}, children=[
        _field('Instrument B', _dd('an-pair-b', placeholder='second instrument', clearable=True)),
        html.Button('Run pair', id='an-pair-run', className='run-btn', n_clicks=0),
    ]),
    dcc.Loading(html.Div(id='an-pair-warnings', style={'margin': '6px 0', 'fontSize': '12px',
                                                       'color': _RED})),
    dcc.Loading(html.Div(id='an-pair-output'), type='default'),
])


# ── populate instrument dropdowns ─────────────────────────────────────

@callback(
    Output('an-ticker', 'options'), Output('an-bench', 'options'),
    Output('an-peer-sector', 'options'), Output('an-peers', 'options'),
    Output('an-pair-b', 'options'),
    Input('an-init', 'data'),
)
def an_populate(_init):
    instruments = [{'label': t, 'value': t} for t in analysis_service._available_instruments()]
    return (instruments, analysis_service._benchmark_options(),
            analysis_service._sector_options(), analysis_service._peers_options(), instruments)


@callback(
    Output('an-peers', 'options', allow_duplicate=True),
    Output('an-peers', 'value'),
    Input('an-peer-sector', 'value'),
    prevent_initial_call=True,
)
def an_filter_peers(sector):
    """Restrict the peers list to the chosen sector (clears the current selection so a
    stale ticker from another sector can't linger)."""
    return analysis_service._peers_options(sector), []


# ── figure builders ───────────────────────────────────────────────────

def _hist_fig(res):
    counts, edges, x_norm, y_norm = res.hist
    if len(counts) == 0:
        return empty_figure(f'{res.ticker}: no returns')
    centers = (edges[:-1] + edges[1:]) / 2
    fig = go.Figure([
        go.Bar(x=centers, y=counts, marker_color=ACCENT, name='returns'),
        go.Scatter(x=x_norm, y=y_norm, mode='lines', line={'color': _RED}, name='normal'),
    ])
    fig.update_layout(base_chart_layout(title=f'{res.ticker} return distribution ({res.freq})',
                                        showlegend=False), height=300)
    return fig


def _qq_fig(theo, samp, slope, intercept, title):
    if theo is None or len(theo) == 0:
        return empty_figure(f'{title}: n/a')
    line = intercept + slope * theo
    fig = go.Figure([
        go.Scatter(x=theo, y=samp, mode='markers', marker={'color': ACCENT, 'size': 4}, name='sample'),
        go.Scatter(x=theo, y=line, mode='lines', line={'color': _RED}, name='normal'),
    ])
    fig.update_layout(scatter_chart_layout(title=title, xaxis_title='theoretical',
                                           yaxis_title='sample', showlegend=False), height=300)
    return fig


def _line_fig(s, title, color=ACCENT, fill=False, ytitle=''):
    if s is None or s.dropna().empty:
        return empty_figure(f'{title}: n/a')
    s = s.dropna()
    tr = go.Scatter(x=list(s.index), y=s.to_numpy(), mode='lines', line={'color': color},
                    fill='tozeroy' if fill else None)
    fig = go.Figure(tr)
    fig.update_layout(base_chart_layout(title=title, yaxis_title=ytitle), height=300)
    return fig


def _acf_fig(acf_tuple, title):
    if acf_tuple is None:
        return empty_figure(f'{title}: n/a')
    vals, confint = acf_tuple
    lags = np.arange(len(vals))
    band = confint[:, 1] - vals          # half-width of the 95% band
    fig = go.Figure(go.Bar(x=lags[1:], y=vals[1:], marker_color=ACCENT))
    fig.add_scatter(x=lags[1:], y=band[1:], mode='lines', line={'color': MUTED, 'dash': 'dot'},
                    name='+95%')
    fig.add_scatter(x=lags[1:], y=-band[1:], mode='lines', line={'color': MUTED, 'dash': 'dot'},
                    name='-95%')
    fig.update_layout(base_chart_layout(title=title, xaxis_title='lag', showlegend=False), height=300)
    return fig


def _scatter_fig(res):
    mm = res.market
    x = mm['bench_aligned'].to_numpy()
    y = mm['stock_aligned'].to_numpy()
    xs = np.array([x.min(), x.max()])
    fit = mm['intercept'] + mm['slope'] * xs
    fig = go.Figure([
        go.Scatter(x=x, y=y, mode='markers', marker={'color': ACCENT, 'size': 4, 'opacity': 0.5},
                   name='obs'),
        go.Scatter(x=xs, y=fit, mode='lines', line={'color': _RED},
                   name=f"β={mm['beta']:.2f}"),
    ])
    fig.update_layout(scatter_chart_layout(
        title=f'{res.ticker} vs {res.benchmark} returns',
        xaxis_title=f'{res.benchmark} return', yaxis_title=f'{res.ticker} return',
        showlegend=False), height=420, width=420)
    fig.update_yaxes(scaleanchor='x', scaleratio=1)   # equal x/y scale → slope reads as beta
    return fig


def _summary_chips(res):
    s = res.summary
    sharpe_c = _GREEN if (np.isfinite(s['sharpe']) and s['sharpe'] > 0) else _RED
    return _row(
        _chip('Ann. return', _fmt_pct(s['ann_return']),
              _GREEN if s['ann_return'] > 0 else _RED),
        _chip('Ann. vol', _fmt_pct(s['ann_vol'])),
        _chip('Sharpe', _fmt_num(s['sharpe']), sharpe_c),
        _chip('Skew', _fmt_num(s['skew'])),
        _chip('Excess kurt', _fmt_num(s['excess_kurtosis'])),
        _chip('VaR 95%', _fmt_pct(s['var95']), _RED),
        _chip('CVaR 95%', _fmt_pct(s['cvar95']), _RED),
        _chip('Hit rate', _fmt_pct(s['hit_rate'])),
        _chip('Obs', str(s['n'])),
        _chip('ADF p', _fmt_num(res.adf[1], 3)),
    )


def _beta_chips(res):
    mm = res.market
    return _row(
        _chip('Beta', _fmt_num(mm['beta'])),
        _chip('Alpha (ann.)', _fmt_pct(mm['alpha']),
              _GREEN if (np.isfinite(mm['alpha']) and mm['alpha'] > 0) else _RED),
        _chip('R²', _fmt_num(mm['r2'])),
        _chip('Residual vol', _fmt_pct(mm['resid_vol'])),
        _chip('Up capture', _fmt_num(mm['up_capture'])),
        _chip('Down capture', _fmt_num(mm['down_capture'])),
        _chip('Overlap obs', str(mm['n'])),
    )


def _build_output(res):
    blocks = [
        _section('1 · Distribution & summary'),
        _summary_chips(res),
        _row(_graph(_hist_fig(res)),
             _graph(_qq_fig(*res.qq, f'{res.ticker} QQ vs normal'))),

        _section('2 · Time-series behavior'),
        _row(_graph(_line_fig(res.cumulative, 'Cumulative log return', ytitle='cum log ret')),
             _graph(_line_fig(res.drawdown, 'Drawdown (underwater)', color=_RED, fill=True))),
        _row(_graph(_line_fig(res.rolling_vol, 'Rolling volatility (annualized)', ytitle='vol')),
             _graph(_acf_fig(res.acf, 'Return autocorrelation'))),
    ]

    if res.market and np.isfinite(res.market.get('beta', np.nan)):
        blocks += [
            _section(f'3 · Benchmark & beta (vs {res.benchmark})'),
            _beta_chips(res),
            _row(_graph(_scatter_fig(res), basis='0 0 auto'),
                 _graph(_line_fig(res.rolling_beta, 'Rolling beta', ytitle='beta'))),
            _section('4 · Residual diagnostics'),
            _row(_graph(_line_fig(res.market['residuals'], 'Market-model residuals', color=MUTED)),
                 _graph(_qq_fig(*(res.resid_qq or (None, None, np.nan, np.nan)), 'Residual QQ'))),
            _row(_graph(_acf_fig(res.resid_acf, 'Residual autocorrelation'),
                        basis='0 1 calc(50% - 8px)')),
        ]

    if res.peers_corr is not None and not res.peers_corr.empty:
        labels = list(res.peers_corr.columns)
        blocks += [
            _section('5 · Peer correlation'),
            _graph(corr_heatmap_figure(res.peers_corr, labels, 'Return correlation'), basis='1 1 100%'),
        ]
    return blocks


# ── run ───────────────────────────────────────────────────────────────

@callback(
    Output('an-build', 'data'),
    Output('an-warnings', 'children'),
    Input('an-run', 'n_clicks'),
    State('an-ticker', 'value'), State('an-bench', 'value'), State('an-peers', 'value'),
    State('an-freq', 'value'), State('an-preset', 'value'),
    running=[(Output('an-run', 'disabled'), True, False),
             (Output('an-run', 'children'), 'Running…', 'Run')],
    prevent_initial_call=True,
)
def an_run(_n, ticker, bench, peers, freq, preset_days):
    if not ticker:
        return None, 'Pick an instrument.'
    end = datetime.date.today()
    start = None if not preset_days else end - datetime.timedelta(days=int(preset_days))
    try:
        res = analysis_service._analyze(ticker, bench, peers, start, end, freq)
    except Exception as exc:
        logger.exception('analysis failed')
        return None, f'Analysis failed: {exc}'
    token = str(abs(hash((ticker, bench, tuple(peers or []), freq, preset_days,
                          str(start), str(end)))))
    _cache_put(token, res)
    warn = '  •  '.join(res.warnings) if res.warnings else ''
    return {'token': token}, warn


@callback(
    Output('an-output', 'children'),
    Input('an-build', 'data'),
)
def an_render(store):
    if not store or store.get('token') not in _RESULT_CACHE:
        return html.P('Pick an instrument (and optionally a benchmark / peers), then Run.',
                      className='no-data')
    return _build_output(_RESULT_CACHE[store['token']])


# ── factor model ──────────────────────────────────────────────────────

_FM_CACHE: dict[str, risk_model_service.RiskModelResult] = {}
_FM_HORIZON = {'Q': 63, 'A': 252}


def _fm_cache_put(key, val):
    _FM_CACHE[key] = val
    while len(_FM_CACHE) > _CACHE_CAP:
        _FM_CACHE.pop(next(iter(_FM_CACHE)))


def _exposure_fig(res):
    reg = res.regression
    factors = res.factors
    betas = [reg['betas'][f] for f in factors]
    tvals = [reg['tvalues'][f] for f in factors]
    colors = [_GREEN if abs(t) >= 2 else MUTED for t in tvals]   # |t|>=2 ~ significant
    text = [f'β={b:.2f}<br>t={t:.1f}' for b, t in zip(betas, tvals, strict=True)]
    fig = go.Figure(go.Bar(x=factors, y=betas, marker_color=colors, text=text,
                           textposition='outside'))
    fig.update_layout(base_chart_layout(title='Factor exposures (beta)', yaxis_title='beta',
                                        showlegend=False), height=320)
    return fig


def _rolling_exposures_fig(res):
    roll = res.rolling_exposures
    if roll is None or roll.dropna(how='all').empty:
        return empty_figure('Rolling exposures: n/a')
    roll = roll.dropna(how='all')
    fig = go.Figure()
    for f in res.factors:
        if f in roll.columns:
            fig.add_scatter(x=list(roll.index), y=roll[f].to_numpy(), mode='lines', name=f)
    fig.update_layout(base_chart_layout(title='Rolling factor betas', yaxis_title='beta'),
                      height=320)
    return fig


def _contrib_fig(series, title, ytitle):
    if series is None or series.empty:
        return empty_figure(f'{title}: n/a')
    vals = series.to_numpy()
    colors = [_GREEN if v >= 0 else _RED for v in vals]
    fig = go.Figure(go.Bar(x=list(series.index), y=vals, marker_color=colors))
    fig.update_layout(base_chart_layout(title=title, yaxis_title=ytitle, showlegend=False),
                      height=320)
    return fig


def _build_fm_output(res):
    if res.n == 0 or not res.regression:
        msg = '  •  '.join(res.warnings) if res.warnings else 'No factor-model result.'
        return html.P(msg, className='no-data')
    reg = res.regression
    chips = _row(
        _chip('Alpha (ann.)', _fmt_pct(reg['alpha']),
              _GREEN if (np.isfinite(reg['alpha']) and reg['alpha'] > 0) else _RED),
        _chip('R²', _fmt_num(reg['r2'])),
        _chip('Periods', str(res.n)),
        _chip('Rebalance', res.freq),
    )
    blocks = [
        chips,
        _row(_graph(_exposure_fig(res)), _graph(_rolling_exposures_fig(res))),
        _row(_graph(_contrib_fig(res.return_contrib, 'Return decomposition (avg per period)',
                                 'log ret')),
             _graph(_contrib_fig(res.risk_contrib, 'Risk attribution (variance share)', 'share'))),
    ]
    if res.factor_corr is not None and not res.factor_corr.empty:
        labels = list(res.factor_corr.columns)
        blocks.append(_graph(corr_heatmap_figure(res.factor_corr, labels, 'Factor-return correlation'),
                             basis='0 1 calc(50% - 8px)'))
    return blocks


@callback(
    Output('an-fm-build', 'data'),
    Output('an-fm-warnings', 'children'),
    Input('an-fm-run', 'n_clicks'),
    State('an-ticker', 'value'), State('an-bench', 'value'),
    State('an-preset', 'value'), State('an-fm-rebalance', 'value'),
    running=[(Output('an-fm-run', 'disabled'), True, False),
             (Output('an-fm-run', 'children'), 'Building…', 'Run factor model')],
    prevent_initial_call=True,
)
def an_run_fm(_n, ticker, bench, preset_days, rebalance):
    if not ticker:
        return None, 'Pick an instrument above.'
    end = datetime.date.today()
    # factor returns need history; default to 10y when the page period is "Max".
    days = int(preset_days) if preset_days else 3650
    start = end - datetime.timedelta(days=max(days, 1825))   # at least ~5y for a usable fit
    horizon = _FM_HORIZON.get(rebalance, 63)
    try:
        res = risk_model_service._risk_model(ticker, start, end, freq=rebalance,
                                             horizon=horizon, benchmark=bench)
    except Exception as exc:
        logger.exception('factor model failed')
        return None, f'Factor model failed: {exc}'
    token = str(abs(hash((ticker, bench, str(start), str(end), rebalance, horizon))))
    _fm_cache_put(token, res)
    warn = '  •  '.join(res.warnings) if res.warnings else ''
    return {'token': token}, warn


@callback(
    Output('an-fm-output', 'children'),
    Input('an-fm-build', 'data'),
)
def an_render_fm(store):
    if not store or store.get('token') not in _FM_CACHE:
        return html.P('Pick an instrument above, then Run factor model.', className='no-data')
    return _build_fm_output(_FM_CACHE[store['token']])


# ── pair / cointegration ──────────────────────────────────────────────

_PAIR_CACHE: dict[str, analysis_service.PairResult] = {}


def _pair_cache_put(key, val):
    _PAIR_CACHE[key] = val
    while len(_PAIR_CACHE) > _CACHE_CAP:
        _PAIR_CACHE.pop(next(iter(_PAIR_CACHE)))


def _zscore_fig(res):
    z = res.zscore.dropna()
    if z.empty:
        return empty_figure('Spread z-score: n/a')
    fig = go.Figure(go.Scatter(x=list(z.index), y=z.to_numpy(), mode='lines',
                               line={'color': ACCENT}))
    for lvl, dstyle in ((2, 'dash'), (-2, 'dash'), (0, 'dot')):
        fig.add_hline(y=lvl, line_dash=dstyle, line_color=MUTED)
    fig.update_layout(base_chart_layout(title='Spread z-score (±2σ bands)', yaxis_title='z'),
                      height=300)
    return fig


def _overlay_fig(res):
    o = res.overlay
    if o is None or o.empty:
        return empty_figure('Price overlay: n/a')
    fig = go.Figure([
        go.Scatter(x=list(o.index), y=o['a_norm'].to_numpy(), mode='lines', name=res.a,
                   line={'color': ACCENT}),
        go.Scatter(x=list(o.index), y=o['b_norm'].to_numpy(), mode='lines', name=res.b,
                   line={'color': '#e0a040'}),
    ])
    fig.update_layout(base_chart_layout(title=f'{res.a} vs {res.b} (rebased to 100)',
                                        yaxis_title='index'), height=300)
    return fig


def _hedge_scatter_fig(res):
    la, lb = res.log_a, res.log_b
    if la is None or la.empty:
        return empty_figure('Hedge scatter: n/a')
    eg = res.eg
    x = lb.to_numpy()
    y = la.to_numpy()
    xs = np.array([x.min(), x.max()])
    line = eg['const'] + eg['hedge_ratio'] * xs
    fig = go.Figure([
        go.Scatter(x=x, y=y, mode='markers', marker={'color': ACCENT, 'size': 4, 'opacity': 0.4},
                   name='obs'),
        go.Scatter(x=xs, y=line, mode='lines', line={'color': _RED},
                   name=f"hedge={eg['hedge_ratio']:.2f}"),
    ])
    fig.update_layout(scatter_chart_layout(title=f'log {res.a} vs log {res.b} (hedge fit)',
                                           xaxis_title=f'log {res.b}', yaxis_title=f'log {res.a}',
                                           showlegend=False), height=300)
    return fig


def _leadlag_fig(res):
    lags, xcorr, best = res.leadlag
    if not len(lags):
        return empty_figure('Lead-lag: n/a')
    colors = [_GREEN if lg == best else ACCENT for lg in lags]
    fig = go.Figure(go.Bar(x=list(lags), y=list(xcorr), marker_color=colors))
    fig.update_layout(base_chart_layout(
        title=f'Lead-lag cross-correlation (best lag {best:+d})',
        xaxis_title='lag (B leads A when >0)', yaxis_title='corr', showlegend=False), height=300)
    return fig


def _build_pair_output(res):
    if res.n < 20 or not res.eg:
        msg = '  •  '.join(res.warnings) if res.warnings else 'Not enough overlapping history.'
        return html.P(msg, className='no-data')
    eg = res.eg
    hl = res.half_life
    cointegrated = np.isfinite(eg['pvalue']) and eg['pvalue'] < 0.05
    # Half-life is only meaningful when the spread is cointegrated (p<0.05); grey it
    # out otherwise so a non-reverting spread isn't misread as tradeable.
    hl_txt = '—' if not np.isfinite(hl) else f'{hl:.0f}'
    chips = _row(
        _chip('Cointegration p', _fmt_num(eg['pvalue'], 3), _GREEN if cointegrated else _RED),
        _chip('Hedge ratio', _fmt_num(eg['hedge_ratio'])),
        _chip('Half-life (days)', hl_txt, ACCENT if cointegrated else MUTED),
        _chip('Best lead-lag', f"{res.leadlag[2]:+d}"),
        _chip('Days', str(res.n)),
    )
    return [
        chips,
        _row(_graph(_zscore_fig(res)), _graph(_overlay_fig(res))),
        _row(_graph(_hedge_scatter_fig(res)), _graph(_leadlag_fig(res))),
    ]


@callback(
    Output('an-pair-build', 'data'),
    Output('an-pair-warnings', 'children'),
    Input('an-pair-run', 'n_clicks'),
    State('an-ticker', 'value'), State('an-pair-b', 'value'), State('an-preset', 'value'),
    running=[(Output('an-pair-run', 'disabled'), True, False),
             (Output('an-pair-run', 'children'), 'Running…', 'Run pair')],
    prevent_initial_call=True,
)
def an_run_pair(_n, a, b, preset_days):
    if not a or not b:
        return None, 'Pick Instrument A (above) and Instrument B.'
    if a == b:
        return None, 'Pick two different instruments.'
    end = datetime.date.today()
    start = None if not preset_days else end - datetime.timedelta(days=int(preset_days))
    try:
        res = analysis_service._pair_analysis(a, b, start, end)
    except Exception as exc:
        logger.exception('pair analysis failed')
        return None, f'Pair analysis failed: {exc}'
    if res is None:
        return None, 'No overlapping price history for that pair.'
    token = str(abs(hash((a, b, preset_days, str(start), str(end)))))
    _pair_cache_put(token, res)
    warn = '  •  '.join(res.warnings) if res.warnings else ''
    return {'token': token}, warn


@callback(
    Output('an-pair-output', 'children'),
    Input('an-pair-build', 'data'),
)
def an_render_pair(store):
    if not store or store.get('token') not in _PAIR_CACHE:
        return html.P('Pick Instrument B, then Run pair.', className='no-data')
    return _build_pair_output(_PAIR_CACHE[store['token']])

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
from irp.ui.services import analysis_service
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
    dcc.Store(id='an-build'),     # {token}

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
])


# ── populate instrument dropdowns ─────────────────────────────────────

@callback(
    Output('an-ticker', 'options'), Output('an-bench', 'options'),
    Output('an-peer-sector', 'options'), Output('an-peers', 'options'),
    Input('an-init', 'data'),
)
def an_populate(_init):
    instruments = [{'label': t, 'value': t} for t in analysis_service._available_instruments()]
    return (instruments, analysis_service._benchmark_options(),
            analysis_service._sector_options(), analysis_service._peers_options())


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

"""Backtest page: factor IC series and quintile cumulative returns."""
import datetime
import logging
from math import isfinite, isnan
from typing import Any

import dash
import pandas as pd
import plotly.graph_objects as go
from dash import Input, Output, State, callback, dcc, html
from dash.exceptions import PreventUpdate

from irp.factors.compute import run_backtest
from irp.ui.factor_meta import FACTOR_OPTIONS
from irp.ui.theme import ACCENT, GRID, HOVER_LABEL, MUTED, TABLE_STYLE

dash.register_page(__name__, path='/backtest', name='Backtest')

logger = logging.getLogger(__name__)

_HORIZON_OPTIONS = [
    {'label': '1m  (21d)',  'value': 21},
    {'label': '3m  (63d)',  'value': 63},
    {'label': '6m  (126d)', 'value': 126},
    {'label': '12m (252d)', 'value': 252},
]
_VARIANT_OPTIONS  = [{'label': 'Annual', 'value': 'A'}, {'label': 'Quarterly', 'value': 'Q'}]
_FREQ_OPTIONS     = [{'label': 'Quarterly', 'value': 'Q'}, {'label': 'Annual', 'value': 'A'}]
_CURRENT_YEAR     = datetime.date.today().year
_QUINTILE_COLOURS = ['#e05252', '#e09952', MUTED, '#52a0e0', ACCENT]


def _empty_figure(message: str = 'No data') -> go.Figure:
    fig = go.Figure()
    fig.update_layout(
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)',
        annotations=[dict(
            text=message, x=0.5, y=0.5, xref='paper', yref='paper',
            showarrow=False, font=dict(color=MUTED, size=13),
        )],
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
        margin=dict(l=0, r=0, t=28, b=0),
        hovermode='x unified',
        xaxis=dict(gridcolor=GRID, linecolor=GRID, tickfont=dict(color=MUTED, size=11),
                   zeroline=False, showline=True),
        yaxis=dict(gridcolor=GRID, linecolor=GRID, tickfont=dict(color=MUTED, size=11),
                   zeroline=True, showline=False),
        legend=dict(bgcolor='rgba(0,0,0,0)', font=dict(color=MUTED, size=11)),
        **extra,
    )


def _stat_chip(label: str, value: str) -> html.Div:
    return html.Div(className='stat-chip', children=[
        html.Span(label, className='stat-label'),
        html.Span(value, className='stat-value'),
    ])


# ── Layout ────────────────────────────────────────────────────────────

layout = html.Div(className='backtest-page', children=[
    dcc.Store(id='backtest-store'),

    html.H2('Factor Backtest', className='page-title'),
    html.P(
        'Spearman IC series and equal-weight quintile cumulative returns '
        'for any factor over a historical window.',
        className='page-subtitle',
    ),

    # Controls
    html.Div(className='bt-controls', children=[
        html.Div(className='bt-control-group', children=[
            html.Label('Factor', className='control-label'),
            dcc.Dropdown(
                id='bt-factor',
                options=FACTOR_OPTIONS,
                value='pe',
                clearable=False,
                className='control-dropdown',
            ),
        ]),
        html.Div(className='bt-control-group', children=[
            html.Label('Horizon', className='control-label'),
            dcc.Dropdown(
                id='bt-horizon',
                options=_HORIZON_OPTIONS,
                value=252,
                clearable=False,
                className='control-dropdown',
            ),
        ]),
        html.Div(className='bt-control-group', children=[
            html.Label('Fundamentals', className='control-label'),
            dcc.Dropdown(
                id='bt-variant',
                options=_VARIANT_OPTIONS,
                value='A',
                clearable=False,
                className='control-dropdown',
            ),
        ]),
        html.Div(className='bt-control-group', children=[
            html.Label('Rebalance', className='control-label'),
            dcc.Dropdown(
                id='bt-freq',
                options=_FREQ_OPTIONS,
                value='Q',
                clearable=False,
                className='control-dropdown',
            ),
        ]),
        html.Div(className='bt-control-group bt-control-narrow', children=[
            html.Label('Start year', className='control-label'),
            dcc.Input(id='bt-start-year', type='number', value=2015,
                      min=2000, max=_CURRENT_YEAR - 1, step=1,
                      className='control-input'),
        ]),
        html.Div(className='bt-control-group bt-control-narrow', children=[
            html.Label('End year', className='control-label'),
            dcc.Input(id='bt-end-year', type='number', value=_CURRENT_YEAR,
                      min=2001, max=_CURRENT_YEAR, step=1,
                      className='control-input'),
        ]),
        html.Button('Run', id='bt-run-btn', className='run-btn', n_clicks=0),
    ]),

    # Summary chips (populated by callback)
    dcc.Loading(
        id='bt-loading',
        type='circle',
        color=ACCENT,
        children=html.Div(id='bt-summary', className='bt-summary'),
    ),

    # IC chart
    html.Div(className='chart-container', children=[
        html.H3('Information Coefficient (Spearman)', className='chart-title'),
        dcc.Graph(id='bt-ic-chart', figure=_empty_figure('Run backtest to see results'),
                  config={'displayModeBar': False}),
    ]),

    # Quintile chart
    html.Div(className='chart-container', children=[
        html.H3('Quintile Cumulative Log Returns', className='chart-title'),
        dcc.Graph(id='bt-quintile-chart', figure=_empty_figure('Run backtest to see results'),
                  config={'displayModeBar': False}),
    ]),
])


# ── Callbacks ─────────────────────────────────────────────────────────

@callback(
    Output('backtest-store', 'data'),
    Input('bt-run-btn', 'n_clicks'),
    State('bt-factor',     'value'),
    State('bt-horizon',    'value'),
    State('bt-variant',    'value'),
    State('bt-freq',       'value'),
    State('bt-start-year', 'value'),
    State('bt-end-year',   'value'),
    prevent_initial_call=True,
)
def run_bt(n_clicks, factor, horizon, variant, freq, start_yr, end_yr):
    if not n_clicks or not factor or not horizon:
        raise PreventUpdate
    try:
        start_yr = int(start_yr or 2015)
        end_yr   = int(end_yr   or _CURRENT_YEAR)
        result = run_backtest(
            factor=factor,
            horizon_days=int(horizon),
            start_date=datetime.date(start_yr, 1, 1),
            end_date=datetime.date(end_yr, 12, 31),
            variant=variant or 'A',
            freq=freq or 'Q',
        )
        # Serialise pandas objects → JSON-safe
        ic = result['ic_series']
        ic_records = [
            {'date': d.isoformat(), 'ic': None if isnan(v) else float(v)}
            for d, v in zip(ic.index, ic.values)
        ]
        qcr = result['quintile_cumret']
        qcr_records = []
        if not qcr.empty:
            qcr = qcr.reset_index()
            qcr.columns = ['date'] + list(qcr.columns[1:])
            for row in qcr.itertuples(index=False):
                rec = {'date': pd.Timestamp(row.date).isoformat()}
                for col in ['Q1', 'Q2', 'Q3', 'Q4', 'Q5']:
                    v = getattr(row, col, None)
                    rec[col] = None if v is None or (isinstance(v, float) and isnan(v)) else float(v)
                qcr_records.append(rec)
        mean_ic = result['mean_ic']
        ic_tstat = result['ic_tstat']
        return {
            'ic': ic_records,
            'qcr': qcr_records,
            'mean_ic': None if isnan(mean_ic) else float(mean_ic),
            'ic_tstat': None if isnan(ic_tstat) else float(ic_tstat),
            'n_dates': result['n_dates'],
            'factor': factor,
        }
    except Exception as exc:
        logger.exception(f'Backtest error: {exc}')
        return {'error': str(exc)}


@callback(
    Output('bt-summary', 'children'),
    Output('bt-ic-chart', 'figure'),
    Input('backtest-store', 'data'),
)
def render_ic(data):
    if not data or 'error' in (data or {}):
        msg = data.get('error', 'No data') if data else 'No data'
        return html.Span(msg, style={'color': 'red'}), _empty_figure(msg)

    ic_records = data.get('ic', [])
    mean_ic    = data.get('mean_ic')
    ic_tstat   = data.get('ic_tstat')
    n_dates    = data.get('n_dates', 0)

    # Summary chips
    def _fmt_ic(v):
        return f'{v:.3f}' if v is not None and isfinite(v) else 'N/A'

    chips = html.Div(className='stat-chips', children=[
        _stat_chip('Mean IC',   _fmt_ic(mean_ic)),
        _stat_chip('IC t-stat', _fmt_ic(ic_tstat) if ic_tstat is not None else 'N/A'),
        _stat_chip('Dates',     str(n_dates)),
    ])

    if not ic_records:
        return chips, _empty_figure('No IC data')

    dates  = [r['date'] for r in ic_records]
    ic_vals = [r['ic'] for r in ic_records]
    colours = [ACCENT if (v is not None and v >= 0) else '#e05252' for v in ic_vals]

    fig = go.Figure(layout=_chart_layout(title=dict(text='', x=0)))
    fig.add_trace(go.Bar(
        x=dates, y=ic_vals,
        marker_color=colours,
        name='IC',
        hovertemplate='%{x}<br>IC: %{y:.3f}<extra></extra>',
    ))
    if mean_ic is not None and isfinite(mean_ic):
        fig.add_hline(
            y=mean_ic,
            line_dash='dash',
            line_color=MUTED,
            annotation_text=f'Mean {mean_ic:.3f}',
            annotation_font_color=MUTED,
        )
    fig.update_layout(yaxis_title='Spearman IC', showlegend=False)
    return chips, fig


@callback(
    Output('bt-quintile-chart', 'figure'),
    Input('backtest-store', 'data'),
)
def render_quintiles(data):
    if not data or 'error' in (data or {}) or not data.get('qcr'):
        return _empty_figure('No quintile data')

    qcr_records = data['qcr']
    df = pd.DataFrame(qcr_records)
    df['date'] = pd.to_datetime(df['date'])

    fig = go.Figure(layout=_chart_layout())
    for i, col in enumerate(['Q1', 'Q2', 'Q3', 'Q4', 'Q5']):
        if col not in df.columns:
            continue
        fig.add_trace(go.Scatter(
            x=df['date'], y=df[col],
            mode='lines',
            name=col,
            line=dict(color=_QUINTILE_COLOURS[i], width=1.5 if col in ('Q1', 'Q5') else 1),
            hovertemplate=f'{col}: %{{y:.3f}}<extra></extra>',
        ))
    fig.update_layout(yaxis_title='Cumulative log return')
    return fig

"""Backtest page: single-factor IC/quintile analysis and multi-factor composite models."""

import datetime
import logging
from math import isfinite, isnan
from typing import Any

import dash
import pandas as pd
import plotly.graph_objects as go
from dash import Input, Output, State, callback, dcc, html
from dash.exceptions import PreventUpdate

from irp.core.config import config
from irp.features.composite import PRESETS
from irp.ui.charts import base_chart_layout as _chart_layout, empty_figure
from irp.ui.factor_meta import FACTOR_OPTIONS
from irp.ui.services import backtest_service, universe_service, watchlist_service
from irp.ui.theme import ACCENT, MUTED

dash.register_page(__name__, path='/backtest', name='Backtest')

logger = logging.getLogger(__name__)

_HORIZON_OPTIONS = [
    {'label': '1m  (21d)', 'value': 21},
    {'label': '3m  (63d)', 'value': 63},
    {'label': '6m  (126d)', 'value': 126},
    {'label': '12m (252d)', 'value': 252},
]
_VARIANT_OPTIONS = [
    {'label': 'Annual', 'value': 'A'},
    {'label': 'Quarterly', 'value': 'Q'},
]
_FREQ_OPTIONS = [
    {'label': 'Quarterly', 'value': 'Q'},
    {'label': 'Annual', 'value': 'A'},
]
_CURRENT_YEAR = datetime.date.today().year
_QUINTILE_COLOURS = ['#e05252', '#e09952', MUTED, '#52a0e0', ACCENT]

_PRESET_OPTIONS = [
    {'label': 'Value  (P/E + P/B + P/S + FCF Yield)', 'value': 'value'},
    {'label': 'Quality  (ROE + ROIC + Gross Margin + FCF Margin)', 'value': 'quality'},
    {'label': 'Momentum  (12-1m + 6-1m)', 'value': 'momentum'},
    {'label': 'Composite  (Value + Quality + Momentum)', 'value': 'composite'},
]
_NORMALIZE_OPTIONS = [
    {'label': 'Z-score', 'value': 'zscore'},
    {'label': 'Rank norm', 'value': 'rank'},
]

_HIDE = {'display': 'none'}
_SHOW = {
    'display': 'flex',
    'flexWrap': 'wrap',
    'gap': '12px',
    'alignItems': 'flex-end',
    'width': '100%',
}


def _stat_chip(label: str, value: str, value_color: str | None = None) -> html.Div:
    value_style = {'color': value_color} if value_color else {}
    return html.Div(
        className='stat-chip',
        children=[
            html.Span(label, className='stat-label'),
            html.Span(value, className='stat-value', style=value_style),
        ],
    )


def _filtered_tickers(market: str, sector: str | None) -> list[str] | None:
    """Backwards-compatible shim — delegates to universe_service.filter_tickers."""
    return universe_service._filter_tickers(market=market or None, sector=sector)


# ── Layout ────────────────────────────────────────────────────────────

layout = html.Div(
    className='backtest-page',
    children=[
        dcc.Store(id='backtest-store'),
        dcc.Store(id='backtest-init', data=1),
        dcc.Store(id='backtest-wl-trigger', data=0),
        dcc.Store(id='bt-decay-store'),
        html.H2('Factor Backtest', className='page-title'),
        html.P(
            'Spearman IC series and equal-weight quintile cumulative returns '
            'for any factor or composite model over a historical window.',
            className='page-subtitle',
        ),
        # Mode selector tabs
        dcc.Tabs(
            id='bt-mode-tabs',
            value='single',
            className='bt-mode-tabs',
            children=[
                dcc.Tab(
                    label='Single Factor',
                    value='single',
                    className='bt-mode-tab',
                    selected_className='bt-mode-tab--active',
                ),
                dcc.Tab(
                    label='Composite Model',
                    value='composite',
                    className='bt-mode-tab',
                    selected_className='bt-mode-tab--active',
                ),
            ],
        ),
        # Controls
        html.Div(
            className='bt-controls',
            children=[
                # Row 1: Signal
                html.Div(
                    className='control-row',
                    children=[
                        html.Span('Signal', className='control-group-label'),
                        html.Div(
                            id='bt-single-controls',
                            style=_SHOW,
                            children=[
                                html.Div(
                                    className='bt-control-group',
                                    children=[
                                        html.Label('Factor', className='control-label'),
                                        dcc.Dropdown(
                                            id='bt-factor',
                                            options=FACTOR_OPTIONS,
                                            value='pe',
                                            clearable=False,
                                            className='control-dropdown',
                                        ),
                                    ],
                                ),
                            ],
                        ),
                        html.Div(
                            id='bt-composite-controls',
                            style=_HIDE,
                            children=[
                                html.Div(
                                    className='bt-control-group',
                                    children=[
                                        html.Label('Preset', className='control-label'),
                                        dcc.Dropdown(
                                            id='bt-preset',
                                            options=_PRESET_OPTIONS,
                                            value='composite',
                                            clearable=False,
                                            className='control-dropdown',
                                            style={'minWidth': '260px'},
                                        ),
                                    ],
                                ),
                                html.Div(
                                    className='bt-control-group',
                                    children=[
                                        html.Label(
                                            'Normalize', className='control-label'
                                        ),
                                        dcc.Dropdown(
                                            id='bt-normalize',
                                            options=_NORMALIZE_OPTIONS,
                                            value='zscore',
                                            clearable=False,
                                            className='control-dropdown',
                                        ),
                                    ],
                                ),
                                dcc.Checklist(
                                    id='bt-sector-neutral',
                                    options=[
                                        {
                                            'label': ' Sector neutral',
                                            'value': 'sector_neutral',
                                        }
                                    ],
                                    value=[],
                                    labelClassName='check-item',
                                    style={
                                        'alignSelf': 'flex-end',
                                        'paddingBottom': '6px',
                                    },
                                ),
                            ],
                        ),
                    ],
                ),
                # Row 2: Universe + Period + Run
                html.Div(
                    className='control-row',
                    children=[
                        html.Span('Universe', className='control-group-label'),
                        html.Div(
                            className='bt-control-group',
                            children=[
                                html.Label('Market', className='control-label'),
                                dcc.Dropdown(
                                    id='bt-market',
                                    options=[
                                        {'label': 'All markets', 'value': ''},
                                        {'label': 'Stocks only', 'value': 'stocks'},
                                        {'label': 'ETFs only', 'value': 'etfs'},
                                    ],
                                    value='',
                                    clearable=False,
                                    className='control-dropdown',
                                ),
                            ],
                        ),
                        html.Div(
                            className='bt-control-group',
                            children=[
                                html.Label('Sector', className='control-label'),
                                dcc.Dropdown(
                                    id='bt-sector',
                                    placeholder='All sectors',
                                    clearable=True,
                                    className='control-dropdown',
                                    style={'minWidth': '170px'},
                                ),
                            ],
                        ),
                        html.Div(
                            className='bt-control-group',
                            children=[
                                html.Label('Watchlist', className='control-label'),
                                dcc.Dropdown(
                                    id='bt-watchlist',
                                    placeholder='None (use market/sector)',
                                    clearable=True,
                                    className='control-dropdown',
                                    style={'minWidth': '190px'},
                                ),
                            ],
                        ),
                        html.Div(className='control-row-sep'),
                        html.Span('Period', className='control-group-label'),
                        html.Div(
                            className='bt-control-group',
                            children=[
                                html.Label('Horizon', className='control-label'),
                                dcc.Dropdown(
                                    id='bt-horizon',
                                    options=_HORIZON_OPTIONS,
                                    value=252,
                                    clearable=False,
                                    className='control-dropdown',
                                ),
                            ],
                        ),
                        html.Div(
                            className='bt-control-group',
                            children=[
                                html.Label('Filing', className='control-label'),
                                dcc.Dropdown(
                                    id='bt-variant',
                                    options=_VARIANT_OPTIONS,
                                    value='A',
                                    clearable=False,
                                    className='control-dropdown',
                                ),
                            ],
                        ),
                        html.Div(
                            className='bt-control-group',
                            children=[
                                html.Label('Rebalance', className='control-label'),
                                dcc.Dropdown(
                                    id='bt-freq',
                                    options=_FREQ_OPTIONS,
                                    value='Q',
                                    clearable=False,
                                    className='control-dropdown',
                                ),
                            ],
                        ),
                        html.Div(
                            className='bt-control-group bt-control-narrow',
                            children=[
                                html.Label('Start year', className='control-label'),
                                dcc.Input(
                                    id='bt-start-year',
                                    type='number',
                                    value=1995,
                                    min=1991,
                                    max=_CURRENT_YEAR - 1,
                                    step=1,
                                    className='control-input',
                                ),
                            ],
                        ),
                        html.Div(
                            className='bt-control-group bt-control-narrow',
                            children=[
                                html.Label('End year', className='control-label'),
                                dcc.Input(
                                    id='bt-end-year',
                                    type='number',
                                    value=_CURRENT_YEAR,
                                    min=2001,
                                    max=_CURRENT_YEAR,
                                    step=1,
                                    className='control-input',
                                ),
                            ],
                        ),
                        html.Div(
                            className='bt-control-group bt-control-narrow',
                            children=[
                                html.Label('Cost (bps)', className='control-label'),
                                dcc.Input(
                                    id='bt-cost-bps',
                                    type='number',
                                    value=config.factors.default_cost_bps,
                                    min=0,
                                    max=200,
                                    step=1,
                                    className='control-input',
                                ),
                            ],
                        ),
                        html.Button(
                            'Run', id='bt-run-btn', className='run-btn', n_clicks=0
                        ),
                    ],
                ),
            ],
        ),
        # Summary chips (populated by callback)
        dcc.Loading(
            id='bt-loading',
            type='circle',
            color=ACCENT,
            children=html.Div(id='bt-summary', className='bt-summary'),
        ),
        # IC chart
        html.Div(
            className='chart-container',
            children=[
                html.H3('Information Coefficient (Spearman)', className='chart-title'),
                dcc.Graph(
                    id='bt-ic-chart',
                    figure=empty_figure('Run backtest to see results'),
                    config={'displayModeBar': False},
                ),
            ],
        ),
        # Quintile chart
        html.Div(
            className='chart-container',
            children=[
                html.H3('Quintile Cumulative Log Returns', className='chart-title'),
                dcc.Graph(
                    id='bt-quintile-chart',
                    figure=empty_figure('Run backtest to see results'),
                    config={'displayModeBar': False},
                ),
            ],
        ),
        # Spread chart
        html.Div(
            className='chart-container',
            children=[
                html.H3('Quintile Excess Return vs EW', className='chart-title'),
                dcc.Graph(
                    id='bt-spread-chart',
                    figure=empty_figure('Run backtest to see results'),
                    config={'displayModeBar': False},
                ),
            ],
        ),
        # Factor Decay section
        html.Div(
            className='chart-container',
            children=[
                html.H3('Factor Decay', className='chart-title'),
                html.P(
                    'IC and ICIR at multiple horizons — shows how quickly the signal decays.',
                    className='page-subtitle',
                    style={'marginBottom': '8px'},
                ),
                html.Div(
                    className='control-row',
                    style={'marginBottom': '12px'},
                    children=[
                        dcc.Checklist(
                            id='bt-decay-horizons',
                            options=[
                                {'label': ' 1m (21d)', 'value': 21},
                                {'label': ' 3m (63d)', 'value': 63},
                                {'label': ' 6m (126d)', 'value': 126},
                                {'label': ' 12m (252d)', 'value': 252},
                            ],
                            value=config.factors.decay_horizons,
                            inline=True,
                            labelClassName='check-item',
                            style={'alignSelf': 'center'},
                        ),
                        html.Button(
                            'Run Decay',
                            id='bt-decay-btn',
                            className='run-btn',
                            n_clicks=0,
                            style={'marginLeft': '16px'},
                        ),
                    ],
                ),
                dcc.Loading(
                    type='circle',
                    color=ACCENT,
                    children=dcc.Graph(
                        id='bt-decay-chart',
                        figure=empty_figure('Run decay analysis to see results'),
                        config={'displayModeBar': False},
                    ),
                ),
            ],
        ),
    ],
)


# ── Callbacks ─────────────────────────────────────────────────────────


@callback(
    Output('bt-sector', 'options'),
    Output('bt-watchlist', 'options'),
    Input('backtest-init', 'data'),
    Input('backtest-wl-trigger', 'data'),
)
def load_sector_options(_: Any, _wl: Any) -> tuple[list, list]:
    sectors = universe_service._get_sectors()
    sector_opts = [{'label': s, 'value': s} for s in sectors]
    wl_df = watchlist_service.list_watchlists()
    wl_opts = (
        [
            {'label': f'{r["name"]} ({r["n"]})', 'value': r['name']}
            for _, r in wl_df.iterrows()
        ]
        if not wl_df.empty
        else []
    )
    return sector_opts, wl_opts


@callback(
    Output('bt-sector', 'disabled'),
    Output('bt-market', 'disabled'),
    Input('bt-watchlist', 'value'),
)
def toggle_bt_filters(watchlist: str | None) -> tuple[bool, bool]:
    active = bool(watchlist)
    return active, active


@callback(
    Output('bt-single-controls', 'style'),
    Output('bt-composite-controls', 'style'),
    Input('bt-mode-tabs', 'value'),
)
def toggle_mode_controls(mode: str) -> tuple[dict, dict]:
    if mode == 'composite':
        return _HIDE, _SHOW
    return _SHOW, _HIDE


@callback(
    Output('backtest-store', 'data'),
    Input('bt-run-btn', 'n_clicks'),
    State('bt-mode-tabs', 'value'),
    State('bt-factor', 'value'),
    State('bt-preset', 'value'),
    State('bt-normalize', 'value'),
    State('bt-sector-neutral', 'value'),
    State('bt-horizon', 'value'),
    State('bt-variant', 'value'),
    State('bt-freq', 'value'),
    State('bt-start-year', 'value'),
    State('bt-end-year', 'value'),
    State('bt-market', 'value'),
    State('bt-sector', 'value'),
    State('bt-watchlist', 'value'),
    State('bt-cost-bps', 'value'),
    running=[
        (Output('bt-run-btn', 'disabled'), True, False),
        (Output('bt-run-btn', 'children'), 'Running...', 'Run'),
    ],
    prevent_initial_call=True,
)
def run_bt(
    n_clicks,
    mode,
    factor,
    preset,
    normalize,
    sector_neutral,
    horizon,
    variant,
    freq,
    start_yr,
    end_yr,
    market,
    sector,
    watchlist,
    cost_bps_input,
):
    if not n_clicks or not horizon:
        raise PreventUpdate
    try:
        start_yr = int(start_yr or 2015)
        end_yr = int(end_yr or _CURRENT_YEAR)
        cost_bps = float(cost_bps_input or 0)
        if end_yr <= start_yr:
            return {
                'error': f'End year ({end_yr}) must be after start year ({start_yr})'
            }
        if watchlist:
            try:
                # Validate watchlist exists; service handles the actual filtering
                watchlist_service.load_watchlist(watchlist)
            except KeyError:
                return {'error': f'Watchlist "{watchlist}" not found.'}

        if mode == 'composite':
            weights = PRESETS.get(preset or 'composite', PRESETS['composite'])
            result = backtest_service._run_composite(
                weights=weights,
                horizon_days=int(horizon),
                start_date=datetime.date(start_yr, 1, 1),
                end_date=datetime.date(end_yr, 12, 31),
                variant=variant or 'A',
                freq=freq or 'Q',
                normalize=normalize or 'zscore',
                use_sector_neutral=bool(sector_neutral),
                cost_bps=cost_bps,
                market=market or None,
                sector=sector,
                watchlist=watchlist,
            )
            label = f'{preset or "composite"} ({normalize or "zscore"}{"  sector-neutral" if sector_neutral else ""})'
        else:
            if not factor:
                raise PreventUpdate
            result = backtest_service._run_factor(
                factor=factor,
                horizon_days=int(horizon),
                start_date=datetime.date(start_yr, 1, 1),
                end_date=datetime.date(end_yr, 12, 31),
                variant=variant or 'A',
                freq=freq or 'Q',
                cost_bps=cost_bps,
                market=market or None,
                sector=sector,
                watchlist=watchlist,
            )
            label = factor

        def _ser_cumret(ser_or_df):
            if ser_or_df is None or (hasattr(ser_or_df, 'empty') and ser_or_df.empty):
                return []
            if isinstance(ser_or_df, pd.Series):
                return [
                    {
                        'date': pd.Timestamp(d).isoformat(),
                        'v': None if (isinstance(v, float) and isnan(v)) else float(v),
                    }
                    for d, v in zip(ser_or_df.index, ser_or_df.values, strict=False)
                ]
            df = ser_or_df.reset_index()
            df.columns = ['date'] + list(df.columns[1:])
            rows_out = []
            for row in df.itertuples(index=False):
                rec: dict[str, Any] = {'date': pd.Timestamp(row.date).isoformat()}
                for c in ['Q1', 'Q2', 'Q3', 'Q4', 'Q5']:
                    v = getattr(row, c, None)
                    rec[c] = (
                        None
                        if v is None or (isinstance(v, float) and isnan(v))
                        else float(v)
                    )
                rows_out.append(rec)
            return rows_out

        def _safe(v):
            return (
                None
                if v is None or (isinstance(v, float) and (isnan(v) or not isfinite(v)))
                else float(v)
            )

        ic = result['ic_series']
        filtered = universe_service._filter_tickers(
            market=market or None,
            sector=sector,
            watchlist=watchlist,
        )
        n_tickers = len(filtered) if filtered is not None else None
        mean_ic = result['mean_ic']
        ic_tstat = result['ic_tstat']
        icir = result.get('icir', float('nan'))

        q_ann_ret = {q: _safe(v) for q, v in result.get('quintile_ann_ret', {}).items()}
        q_sharpe = {q: _safe(v) for q, v in result.get('quintile_sharpe', {}).items()}
        q_max_dd = {q: _safe(v) for q, v in result.get('quintile_max_dd', {}).items()}

        ls = result.get('ls_cumret', pd.Series(dtype=float))
        ls_records = []
        if not ls.empty:
            for d, v in zip(ls.index, ls.values, strict=False):
                ls_records.append({'date': pd.Timestamp(d).isoformat(), 'v': _safe(v)})

        # Annualized L/S return from Q1 and Q5 annualized returns
        q1_ann = q_ann_ret.get('Q1')
        q5_ann = q_ann_ret.get('Q5')
        ls_ann_ret = (
            (q1_ann - q5_ann) if (q1_ann is not None and q5_ann is not None) else None
        )

        return {
            'ic': [
                {'date': d.isoformat(), 'ic': _safe(v)}
                for d, v in zip(ic.index, ic.values, strict=False)
            ],
            'qcr': _ser_cumret(result['quintile_cumret']),
            'qcr_net': _ser_cumret(result.get('quintile_cumret_net')),
            'ew': [
                {'date': pd.Timestamp(d).isoformat(), 'ew': _safe(v)}
                for d, v in zip(
                    result.get('ew_cumret', pd.Series()).index,
                    result.get('ew_cumret', pd.Series()).values, strict=False,
                )
            ],
            'ls': ls_records,
            'mean_ic': _safe(mean_ic),
            'ic_tstat': _safe(ic_tstat),
            'icir': _safe(icir),
            'n_dates': result['n_dates'],
            'n_tickers': n_tickers,
            'label': label,
            'q_ann_ret': q_ann_ret,
            'q_sharpe': q_sharpe,
            'q_max_dd': q_max_dd,
            'mean_turnover_q1': _safe(result.get('mean_turnover_q1', float('nan'))),
            'mean_turnover_q5': _safe(result.get('mean_turnover_q5', float('nan'))),
            'ls_ann_ret': ls_ann_ret,
            'cost_bps': cost_bps,
        }
    except PreventUpdate:
        raise
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
        return html.Span(msg, style={'color': 'red'}), empty_figure(msg)

    ic_records = data.get('ic', [])
    mean_ic = data.get('mean_ic')
    ic_tstat = data.get('ic_tstat')
    n_dates = data.get('n_dates', 0)
    n_tickers = data.get('n_tickers')

    def _fmt_ic(v):
        return f'{v:.3f}' if v is not None and isfinite(v) else 'N/A'

    ic_color = (
        ACCENT
        if (mean_ic is not None and isfinite(mean_ic) and mean_ic > 0.02)
        else '#e05252'
        if (mean_ic is not None and isfinite(mean_ic) and mean_ic < -0.02)
        else MUTED
    )
    tstat_color = (
        '#3fb950'
        if (ic_tstat is not None and isfinite(ic_tstat) and abs(ic_tstat) > 2)
        else '#d29922'
        if (ic_tstat is not None and isfinite(ic_tstat) and abs(ic_tstat) >= 1)
        else '#e05252'
        if (ic_tstat is not None and isfinite(ic_tstat))
        else MUTED
    )
    icir = data.get('icir')
    q_sharpe = data.get('q_sharpe', {})
    q_ann_ret = data.get('q_ann_ret', {})
    mean_to_q1 = data.get('mean_turnover_q1')
    ls_ann_ret = data.get('ls_ann_ret')
    cost_bps = data.get('cost_bps', 0)

    def _fmt_pct(v, decimals=1):
        return f'{v * 100:+.{decimals}f}%' if v is not None and isfinite(v) else 'N/A'

    icir_color = (
        '#3fb950'
        if (icir is not None and isfinite(icir) and icir > 0.5)
        else '#d29922'
        if (icir is not None and isfinite(icir) and icir > 0.3)
        else '#e05252'
        if (icir is not None and isfinite(icir))
        else MUTED
    )
    tickers_label = str(n_tickers) if n_tickers is not None else 'All'
    chips_row1 = [
        _stat_chip('Mean IC', _fmt_ic(mean_ic), ic_color),
        _stat_chip(
            'IC t-stat',
            _fmt_ic(ic_tstat) if ic_tstat is not None else 'N/A',
            tstat_color,
        ),
        _stat_chip('ICIR', _fmt_ic(icir), icir_color),
        _stat_chip('Dates', str(n_dates)),
        _stat_chip('Tickers', tickers_label),
    ]

    q1_sharpe = q_sharpe.get('Q1')
    q5_sharpe = q_sharpe.get('Q5')
    q1_sharpe_color = (
        '#3fb950'
        if (q1_sharpe is not None and isfinite(q1_sharpe) and q1_sharpe > 0.5)
        else '#d29922'
        if (q1_sharpe is not None and isfinite(q1_sharpe) and q1_sharpe > 0)
        else '#e05252'
        if (q1_sharpe is not None and isfinite(q1_sharpe))
        else MUTED
    )
    chips_row2 = [
        _stat_chip(
            'Q1 Ann Ret',
            _fmt_pct(q_ann_ret.get('Q1')),
            '#3fb950' if (q_ann_ret.get('Q1') or 0) > 0 else '#e05252',
        ),
        _stat_chip('Q5 Ann Ret', _fmt_pct(q_ann_ret.get('Q5'))),
        _stat_chip('Q1 Sharpe', _fmt_ic(q1_sharpe), q1_sharpe_color),
        _stat_chip('Q5 Sharpe', _fmt_ic(q5_sharpe)),
        _stat_chip(
            'L/S Ann Ret',
            _fmt_pct(ls_ann_ret),
            '#3fb950' if (ls_ann_ret or 0) > 0 else '#e05252',
        ),
        _stat_chip(
            'Q1 Turnover',
            f'{mean_to_q1 * 100:.0f}%'
            if mean_to_q1 is not None and isfinite(mean_to_q1)
            else 'N/A',
        ),
    ]
    if cost_bps:
        chips_row2.append(_stat_chip('Cost bps', str(int(cost_bps)), MUTED))

    chips = html.Div(
        children=[
            html.Div(className='stat-chips', children=chips_row1),
            html.Div(
                className='stat-chips', style={'marginTop': '6px'}, children=chips_row2
            ),
        ],
    )

    if not ic_records:
        return chips, empty_figure('No IC data')

    dates = [r['date'] for r in ic_records]
    ic_vals = [r['ic'] for r in ic_records]
    colours = [ACCENT if (v is not None and v >= 0) else '#e05252' for v in ic_vals]

    fig = go.Figure(layout=_chart_layout(title=dict(text='', x=0)))
    fig.add_trace(
        go.Bar(
            x=dates,
            y=ic_vals,
            marker_color=colours,
            name='IC',
            hovertemplate='%{x}<br>IC: %{y:.3f}<extra></extra>',
        )
    )
    rolling = (
        pd.Series(ic_vals, index=pd.to_datetime(dates)).rolling(4, min_periods=2).mean()
    )
    fig.add_trace(
        go.Scatter(
            x=[d.isoformat() for d in rolling.index],
            y=rolling.tolist(),
            mode='lines',
            name='4-period rolling',
            line=dict(color=MUTED, width=1.5, dash='dot'),
            hovertemplate='%{x}<br>Rolling IC: %{y:.3f}<extra></extra>',
        )
    )
    if mean_ic is not None and isfinite(mean_ic):
        fig.add_hline(
            y=mean_ic,
            line_dash='dash',
            line_color=MUTED,
            annotation_text=f'Mean {mean_ic:.3f}',
            annotation_font_color=MUTED,
        )
    fig.add_hrect(
        y0=0.05, y1=1.0, fillcolor='rgba(63,185,80,0.04)', line_width=0, layer='below'
    )
    fig.add_hrect(
        y0=-1.0, y1=-0.05, fillcolor='rgba(224,82,82,0.04)', line_width=0, layer='below'
    )
    fig.add_hline(
        y=0.05,
        line_dash='dot',
        line_color='rgba(63,185,80,0.25)',
        line_width=1,
        annotation_text='0.05',
        annotation_font_color='rgba(63,185,80,0.5)',
        annotation_font_size=9,
        annotation_position='top right',
    )
    fig.add_hline(
        y=-0.05, line_dash='dot', line_color='rgba(224,82,82,0.25)', line_width=1
    )
    fig.update_layout(yaxis_title='Spearman IC', showlegend=True)
    return chips, fig


@callback(
    Output('bt-quintile-chart', 'figure'),
    Input('backtest-store', 'data'),
)
def render_quintiles(data):
    if not data or 'error' in (data or {}) or not data.get('qcr'):
        return empty_figure('No quintile data')

    qcr_records = data['qcr']
    df = pd.DataFrame(qcr_records)
    df['date'] = pd.to_datetime(df['date'])

    cost_bps = data.get('cost_bps', 0)
    qcr_net = data.get('qcr_net', [])
    df_net = pd.DataFrame(qcr_net) if qcr_net else pd.DataFrame()
    if not df_net.empty:
        df_net['date'] = pd.to_datetime(df_net['date'])

    fig = go.Figure(layout=_chart_layout())
    for i, col in enumerate(['Q1', 'Q2', 'Q3', 'Q4', 'Q5']):
        if col not in df.columns:
            continue
        fig.add_trace(
            go.Scatter(
                x=df['date'],
                y=df[col],
                mode='lines',
                name=col,
                line=dict(
                    color=_QUINTILE_COLOURS[i], width=1.5 if col in ('Q1', 'Q5') else 1
                ),
                hovertemplate=f'{col}: %{{y:.3f}}<extra></extra>',
            )
        )
        # Net (cost-adjusted) traces for Q1 and Q5 only
        if (
            cost_bps
            and col in ('Q1', 'Q5')
            and not df_net.empty
            and col in df_net.columns
        ):
            fig.add_trace(
                go.Scatter(
                    x=df_net['date'],
                    y=df_net[col],
                    mode='lines',
                    name=f'{col} net',
                    line=dict(color=_QUINTILE_COLOURS[i], width=1, dash='dot'),
                    hovertemplate=f'{col} net: %{{y:.3f}}<extra></extra>',
                )
            )

    ew_cols = [c for c in ['Q1', 'Q2', 'Q3', 'Q4', 'Q5'] if c in df.columns]
    if ew_cols:
        ew_vals = df[ew_cols].mean(axis=1)
        fig.add_trace(
            go.Scatter(
                x=df['date'],
                y=ew_vals,
                mode='lines',
                name='EW',
                line=dict(color=MUTED, width=1, dash='dash'),
                hovertemplate='EW: %{y:.3f}<extra></extra>',
            )
        )

    # L/S spread (Q1 − Q5)
    ls_records = data.get('ls', [])
    if ls_records:
        ls_df = pd.DataFrame(ls_records)
        ls_df['date'] = pd.to_datetime(ls_df['date'])
        fig.add_trace(
            go.Scatter(
                x=ls_df['date'],
                y=ls_df['v'],
                mode='lines',
                name='L/S (Q1−Q5)',
                line=dict(color='#ffffff', width=1.5, dash='longdash'),
                hovertemplate='L/S: %{y:.3f}<extra></extra>',
            )
        )

    # Terminal value labels for Q1 and Q5
    last_date = df['date'].iloc[-1]
    for col, xanchor, xshift, i in [('Q1', 'right', -6, 0), ('Q5', 'left', 6, 4)]:
        if col in df.columns:
            series = df[col].dropna()
            if not series.empty:
                val = float(series.iloc[-1])
                fig.add_annotation(
                    x=last_date,
                    y=val,
                    text=f'{val:+.2f}',
                    showarrow=False,
                    font=dict(color=_QUINTILE_COLOURS[i], size=11),
                    xanchor=xanchor,
                    xshift=xshift,
                )
    fig.update_layout(yaxis_title='Cumulative log return')
    return fig


@callback(
    Output('bt-spread-chart', 'figure'),
    Input('backtest-store', 'data'),
)
def render_spread(data):
    if not data or 'error' in (data or {}) or not data.get('qcr'):
        return empty_figure('No quintile data')

    df = pd.DataFrame(data['qcr'])
    df['date'] = pd.to_datetime(df['date'])

    ew_cols = [c for c in ['Q1', 'Q2', 'Q3', 'Q4', 'Q5'] if c in df.columns]
    if not ew_cols:
        return empty_figure('No quintile data')

    ew_vals = df[ew_cols].mean(axis=1)
    fig = go.Figure(layout=_chart_layout())
    for i, col in enumerate(['Q1', 'Q2', 'Q3', 'Q4', 'Q5']):
        if col not in df.columns:
            continue
        spread = df[col] - ew_vals
        fig.add_trace(
            go.Scatter(
                x=df['date'],
                y=spread,
                mode='lines',
                name=col,
                line=dict(
                    color=_QUINTILE_COLOURS[i], width=1.5 if col in ('Q1', 'Q5') else 1
                ),
                hovertemplate=f'{col}: %{{y:.3f}}<extra></extra>',
            )
        )
    fig.add_hline(y=0, line_color=MUTED, line_width=1)
    last_date = df['date'].iloc[-1]
    for col, xanchor, xshift, i in [('Q1', 'right', -6, 0), ('Q5', 'left', 6, 4)]:
        if col in df.columns:
            series = (df[col] - ew_vals).dropna()
            if not series.empty:
                val = float(series.iloc[-1])
                fig.add_annotation(
                    x=last_date,
                    y=val,
                    text=f'{val:+.2f}',
                    showarrow=False,
                    font=dict(color=_QUINTILE_COLOURS[i], size=11),
                    xanchor=xanchor,
                    xshift=xshift,
                )
    fig.update_layout(yaxis_title='Excess cumulative log return vs EW')
    return fig


@callback(
    Output('bt-decay-store', 'data'),
    Input('bt-decay-btn', 'n_clicks'),
    State('bt-decay-horizons', 'value'),
    State('bt-mode-tabs', 'value'),
    State('bt-factor', 'value'),
    State('bt-variant', 'value'),
    State('bt-freq', 'value'),
    State('bt-start-year', 'value'),
    State('bt-end-year', 'value'),
    State('bt-market', 'value'),
    State('bt-sector', 'value'),
    State('bt-watchlist', 'value'),
    running=[
        (Output('bt-decay-btn', 'disabled'), True, False),
        (Output('bt-decay-btn', 'children'), 'Running...', 'Run Decay'),
    ],
    prevent_initial_call=True,
)
def run_decay_analysis(
    n_clicks,
    horizons,
    mode,
    factor,
    variant,
    freq,
    start_yr,
    end_yr,
    market,
    sector,
    watchlist,
):
    if not n_clicks or not horizons or mode == 'composite' or not factor:
        raise PreventUpdate
    try:
        start_yr = int(start_yr or 2015)
        end_yr = int(end_yr or _CURRENT_YEAR)
        if watchlist:
            try:
                watchlist_service.load_watchlist(watchlist)
            except KeyError:
                return {'error': f'Watchlist "{watchlist}" not found.'}
        df = backtest_service._run_decay(
            factor=factor,
            horizons=sorted(horizons),
            start_date=datetime.date(start_yr, 1, 1),
            end_date=datetime.date(end_yr, 12, 31),
            variant=variant or 'A',
            freq=freq or 'Q',
            market=market or None,
            sector=sector,
            watchlist=watchlist,
        )
        return df.to_dict(orient='records')
    except PreventUpdate:
        raise
    except Exception as exc:
        logger.exception(f'Decay analysis error: {exc}')
        return {'error': str(exc)}


@callback(
    Output('bt-decay-chart', 'figure'),
    Input('bt-decay-store', 'data'),
)
def render_decay(data):
    if not data:
        return empty_figure('Run decay analysis to see results')
    if isinstance(data, dict) and 'error' in data:
        return empty_figure(data['error'])
    df = pd.DataFrame(data)
    if df.empty:
        return empty_figure('No data')

    valid_icir = df['icir'].dropna()

    fig = go.Figure(layout=_chart_layout())
    fig.add_trace(
        go.Scatter(
            x=df['horizon'],
            y=df['mean_ic'],
            mode='lines+markers',
            name='Mean IC',
            line=dict(color=ACCENT, width=2),
            marker=dict(size=8),
            hovertemplate='Horizon %{x}d<br>Mean IC: %{y:.3f}<extra></extra>',
        )
    )
    if not valid_icir.empty:
        fig.add_trace(
            go.Scatter(
                x=df['horizon'],
                y=df['icir'],
                mode='lines+markers',
                name='ICIR',
                line=dict(color='#3fb950', width=2),
                marker=dict(size=8),
                yaxis='y2',
                hovertemplate='Horizon %{x}d<br>ICIR: %{y:.2f}<extra></extra>',
            )
        )
        fig.update_layout(
            yaxis2=dict(
                title='ICIR',
                overlaying='y',
                side='right',
                gridcolor='rgba(0,0,0,0)',
                tickfont=dict(color=MUTED, size=11),
                zeroline=False,
            ),
        )
    fig.update_layout(
        xaxis_title='Horizon (days)',
        yaxis_title='Mean IC',
    )
    return fig

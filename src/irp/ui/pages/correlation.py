"""Correlation matrix page: factor-to-factor or price-return pairwise correlations."""

import datetime
import logging

import dash
import numpy as np
import pandas as pd
import plotly.graph_objects as go
from dash import Input, Output, State, callback, dcc, html
from dash.exceptions import PreventUpdate

from irp.core.config import config
from irp.factors.registry import all_factors
from irp.panel.load import load_prices_wide
from irp.ui.charts import base_chart_layout as _chart_layout
from irp.ui.charts import empty_figure as _empty_figure
from irp.ui.services import factors_service, universe_service, watchlist_service
from irp.ui.theme import ACCENT, MUTED

dash.register_page(__name__, path='/correlation', name='Correlation')

logger = logging.getLogger(__name__)

_DEFAULT_DATE = (datetime.date.today() - datetime.timedelta(days=90)).isoformat()
_VARIANT_OPTIONS = [
    {'label': 'Annual', 'value': 'A'},
    {'label': 'Quarterly', 'value': 'Q'},
]
_WINDOW_OPTIONS = [
    {'label': '1 year', 'value': 252},
    {'label': '2 years', 'value': 504},
    {'label': '5 years', 'value': 1260},
]
_MAX_RETURN_TICKERS = config.factors.max_return_tickers
_HIDE = {'display': 'none'}
_SHOW = {'display': 'flex', 'gap': '12px', 'alignItems': 'flex-end', 'flexWrap': 'wrap'}


# ── Helpers ───────────────────────────────────────────────────────────


def _factor_label_map() -> dict[str, str]:
    return {f.name: f.label for f in all_factors()}


def _factor_corr(
    date: datetime.date,
    variant: str,
    market: str | None,
    sector: str | None,
    watchlist: str | None,
) -> tuple[pd.DataFrame, list[str], str | None]:
    """Compute factor × factor Pearson correlation from a cross-section snapshot.

    Returns (corr_df, labels, error_msg). corr_df has raw factor names as index/cols;
    labels has the display labels in matching order.
    """
    xs = factors_service.load_cross_section(
        date, variant, market, sector, watchlist, enrich_company_columns=False,
    )
    if xs.empty:
        return pd.DataFrame(), [], 'No data for selected filters'

    label_map = _factor_label_map()
    factor_cols = [f for f in xs.columns if f in label_map]
    if not factor_cols:
        return pd.DataFrame(), [], 'No factor columns in cross-section'

    corr = xs[factor_cols].corr(method='pearson')
    labels = [label_map[c] for c in factor_cols]
    n = len(xs)
    return corr, labels, None if n >= 10 else f'Only {n} tickers — correlation may be unreliable'


def _return_corr(
    as_of_date: datetime.date,
    window_days: int,
    market: str | None,
    sector: str | None,
    watchlist: str | None,
) -> tuple[pd.DataFrame, list[str], str | None]:
    """Compute ticker × ticker return correlation from the price panel.

    Returns (corr_df, ticker_labels, error_msg).
    """
    tickers = universe_service.filter_tickers(market=market, sector=sector, watchlist=watchlist)
    if tickers is not None and len(tickers) > _MAX_RETURN_TICKERS:
        return pd.DataFrame(), [], (
            f'Too many tickers ({len(tickers)}) — add market/sector/watchlist filter '
            f'to narrow to ≤{_MAX_RETURN_TICKERS}'
        )

    panel = load_prices_wide('Close')
    dates_np = panel.dates
    # Convert as_of_date to numpy datetime64 for comparison
    as_of_np = np.datetime64(as_of_date, 'D')
    end_idx = int(np.searchsorted(dates_np, as_of_np, side='right'))
    start_idx = max(0, end_idx - window_days)
    if end_idx - start_idx < 20:
        return pd.DataFrame(), [], 'Not enough price history in selected window'

    # Build price matrix slice
    all_tickers = list(panel.tickers)
    if tickers is not None:
        use_tickers = [t for t in tickers if t in panel.ticker_to_idx]
    else:
        return pd.DataFrame(), [], (
            f'Returns correlation requires a market/sector/watchlist filter '
            f'(full universe has {len(all_tickers):,} tickers)'
        )

    if not use_tickers:
        return pd.DataFrame(), [], 'No tickers found in price panel for selected filters'

    idxs = [panel.ticker_to_idx[t] for t in use_tickers]
    price_slice = panel.values[start_idx:end_idx, :][:, idxs]  # (days, n_tickers)

    prices_df = pd.DataFrame(price_slice, columns=use_tickers)
    # Drop tickers with >20% missing values
    min_obs = int(price_slice.shape[0] * 0.8)
    prices_df = prices_df.dropna(axis=1, thresh=min_obs)
    if prices_df.shape[1] < 2:
        return pd.DataFrame(), [], 'Too few tickers with sufficient price history'

    log_ret = np.log(prices_df / prices_df.shift(1)).dropna(how='all')
    corr = log_ret.corr(method='pearson')

    n_days = end_idx - start_idx
    warning = None
    if len(prices_df.columns) > 80:
        warning = f'{len(prices_df.columns)} tickers — heatmap may be dense'
    return corr, list(corr.columns), warning


def _heatmap_figure(
    corr: pd.DataFrame,
    labels: list[str],
    title: str,
    warning: str | None = None,
) -> go.Figure:
    z = corr.values
    n = len(labels)

    # Annotations for high |corr| cells (off-diagonal only)
    annotations = []
    if n <= 40:  # only annotate when matrix is small enough to read
        for i in range(n):
            for j in range(n):
                if i != j and abs(z[i, j]) >= 0.7:
                    annotations.append(dict(
                        x=j, y=i,
                        text=f'{z[i, j]:.2f}',
                        showarrow=False,
                        font=dict(size=9, color='white' if abs(z[i, j]) > 0.85 else MUTED),
                        xref='x', yref='y',
                    ))

    heatmap = go.Heatmap(
        z=z,
        x=labels,
        y=labels,
        colorscale='RdBu_r',
        zmin=-1,
        zmax=1,
        zmid=0,
        colorbar=dict(
            thickness=14,
            len=0.9,
            tickfont=dict(color=MUTED, size=10),
            outlinewidth=0,
        ),
        hovertemplate='%{y} / %{x}: %{z:.3f}<extra></extra>',
        xgap=1,
        ygap=1,
    )

    subtitle = f' — {warning}' if warning else ''
    layout = _chart_layout(
        title=dict(
            text=f'{title}{subtitle}',
            font=dict(size=13, color=MUTED),
            x=0,
            pad=dict(l=0),
        ),
        hovermode='closest',
        xaxis=dict(
            tickfont=dict(size=10, color=MUTED),
            side='bottom',
            tickangle=-40,
            showgrid=False,
            zeroline=False,
        ),
        yaxis=dict(
            tickfont=dict(size=10, color=MUTED),
            autorange='reversed',
            showgrid=False,
            zeroline=False,
            scaleanchor='x',
        ),
        margin=dict(l=0, r=0, t=40, b=0),
        annotations=annotations,
        height=max(400, n * 26 + 80),
    )

    return go.Figure(data=[heatmap], layout=layout)


# ── Layout ────────────────────────────────────────────────────────────

layout = html.Div(
    className='corr-page',
    children=[
        dcc.Store(id='corr-wl-trigger', data=0),
        html.H2('Correlation Matrix', className='page-title'),
        html.P(
            'Factor-to-factor collinearity (left) or price-return co-movement between tickers (right).',
            className='page-subtitle',
        ),
        # Controls
        html.Div(
            className='control-row sticky-controls',
            children=[
                dcc.RadioItems(
                    id='corr-mode',
                    options=[
                        {'label': 'Factor correlation', 'value': 'factor'},
                        {'label': 'Return correlation', 'value': 'returns'},
                    ],
                    value='factor',
                    inline=True,
                    labelClassName='check-item',
                    style={'paddingBottom': '6px'},
                ),
                dcc.DatePickerSingle(
                    id='corr-date',
                    date=_DEFAULT_DATE,
                    display_format='YYYY-MM-DD',
                    style={'fontSize': '13px'},
                ),
                dcc.RadioItems(
                    id='corr-variant',
                    options=_VARIANT_OPTIONS,
                    value='A',
                    inline=True,
                    labelClassName='check-item',
                ),
                dcc.Dropdown(
                    id='corr-market',
                    options=[
                        {'label': 'All markets', 'value': ''},
                        {'label': 'Stocks only', 'value': 'stocks'},
                        {'label': 'ETFs only', 'value': 'etfs'},
                    ],
                    value='',
                    clearable=False,
                    className='filter-dropdown',
                    style={'minWidth': '140px'},
                ),
                dcc.Dropdown(
                    id='corr-sector',
                    placeholder='All sectors',
                    clearable=True,
                    className='filter-dropdown',
                    style={'minWidth': '160px'},
                ),
                dcc.Dropdown(
                    id='corr-watchlist',
                    placeholder='Watchlist…',
                    clearable=True,
                    className='filter-dropdown',
                    style={'minWidth': '150px'},
                ),
                # Return window — only shown in Returns mode
                html.Div(
                    id='corr-window-wrap',
                    style=_HIDE,
                    children=[
                        dcc.Dropdown(
                            id='corr-window',
                            options=_WINDOW_OPTIONS,
                            value=252,
                            clearable=False,
                            className='filter-dropdown',
                            style={'minWidth': '110px'},
                        ),
                    ],
                ),
                html.Button('Run', id='corr-run-btn', className='run-btn', n_clicks=0),
            ],
        ),
        # Heatmap output
        dcc.Loading(
            id='corr-loading',
            type='circle',
            color=ACCENT,
            children=dcc.Graph(
                id='corr-heatmap',
                figure=_empty_figure('Select filters and click Run'),
                config={'displayModeBar': False},
                style={'minHeight': '400px'},
            ),
        ),
    ],
)


# ── Callbacks ─────────────────────────────────────────────────────────


@callback(
    Output('corr-window-wrap', 'style'),
    Input('corr-mode', 'value'),
)
def _toggle_window(mode: str) -> dict:
    return _SHOW if mode == 'returns' else _HIDE


@callback(
    Output('corr-sector', 'options'),
    Input('corr-wl-trigger', 'data'),
)
def _load_sectors(_: int) -> list[dict]:
    try:
        sectors = universe_service.get_sectors()
        return [{'label': s, 'value': s} for s in sectors]
    except Exception:
        return []


@callback(
    Output('corr-watchlist', 'options'),
    Input('corr-wl-trigger', 'data'),
)
def _load_watchlists(_: int) -> list[dict]:
    try:
        wl_df = watchlist_service.list_watchlists()
        if wl_df.empty:
            return []
        return [
            {'label': f'{r["name"]} ({r["n"]})', 'value': r['name']}
            for _, r in wl_df.iterrows()
        ]
    except Exception:
        return []


@callback(
    Output('corr-heatmap', 'figure'),
    Input('corr-run-btn', 'n_clicks'),
    State('corr-mode', 'value'),
    State('corr-date', 'date'),
    State('corr-variant', 'value'),
    State('corr-market', 'value'),
    State('corr-sector', 'value'),
    State('corr-watchlist', 'value'),
    State('corr-window', 'value'),
    prevent_initial_call=True,
)
def _run(
    n_clicks: int,
    mode: str,
    date_str: str | None,
    variant: str,
    market: str,
    sector: str | None,
    watchlist: str | None,
    window: int,
) -> go.Figure:
    if not n_clicks or not date_str:
        raise PreventUpdate

    date = datetime.date.fromisoformat(date_str[:10])
    market_filter = market or None

    try:
        if mode == 'factor':
            corr, labels, err = _factor_corr(date, variant, market_filter, sector, watchlist)
            title = f'Factor correlation — {date_str[:10]} ({variant})'
        else:
            corr, labels, err = _return_corr(date, window or 252, market_filter, sector, watchlist)
            w_label = next((o['label'] for o in _WINDOW_OPTIONS if o['value'] == window), str(window))
            title = f'Return correlation — {w_label} ending {date_str[:10]}'

        if corr.empty:
            return _empty_figure(err or 'No data')

        return _heatmap_figure(corr, labels, title, warning=err)

    except Exception as exc:
        logger.exception('correlation callback error')
        return _empty_figure(f'Error: {exc}')

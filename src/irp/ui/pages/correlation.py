"""Correlation matrix page: factor-to-factor or price-return pairwise correlations."""

import datetime
import logging
from typing import Literal

import dash
import pandas as pd
import plotly.graph_objects as go
from dash import Input, Output, State, callback, dcc, html
from dash.exceptions import PreventUpdate

from irp.core.config import config
from irp.factors.registry import all_factors
from irp.ui.charts import corr_heatmap_figure as _heatmap_figure
from irp.ui.charts import empty_figure
from irp.ui.services import factors_service, universe_service, watchlist_service
from irp.ui.theme import ACCENT

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
    variant: Literal['A', 'Q'],
    market: str | None,
    sector: str | None,
    watchlist: str | None,
) -> tuple[pd.DataFrame, list[str], str | None]:
    """Compute factor x factor Pearson correlation from a cross-section snapshot.

    Returns (corr_df, labels, error_msg). corr_df has raw factor names as index/cols;
    labels has the display labels in matching order.
    """
    xs = factors_service._load_cross_section(
        date,
        variant,
        market,
        sector,
        watchlist,
        enrich_company_columns=False,
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
    return (
        corr,
        labels,
        None if n >= 10 else f'Only {n} tickers — correlation may be unreliable',
    )


def _return_corr(
    as_of_date: datetime.date,
    window_days: int,
    market: str | None,
    sector: str | None,
    watchlist: str | None,
) -> tuple[pd.DataFrame, list[str], str | None]:
    """Compute ticker × ticker return correlation from the price panel."""
    tickers = universe_service._filter_tickers(
        market=market, sector=sector, watchlist=watchlist
    )
    if tickers is None:
        from irp.panel.load import load_prices_wide

        n_all = len(load_prices_wide('Close').tickers)
        return (
            pd.DataFrame(),
            [],
            (
                f'Returns correlation requires a market/sector/watchlist filter '
                f'(full universe has {n_all:,} tickers)'
            ),
        )
    if len(tickers) > _MAX_RETURN_TICKERS:
        return (
            pd.DataFrame(),
            [],
            (
                f'Too many tickers ({len(tickers)}) — add market/sector/watchlist filter '
                f'to narrow to ≤{_MAX_RETURN_TICKERS}'
            ),
        )
    return factors_service._compute_return_corr(tickers, window_days, as_of_date)


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
                figure=empty_figure('Select filters and click Run'),
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
        sectors = universe_service._get_sectors()
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
    variant: Literal['A', 'Q'],
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
            corr, labels, err = _factor_corr(
                date, variant, market_filter, sector, watchlist
            )
            title = f'Factor correlation — {date_str[:10]} ({variant})'
        else:
            corr, labels, err = _return_corr(
                date, window or 252, market_filter, sector, watchlist
            )
            w_label = next(
                (o['label'] for o in _WINDOW_OPTIONS if o['value'] == window),
                str(window),
            )
            title = f'Return correlation — {w_label} ending {date_str[:10]}'

        if corr.empty:
            return empty_figure(err or 'No data')

        return _heatmap_figure(corr, labels, title, warning=err)

    except Exception as exc:
        logger.exception('correlation callback error')
        return empty_figure(f'Error: {exc}')

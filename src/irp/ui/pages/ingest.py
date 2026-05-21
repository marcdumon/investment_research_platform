import logging
import threading
from typing import Any

import dash
from dash import Input, Output, State, callback, dcc, html
from dash.exceptions import PreventUpdate

from irp.core import cancel as _cancel
from irp.core.logging import LEVEL_COLORS_HEX as _LEVEL_COLORS
from irp.core.logging import LOG_DATEFMT, LOG_FMT
from irp.ui import log_handler as lh

logger = logging.getLogger(__name__)

dash.register_page(__name__, path='/ingest', name='Ingest')

_HIDE = {'display': 'none'}
_SHOW = {'display': 'block'}

layout = html.Div(className='ingest-page', children=[
    # Sidebar: controls
    html.Div(className='ingest-sidebar', children=[
        html.Label('Providers', className='section-label'),
        dcc.Checklist(
            id='providers',
            options=[
                {'label': ' simfin', 'value': 'simfin'},
                {'label': ' stooq', 'value': 'stooq'},
                {'label': ' yahoo', 'value': 'yahoo'},
            ],
            value=['simfin', 'stooq'],
            labelClassName='check-item',
        ),

        html.Div(id='yahoo-options', style=_HIDE, children=[
            html.Label('Yahoo content', className='section-label'),
            dcc.Checklist(
                id='yahoo-content',
                options=[
                    {'label': ' actions (dividends + splits)', 'value': 'actions'},
                    {'label': ' prices (OHLCV)', 'value': 'prices'},
                ],
                value=['actions', 'prices'],
                labelClassName='check-item',
            ),
            html.Div(id='prices-mode-container', children=[
                html.Label('Yahoo prices mode', className='section-label'),
                dcc.RadioItems(
                    id='yahoo-prices-mode',
                    options=[
                        {'label': ' batch (~10x faster)', 'value': 'batch'},
                        {'label': ' ticker (per-ticker history)', 'value': 'ticker'},
                    ],
                    value='batch',
                    labelClassName='check-item',
                ),
            ]),
        ]),

        html.Label('Feed', className='section-label'),
        dcc.RadioItems(
            id='feed',
            options=[
                {'label': ' bulk', 'value': 'bulk'},
                {'label': ' update', 'value': 'update'},
            ],
            value='bulk',
            labelClassName='check-item',
        ),

        html.Label('Steps', className='section-label'),
        dcc.Checklist(
            id='steps',
            options=[
                {'label': ' fetch', 'value': 'fetch'},
                {'label': ' transform', 'value': 'transform'},
                {'label': ' store', 'value': 'store'},
                {'label': ' cleanup', 'value': 'cleanup'},
                {'label': ' seed-universe', 'value': 'seed-universe'},
                {'label': ' universe', 'value': 'universe'},
                {'label': ' catalog', 'value': 'catalog'},
                {'label': ' clear factor cache', 'value': 'clear-factor-cache'},
                {'label': ' rebuild factor cache', 'value': 'rebuild-factor-cache'},
            ],
            value=['fetch', 'transform', 'store'],
            labelClassName='check-item',
        ),

        dcc.Checklist(
            id='force',
            options=[{'label': ' Force re-run', 'value': 'force'}],
            value=[],
            labelClassName='check-item',
            style={'marginTop': '12px'},
        ),

        html.Div(style={'marginTop': '16px', 'display': 'flex', 'gap': '8px'}, children=[
            html.Button('Run', id='run-btn', n_clicks=0, className='btn btn-primary'),
            html.Button('Break', id='break-btn', n_clicks=0, disabled=True, className='btn btn-danger'),
        ]),
        html.Div(id='run-status', className='run-status'),
    ]),

    # Log panel
    html.Div(className='ingest-log', children=[
        dcc.Interval(id='log-interval', interval=1000),
        html.Div(id='log-output', children=[]),
    ]),
])


@callback(
    Output('yahoo-options', 'style'),
    Input('providers', 'value'),
)
def toggle_yahoo_options(providers: list[str]) -> dict:
    return _SHOW if providers and 'yahoo' in providers else _HIDE


@callback(
    Output('prices-mode-container', 'style'),
    Input('yahoo-content', 'value'),
)
def toggle_prices_mode(yahoo_content: list[str]) -> dict:
    return _SHOW if yahoo_content and 'prices' in yahoo_content else _HIDE


@callback(
    Output('run-btn', 'disabled', allow_duplicate=True),
    Output('break-btn', 'disabled', allow_duplicate=True),
    Output('run-status', 'children', allow_duplicate=True),
    Output('log-output', 'children', allow_duplicate=True),
    Input('run-btn', 'n_clicks'),
    State('providers', 'value'),
    State('yahoo-content', 'value'),
    State('yahoo-prices-mode', 'value'),
    State('feed', 'value'),
    State('steps', 'value'),
    State('force', 'value'),
    prevent_initial_call=True,
)
def start_run(
    n_clicks: int,
    providers: list[str],
    yahoo_content: list[str],
    yahoo_prices_mode: str,
    feed: str,
    steps: list[str],
    force: list[str],
) -> tuple[Any, ...]:
    if not n_clicks:
        raise PreventUpdate

    lh._log_buffer.clear()
    lh._run_active = True
    lh._run_done = False
    _cancel.clear()

    handler = lh.DequeHandler()
    handler.setFormatter(logging.Formatter(fmt=LOG_FMT, datefmt=LOG_DATEFMT))
    irp_logger = logging.getLogger('irp')
    irp_logger.addHandler(handler)

    def _run() -> None:
        from irp.core.db import db_close
        db_close()
        try:
            _run_pipeline(
                providers or [],
                yahoo_content or ['actions', 'prices'],
                yahoo_prices_mode or 'batch',
                feed or 'bulk',
                steps or [],
                bool(force),
            )
        except Exception as exc:
            import traceback
            logger.error(f'Pipeline error: {exc}\n{traceback.format_exc()}')
            logger.error('Pipeline finished with errors.')
        else:
            logger.info('Pipeline finished.')
        finally:
            irp_logger.removeHandler(handler)
            lh._run_done = True
            lh._run_active = False

    threading.Thread(target=_run, daemon=True).start()
    return True, False, 'Running...', []


@callback(
    Output('break-btn', 'disabled', allow_duplicate=True),
    Output('run-status', 'children', allow_duplicate=True),
    Input('break-btn', 'n_clicks'),
    prevent_initial_call=True,
)
def cancel_run(n_clicks: int) -> tuple[Any, ...]:
    if not n_clicks:
        raise PreventUpdate
    _cancel.request_cancel()
    return True, 'Cancelling...'


@callback(
    Output('log-output', 'children'),
    Output('run-btn', 'disabled'),
    Output('break-btn', 'disabled'),
    Output('run-status', 'children'),
    Input('log-interval', 'n_intervals'),
    prevent_initial_call=True,
)
def poll_log(n_intervals: int) -> tuple[Any, ...]:
    lines = [
        html.Div(text, style={'color': _LEVEL_COLORS.get(level, '#d4d4d4'), 'lineHeight': '1.6'})
        for level, text in list(lh._log_buffer)
    ]
    active = lh._run_active
    done = lh._run_done
    cancelled = _cancel.is_cancelled()
    if active and cancelled:
        status = 'Cancelling...'
    elif active:
        status = 'Running...'
    elif cancelled:
        status = 'Cancelled'
    elif done:
        status = 'Done'
    else:
        status = ''
    return lines, active, not active, status


def _run_pipeline(
    providers: list[str],
    yahoo_content: list[str],
    yahoo_prices_mode: str,
    feed: str,
    steps: list[str],
    force: bool,
) -> None:
    from irp.cli import _delete_markers, _make_source

    def cancelled() -> bool:
        return _cancel.is_cancelled()

    if force:
        for name in providers:
            _delete_markers(name, feed)

    for name in providers:
        if cancelled():
            break
        src = _make_source(name, yahoo_content=yahoo_content, yahoo_prices_mode=yahoo_prices_mode)
        logger.info(f'-- {name} --')
        if feed not in src.SUPPORTED_FEEDS:
            logger.warning(f'feed {feed!r} not supported by {name}, skipping')
            continue
        if 'fetch' in steps and not cancelled():
            logger.info(f'fetch ({feed})')
            src.fetch_bulk() if feed == 'bulk' else src.update()
        if 'transform' in steps and not cancelled():
            logger.info(f'transform ({feed})')
            src.transform(feed)
        if 'store' in steps and not cancelled():
            logger.info(f'store ({feed})')
            src.store(feed)
        if 'cleanup' in steps and not cancelled():
            logger.info('cleanup')
            src.cleanup()

    if 'seed-universe' in steps and not cancelled():
        from irp.query.universe import seed as _seed_universe
        logger.info('-- seed-universe --')
        n = _seed_universe()
        logger.info(f'{n:,} tickers written to universe.csv')

    if 'universe' in steps and not cancelled():
        from irp.query.universe import refresh as _refresh_universe
        logger.info('-- universe --')
        n = _refresh_universe()
        logger.info(f'{n:,} tickers')

    if 'catalog' in steps and not cancelled():
        from irp.query.catalog import refresh as _refresh_catalog
        logger.info('-- catalog --')
        n = _refresh_catalog()
        logger.info(f'{n:,} tickers')

    if 'clear-factor-cache' in steps and not cancelled():
        from irp.factors.cache import clear as _clear_cache
        logger.info('-- clear factor cache --')
        n = _clear_cache()
        logger.info(f'factor cache cleared ({n} snapshot{"s" if n != 1 else ""} removed)')

    if 'rebuild-factor-cache' in steps and not cancelled():
        from irp.factors.cache import precompute_all
        import datetime as _dt
        logger.info('-- rebuild factor cache --')
        n = precompute_all(
            start_date=_dt.date(2010, 1, 1),
            end_date=_dt.date.today(),
        )
        logger.info(f'factor cache rebuilt ({n} new snapshot{"s" if n != 1 else ""} written)')

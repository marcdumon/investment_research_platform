"""Feature Engineering page: compose ML features from cached snapshots + export.

Build a long (Date, Ticker) feature matrix by stacking cached factor snapshots
over a date grid, applying an ordered step stack (temporal / math / norm),
attaching a configurable forward-return label, and exporting to parquet/CSV.
Specs are saveable as named recipes.

Compute is synchronous and cache-backed: a cold snapshot cache short-circuits
with a "run precompute_all first" message instead of recomputing inline.
"""
import logging
from typing import Any

import dash
import pandas as pd
from dash import ALL, Input, Output, State, callback, ctx, dash_table as _dt, dcc, html
from dash.exceptions import PreventUpdate

from irp.factors.registry import all_factors
from irp.ui.factor_meta import FACTOR_LABELS, FACTOR_OPTIONS
from irp.ui.services import features_service, watchlist_service
from irp.ui.tables import column_format as _col_fmt
from irp.ui.theme import ACCENT, MUTED, TABLE_STYLE

dash.register_page(__name__, path='/features', name='Feature Engineering')

logger = logging.getLogger(__name__)

# Server-side cache of the most recent builds (token -> DataFrame). Avoids
# shipping a huge panel through dcc.Store; export re-uses or rebuilds from spec.
_BUILD_CACHE: dict[str, pd.DataFrame] = {}
_BUILD_CACHE_CAP = 4

_VARIANT_OPTIONS = [{'label': ' Annual (FY)', 'value': 'A'}, {'label': ' Quarterly (TTM)', 'value': 'Q'}]
_FREQ_OPTIONS = [
    {'label': ' Yearly', 'value': 'A'}, {'label': ' Quarterly', 'value': 'Q'},
    {'label': ' Monthly', 'value': 'M'}, {'label': ' Weekly', 'value': 'W'},
    {'label': ' Daily', 'value': 'D'},
]
_DENSE_FREQS = {'M', 'W', 'D'}
_OP_OPTIONS = [
    {'label': 'Base column', 'value': 'base'},
    {'label': 'Lag', 'value': 'lag'},
    {'label': 'Lag window (p0..p-n)', 'value': 'lagwin'},
    {'label': 'Diff', 'value': 'diff'},
    {'label': 'Pct change', 'value': 'pct_change'},
    {'label': 'Rolling', 'value': 'rolling'},
    {'label': 'Ratio (a / b)', 'value': 'ratio'},
    {'label': 'Product (a × b)', 'value': 'product'},
    {'label': 'Log', 'value': 'log'},
    {'label': 'Winsorize', 'value': 'winsorize'},
    {'label': 'Normalize (per date)', 'value': 'norm'},
]
_ROLL_FN_OPTIONS = [{'label': f, 'value': f} for f in ('mean', 'std', 'min', 'max', 'sum')]
_NORM_OPTIONS = [
    {'label': 'Z-score', 'value': 'zscore'},
    {'label': 'Rank', 'value': 'rank'},
    {'label': 'Sector-neutral', 'value': 'sector'},
]
_HORIZON_OPTIONS = [
    {'label': '21d', 'value': 21}, {'label': '63d', 'value': 63},
    {'label': '126d', 'value': 126}, {'label': '252d', 'value': 252},
]
_LABEL_MODE_OPTIONS = [
    {'label': ' None (features only)', 'value': 'none'},
    {'label': ' Return (number)', 'value': 'continuous'},
    {'label': ' Up/Down (0/1)', 'value': 'binary'},
    {'label': ' Quantile bucket', 'value': 'quantile'},
]
_SCALE_METHOD_OPTIONS = [
    {'label': ' None', 'value': 'none'},
    {'label': ' MinMax [0,1]', 'value': 'minmax'},
    {'label': ' Robust (median/IQR)', 'value': 'robust'},
]
_SCALE_SCOPE_OPTIONS = [
    {'label': ' Per-date (leak-free)', 'value': 'date'},
    {'label': ' Per-ticker', 'value': 'ticker'},
    {'label': ' Global', 'value': 'global'},
]
_SPLIT_METHOD_OPTIONS = [
    {'label': ' None (single file)', 'value': 'none'},
    {'label': ' By date (train < valid < test)', 'value': 'date'},
    {'label': ' Leave tickers out', 'value': 'ticker'},
]


def _num(id_: str, placeholder: str, value=None, width='90px'):
    return dcc.Input(
        id=id_, type='number', placeholder=placeholder, value=value,
        className='filter-input', style={'width': width},
    )


def _dd(id_: str, options, value=None, placeholder='', width='150px', clearable=False):
    return dcc.Dropdown(
        id=id_, options=options, value=value, placeholder=placeholder,
        clearable=clearable, className='filter-dropdown', style={'minWidth': width},
    )


_SHOW = {'display': 'flex', 'flexDirection': 'column', 'gap': '2px'}
_HIDE = {'display': 'none'}


def _field(label: str, control, wrap_id: str | None = None):
    """A captioned control: small label stacked above the input."""
    kw = {'id': wrap_id} if wrap_id else {}
    return html.Div(
        style=dict(_SHOW), **kw,
        children=[html.Label(label, style={'color': MUTED, 'fontSize': '11px'}), control],
    )


# Which step inputs each operation needs (drives Row-2 visibility).
_OP_FIELDS: dict[str, set[str]] = {
    'base': {'col'},
    'lag': {'col', 'k'}, 'diff': {'col', 'k'}, 'pct_change': {'col', 'k'},
    'lagwin': {'col', 'window'},
    'rolling': {'col', 'window', 'fn'},
    'ratio': {'col', 'colb'}, 'product': {'col', 'colb'},
    'log': {'col'},
    'winsorize': {'col', 'p'},
    'norm': {'col', 'method'},
}
_OP_HINT: dict[str, str] = {
    'base': 'Include a raw column as-is.',
    'lag': 'Value k periods earlier (per ticker, along the date grid).',
    'lagwin': 'Price window: level + n lags as columns (col, col_lag1 … col_lag_n). '
              'Use Window = n. Daily freq → past-day prices p0..p-n in each row.',
    'diff': 'Change vs k periods earlier.',
    'pct_change': 'Percent change vs k periods earlier.',
    'rolling': 'Rolling-window statistic over the last N grid periods.',
    'ratio': 'Column A ÷ Column B.',
    'product': 'Column A × Column B.',
    'log': 'Natural log of the column.',
    'winsorize': 'Clip extreme tails at the p / (1−p) quantiles.',
    'norm': 'Normalize across all tickers within each date.',
}
_FIELD_IDS = ('col', 'colb', 'k', 'window', 'fn', 'method', 'p')

# Preset packs: base-include every factor in a registry group (or all).
_ALL_BASE_COLS = [f.name for f in all_factors()]
_GROUP_COLS: dict[str, list[str]] = {}
for _f in all_factors():
    if _f.group:
        _GROUP_COLS.setdefault(_f.group, []).append(_f.name)
_PACK_OPTIONS = [{'label': 'All base columns', 'value': '__all__'}] + [
    {'label': f'{g.title()} ({len(cols)})', 'value': g}
    for g, cols in sorted(_GROUP_COLS.items())
]


# Column palette for the dropdowns: registry factors + raw price/volume.
_PRICE_VOL_OPTS = [
    {'label': 'Close price', 'value': 'close'},
    {'label': 'Volume', 'value': 'volume'},
]
_COL_OPTIONS = FACTOR_OPTIONS + _PRICE_VOL_OPTS
# Dense mode drops price-dependent factors (incoherent next to a daily close).
_DENSE_EXCL = {'mktcap', 'pe', 'pb', 'ps', 'ev_ebitda', 'ev_ebit', 'ev_sales',
               'fcf_yield', 'mom_12_1', 'mom_6_1', 'vol_21d', 'ma200_ratio'}
_COL_OPTIONS_DENSE = [o for o in FACTOR_OPTIONS if o['value'] not in _DENSE_EXCL] + _PRICE_VOL_OPTS


def _parse_ints(text, default: int) -> list[int]:
    """Parse an int list with comma + range syntax.

    '1,2,4'      -> [1, 2, 4]
    '1-5'        -> [1, 2, 3, 4, 5]          (inclusive range, for price windows)
    '1-10:2'     -> [1, 3, 5, 7, 9]          (range with step)
    '1,5-7,20'   -> [1, 5, 6, 7, 20]         (mixed)
    blank        -> [default]
    Dedups, preserves order.
    """
    if text is None or str(text).strip() == '':
        return [default]
    out: list[int] = []

    def _add(v: int) -> None:
        if v not in out:
            out.append(v)

    for tok in str(text).replace(' ', '').split(','):
        if tok == '':
            continue
        if '-' in tok.lstrip('-'):  # a range like 1-5 or 1-10:2 (not a lone negative)
            body, _, step_s = tok.partition(':')
            a_s, _, b_s = body.partition('-')
            try:
                a, b = int(a_s), int(b_s)
                step = int(step_s) if step_s else 1
            except ValueError:
                continue
            if step <= 0:
                step = 1
            for v in range(a, b + 1, step):
                _add(v)
        else:
            try:
                _add(int(float(tok)))
            except ValueError:
                continue
    return out or [default]


def _as_list(v) -> list:
    if v is None:
        return []
    return v if isinstance(v, list) else [v]


# ── Layout ────────────────────────────────────────────────────────────

layout = html.Div(
    className='features-page',
    children=[
        dcc.Store(id='features-init', data=1),
        dcc.Store(id='features-steps-store', data=[]),
        dcc.Store(id='features-build-store'),  # {token, spec, n_rows, n_cols, missing, head}
        dcc.Store(id='features-recipe-trigger', data=0),
        html.H2('Feature Engineering', className='page-title'),
        html.P(
            'Compose ML features from cached factor snapshots, attach a label, export.',
            className='home-subtitle',
        ),

        # ── Step 1: universe & date grid ──────────────────────────────
        html.H4('1 · Universe & dates', className='section-title',
                style={'margin': '14px 0 4px', 'color': ACCENT}),
        html.P('Yearly/Quarterly = cross-section at filing snapshots. '
               'Monthly/Weekly/Daily = dense price SEQUENCE per ticker (close/volume/TA '
               'dense; fundamentals carried forward from last filing; price-ratio factors '
               'like P/E unavailable in dense mode — build them from Close).',
               style={'color': MUTED, 'fontSize': '12px', 'margin': '0 0 6px'}),
        html.Div(className='control-row', style={'alignItems': 'flex-end'}, children=[
            _field('From year', _num('feat-start-year', '2015', 2015, '90px')),
            _field('To year', _num('feat-end-year', '2025', 2025, '90px')),
            _field('Sample a row every',
                   dcc.RadioItems(id='feat-freq', options=_FREQ_OPTIONS, value='A',
                                  inline=True, labelClassName='check-item')),
            _field('Fundamentals source',
                   dcc.RadioItems(id='feat-variant', options=_VARIANT_OPTIONS, value='A',
                                  inline=True, labelClassName='check-item')),
            _field('Market', _dd('feat-market', [], placeholder='All', width='120px', clearable=True)),
            _field('Sector', _dd('feat-sector', [], placeholder='All', width='150px', clearable=True)),
            _field('Watchlist', _dd('feat-watchlist', [], placeholder='None', width='150px', clearable=True)),
            _field('Cache', html.Button('Precompute', id='feat-precompute',
                                        className='run-btn', n_clicks=0)),
        ]),
        html.P('Precompute builds the snapshot cache for the chosen years + fundamentals '
               'source. Run it once if a build reports cold/missing dates.',
               style={'color': MUTED, 'fontSize': '12px', 'margin': '4px 0 0'}),

        # ── Step 2: add feature steps ─────────────────────────────────
        html.H4('2 · Add features', className='section-title',
                style={'margin': '16px 0 4px', 'color': ACCENT}),
        html.P('Pick an operation; select one or MANY columns; k/window accept lists '
               'like 1,2,4. Add expands every combination at once.',
               style={'color': MUTED, 'fontSize': '12px', 'margin': '0 0 6px'}),
        html.Div(className='control-row', style={'alignItems': 'flex-end'}, children=[
            _field('Operation', _dd('feat-op', _OP_OPTIONS, value='lag', width='180px')),
            _field('Column(s)', dcc.Dropdown(
                id='feat-col-a', options=_COL_OPTIONS, value=['roe'], multi=True,
                className='filter-dropdown', style={'minWidth': '220px'}),
                wrap_id='feat-f-col'),
            _field('Column(s) B', dcc.Dropdown(
                id='feat-col-b', options=_COL_OPTIONS, value=['revenue'], multi=True,
                className='filter-dropdown', style={'minWidth': '200px'}),
                wrap_id='feat-f-colb'),
            _field('Periods back (k: 1,2,4 or 1-20)',
                   dcc.Input(id='feat-k', type='text', value='1', className='filter-input',
                             style={'width': '150px'}), wrap_id='feat-f-k'),
            _field('Window (4,8 or 1-20:2)',
                   dcc.Input(id='feat-window', type='text', value='4', className='filter-input',
                             style={'width': '150px'}), wrap_id='feat-f-window'),
            _field('Statistic', _dd('feat-fn', _ROLL_FN_OPTIONS, value='mean', width='100px'),
                   wrap_id='feat-f-fn'),
            _field('Method', _dd('feat-method', _NORM_OPTIONS, value='zscore', width='140px'),
                   wrap_id='feat-f-method'),
            _field('Tail fraction (p)', _num('feat-p', '0.01', 0.01, '90px'), wrap_id='feat-f-p'),
            html.Button('Add', id='feat-add-step', className='run-btn', n_clicks=0),
        ]),
        html.P(id='feat-op-hint', style={'color': MUTED, 'fontSize': '12px', 'margin': '4px 0'}),
        # Quick add: preset packs (base-include a whole factor group)
        html.Div(className='control-row', style={'alignItems': 'flex-end'}, children=[
            _field('Quick pack', _dd('feat-pack', _PACK_OPTIONS, placeholder='Pick a pack…',
                                     width='200px', clearable=True)),
            html.Button('Add pack', id='feat-add-pack', className='run-btn', n_clicks=0),
            html.Button('Clear all', id='feat-clear-steps', className='run-btn', n_clicks=0),
        ]),
        html.Div(id='feat-step-stack', style={'margin': '8px 0', 'maxWidth': '900px'}),

        # ── Step 3: label (prediction target) ─────────────────────────
        html.H4('3 · Label (what the model predicts)', className='section-title',
                style={'margin': '16px 0 4px', 'color': ACCENT}),
        html.P('Forward return over the horizon, optionally turned into a class.',
               style={'color': MUTED, 'fontSize': '12px', 'margin': '0 0 6px'}),
        html.Div(className='control-row', style={'alignItems': 'flex-end'}, children=[
            _field('Forward horizon', _dd('feat-horizon', _HORIZON_OPTIONS, value=252, width='100px')),
            _field('Target type',
                   dcc.RadioItems(id='feat-label-mode', options=_LABEL_MODE_OPTIONS,
                                  value='continuous', inline=True, labelClassName='check-item')),
            _field('Quantile buckets', _num('feat-buckets', '5', 5, '110px')),
        ]),

        # ── Step 3b: scaling (model preprocessing) ────────────────────
        html.H4('3b · Scale features (for the model)', className='section-title',
                style={'margin': '16px 0 4px', 'color': ACCENT}),
        html.P('Most models want scaled inputs. Scales every feature column (not the '
               'label). Per-date fits within each date — leak-free for walk-forward. '
               'Per-ticker / Global fit on the train years (≤ cutoff) only; that is a '
               'single train/test split, not expanding walk-forward (for the latter, '
               'put the scaler in the notebook\'s sklearn Pipeline instead).',
               style={'color': MUTED, 'fontSize': '12px', 'margin': '0 0 6px'}),
        html.Div(className='control-row', style={'alignItems': 'flex-end'}, children=[
            _field('Method',
                   dcc.RadioItems(id='feat-scale-method', options=_SCALE_METHOD_OPTIONS,
                                  value='none', inline=True, labelClassName='check-item')),
            _field('Fit scope',
                   dcc.RadioItems(id='feat-scale-scope', options=_SCALE_SCOPE_OPTIONS,
                                  value='date', inline=True, labelClassName='check-item')),
            _field('Train cutoff year', _num('feat-scale-cutoff', '2020', None, '110px'),
                   wrap_id='feat-f-scale-cutoff'),
        ]),
        html.P('Heavy-tailed columns (named in build notes after a build) stay huge '
               'after robust/minmax. Pick them here to clip / log before scaling, or '
               'drop them. Clip & Log fit on the train rows.',
               style={'color': MUTED, 'fontSize': '12px', 'margin': '6px 0 4px'}),
        html.Div(className='control-row', style={'alignItems': 'flex-end'},
                 id='feat-tame-row', children=[
            _field('Heavy tails',
                   dcc.RadioItems(id='feat-scale-tame-action',
                                  options=[{'label': 'None', 'value': 'none'},
                                           {'label': 'Clip', 'value': 'clip'},
                                           {'label': 'Log', 'value': 'log'},
                                           {'label': 'Drop', 'value': 'drop'}],
                                  value='none', inline=True, labelClassName='check-item')),
            _field('Columns',
                   dcc.Dropdown(id='feat-scale-tame-cols', options=[], value=[],
                                multi=True, placeholder='columns to tame…',
                                style={'width': '320px'})),
            _field('Clip p', _num('feat-scale-tame-p', '0.01', 0.01, '90px'),
                   wrap_id='feat-f-tame-p'),
        ]),

        # ── Step 3c: train / valid / test split ───────────────────────
        html.H4('3c · Train / valid / test split', className='section-title',
                style={'margin': '16px 0 4px', 'color': ACCENT}),
        html.P('Optional. Exports 3 files (train/valid/test) instead of 1. '
               'By date = chronological (test most recent, no look-ahead). Leave '
               'tickers out = whole tickers held out (seeded). When scaling is on, '
               'the scaler fits on the train split only.',
               style={'color': MUTED, 'fontSize': '12px', 'margin': '0 0 6px'}),
        html.Div(className='control-row', style={'alignItems': 'flex-end'}, children=[
            _field('Method',
                   dcc.RadioItems(id='feat-split-method', options=_SPLIT_METHOD_OPTIONS,
                                  value='none', inline=True, labelClassName='check-item')),
            _field('Train frac', _num('feat-split-train', '0.7', 0.7, '80px'),
                   wrap_id='feat-f-split-train'),
            _field('Valid frac', _num('feat-split-valid', '0.15', 0.15, '80px'),
                   wrap_id='feat-f-split-valid'),
            _field('Test frac', _num('feat-split-test', '0.15', 0.15, '80px'),
                   wrap_id='feat-f-split-test'),
            _field('Seed', _num('feat-split-seed', '0', 0, '70px'),
                   wrap_id='feat-f-split-seed'),
        ]),

        # ── Step 4: build, export, save ───────────────────────────────
        html.H4('4 · Build & export', className='section-title',
                style={'margin': '16px 0 4px', 'color': ACCENT}),
        html.Div(className='control-row', children=[
            html.Button('Run Build', id='feat-run', className='run-btn', n_clicks=0),
            html.Button('Export parquet', id='feat-export-parquet', className='run-btn', n_clicks=0),
            html.Button('Export CSV', id='feat-export-csv', className='run-btn', n_clicks=0),
            dcc.Input(id='feat-recipe-name', type='text', placeholder='recipe name',
                      className='filter-input', style={'width': '160px'}),
            html.Button('Save recipe', id='feat-save-recipe', className='run-btn', n_clicks=0),
            _dd('feat-recipe-load', [], placeholder='Load recipe…', width='180px', clearable=True),
        ]),

        dcc.Loading(
            html.Div(id='feat-status', style={'margin': '8px 0', 'fontSize': '13px', 'color': ACCENT}),
            type='dot', delay_show=150,
        ),
        dcc.Loading(html.Div(id='feat-chips', style={'margin': '8px 0'})),
        dcc.Loading(html.Div(id='feat-preview')),
    ],
)


# ── Callbacks ─────────────────────────────────────────────────────────

@callback(
    Output('feat-f-col', 'style'), Output('feat-f-colb', 'style'),
    Output('feat-f-k', 'style'), Output('feat-f-window', 'style'),
    Output('feat-f-fn', 'style'), Output('feat-f-method', 'style'),
    Output('feat-f-p', 'style'),
    Output('feat-op-hint', 'children'),
    Input('feat-op', 'value'),
)
def toggle_step_inputs(op):
    """Show only the inputs the chosen operation needs; update the hint."""
    needed = _OP_FIELDS.get(op, set())
    styles = [dict(_SHOW) if f in needed else dict(_HIDE) for f in _FIELD_IDS]
    return (*styles, _OP_HINT.get(op, ''))


@callback(
    Output('feat-f-scale-cutoff', 'style'),
    Input('feat-scale-scope', 'value'),
)
def toggle_scale_cutoff(scope):
    """Train cutoff only matters for the global / per-ticker fit scopes."""
    return dict(_SHOW) if scope in ('global', 'ticker') else dict(_HIDE)


@callback(
    Output('feat-tame-row', 'style'),
    Output('feat-f-tame-p', 'style'),
    Input('feat-scale-method', 'value'),
    Input('feat-scale-tame-action', 'value'),
)
def toggle_tame_inputs(method, action):
    """Tame row only when scaling is on; Clip p only for the clip action."""
    row = dict(_HIDE) if method in (None, 'none') else {'display': 'flex'}
    p = dict(_SHOW) if action == 'clip' else dict(_HIDE)
    return row, p


@callback(
    Output('feat-f-split-train', 'style'), Output('feat-f-split-valid', 'style'),
    Output('feat-f-split-test', 'style'), Output('feat-f-split-seed', 'style'),
    Input('feat-split-method', 'value'),
)
def toggle_split_inputs(method):
    """Show fraction inputs when splitting; seed only for the ticker leave-out."""
    on = method in ('date', 'ticker')
    frac = dict(_SHOW) if on else dict(_HIDE)
    seed = dict(_SHOW) if method == 'ticker' else dict(_HIDE)
    return frac, frac, frac, seed


@callback(
    Output('feat-col-a', 'options'), Output('feat-col-b', 'options'),
    Output('feat-scale-tame-cols', 'options'),
    Input('feat-freq', 'value'),
)
def update_col_options(freq):
    """Dense frequencies expose the fundamentals-only + price/volume palette."""
    opts = _COL_OPTIONS_DENSE if freq in _DENSE_FREQS else _COL_OPTIONS
    return opts, opts, opts


@callback(
    Output('feat-market', 'options'),
    Output('feat-sector', 'options'),
    Output('feat-watchlist', 'options'),
    Output('feat-recipe-load', 'options'),
    Input('features-init', 'data'),
    Input('features-recipe-trigger', 'data'),
)
def load_options(_: Any, _trigger: Any):
    from irp.ui.services import universe_service
    try:
        comp = universe_service._get_companies()
        markets = [{'label': m, 'value': m} for m in sorted(comp['Market'].dropna().unique())]
        sectors = [{'label': s, 'value': s} for s in sorted(comp['Sector'].dropna().unique())]
    except Exception:
        logger.exception('features load_options failed')
        markets, sectors = [], []
    wl = watchlist_service.list_watchlists()
    wl_opts = [{'label': f'{r["name"]} ({r["n"]})', 'value': r['name']} for _, r in wl.iterrows()]
    rc = features_service.list_recipes()
    rc_opts = [{'label': r['name'], 'value': r['name']} for _, r in rc.iterrows()]
    return markets, sectors, wl_opts, rc_opts


def _step_label(step: dict) -> str:
    op = step['op']
    if op == 'base':
        return f'base · {step["col"]}'
    if op in ('lag', 'diff', 'pct_change'):
        return f'{op} · {step["col"]} k={step["k"]}'
    if op == 'lagwin':
        return f'window · {step["col"]} p0..p-{step.get("n", step.get("window"))}'
    if op == 'rolling':
        return f'rolling · {step["fn"]}({step["col"]}, {step["window"]})'
    if op in ('ratio', 'product'):
        sym = '/' if op == 'ratio' else '×'
        return f'{step["a"]} {sym} {step["b"]}'
    if op == 'log':
        return f'log · {step["col"]}'
    if op == 'winsorize':
        return f'winsorize · {step["col"]} p={step["p"]}'
    if op == 'norm':
        return f'norm · {step["method"]}({step["cols"][0]})'
    return op


def _expand_add(op, cols_a, cols_b, ks, windows, fn, method, p) -> list[dict]:
    """Cartesian-expand the add-controls into many step dicts."""
    if op == 'base':
        return [{'op': 'base', 'col': c} for c in cols_a]
    if op in ('lag', 'diff', 'pct_change'):
        return [{'op': op, 'col': c, 'k': k} for c in cols_a for k in ks]
    if op == 'lagwin':
        n = max(windows) if windows else 1
        return [{'op': 'lagwin', 'col': c, 'n': int(n)} for c in cols_a]
    if op == 'rolling':
        return [{'op': 'rolling', 'col': c, 'window': w, 'fn': fn or 'mean'}
                for c in cols_a for w in windows]
    if op in ('ratio', 'product'):
        return [{'op': op, 'a': a, 'b': b}
                for a in cols_a for b in cols_b if a != b]
    if op == 'log':
        return [{'op': 'log', 'col': c} for c in cols_a]
    if op == 'winsorize':
        return [{'op': 'winsorize', 'col': c, 'p': float(p or 0.01)} for c in cols_a]
    if op == 'norm':
        return [{'op': 'norm', 'cols': [c], 'method': method or 'zscore'} for c in cols_a]
    return []


def _append_dedup(steps: list[dict], new: list[dict]) -> list[dict]:
    for s in new:
        if s not in steps:
            steps.append(s)
    return steps


@callback(
    Output('features-steps-store', 'data'),
    Input('feat-add-step', 'n_clicks'),
    Input('feat-add-pack', 'n_clicks'),
    Input('feat-clear-steps', 'n_clicks'),
    Input({'type': 'feat-del-step', 'index': ALL}, 'n_clicks'),
    Input('feat-recipe-load', 'value'),
    State('feat-op', 'value'),
    State('feat-col-a', 'value'),
    State('feat-col-b', 'value'),
    State('feat-k', 'value'),
    State('feat-window', 'value'),
    State('feat-fn', 'value'),
    State('feat-method', 'value'),
    State('feat-p', 'value'),
    State('feat-pack', 'value'),
    State('features-steps-store', 'data'),
    prevent_initial_call=True,
)
def mutate_steps(add_n, pack_n, clear_n, del_clicks, recipe_name, op, col_a, col_b,
                 k, window, fn, method, p, pack, steps):
    steps = list(steps or [])
    trig = ctx.triggered_id

    if trig == 'feat-recipe-load':
        if not recipe_name:
            raise PreventUpdate
        try:
            spec = features_service.load_recipe(recipe_name)
        except KeyError:
            raise PreventUpdate from None
        return spec.get('steps', [])

    if trig == 'feat-clear-steps':
        return []

    if isinstance(trig, dict) and trig.get('type') == 'feat-del-step':
        if not any(del_clicks or []):
            raise PreventUpdate
        idx = trig['index']
        if 0 <= idx < len(steps):
            steps.pop(idx)
        return steps

    if trig == 'feat-add-pack':
        if not pack:
            raise PreventUpdate
        cols = _ALL_BASE_COLS if pack == '__all__' else _GROUP_COLS.get(pack, [])
        return _append_dedup(steps, [{'op': 'base', 'col': c} for c in cols])

    if trig == 'feat-add-step':
        cols_a = _as_list(col_a)
        if op in ('ratio', 'product'):
            cols_b = _as_list(col_b)
            if not cols_a or not cols_b:
                raise PreventUpdate
        else:
            cols_b = []
            if not cols_a and op != 'base':
                raise PreventUpdate
            if not cols_a:
                raise PreventUpdate
        new = _expand_add(op, cols_a, cols_b, _parse_ints(k, 1),
                          _parse_ints(window, 4), fn, method, p)
        if not new:
            raise PreventUpdate
        return _append_dedup(steps, new)

    raise PreventUpdate


@callback(
    Output('feat-step-stack', 'children'),
    Input('features-steps-store', 'data'),
)
def render_step_stack(steps):
    if not steps:
        return html.Span('No feature steps yet.', style={'color': MUTED, 'fontSize': '12px'})
    rows = []
    for i, step in enumerate(steps):
        rows.append(html.Div(
            style={'display': 'flex', 'alignItems': 'center', 'gap': '12px',
                   'padding': '4px 8px', 'background': 'var(--surface-2)',
                   'borderRadius': '4px', 'marginBottom': '4px', 'fontSize': '12px'},
            children=[
                html.Span(f'{i + 1}.', style={'color': MUTED}),
                html.Span(_step_label(step), style={'color': 'var(--text)', 'flex': '1'}),
                html.Button('×', id={'type': 'feat-del-step', 'index': i}, n_clicks=0,
                            style={'background': 'none', 'border': 'none', 'color': MUTED,
                                   'cursor': 'pointer', 'fontSize': '14px', 'padding': '0 4px'}),
            ],
        ))
    return rows


def _spec(start, end, freq, variant, steps, horizon, mode, buckets,
          market, sector, watchlist, scale_method='none', scale_scope='date',
          scale_cutoff=None, tame_action='none', tame_cols=None, tame_p=0.01,
          split_method='none', split_train=0.7, split_valid=0.15,
          split_test=0.15, split_seed=0) -> dict:
    return {
        'start': int(start or 2015), 'end': int(end or 2025),
        'freq': freq, 'variant': variant, 'steps': steps or [],
        'label': {'mode': mode, 'horizon_days': int(horizon or 252),
                  'n_buckets': int(buckets or 5)},
        'scale': {'method': scale_method or 'none', 'scope': scale_scope or 'date',
                  'train_cutoff': int(scale_cutoff) if scale_cutoff not in (None, '') else None,
                  'tame_action': tame_action or 'none', 'tame_cols': tame_cols or [],
                  'tame_p': float(tame_p) if tame_p not in (None, '') else 0.01},
        'split': {'method': split_method or 'none',
                  'train': float(split_train or 0.7), 'valid': float(split_valid or 0.15),
                  'test': float(split_test or 0.15), 'seed': int(split_seed or 0)},
        'filters': {'market': market, 'sector': sector, 'watchlist': watchlist},
    }


def _run_spec(spec: dict):
    f = spec['filters']
    return features_service.build_panel(
        spec['start'], spec['end'], spec['freq'], spec['variant'],
        spec['steps'], spec['label'],
        market=f.get('market'), sector=f.get('sector'), watchlist=f.get('watchlist'),
        scale_cfg=spec.get('scale'), split_cfg=spec.get('split'),
    )


def _cache_put(spec: dict, df: pd.DataFrame) -> str:
    token = str(abs(hash(repr(spec))))
    _BUILD_CACHE[token] = df
    while len(_BUILD_CACHE) > _BUILD_CACHE_CAP:
        _BUILD_CACHE.pop(next(iter(_BUILD_CACHE)))
    return token


@callback(
    Output('feat-status', 'children', allow_duplicate=True),
    Input('feat-precompute', 'n_clicks'),
    State('feat-start-year', 'value'), State('feat-end-year', 'value'),
    State('feat-variant', 'value'),
    running=[(Output('feat-precompute', 'disabled'), True, False),
             (Output('feat-precompute', 'children'), 'Computing…', 'Precompute')],
    prevent_initial_call=True,
)
def precompute_cache(n, start, end, variant):
    try:
        wrote = features_service.precompute(start or 2015, end or 2025, variant or 'A')
    except Exception as exc:
        logger.exception('precompute failed')
        return f'Precompute failed: {exc}'
    return (f'Precompute done for variant "{variant}", {start}–{end}: '
            f'{wrote} new snapshot(s) written. Now click Run Build.')


@callback(
    Output('features-build-store', 'data'),
    Output('feat-status', 'children'),
    Output('feat-chips', 'children'),
    Input('feat-run', 'n_clicks'),
    State('feat-start-year', 'value'), State('feat-end-year', 'value'),
    State('feat-freq', 'value'), State('feat-variant', 'value'),
    State('features-steps-store', 'data'),
    State('feat-horizon', 'value'), State('feat-label-mode', 'value'),
    State('feat-buckets', 'value'),
    State('feat-market', 'value'), State('feat-sector', 'value'),
    State('feat-watchlist', 'value'),
    State('feat-scale-method', 'value'), State('feat-scale-scope', 'value'),
    State('feat-scale-cutoff', 'value'),
    State('feat-scale-tame-action', 'value'), State('feat-scale-tame-cols', 'value'),
    State('feat-scale-tame-p', 'value'),
    State('feat-split-method', 'value'), State('feat-split-train', 'value'),
    State('feat-split-valid', 'value'), State('feat-split-test', 'value'),
    State('feat-split-seed', 'value'),
    running=[(Output('feat-run', 'disabled'), True, False),
             (Output('feat-run', 'children'), 'Building…', 'Run Build')],
    prevent_initial_call=True,
)
def run_build(n, start, end, freq, variant, steps, horizon, mode, buckets,
              market, sector, watchlist, scale_method, scale_scope, scale_cutoff,
              tame_action, tame_cols, tame_p,
              split_method, split_train, split_valid, split_test, split_seed):
    spec = _spec(start, end, freq, variant, steps, horizon, mode, buckets,
                 market, sector, watchlist, scale_method, scale_scope, scale_cutoff,
                 tame_action, tame_cols, tame_p,
                 split_method, split_train, split_valid, split_test, split_seed)
    try:
        df, missing = _run_spec(spec)
    except Exception as exc:
        logger.exception('features build failed')
        return None, f'Build failed: {exc}', None

    if df.empty:
        if missing:
            # cache fully cold for this range
            msg = (f'No cached snapshots for the {len(missing)} grid date(s) of variant '
                   f'"{variant}". Run precompute_all for "{variant}" over this range first '
                   f'(e.g. {", ".join(missing[:3])}).')
        else:
            msg = 'No rows — check filters / date range.'
        return None, msg, None

    token = _cache_put(spec, df)
    # Representative preview: spread across the (Ticker, Date)-sorted frame so the
    # sample shows many tickers, not just the alphabetically-first one.
    step = max(1, len(df) // 50)
    head = df.iloc[::step].head(50).to_dict('records')
    store = {'token': token, 'spec': spec, 'n_rows': len(df),
             'n_cols': df.shape[1], 'head': head, 'columns': list(df.columns)}

    status = 'Build complete.'
    if missing:
        # coarse path → list of skipped dates; dense path → free-text warnings
        looks_like_dates = all('-' in str(m) and str(m)[:4].isdigit() for m in missing)
        if looks_like_dates:
            status = (f'Build complete — skipped {len(missing)} uncached date(s) '
                      f'(e.g. {", ".join(missing[:3])}); built from the rest.')
        else:
            status = 'Build complete. Note: ' + '; '.join(missing)
    if len(df) > 2_000_000:
        status += f'  ⚠ large dataset ({len(df):,} rows) — export may be slow/big.'

    def chip(text):
        return html.Span(text, style={
            'background': 'var(--surface-2)', 'color': ACCENT, 'padding': '4px 10px',
            'borderRadius': '4px', 'marginRight': '8px', 'fontSize': '13px'})

    chips = [chip(f'{len(df):,} rows'), chip(f'{df.shape[1]} columns'),
             chip(f'{df["Ticker"].nunique():,} tickers')]
    if 'split' in df.columns:
        vc = df['split'].value_counts()
        chips.append(chip('split  ' + '  '.join(
            f'{s}:{int(vc.get(s, 0)):,}' for s in ('train', 'valid', 'test'))))
    return store, status, chips


@callback(
    Output('feat-preview', 'children'),
    Input('features-build-store', 'data'),
)
def render_preview(store):
    if not store or not store.get('head'):
        return html.P('Configure steps and click Run Build.', className='no-data')
    df = pd.DataFrame(store['head'])
    cols = []
    for c in df.columns:
        col = {'name': FACTOR_LABELS.get(c, c), 'id': c}
        if c not in ('Date', 'Ticker'):
            col.update(_col_fmt(c))
        cols.append(col)
    left = [{'if': {'column_id': c}, 'textAlign': 'left'} for c in ('Date', 'Ticker')]
    return _dt.DataTable(
        data=df.round(4).to_dict('records'), columns=cols,
        sort_action='native', page_size=25,
        **{**TABLE_STYLE,
           'style_cell_conditional': TABLE_STYLE.get('style_cell_conditional', []) + left},
    )


@callback(
    Output('feat-status', 'children', allow_duplicate=True),
    Input('feat-export-parquet', 'n_clicks'),
    Input('feat-export-csv', 'n_clicks'),
    State('features-build-store', 'data'),
    State('feat-recipe-name', 'value'),
    prevent_initial_call=True,
)
def export(parquet_n, csv_n, store, name):
    if not store or not store.get('token'):
        return 'Nothing to export — run a build first.'
    fmt = 'parquet' if ctx.triggered_id == 'feat-export-parquet' else 'csv'
    df = _BUILD_CACHE.get(store['token'])
    if df is None:
        df, missing = _run_spec(store['spec'])
        if missing or df.empty:
            return 'Could not rebuild for export — re-run the build.'
    out = features_service.export_panel(df, fmt, name=name or 'features')
    if isinstance(out, list):
        names = ', '.join(p.name for p in out)
        return f'Exported {len(df):,} rows → {len(out)} split files: {names}'
    return f'Exported {len(df):,} rows → {out}'


@callback(
    Output('features-recipe-trigger', 'data'),
    Output('feat-status', 'children', allow_duplicate=True),
    Input('feat-save-recipe', 'n_clicks'),
    State('feat-recipe-name', 'value'),
    State('feat-start-year', 'value'), State('feat-end-year', 'value'),
    State('feat-freq', 'value'), State('feat-variant', 'value'),
    State('features-steps-store', 'data'),
    State('feat-horizon', 'value'), State('feat-label-mode', 'value'),
    State('feat-buckets', 'value'),
    State('feat-market', 'value'), State('feat-sector', 'value'),
    State('feat-watchlist', 'value'),
    State('feat-scale-method', 'value'), State('feat-scale-scope', 'value'),
    State('feat-scale-cutoff', 'value'),
    State('feat-scale-tame-action', 'value'), State('feat-scale-tame-cols', 'value'),
    State('feat-scale-tame-p', 'value'),
    State('feat-split-method', 'value'), State('feat-split-train', 'value'),
    State('feat-split-valid', 'value'), State('feat-split-test', 'value'),
    State('feat-split-seed', 'value'),
    State('features-recipe-trigger', 'data'),
    prevent_initial_call=True,
)
def save_recipe(n, name, start, end, freq, variant, steps, horizon, mode, buckets,
                market, sector, watchlist, scale_method, scale_scope, scale_cutoff,
                tame_action, tame_cols, tame_p,
                split_method, split_train, split_valid, split_test, split_seed, trigger):
    if not name:
        return trigger, 'Enter a recipe name to save.'
    spec = _spec(start, end, freq, variant, steps, horizon, mode, buckets,
                 market, sector, watchlist, scale_method, scale_scope, scale_cutoff,
                 tame_action, tame_cols, tame_p,
                 split_method, split_train, split_valid, split_test, split_seed)
    features_service.save_recipe(name, spec)
    return (trigger or 0) + 1, f'Saved recipe "{name}".'


@callback(
    Output('feat-start-year', 'value'), Output('feat-end-year', 'value'),
    Output('feat-freq', 'value'), Output('feat-variant', 'value'),
    Output('feat-horizon', 'value'), Output('feat-label-mode', 'value'),
    Output('feat-buckets', 'value'),
    Output('feat-market', 'value'), Output('feat-sector', 'value'),
    Output('feat-watchlist', 'value'),
    Output('feat-scale-method', 'value'), Output('feat-scale-scope', 'value'),
    Output('feat-scale-cutoff', 'value'),
    Output('feat-scale-tame-action', 'value'), Output('feat-scale-tame-cols', 'value'),
    Output('feat-scale-tame-p', 'value'),
    Output('feat-split-method', 'value'), Output('feat-split-train', 'value'),
    Output('feat-split-valid', 'value'), Output('feat-split-test', 'value'),
    Output('feat-split-seed', 'value'),
    Input('feat-recipe-load', 'value'),
    prevent_initial_call=True,
)
def load_recipe_controls(name):
    if not name:
        raise PreventUpdate
    try:
        s = features_service.load_recipe(name)
    except KeyError:
        raise PreventUpdate from None
    lbl = s.get('label', {})
    f = s.get('filters', {})
    sc = s.get('scale', {})
    sp = s.get('split', {})
    return (s.get('start', 2015), s.get('end', 2025), s.get('freq', 'A'),
            s.get('variant', 'A'), lbl.get('horizon_days', 252), lbl.get('mode', 'continuous'),
            lbl.get('n_buckets', 5), f.get('market'), f.get('sector'), f.get('watchlist'),
            sc.get('method', 'none'), sc.get('scope', 'date'), sc.get('train_cutoff'),
            sc.get('tame_action', 'none'), sc.get('tame_cols', []), sc.get('tame_p', 0.01),
            sp.get('method', 'none'), sp.get('train', 0.7), sp.get('valid', 0.15),
            sp.get('test', 0.15), sp.get('seed', 0))

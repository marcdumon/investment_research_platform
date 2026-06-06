"""Feature Engineering page: load a built dataset, INSPECT heavy tails (see the
rows/columns/tickers causing trouble), decide and apply per-column taming
(drop/log/clip) + exclude bad tickers, then scale + split and export train/valid/
test. Scaling/splitting/cleaning live here; the Dataset Builder (/features) only
produces the raw dataset.
"""
import logging

import dash
import numpy as np
import pandas as pd
import plotly.graph_objects as go
from dash import ALL, Input, Output, State, callback, ctx, dash_table as _dt, dcc, html
from dash.exceptions import PreventUpdate

from irp.ui.charts import base_chart_layout, empty_figure
from irp.ui.services import features_service
from irp.ui.tables import column_format as _col_fmt
from irp.ui.theme import ACCENT, MUTED, TABLE_STYLE

dash.register_page(__name__, path='/feature-engineering', name='Feature Engineering')

logger = logging.getLogger(__name__)

# Server-side caches (avoid shipping big frames through dcc.Store).
_SRC_CACHE: dict[str, pd.DataFrame] = {}     # loaded raw datasets, by token
_CUR_CACHE: dict[str, pd.DataFrame] = {}     # current cleaned (exclude+tame) frames
_PREP_CACHE: dict[str, pd.DataFrame] = {}    # prepared (tamed/scaled/split) frames
_CACHE_CAP = 4

_RESERVED = ('Date', 'Ticker', 'fwd_ret', 'label', 'split')

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
_TAME_ACTION_OPTIONS = [
    {'label': ' None', 'value': 'none'},
    {'label': ' Clip', 'value': 'clip'},
    {'label': ' Log', 'value': 'log'},
    {'label': ' Drop column', 'value': 'drop'},
]

_SHOW = {'display': 'flex', 'flexDirection': 'column', 'gap': '2px'}
_HIDE = {'display': 'none'}


def _num(id_, placeholder, value=None, width='90px'):
    return dcc.Input(id=id_, type='number', placeholder=placeholder, value=value,
                     className='filter-input', style={'width': width})


def _dd(id_, options, value=None, placeholder='', width='150px', clearable=False, multi=False):
    return dcc.Dropdown(id=id_, options=options, value=value, placeholder=placeholder,
                        clearable=clearable, multi=multi, className='filter-dropdown',
                        style={'minWidth': width})


def _field(label, control, wrap_id=None):
    kw = {'id': wrap_id} if wrap_id else {}
    return html.Div(style=dict(_SHOW), **kw,
                    children=[html.Label(label, style={'color': MUTED, 'fontSize': '11px'}),
                              control])


def _cache_put(cache, key, df):
    cache[key] = df
    while len(cache) > _CACHE_CAP:
        cache.pop(next(iter(cache)))


def _chip(text):
    return html.Span(text, style={
        'background': 'var(--surface-2)', 'color': ACCENT, 'padding': '4px 10px',
        'borderRadius': '4px', 'marginRight': '8px', 'fontSize': '13px'})


def _table(df, page_size=12):
    cols = [{'name': c, 'id': c} for c in df.columns]
    return _dt.DataTable(
        data=df.round(4).to_dict('records') if not df.empty else [],
        columns=cols, sort_action='native', page_size=page_size, **TABLE_STYLE,
    )


# ── layout ────────────────────────────────────────────────────────────

layout = html.Div(className='page', children=[
    dcc.Store(id='fe-init', data=1),
    dcc.Store(id='fe-source-store'),       # {token, n_rows, n_cols, name}
    dcc.Store(id='fe-tame-store', data=[]),  # list of {col, action, p}
    dcc.Store(id='fe-build-store'),         # {token, n_rows, n_cols, head, columns}
    dcc.Store(id='fe-recipe-trigger', data=0),

    html.H2('Feature Engineering', className='page-title'),
    html.P('Load a dataset built on the Dataset Builder, see which columns/rows/tickers '
           'are pathological, then clean (clip / log / drop / exclude ticker), scale and '
           'split. All transforms are explicit; nothing is applied without you choosing it.',
           style={'color': MUTED, 'fontSize': '13px'}),

    # ── 1 · Source ───────────────────────────────────────────────────
    html.H4('1 · Source dataset', className='section-title',
            style={'margin': '14px 0 4px', 'color': ACCENT}),
    html.Div(className='control-row', style={'alignItems': 'flex-end'}, children=[
        html.Button('Use last build', id='fe-use-last', className='run-btn', n_clicks=0),
        _field('Or load an export', _dd('fe-dataset', [], placeholder='Pick a dataset…',
                                        width='280px', clearable=True)),
        html.Button('Load', id='fe-load', className='run-btn', n_clicks=0),
    ]),
    dcc.Loading(html.Div(id='fe-source-status',
                         style={'margin': '6px 0', 'fontSize': '13px', 'color': ACCENT})),

    # ── 2 · Inspect & clean (per column) ─────────────────────────────
    html.H4('2 · Inspect & clean', className='section-title',
            style={'margin': '16px 0 4px', 'color': ACCENT}),
    html.P('Run Inspect to flag fat-tailed columns. Pick a column to see its worst rows, '
           'which tickers own the outliers, and its distribution; choose an action and '
           'Apply — the view below updates for THAT column so you confirm the fix, then '
           'move to the next column. (Run Inspect again to refresh the overview table.)',
           style={'color': MUTED, 'fontSize': '12px', 'margin': '0 0 6px'}),
    html.Div(className='control-row', style={'alignItems': 'flex-end'}, children=[
        html.Button('Run Inspect', id='fe-inspect', className='run-btn', n_clicks=0),
    ]),
    dcc.Loading(html.Div(id='fe-summary', style={'margin': '8px 0'})),

    # column workspace: select one or MANY columns; the drill-down shows the first
    # selected, Apply hits every selected column at once.
    html.Div(className='control-row', style={'alignItems': 'flex-end'}, children=[
        _field('Column(s)', _dd('fe-col', [], placeholder='columns…', width='280px',
                                clearable=True, multi=True)),
        _field('Action', dcc.RadioItems(id='fe-tame-action', options=_TAME_ACTION_OPTIONS,
                                        value='none', inline=True, labelClassName='check-item')),
        _field('Clip p', _num('fe-tame-p', '0.01', 0.01, '90px'), wrap_id='fe-f-tame-p'),
        html.Button('Apply to selected', id='fe-tame-add', className='run-btn', n_clicks=0),
    ]),
    html.P('One histogram per selected column. For Clip, dashed lines mark the '
           '[p, 1−p] cut points; for Log, the curve is the signed-log result.',
           style={'color': MUTED, 'fontSize': '11px', 'margin': '4px 0'}),
    dcc.Loading(html.Div(id='fe-hist-container',
                         style={'display': 'flex', 'gap': '16px', 'flexWrap': 'wrap'})),

    # cleaning plan (queued actions) + exclude tickers
    html.Div(className='control-row', style={'alignItems': 'flex-end', 'marginTop': '8px'}, children=[
        html.Button('Clear plan', id='fe-tame-clear', className='run-btn', n_clicks=0),
        _field('Exclude tickers (comma list)',
               dcc.Input(id='fe-exclude', type='text', placeholder='e.g. BADCO, XYZ',
                         className='filter-input', style={'width': '320px'})),
    ]),
    html.Div(id='fe-tame-stack', style={'margin': '8px 0', 'maxWidth': '700px'}),

    # ── 4 · Scale + split ────────────────────────────────────────────
    html.H4('4 · Scale & split', className='section-title',
            style={'margin': '16px 0 4px', 'color': ACCENT}),
    html.Div(className='control-row', style={'alignItems': 'flex-end'}, children=[
        _field('Scale method', dcc.RadioItems(id='fe-scale-method', options=_SCALE_METHOD_OPTIONS,
                                              value='none', inline=True, labelClassName='check-item')),
        _field('Fit scope', dcc.RadioItems(id='fe-scale-scope', options=_SCALE_SCOPE_OPTIONS,
                                           value='date', inline=True, labelClassName='check-item')),
        _field('Train cutoff year', _num('fe-scale-cutoff', '2020', None, '110px'),
               wrap_id='fe-f-scale-cutoff'),
    ]),
    html.Div(className='control-row', style={'alignItems': 'flex-end'}, children=[
        _field('Split method', dcc.RadioItems(id='fe-split-method', options=_SPLIT_METHOD_OPTIONS,
                                              value='none', inline=True, labelClassName='check-item')),
        _field('Train frac', _num('fe-split-train', '0.7', 0.7, '80px'), wrap_id='fe-f-split-train'),
        _field('Valid frac', _num('fe-split-valid', '0.15', 0.15, '80px'), wrap_id='fe-f-split-valid'),
        _field('Test frac', _num('fe-split-test', '0.15', 0.15, '80px'), wrap_id='fe-f-split-test'),
        _field('Seed', _num('fe-split-seed', '0', 0, '70px'), wrap_id='fe-f-split-seed'),
    ]),

    # ── 5 · Prepare + export ─────────────────────────────────────────
    html.H4('5 · Prepare & export', className='section-title',
            style={'margin': '16px 0 4px', 'color': ACCENT}),
    html.Div(className='control-row', children=[
        html.Button('Run Prepare', id='fe-prepare', className='run-btn', n_clicks=0),
        html.Button('Export parquet', id='fe-export-parquet', className='run-btn', n_clicks=0),
        html.Button('Export CSV', id='fe-export-csv', className='run-btn', n_clicks=0),
        dcc.Input(id='fe-recipe-name', type='text', placeholder='FE recipe name',
                  className='filter-input', style={'width': '160px'}),
        html.Button('Save recipe', id='fe-save-recipe', className='run-btn', n_clicks=0),
        _dd('fe-recipe-load', [], placeholder='Load FE recipe…', width='180px', clearable=True),
    ]),
    dcc.Loading(html.Div(id='fe-status', style={'margin': '8px 0', 'fontSize': '13px', 'color': ACCENT}),
                type='dot', delay_show=150),
    dcc.Loading(html.Div(id='fe-chips', style={'margin': '8px 0'})),
    dcc.Loading(html.Div(id='fe-preview')),
])


# ── callbacks ─────────────────────────────────────────────────────────

@callback(
    Output('fe-dataset', 'options'),
    Output('fe-recipe-load', 'options'),
    Input('fe-init', 'data'),
    Input('fe-recipe-trigger', 'data'),
)
def fe_populate_dropdowns(_init, _trig):
    ds = features_service.list_datasets()
    ds_opts = [{'label': f'{r["file"]} ({r["mb"]} MB)', 'value': r['file']}
               for _, r in ds.iterrows()]
    rc = features_service.list_fe_recipes()
    rc_opts = [{'label': r['name'], 'value': r['name']} for _, r in rc.iterrows()]
    return ds_opts, rc_opts


def _numeric_cols(df):
    return [c for c in df.columns
            if c not in _RESERVED and pd.api.types.is_numeric_dtype(df[c])]


def _cur_frame(src, plan, exclude):
    """The dataset AS CURRENTLY CLEANED (source → exclude tickers → tame plan), with
    no scaling/splitting. Inspect runs on this so the loop is iterative: clean a
    column, re-inspect, see whether the problem is gone. Cached per (source, plan,
    exclude) so a drill-down doesn't re-clean."""
    df = _SRC_CACHE[src['token']]
    key = str(abs(hash((src['token'], repr(plan), repr(exclude)))))
    if key not in _CUR_CACHE:
        cur, _ = features_service.prepare_features(
            df, tame_plan=plan or [], exclude_tickers=_parse_exclude(exclude),
            scale_cfg=None, split_cfg=None)
        _cache_put(_CUR_CACHE, key, cur)
    return _CUR_CACHE[key]


@callback(
    Output('fe-source-store', 'data'),
    Output('fe-source-status', 'children'),
    Output('fe-col', 'options'),
    Input('fe-use-last', 'n_clicks'),
    Input('fe-load', 'n_clicks'),
    State('fe-dataset', 'value'),
    prevent_initial_call=True,
)
def fe_load_source(_last, _load, dataset):
    trig = ctx.triggered_id
    try:
        if trig == 'fe-use-last':
            last = features_service.get_last_build()
            if not last:
                return None, 'No in-memory build — build one on the Dataset Builder first.', []
            df, name = last['df'], 'last build'
        else:
            if not dataset:
                return None, 'Pick a dataset to load.', []
            df, name = features_service.load_dataset(dataset), dataset
    except Exception as exc:
        logger.exception('fe load source failed')
        return None, f'Load failed: {exc}', []

    token = str(abs(hash((name, df.shape[0], df.shape[1]))))
    _cache_put(_SRC_CACHE, token, df)
    opts = [{'label': c, 'value': c} for c in _numeric_cols(df)]
    msg = f'Loaded "{name}": {len(df):,} rows × {df.shape[1]} cols, {df["Ticker"].nunique():,} tickers.'
    return {'token': token, 'n_rows': len(df), 'n_cols': df.shape[1], 'name': name}, msg, opts


@callback(
    Output('fe-summary', 'children'),
    Output('fe-col', 'value'),
    Input('fe-inspect', 'n_clicks'),
    State('fe-source-store', 'data'),
    State('fe-tame-store', 'data'),
    State('fe-exclude', 'value'),
    prevent_initial_call=True,
)
def fe_run_inspect(_n, src, plan, exclude):
    if not src or src['token'] not in _SRC_CACHE:
        return html.P('Load a dataset first.', className='no-data'), None
    cur = _cur_frame(src, plan, exclude)   # overview of the CURRENTLY-cleaned dataset
    rep = features_service.inspect_dataset(cur, with_detail=False)
    if rep.summary.empty:
        return html.P('No heavy-tailed columns remain — the dataset looks clean.',
                      className='no-data'), None
    summ = rep.summary[['col', 'n', 'median', 'p1', 'p99', 'min', 'max', 'iqr',
                        'tail_ratio', 'n_outliers', 'worst_ticker']]
    return _table(summ, page_size=15), [summ.iloc[0]['col']]   # auto-drill the worst column


def _hist_figure(col, action, p, series):
    """Server-side-binned histogram (60 bars) for one column, previewing `action`.
    Clip → raw distribution with dashed lines at the [p, 1−p] cut points; Log → the
    signed-log result; None → raw."""
    vals = pd.to_numeric(series, errors='coerce').dropna().to_numpy() if series is not None else np.array([])
    if not vals.size:
        return empty_figure(f'{col}: no numeric values')
    counts, edges = np.histogram(vals, bins=60)
    centers = (edges[:-1] + edges[1:]) / 2
    fig = go.Figure(go.Bar(x=centers, y=counts, marker_color=ACCENT))
    suffix = {'clip': ' (clip bounds)', 'log': ' (after log)'}.get(action, '')
    fig.update_layout(base_chart_layout(title=f'{col}{suffix}', yaxis_title='count',
                                        xaxis_title=col))
    if action == 'clip':
        pc = min(max(float(p), 0.0), 0.4999)   # clamp to a valid tail fraction
        lo, hi = np.quantile(vals, pc), np.quantile(vals, 1 - pc)
        for x in (lo, hi):
            fig.add_vline(x=x, line_dash='dash', line_color='#e45756')
    fig.update_layout(height=300, margin={'l': 40, 'r': 10, 't': 36, 'b': 36})
    return fig


_MAX_HISTS = 12   # cap rendered charts so a huge multi-select can't flood the page


@callback(
    Output('fe-hist-container', 'children'),
    Input('fe-col', 'value'),
    Input('fe-tame-action', 'value'),
    Input('fe-tame-p', 'value'),
    State('fe-source-store', 'data'),
    State('fe-exclude', 'value'),
    prevent_initial_call=True,
)
def fe_histograms(cols, action, p, src, exclude):
    """One histogram per selected column, previewing the live Action/Clip-p so you see
    exactly what Apply will do. Computed per column via `column_preview` — no panel
    rebuild. (Tables removed; the charts carry the signal.)"""
    cols = cols if isinstance(cols, list) else ([cols] if cols else [])
    if not cols or not src or src['token'] not in _SRC_CACHE:
        raise PreventUpdate
    df = _SRC_CACHE[src['token']]
    excl = _parse_exclude(exclude)
    pf = float(p) if p not in (None, '') else 0.01
    graphs = []
    for col in cols[:_MAX_HISTS]:
        if action == 'drop':
            graphs.append(html.Div(html.P(f'"{col}" → DROP', className='no-data'),
                                   style={'flex': '1 1 360px'}))
            continue
        # Log previews the transformed series; Clip/None show the raw series.
        entry = {'col': col, 'action': 'log'} if action == 'log' else None
        series, _ = features_service.column_preview(df, col, tame_entry=entry, exclude_tickers=excl)
        fig = _hist_figure(col, action, pf, series)
        graphs.append(dcc.Graph(figure=fig, config={'displayModeBar': False},
                                style={'flex': '1 1 360px'}))
    if len(cols) > _MAX_HISTS:
        graphs.append(html.P(f'… {len(cols) - _MAX_HISTS} more selected (charts capped '
                             f'at {_MAX_HISTS}).', style={'color': MUTED, 'fontSize': '12px'}))
    return graphs


# ── tame plan editor ──────────────────────────────────────────────────

@callback(
    Output('fe-f-tame-p', 'style'),
    Input('fe-tame-action', 'value'),
)
def fe_toggle_tame_p(action):
    return dict(_SHOW) if action == 'clip' else dict(_HIDE)


@callback(
    Output('fe-tame-store', 'data'),
    Input('fe-tame-add', 'n_clicks'),
    Input('fe-tame-clear', 'n_clicks'),
    Input({'type': 'fe-tame-del', 'index': ALL}, 'n_clicks'),
    State('fe-col', 'value'),
    State('fe-tame-action', 'value'),
    State('fe-tame-p', 'value'),
    State('fe-tame-store', 'data'),
    prevent_initial_call=True,
)
def fe_edit_tame(_add, _clear, _dels, cols, action, p, plan):
    plan = plan or []
    trig = ctx.triggered_id
    if trig == 'fe-tame-clear':
        return []
    if isinstance(trig, dict) and trig.get('type') == 'fe-tame-del':
        i = trig['index']
        return [s for j, s in enumerate(plan) if j != i]
    if trig == 'fe-tame-add' and cols:
        cols = cols if isinstance(cols, list) else [cols]
        plan = [s for s in plan if s['col'] not in cols]       # one action per column
        if action and action != 'none':                        # 'none' just clears them
            pval = float(p) if p not in (None, '') else 0.01
            for c in cols:
                step = {'col': c, 'action': action}
                if action == 'clip':
                    step['p'] = pval
                plan = plan + [step]
    return plan


@callback(
    Output('fe-tame-stack', 'children'),
    Input('fe-tame-store', 'data'),
)
def fe_render_tame(plan):
    if not plan:
        return html.P('No cleaning actions yet.', style={'color': MUTED, 'fontSize': '12px'})
    rows = []
    for i, s in enumerate(plan):
        label = (f'{s["action"]} · {s["col"]}'
                 + (f' p={s["p"]}' if s['action'] == 'clip' else ''))
        rows.append(html.Div(style={'display': 'flex', 'alignItems': 'center', 'gap': '8px',
                                    'padding': '2px 0'}, children=[
            html.Span(label, style={'fontSize': '13px', 'color': ACCENT}),
            html.Button('×', id={'type': 'fe-tame-del', 'index': i},
                        style={'background': 'none', 'border': 'none', 'color': MUTED,
                               'cursor': 'pointer', 'fontSize': '14px', 'padding': '0 4px'}),
        ]))
    return rows


# ── visibility toggles ────────────────────────────────────────────────

@callback(
    Output('fe-f-scale-cutoff', 'style'),
    Input('fe-scale-scope', 'value'),
)
def fe_toggle_scale_cutoff(scope):
    return dict(_SHOW) if scope in ('global', 'ticker') else dict(_HIDE)


@callback(
    Output('fe-f-split-train', 'style'), Output('fe-f-split-valid', 'style'),
    Output('fe-f-split-test', 'style'), Output('fe-f-split-seed', 'style'),
    Input('fe-split-method', 'value'),
)
def fe_toggle_split(method):
    on = method in ('date', 'ticker')
    frac = dict(_SHOW) if on else dict(_HIDE)
    seed = dict(_SHOW) if method == 'ticker' else dict(_HIDE)
    return frac, frac, frac, seed


# ── prepare + export ──────────────────────────────────────────────────

def _scale_cfg(method, scope, cutoff):
    return {'method': method or 'none', 'scope': scope or 'date',
            'train_cutoff': int(cutoff) if cutoff not in (None, '') else None}


def _split_cfg(method, train, valid, test, seed):
    return {'method': method or 'none', 'train': float(train or 0.7),
            'valid': float(valid or 0.15), 'test': float(test or 0.15),
            'seed': int(seed or 0)}


def _parse_exclude(text):
    if not text:
        return []
    return [t.strip() for t in str(text).replace(';', ',').split(',') if t.strip()]


@callback(
    Output('fe-build-store', 'data'),
    Output('fe-status', 'children'),
    Output('fe-chips', 'children'),
    Input('fe-prepare', 'n_clicks'),
    State('fe-source-store', 'data'),
    State('fe-tame-store', 'data'),
    State('fe-exclude', 'value'),
    State('fe-scale-method', 'value'), State('fe-scale-scope', 'value'),
    State('fe-scale-cutoff', 'value'),
    State('fe-split-method', 'value'), State('fe-split-train', 'value'),
    State('fe-split-valid', 'value'), State('fe-split-test', 'value'),
    State('fe-split-seed', 'value'),
    running=[(Output('fe-prepare', 'disabled'), True, False),
             (Output('fe-prepare', 'children'), 'Preparing…', 'Run Prepare')],
    prevent_initial_call=True,
)
def fe_run_prepare(_n, src, plan, exclude, sm, ss, sc, spm, st, sv, se, sd):
    if not src or src['token'] not in _SRC_CACHE:
        return None, 'Load a dataset first.', None
    df = _SRC_CACHE[src['token']]
    scale_cfg = _scale_cfg(sm, ss, sc)
    split_cfg = _split_cfg(spm, st, sv, se, sd)
    try:
        out, notes = features_service.prepare_features(
            df, tame_plan=plan or [], exclude_tickers=_parse_exclude(exclude),
            scale_cfg=scale_cfg, split_cfg=split_cfg)
    except Exception as exc:
        logger.exception('fe prepare failed')
        return None, f'Prepare failed: {exc}', None
    if out.empty:
        return None, 'Nothing left after cleaning — check exclude list / actions.', None

    token = str(abs(hash((src['token'], repr(plan), repr(exclude),
                          repr(scale_cfg), repr(split_cfg)))))
    _cache_put(_PREP_CACHE, token, out)
    step = max(1, len(out) // 50)
    head = out.iloc[::step].head(50).to_dict('records')
    store = {'token': token, 'n_rows': len(out), 'n_cols': out.shape[1],
             'head': head, 'columns': list(out.columns)}

    status = 'Prepared.'
    if notes:
        status += ' Note: ' + '; '.join(notes)
    chips = [_chip(f'{len(out):,} rows'), _chip(f'{out.shape[1]} columns'),
             _chip(f'{out["Ticker"].nunique():,} tickers')]
    if 'split' in out.columns:
        vc = out['split'].value_counts()
        chips.append(_chip('split  ' + '  '.join(
            f'{s}:{int(vc.get(s, 0)):,}' for s in ('train', 'valid', 'test'))))
    return store, status, chips


@callback(
    Output('fe-preview', 'children'),
    Input('fe-build-store', 'data'),
)
def fe_render_preview(store):
    if not store or not store.get('head'):
        return html.P('Configure cleaning + scaling, then Run Prepare.', className='no-data')
    df = pd.DataFrame(store['head'])
    cols = []
    for c in df.columns:
        col = {'name': c, 'id': c}
        if c not in ('Date', 'Ticker', 'split'):
            col.update(_col_fmt(c))
        cols.append(col)
    return _dt.DataTable(data=df.round(4).to_dict('records'), columns=cols,
                         sort_action='native', page_size=25, **TABLE_STYLE)


@callback(
    Output('fe-status', 'children', allow_duplicate=True),
    Input('fe-export-parquet', 'n_clicks'),
    Input('fe-export-csv', 'n_clicks'),
    State('fe-build-store', 'data'),
    State('fe-recipe-name', 'value'),
    prevent_initial_call=True,
)
def fe_export(_p, _c, store, name):
    if not store or store.get('token') not in _PREP_CACHE:
        return 'Nothing to export — Run Prepare first.'
    fmt = 'parquet' if ctx.triggered_id == 'fe-export-parquet' else 'csv'
    df = _PREP_CACHE[store['token']]
    out = features_service.export_panel(df, fmt, name=name or 'fe_dataset')
    if isinstance(out, list):
        return f'Exported {len(df):,} rows → {len(out)} split files: ' + ', '.join(p.name for p in out)
    return f'Exported {len(df):,} rows → {out}'


# ── FE recipe save / load ─────────────────────────────────────────────

def _fe_spec(plan, exclude, sm, ss, sc, spm, st, sv, se, sd):
    return {
        'tame_plan': plan or [], 'exclude_tickers': _parse_exclude(exclude),
        'scale': _scale_cfg(sm, ss, sc), 'split': _split_cfg(spm, st, sv, se, sd),
    }


@callback(
    Output('fe-recipe-trigger', 'data'),
    Output('fe-status', 'children', allow_duplicate=True),
    Input('fe-save-recipe', 'n_clicks'),
    State('fe-recipe-name', 'value'),
    State('fe-tame-store', 'data'), State('fe-exclude', 'value'),
    State('fe-scale-method', 'value'), State('fe-scale-scope', 'value'),
    State('fe-scale-cutoff', 'value'),
    State('fe-split-method', 'value'), State('fe-split-train', 'value'),
    State('fe-split-valid', 'value'), State('fe-split-test', 'value'),
    State('fe-split-seed', 'value'),
    State('fe-recipe-trigger', 'data'),
    prevent_initial_call=True,
)
def fe_save_recipe(_n, name, plan, exclude, sm, ss, sc, spm, st, sv, se, sd, trig):
    if not name:
        return trig, 'Enter an FE recipe name to save.'
    features_service.save_fe_recipe(name, _fe_spec(plan, exclude, sm, ss, sc, spm, st, sv, se, sd))
    return (trig or 0) + 1, f'Saved FE recipe "{name}".'


@callback(
    Output('fe-tame-store', 'data', allow_duplicate=True),
    Output('fe-exclude', 'value'),
    Output('fe-scale-method', 'value'), Output('fe-scale-scope', 'value'),
    Output('fe-scale-cutoff', 'value'),
    Output('fe-split-method', 'value'), Output('fe-split-train', 'value'),
    Output('fe-split-valid', 'value'), Output('fe-split-test', 'value'),
    Output('fe-split-seed', 'value'),
    Input('fe-recipe-load', 'value'),
    prevent_initial_call=True,
)
def fe_load_recipe(name):
    if not name:
        raise PreventUpdate
    try:
        s = features_service.load_fe_recipe(name)
    except KeyError:
        raise PreventUpdate from None
    sc = s.get('scale', {})
    sp = s.get('split', {})
    return (s.get('tame_plan', []), ', '.join(s.get('exclude_tickers', [])),
            sc.get('method', 'none'), sc.get('scope', 'date'), sc.get('train_cutoff'),
            sp.get('method', 'none'), sp.get('train', 0.7), sp.get('valid', 0.15),
            sp.get('test', 0.15), sp.get('seed', 0))

ACCENT = '#58a6ff'
MUTED = '#7d8590'
GRID = 'rgba(128,128,128,0.18)'
DIV_COLOR = '#4ec94e'
SPLIT_COLOR = '#e0a040'

HOVER_LABEL: dict = {
    'bgcolor': '#ffffff',
    'font': {'color': '#517198', 'size': 12},
    'namelength': 0,
}

TABLE_STYLE: dict = {
    'style_table': {'overflowX': 'auto', 'marginTop': '16px'},
    'style_header': {
        'backgroundColor': 'var(--surface-2)',
        'color': 'var(--muted)',
        'fontWeight': '600',
        'fontSize': '11px',
        'textTransform': 'uppercase',
        'border': '1px solid var(--border)',
    },
    'style_cell': {
        'backgroundColor': 'var(--surface)',
        'color': 'var(--text)',
        'border': '1px solid var(--border)',
        'fontSize': '12px',
        'fontFamily': '"SF Mono","Cascadia Code",monospace',
        'padding': '5px 12px',
        'textAlign': 'right',
    },
    'style_cell_conditional': [
        {'if': {'column_id': 'Date'}, 'textAlign': 'left'},
    ],
}

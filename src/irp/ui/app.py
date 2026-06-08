from pathlib import Path

import dash
from dash import dcc, html

from irp.ui.components import navbar

_HERE = Path(__file__).parent

app = dash.Dash(
    __name__,
    use_pages=True,
    pages_folder=str(_HERE / 'pages'),
    assets_folder=str(_HERE / 'assets'),
    suppress_callback_exceptions=True,
)
app.title = 'IRP'

app.layout = html.Div([
    # Cross-page shared context: a session-scoped "current selection" (e.g. ticker) that
    # pages prefill from on load and write to on change, plus a Location for programmatic
    # navigation (e.g. /today row click → /analysis). See pages that read `workspace`.
    dcc.Location(id='url-redirect', refresh=False),
    dcc.Store(id='workspace', storage_type='session', data={}),
    navbar(),
    html.Main(className='main-content', children=[dash.page_container]),
])

from pathlib import Path

import dash
from dash import html

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
    navbar(),
    html.Main(className='main-content', children=[dash.page_container]),
])

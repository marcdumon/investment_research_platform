from dash import dcc, html


def navbar() -> html.Nav:
    return html.Nav(className='nav', children=[
        dcc.Link('IRP', href='/', className='nav-logo'),
        html.Div(className='nav-links', children=[
            dcc.Link('Home', href='/', className='nav-link'),
            dcc.Link('Ingest', href='/ingest', className='nav-link'),
            dcc.Link('Ticker', href='/ticker', className='nav-link'),
        ]),
    ])

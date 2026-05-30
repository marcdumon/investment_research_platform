from dash import dcc, html


def _navbar() -> html.Nav:
    return html.Nav(className='nav', children=[
        dcc.Link('IRP', href='/', className='nav-logo'),
        html.Div(className='nav-links', children=[
            dcc.Link('Home', href='/', className='nav-link'),
            dcc.Link('Ingest', href='/ingest', className='nav-link'),
            dcc.Link('Ticker', href='/ticker', className='nav-link'),
            dcc.Link('Factors', href='/factors', className='nav-link'),
            dcc.Link('Backtest', href='/backtest', className='nav-link'),
            dcc.Link('Screener', href='/screener', className='nav-link'),
            dcc.Link('Correlation', href='/correlation', className='nav-link'),
            dcc.Link('Data Quality', href='/data-quality', className='nav-link'),
        ]),
    ])

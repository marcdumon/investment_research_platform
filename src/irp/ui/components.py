from dash import dcc, html


def navbar() -> html.Nav:
    return html.Nav(className='nav', children=[
        dcc.Link('IRP', href='/', className='nav-logo'),
        html.Div(className='nav-links', children=[
            dcc.Link('Home', href='/', className='nav-link'),
            dcc.Link('Today', href='/today', className='nav-link'),
            dcc.Link('Ingest', href='/ingest', className='nav-link'),
            dcc.Link('Ticker', href='/ticker', className='nav-link'),
            dcc.Link('Factors', href='/factors', className='nav-link'),
            dcc.Link('Backtest', href='/backtest', className='nav-link'),
            dcc.Link('Screener', href='/screener', className='nav-link'),
            dcc.Link('Analysis', href='/analysis', className='nav-link'),
            dcc.Link('Regime', href='/regime', className='nav-link'),
            dcc.Link('Dataset Builder', href='/features', className='nav-link'),
            dcc.Link('Feature Engineering', href='/feature-engineering', className='nav-link'),
            dcc.Link('Correlation', href='/correlation', className='nav-link'),
            dcc.Link('Data Quality', href='/data-quality', className='nav-link'),
        ]),
    ])

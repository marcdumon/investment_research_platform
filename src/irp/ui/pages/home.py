import dash
from dash import dcc, html

dash.register_page(__name__, path='/', name='Home')


def _card(icon: str, title: str, desc: str, href: str, available: bool = True) -> html.Div:
    footer = (
        dcc.Link('Open →', href=href, className='card-link')
        if available
        else html.Span('Coming soon', className='card-soon')
    )
    return html.Div(
        className='card' + ('' if available else ' card-disabled'),
        children=[
            html.Div(icon, className='card-icon'),
            html.H3(title, className='card-title'),
            html.P(desc, className='card-desc'),
            footer,
        ],
    )


layout = html.Div(className='home-page', children=[
    html.H1('Investment Research Platform', className='home-title'),
    html.P('Data pipelines, fundamentals analysis, anomaly detection.', className='home-subtitle'),
    html.Div(className='card-grid', children=[
        _card('\U0001f4e5', 'Ingest', 'Run SimFin, Stooq and Yahoo data pipelines.', '/ingest'),
        _card('\U0001f4ca', 'Analysis', 'Anomaly detection and fundamentals review.', '/analysis', available=False),
        _card('\U0001f9ea', 'Research', 'Custom signals and factor models.', '/research', available=False),
    ]),
])

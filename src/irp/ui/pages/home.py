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
    html.P('Data pipelines, fundamentals analysis, factor research.', className='home-subtitle'),
    html.Div(className='card-grid', children=[
        _card('\U0001f4e5', 'Ingest',   'Run SimFin, Stooq and Yahoo data pipelines.', '/ingest'),
        _card('\U0001f50d', 'Ticker',   'Per-ticker fundamentals, prices, and actions.', '/ticker'),
        _card('\U0001f4ca', 'Factors',  'Cross-section screening and factor history.', '/factors'),
        _card('\U0001f4c8', 'Backtest', 'IC and quintile return analysis for any factor.', '/backtest'),
        _card('\U0001f50e', 'Screener', 'Progressive filter stack, scatter/histogram charts, watchlists.', '/screener'),
        _card('\U0001f4c9', 'Analysis', 'Return stats, QQ, drawdown, beta + residuals vs a benchmark.', '/analysis'),
        _card('\U0001f30d', 'Regime', 'Cross-asset macro state; conditioned factors, gated backtest, tactical.', '/regime'),
        _card('\U0001f9f0', 'Dataset Builder', 'Compose ML features from snapshots, attach labels, export raw.', '/features'),
        _card('\U0001f9ea', 'Feature Engineering', 'Inspect heavy tails, clean/scale/split a dataset, export train/valid/test.', '/feature-engineering'),
        _card('\U0001f9ee', 'Correlation', 'Factor collinearity and price-return co-movement heatmaps.', '/correlation'),
        _card('\U0001f50d', 'Data Quality', 'Triage fundamental and price anomalies; mark reviewed; add flags.', '/data-quality'),
    ]),
])

"""Price & return statistics (pure functions; no DB, no Dash).

Powers the `/analysis` page: single-instrument return diagnostics (distribution,
QQ, drawdown, rolling vol, autocorrelation) and the market-model relationship to a
benchmark (beta/alpha/residuals). All compute lives in `stats.py`; the UI service
(`irp.ui.services.analysis_service`) fetches price series from the panel and calls
these.
"""

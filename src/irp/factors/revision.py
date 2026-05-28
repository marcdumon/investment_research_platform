"""Earnings revision factors derived from SimFin as-reported vs restated data."""
from irp.factors.registry import register

register('revision_ni',  'NI Revision',  pct=True, group='quality')
register('revision_rev', 'Rev Revision', pct=True, group='quality')

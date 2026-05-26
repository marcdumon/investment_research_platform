"""Shared Plotly chart helpers for UI pages.

Every page used to define its own `_empty_figure` and `_chart_layout`;
this module consolidates the layout dicts + a couple of high-level
chart factories so behaviour stays consistent across pages.
"""
from typing import Any

import plotly.graph_objects as go

from irp.ui.theme import GRID, MUTED


def empty_figure(message: str = 'No data') -> go.Figure:
    """Blank figure displaying a centered muted-text annotation."""
    fig = go.Figure()
    fig.update_layout(
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)',
        annotations=[
            dict(
                text=message,
                x=0.5, y=0.5, xref='paper', yref='paper',
                showarrow=False,
                font=dict(color=MUTED, size=13),
            )
        ],
        xaxis=dict(visible=False),
        yaxis=dict(visible=False),
        margin=dict(l=0, r=0, t=8, b=0),
    )
    return fig


def base_chart_layout(**extra: Any) -> go.Layout:
    """Standard themed Plotly Layout. Extra kwargs override defaults."""
    defaults: dict[str, Any] = dict(
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(128,128,128,0.05)',
        font=dict(color=MUTED, size=11),
        margin=dict(l=0, r=0, t=28, b=0),
        hovermode='x unified',
        xaxis=dict(
            gridcolor=GRID,
            linecolor=GRID,
            tickfont=dict(color=MUTED, size=11),
            zeroline=False,
            showline=True,
        ),
        yaxis=dict(
            gridcolor=GRID,
            linecolor=GRID,
            tickfont=dict(color=MUTED, size=11),
            zeroline=True,
            showline=False,
        ),
        legend=dict(bgcolor='rgba(0,0,0,0)', font=dict(color=MUTED, size=11)),
    )
    defaults.update(extra)
    return go.Layout(**defaults)


def scatter_chart_layout(**extra: Any) -> go.Layout:
    """Like `base_chart_layout` but with `hovermode='closest'` for scatter-style plots."""
    extra.setdefault('hovermode', 'closest')
    return base_chart_layout(**extra)

"""Per-provider fetch-state primitives: dotfile markers, JSON resume-sets, freshness."""
from dataload.state.freshness import is_fresh
from dataload.state.jsonset import JsonSet
from dataload.state.markers import MarkerSet

__all__ = ['JsonSet', 'MarkerSet', 'is_fresh']

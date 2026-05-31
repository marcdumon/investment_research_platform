"""Batch-add helpers on the /features page (pure logic; app import registers pages)."""
import irp.ui.app  # noqa: F401  (instantiates app so register_page works)
from irp.ui.pages import features as pg


def test_parse_ints():
    assert pg._parse_ints('1,2,4', 1) == [1, 2, 4]
    assert pg._parse_ints('', 3) == [3]
    assert pg._parse_ints(None, 5) == [5]
    assert pg._parse_ints('2, 2, 8', 1) == [2, 8]  # dedup, whitespace


def test_parse_ints_ranges():
    assert pg._parse_ints('1-5', 1) == [1, 2, 3, 4, 5]          # price window
    assert pg._parse_ints('1-10:2', 1) == [1, 3, 5, 7, 9]       # stepped range
    assert pg._parse_ints('1,5-7,20', 1) == [1, 5, 6, 7, 20]    # mixed
    assert pg._parse_ints('3-3', 1) == [3]                       # degenerate range


def test_expand_multi_col_multi_k():
    out = pg._expand_add('lag', ['roe', 'pe'], [], [1, 2], [4], None, None, None)
    assert len(out) == 4
    assert {'op': 'lag', 'col': 'pe', 'k': 2} in out


def test_expand_ratio_skips_self_pairs():
    out = pg._expand_add('ratio', ['a', 'b'], ['b', 'c'], [1], [4], None, None, None)
    pairs = {(s['a'], s['b']) for s in out}
    assert ('b', 'b') not in pairs
    assert pairs == {('a', 'b'), ('a', 'c'), ('b', 'c')}


def test_append_dedup():
    steps = [{'op': 'base', 'col': 'roe'}]
    out = pg._append_dedup(steps, [{'op': 'base', 'col': 'roe'}, {'op': 'base', 'col': 'pe'}])
    assert out == [{'op': 'base', 'col': 'roe'}, {'op': 'base', 'col': 'pe'}]


def test_pack_groups_present():
    vals = {o['value'] for o in pg._PACK_OPTIONS}
    assert '__all__' in vals
    assert 'valuation' in vals and 'momentum' in vals
    assert len(pg._ALL_BASE_COLS) >= 30

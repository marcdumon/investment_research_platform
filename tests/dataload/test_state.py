"""Per-provider fetch-state primitives, now owned by the package."""
import os

from dataload.state import JsonSet, MarkerSet, is_fresh


def test_jsonset_roundtrip(tmp_path) -> None:
    js = JsonSet(tmp_path / 's.json')
    assert js.load() == set()
    js.save({'B', 'A', 'C'})
    assert js.load() == {'A', 'B', 'C'}


def test_is_fresh_marker_newer(tmp_path) -> None:
    inp = tmp_path / 'in'
    inp.write_text('x')
    os.utime(inp, (100, 100))
    mk = tmp_path / 'm'
    mk.write_text('y')
    os.utime(mk, (200, 200))
    assert is_fresh(mk, inp) is True


def test_is_fresh_marker_older(tmp_path) -> None:
    inp = tmp_path / 'in'
    inp.write_text('x')
    os.utime(inp, (200, 200))
    mk = tmp_path / 'm'
    mk.write_text('y')
    os.utime(mk, (100, 100))
    assert is_fresh(mk, inp) is False


def test_is_fresh_missing_marker(tmp_path) -> None:
    inp = tmp_path / 'in'
    inp.write_text('x')
    assert is_fresh(tmp_path / 'nope', inp) is False


def test_markerset_touch_and_freshness(tmp_path) -> None:
    ms = MarkerSet(tmp_path)
    up = tmp_path / 'up'
    up.write_text('x')
    os.utime(up, (100, 100))
    assert ms.is_fresh('done', up) is False
    ms.touch('done')
    assert ms.is_fresh('done', up) is True


def test_markerset_clear_feed(tmp_path) -> None:
    ms = MarkerSet(tmp_path)
    ms.touch('fetched')
    ms.touch('transformed_bulk')
    ms.touch('stored_bulk')
    assert ms.clear_feed('bulk') == 3
    assert not ms.exists('fetched')

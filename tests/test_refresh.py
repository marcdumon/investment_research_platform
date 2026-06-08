import datetime

from irp import refresh


def _d(s):
    return datetime.date.fromisoformat(s)


def test_staleness_panel_behind_db():
    out = refresh._staleness(_d('2026-05-22'), _d('2026-03-31'), {'A': _d('2026-03-31')})
    assert out['panel_stale'] is True


def test_staleness_panel_current():
    out = refresh._staleness(_d('2026-05-22'), _d('2026-05-22'), {'A': _d('2026-03-31')})
    assert out['panel_stale'] is False


def test_staleness_cache_behind_panel():
    out = refresh._staleness(_d('2026-05-22'), _d('2026-05-22'),
                             {'A': _d('2026-03-31'), 'Q': None})
    assert out['cache_stale']['A'] is True          # panel newer than cache
    assert out['cache_stale']['Q'] is True          # no cache at all


def test_staleness_cache_current():
    out = refresh._staleness(_d('2026-03-31'), _d('2026-03-31'), {'A': _d('2026-03-31')})
    assert out['cache_stale']['A'] is False


def test_status_returns_expected_keys():
    st = refresh.status()
    assert {'db_latest', 'panel_latest', 'cache_latest', 'panel_stale', 'cache_stale'} <= set(st)

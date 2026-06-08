import pandas as pd

from irp.models import predictions as pr


def _preds():
    return pd.DataFrame({
        'Date': pd.to_datetime(['2024-01-01', '2024-01-01', '2024-06-01', '2024-06-01']),
        'Ticker': ['AAA', 'BBB', 'AAA', 'BBB'],
        'fwd_ret': [0.1, -0.1, 0.2, 0.05],
        'pred': [0.3, 0.1, 0.05, 0.4],
    })


def test_latest_picks_uses_most_recent_date_ranked():
    out = pr.latest_picks(_preds(), n=10)
    assert out['Date'].nunique() == 1                       # only the latest date
    assert out['Date'].iloc[0] == pd.Timestamp('2024-06-01')
    assert out.index[0] == 'BBB'                            # higher pred on top
    assert list(out['Rank']) == [1, 2]


def test_latest_picks_falls_back_to_score_column():
    df = _preds().rename(columns={'pred': 'score'})
    out = pr.latest_picks(df, n=10)
    assert out.index[0] == 'BBB'                            # ranks on 'score' when no 'pred'


def test_save_and_load_roundtrip(tmp_path, monkeypatch):
    monkeypatch.setattr(pr, '_DIR', tmp_path)
    p = pr.save_predictions(_preds(), 'baseline')
    assert p.exists()
    loaded = pr.load_predictions()
    assert len(loaded) == 4
    assert set(loaded.columns) >= {'Date', 'Ticker', 'pred'}


def test_save_requires_pred_or_score(tmp_path, monkeypatch):
    monkeypatch.setattr(pr, '_DIR', tmp_path)
    bad = pd.DataFrame({'Date': [pd.Timestamp('2024-01-01')], 'Ticker': ['AAA']})
    try:
        pr.save_predictions(bad, 'x')
        raised = False
    except ValueError:
        raised = True
    assert raised

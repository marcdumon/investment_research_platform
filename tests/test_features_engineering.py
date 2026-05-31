import datetime

import numpy as np
import pandas as pd
import pytest

from irp.features import engineering as eng


# ── fixtures ──────────────────────────────────────────────────────────

def _snap(date_factors: dict[str, float], tickers: list[str]) -> pd.DataFrame:
    """One snapshot: DataFrame indexed by Ticker with given columns."""
    return pd.DataFrame(date_factors, index=pd.Index(tickers, name='Ticker'))


@pytest.fixture
def snapshots() -> dict[datetime.date, pd.DataFrame]:
    """Two tickers across four annual dates with a couple of base columns."""
    tickers = ['AAA', 'BBB']
    out = {}
    for i, y in enumerate((2020, 2021, 2022, 2023)):
        d = datetime.date(y, 12, 31)
        out[d] = _snap(
            {
                'roe': [0.10 + i * 0.01, 0.20 - i * 0.02],
                'revenue': [100.0 + 10 * i, 50.0 + 5 * i],
                'total_assets': [200.0, 100.0],
            },
            tickers,
        )
    return out


@pytest.fixture
def panel(snapshots) -> pd.DataFrame:
    return eng.assemble_panel(snapshots)


# ── assemble_panel ────────────────────────────────────────────────────

def test_assemble_panel_long_format(panel):
    assert {'Date', 'Ticker', 'roe', 'revenue', 'total_assets'} <= set(panel.columns)
    # 2 tickers × 4 dates
    assert len(panel) == 8


def test_assemble_panel_sorted_by_ticker_then_date(panel):
    keys = list(zip(panel['Ticker'], panel['Date']))
    assert keys == sorted(keys)


# ── temporal ops respect ticker grouping ─────────────────────────────

def test_add_lag_no_cross_ticker_bleed(panel):
    out = eng.add_lag(panel, 'roe', 1)
    col = 'roe_lag1'
    aaa = out[out['Ticker'] == 'AAA'].sort_values('Date')
    # first date of AAA has no prior -> NaN, not BBB's value
    assert pd.isna(aaa[col].iloc[0])
    # second date lag == AAA's first roe (0.10), not BBB
    assert aaa[col].iloc[1] == pytest.approx(0.10)


def test_add_diff(panel):
    out = eng.add_diff(panel, 'roe', 1)
    aaa = out[out['Ticker'] == 'AAA'].sort_values('Date')
    assert aaa['roe_diff1'].iloc[1] == pytest.approx(0.01)


def test_add_pct_change(panel):
    out = eng.add_pct_change(panel, 'revenue', 1)
    aaa = out[out['Ticker'] == 'AAA'].sort_values('Date')
    # 110/100 - 1 = 0.10
    assert aaa['revenue_pct1'].iloc[1] == pytest.approx(0.10)


def test_add_lag_window(panel):
    out = eng.add_lag_window(panel, 'roe', 3)
    assert {'roe_lag1', 'roe_lag2', 'roe_lag3'} <= set(out.columns)
    aaa = out[out['Ticker'] == 'AAA'].sort_values('Date')
    assert aaa['roe_lag1'].iloc[1] == pytest.approx(0.10)  # prior period level
    assert pd.isna(aaa['roe_lag3'].iloc[1])                # not enough history


def test_apply_step_lagwin_outputs(panel):
    step = {'op': 'lagwin', 'col': 'roe', 'n': 2}
    out = eng.apply_step(panel, step)
    assert eng.step_output_cols(step) == ['roe', 'roe_lag1', 'roe_lag2']
    assert {'roe_lag1', 'roe_lag2'} <= set(out.columns)


def test_add_rolling_mean_window(panel):
    out = eng.add_rolling(panel, 'roe', 2, 'mean')
    aaa = out[out['Ticker'] == 'AAA'].sort_values('Date')
    # mean of first two roe values (0.10, 0.11)
    assert aaa['roe_roll2mean'].iloc[1] == pytest.approx(0.105)


# ── math / interaction ops ────────────────────────────────────────────

def test_add_ratio(panel):
    out = eng.add_ratio(panel, 'revenue', 'total_assets')
    aaa = out[out['Ticker'] == 'AAA'].sort_values('Date')
    assert aaa['revenue_over_total_assets'].iloc[0] == pytest.approx(0.5)


def test_add_ratio_div_zero_is_na(panel):
    p = panel.copy()
    p['total_assets'] = 0.0
    out = eng.add_ratio(p, 'revenue', 'total_assets')
    assert out['revenue_over_total_assets'].isna().all()


def test_add_log(panel):
    out = eng.add_log(panel, 'revenue')
    assert out['log_revenue'].iloc[0] == pytest.approx(np.log(panel['revenue'].iloc[0]))


def test_add_winsorize_clips_tails(panel):
    p = panel.copy()
    p.loc[p.index[0], 'roe'] = 999.0  # outlier
    out = eng.add_winsorize(p, 'roe', 0.1)
    assert out['roe_wins'].max() < 999.0


# ── cross-sectional norm applied per Date ─────────────────────────────

def test_normalize_zscore_per_date(panel):
    out = eng.normalize_step(panel, ['roe'], method='zscore')
    # each date group should have ~zero mean across tickers
    g = out.groupby('Date')['roe_z'].mean()
    assert (g.abs() < 1e-9).all()


# ── apply_step dispatch ───────────────────────────────────────────────

def test_apply_step_lag(panel):
    out = eng.apply_step(panel, {'op': 'lag', 'col': 'roe', 'k': 1})
    assert 'roe_lag1' in out.columns


def test_apply_step_unknown_op_raises(panel):
    with pytest.raises(ValueError, match='unknown'):
        eng.apply_step(panel, {'op': 'nope'})


# ── label attachment, per-date bucketing (no look-ahead) ──────────────

@pytest.fixture
def fwd() -> pd.DataFrame:
    """Forward returns where per-date vs pooled bucketing differ.

    Date 2020: AAA=0.01, BBB=0.02  (BBB above median)
    Date 2021: AAA=0.50, BBB=0.40  (AAA above median)
    Pooled median ~0.21 would put both 2021 rows in the top bucket;
    per-date bucketing must split each date independently.
    """
    rows = [
        (datetime.date(2020, 12, 31), 'AAA', 0.01),
        (datetime.date(2020, 12, 31), 'BBB', 0.02),
        (datetime.date(2021, 12, 31), 'AAA', 0.50),
        (datetime.date(2021, 12, 31), 'BBB', 0.40),
    ]
    return pd.DataFrame(rows, columns=['Date', 'Ticker', 'fwd_ret'])


def _two_date_panel() -> pd.DataFrame:
    rows = [
        (datetime.date(2020, 12, 31), 'AAA', 0.1),
        (datetime.date(2020, 12, 31), 'BBB', 0.2),
        (datetime.date(2021, 12, 31), 'AAA', 0.3),
        (datetime.date(2021, 12, 31), 'BBB', 0.4),
    ]
    return pd.DataFrame(rows, columns=['Date', 'Ticker', 'roe'])


def test_attach_label_continuous():
    df = _two_date_panel()
    out = eng.attach_label(df, _fwd_df(), mode='continuous')
    assert 'fwd_ret' in out.columns
    assert out['fwd_ret'].notna().all()


def _fwd_df() -> pd.DataFrame:
    rows = [
        (datetime.date(2020, 12, 31), 'AAA', 0.01),
        (datetime.date(2020, 12, 31), 'BBB', 0.02),
        (datetime.date(2021, 12, 31), 'AAA', 0.50),
        (datetime.date(2021, 12, 31), 'BBB', 0.40),
    ]
    return pd.DataFrame(rows, columns=['Date', 'Ticker', 'fwd_ret'])


def test_attach_label_binary_is_per_date(fwd):
    df = _two_date_panel()
    out = eng.attach_label(df, fwd, mode='binary')
    out = out.set_index(['Date', 'Ticker'])
    # per-date: 2020 winner = BBB, 2021 winner = AAA
    assert out.loc[(datetime.date(2020, 12, 31), 'BBB'), 'label'] == 1
    assert out.loc[(datetime.date(2020, 12, 31), 'AAA'), 'label'] == 0
    assert out.loc[(datetime.date(2021, 12, 31), 'AAA'), 'label'] == 1
    assert out.loc[(datetime.date(2021, 12, 31), 'BBB'), 'label'] == 0


def test_asof_join_is_pit_backward():
    """Carry-forward must use the last value with Date <= grid date (no look-ahead)."""
    spine = pd.DataFrame({
        'Date': pd.to_datetime(['2020-06-30', '2021-06-30', '2022-06-30']),
        'Ticker': ['AAA', 'AAA', 'AAA'],
        'close': [1.0, 2.0, 3.0],
    })
    filings = pd.DataFrame({
        'Date': pd.to_datetime(['2020-12-31', '2021-12-31']),
        'Ticker': ['AAA', 'AAA'],
        'roe': [0.10, 0.20],
    })
    out = eng.asof_join(spine, filings, by='Ticker').set_index('Date')
    # 2020-06-30: no filing on/before -> NaN (no look-ahead to 2020-12-31)
    assert pd.isna(out.loc['2020-06-30', 'roe'])
    # 2021-06-30: last filing <= is 2020-12-31 -> 0.10
    assert out.loc['2021-06-30', 'roe'] == pytest.approx(0.10)
    # 2022-06-30: last filing <= is 2021-12-31 -> 0.20
    assert out.loc['2022-06-30', 'roe'] == pytest.approx(0.20)


def test_asof_join_no_cross_ticker_bleed():
    spine = pd.DataFrame({
        'Date': pd.to_datetime(['2021-06-30', '2021-06-30']),
        'Ticker': ['AAA', 'BBB'], 'close': [1.0, 1.0],
    })
    filings = pd.DataFrame({
        'Date': pd.to_datetime(['2020-12-31']), 'Ticker': ['AAA'], 'roe': [0.5],
    })
    out = eng.asof_join(spine, filings, by='Ticker').set_index('Ticker')
    assert out.loc['AAA', 'roe'] == pytest.approx(0.5)
    assert pd.isna(out.loc['BBB', 'roe'])  # BBB has no filing


def test_attach_label_none_adds_nothing(fwd):
    df = _two_date_panel()
    out = eng.attach_label(df, fwd, mode='none')
    assert 'label' not in out.columns
    assert 'fwd_ret' not in out.columns

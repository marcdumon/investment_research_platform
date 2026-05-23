"""Tests for irp.factors.piotroski.compute_piotroski."""
import datetime

import numpy as np
import pandas as pd
import pytest

from irp.factors._cols import (
    CFO,
    GROSS_PROFIT,
    LONG_TERM_DEBT,
    NET_INCOME,
    PUBLISH_DATE,
    REPORT_DATE,
    REVENUE,
    SHARES_DILUTED_BAL,
    TICKER,
    TOTAL_ASSETS,
    TOTAL_CURRENT_ASSETS,
    TOTAL_CURRENT_LIABILITIES,
)
from irp.factors.piotroski import compute_piotroski

_AS_OF = datetime.date(2023, 12, 31)


def _inc_row(ticker, date, ni, rev, gp) -> dict:
    return {
        TICKER: ticker, REPORT_DATE: date, PUBLISH_DATE: None,
        NET_INCOME: ni, REVENUE: rev, GROSS_PROFIT: gp,
        'Operating Income (Loss)': 0, 'Interest Expense, Net': 0,
    }


def _bal_row(ticker, date, ta, ltd, ca, cl, sh) -> dict:
    return {
        TICKER: ticker, REPORT_DATE: date, PUBLISH_DATE: None,
        TOTAL_ASSETS: ta, LONG_TERM_DEBT: ltd,
        TOTAL_CURRENT_ASSETS: ca, TOTAL_CURRENT_LIABILITIES: cl,
        SHARES_DILUTED_BAL: sh,
        'Total Equity': 50_000, 'Total Liabilities': 60_000,
        'Cash, Cash Equivalents & Short Term Investments': 5_000,
        'Short Term Debt': 2_000,
    }


def _cf_row(ticker, date, cfo) -> dict:
    return {
        TICKER: ticker, REPORT_DATE: date, PUBLISH_DATE: None,
        CFO: cfo,
        'Net Cash from Investing Activities': 0,
        'Depreciation & Amortization': 0,
    }


def _frames(ni_c=10_000, ni_p=8_000, rev_c=100_000, rev_p=90_000,
            gp_c=40_000, gp_p=35_000, ta_c=120_000, ta_p=110_000,
            ltd_c=20_000, ltd_p=25_000, ca_c=30_000, ca_p=25_000,
            cl_c=15_000, cl_p=14_000, sh_c=10_000, sh_p=10_000,
            cfo_c=12_000):
    inc = pd.DataFrame([
        _inc_row('AAPL', '2023-09-30', ni_c, rev_c, gp_c),
        _inc_row('AAPL', '2022-09-30', ni_p, rev_p, gp_p),
    ])
    bal = pd.DataFrame([
        _bal_row('AAPL', '2023-09-30', ta_c, ltd_c, ca_c, cl_c, sh_c),
        _bal_row('AAPL', '2022-09-30', ta_p, ltd_p, ca_p, cl_p, sh_p),
    ])
    cf = pd.DataFrame([
        _cf_row('AAPL', '2023-09-30', cfo_c),
        _cf_row('AAPL', '2022-09-30', 9_000),
    ])
    return inc, bal, cf


class TestComputePiotroski:
    def test_output_indexed_by_ticker(self):
        inc, bal, cf = _frames()
        result = compute_piotroski(inc, bal, cf, _AS_OF, 'A')
        assert result.index.name == TICKER
        assert 'piotroski_fscore' in result.columns

    def test_perfect_score(self):
        # All 9 signals should pass with these values:
        # F1 ROA>0: NI=10k/TA=120k → 0.083 > 0 ✓
        # F2 CFO>0: 12k > 0 ✓
        # F3 ΔROA: 0.083 > 0.073 ✓
        # F4 CFO/TA > ROA: 12k/120k=0.1 > 0.083 ✓
        # F5 ΔLev<0: 25k/110k=0.227 > 20k/120k=0.167 (decreased) ✓
        # F6 ΔCR>0: 30k/15k=2.0 > 25k/14k=1.79 ✓
        # F7 no dilution: sh_p=sh_c → 0 difference → NOT > 0 ✗
        # F8 ΔGM>0: 40k/100k=0.4 > 35k/90k=0.389 ✓
        # F9 ΔAT>0: 100k/120k=0.833 > 90k/110k=0.818 ✓
        inc, bal, cf = _frames()
        result = compute_piotroski(inc, bal, cf, _AS_OF, 'A')
        # 8/9 because shares didn't change (F7 = 0)
        assert result.loc['AAPL', 'piotroski_fscore'] == pytest.approx(8.0)

    def test_f7_passes_when_shares_decreased(self):
        inc, bal, cf = _frames(sh_c=9_000, sh_p=10_000)  # shares decreased
        result = compute_piotroski(inc, bal, cf, _AS_OF, 'A')
        assert result.loc['AAPL', 'piotroski_fscore'] == pytest.approx(9.0)

    def test_score_zero_bad_company(self):
        # All signals fail:
        # F1: ROA = -5k/120k = -0.042 < 0
        # F2: CFO = -6k < 0
        # F3: ROA_curr=-0.042 < ROA_prior=-3k/110k=-0.027 (worsened)
        # F4: CFO/TA=-0.05 < ROA=-0.042 (accruals worse than cash)
        # F5: lev_curr=30k/120k=0.25 > lev_prior=20k/110k=0.18 (leverage increased)
        # F6: CR_curr=20k/15k=1.33 < CR_prior=30k/14k=2.14 (liquidity fell)
        # F7: sh_curr=12k > sh_prior=10k (dilution)
        # F8: GM_curr=20k/80k=0.25 < GM_prior=25k/90k=0.28 (margin fell)
        # F9: AT_curr=80k/120k=0.67 < AT_prior=90k/110k=0.82 (efficiency fell)
        inc, bal, cf = _frames(
            ni_c=-5_000, ni_p=-3_000,
            cfo_c=-6_000,                # CFO/TA=-0.05 < ROA=-0.042 → F4=0
            ltd_c=30_000, ltd_p=20_000,
            ca_c=20_000, ca_p=30_000,
            cl_c=15_000, cl_p=14_000,
            sh_c=12_000, sh_p=10_000,
            gp_c=20_000, gp_p=25_000,
            rev_c=80_000, rev_p=90_000,
        )
        result = compute_piotroski(inc, bal, cf, _AS_OF, 'A')
        assert result.loc['AAPL', 'piotroski_fscore'] == pytest.approx(0.0)

    def test_no_prior_year_partial_score(self):
        # Only current-year data — delta signals (F3,F5,F6,F7,F8,F9) are NaN
        # F1,F2,F4 can still be computed
        inc = pd.DataFrame([_inc_row('AAPL', '2023-09-30', 10_000, 100_000, 40_000)])
        bal = pd.DataFrame([_bal_row('AAPL', '2023-09-30', 120_000, 20_000, 30_000, 15_000, 10_000)])
        cf  = pd.DataFrame([_cf_row('AAPL', '2023-09-30', 12_000)])
        result = compute_piotroski(inc, bal, cf, _AS_OF, 'A')
        score = result.loc['AAPL', 'piotroski_fscore']
        assert not np.isnan(score)
        assert score <= 3.0  # only F1, F2, F4 can fire

    def test_score_bounded_0_to_9(self):
        inc, bal, cf = _frames()
        result = compute_piotroski(inc, bal, cf, _AS_OF, 'A')
        score = result.loc['AAPL', 'piotroski_fscore']
        assert 0.0 <= score <= 9.0

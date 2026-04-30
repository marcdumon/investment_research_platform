import logging

import pandas as pd

from irp.quality.base import Finding, Rule, default_registry

logger = logging.getLogger(__name__)

_PERIOD_TABLES = ("income", "balance", "cashflow")


def run(
    data: dict[str, pd.DataFrame],
    *,
    rules: list[Rule] | None = None,
) -> pd.DataFrame:
    """Run quality rules against *data* and return a findings DataFrame.

    Args:
        data:  Mapping of table name → DataFrame (income, balance, …).
        rules: Explicit list of Rule instances.  When None, all rules
               registered in *default_registry* are used (standard workflow).
               Pass a list to test rules in isolation without touching global state.
    """
    if rules is None:
        import irp.quality.rules  # noqa: F401 — side-effect: auto-registers rules
        rules = default_registry.rules()

    frames: list[pd.DataFrame] = []
    for rule in rules:
        try:
            result = rule.check(data)
            if not result.empty:
                frames.append(result)
            logger.debug("%s: %d findings", rule.name, len(result))
        except Exception:
            logger.exception("Rule %s raised — skipping", rule.name)

    result = pd.concat(frames, ignore_index=True) if frames else Finding.empty_df()
    result = _replace_with_period(result, data)
    return _enrich_company(result, data)


def _replace_with_period(findings: pd.DataFrame, data: dict[str, pd.DataFrame]) -> pd.DataFrame:
    """Replace variant + report_date columns with period (e.g. '2023A', '2023Q1').

    Joins period from the source tables keyed on (table, ticker, variant, report_date).
    Falls back gracefully when period column is absent from the loaded tables.
    """
    lookups: list[pd.DataFrame] = []
    for tbl_name in _PERIOD_TABLES:
        tbl = data.get(tbl_name)
        if tbl is None or "period" not in tbl.columns:
            continue
        lk = (
            tbl[["Ticker", "variant", "Report Date", "period"]]
            .drop_duplicates()
            .assign(
                ticker=lambda d: d["Ticker"],
                report_date=lambda d: d["Report Date"].astype(str).str[:10],
                table=tbl_name,
            )
            [["table", "ticker", "variant", "report_date", "period"]]
        )
        lookups.append(lk)

    if not lookups:
        return findings

    lookup = pd.concat(lookups, ignore_index=True).drop_duplicates(
        subset=["table", "ticker", "variant", "report_date"]
    )
    enriched = findings.merge(lookup, on=["table", "ticker", "variant", "report_date"], how="left")

    # place period where variant was, then drop variant + report_date
    idx = list(findings.columns).index("variant")
    enriched.insert(idx, "period", enriched.pop("period"))
    return enriched.drop(columns=["variant", "report_date"])


def _enrich_company(findings: pd.DataFrame, data: dict[str, pd.DataFrame]) -> pd.DataFrame:
    if findings.empty or "companies" not in data:
        return findings

    meta = (
        data["companies"][["Ticker", "Company Name", "ISIN", "CIK"]]
        .drop_duplicates("Ticker")
        .rename(columns={"Ticker": "ticker", "Company Name": "company_name", "ISIN": "isin"})
    )
    enriched = findings.merge(meta, on="ticker", how="left")

    # insert company_name and isin right after ticker
    ticker_idx = list(findings.drop(columns=["variant", "report_date"], errors="ignore").columns).index("ticker") + 1
    for col in ("isin", "company_name"):
        enriched.insert(ticker_idx, col, enriched.pop(col))

    # edgar_url last — links to exact filing type near the report date
    period_col = enriched.get("period") if "period" in enriched.columns else None
    variant_col = enriched.get("variant") if "variant" in enriched.columns else None
    report_date_col = enriched.get("report_date") if "report_date" in enriched.columns else None

    if variant_col is not None:
        filing_type = variant_col.map({"A": "10-K", "Q": "10-Q"}).fillna("10-K")
        offset_days = variant_col.map({"A": 120, "Q": 60}).fillna(120).astype(int)
        date_base = pd.to_datetime(report_date_col)
    elif period_col is not None:
        # derive variant from period suffix: ends with 'A' → annual, else quarterly
        is_annual = period_col.str.endswith("A")
        filing_type = is_annual.map({True: "10-K", False: "10-Q"})
        offset_days = is_annual.map({True: 120, False: 60}).astype(int)
        # extract year from period (first 4 chars) + use Dec 31 as proxy date
        date_base = pd.to_datetime(period_col.str[:4] + "-12-31", errors="coerce")
    else:
        return enriched.drop(columns=["CIK"])

    dateb = (date_base + pd.to_timedelta(offset_days, unit="D")).dt.strftime("%Y%m%d")
    cik_str = enriched["CIK"].apply(lambda c: f"{int(c):010d}" if pd.notna(c) else None)

    enriched["edgar_url"] = (
        "https://www.sec.gov/cgi-bin/browse-edgar?action=getcompany&CIK="
        + cik_str.fillna("")
        + "&type=" + filing_type
        + "&dateb=" + dateb
        + "&owner=include&count=5"
    ).where(cik_str.notna(), other=None)

    return enriched.drop(columns=["CIK"])

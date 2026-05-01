import logging

import pandas as pd

from irp.anomalies.base import Finding, Rule, default_registry

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
        import irp.anomalies.rules  # noqa: F401 — side-effect: auto-registers rules
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

    # place period where variant was, report_date after it; drop variant only
    idx = list(findings.columns).index("variant")
    enriched.insert(idx, "period", enriched.pop("period"))
    enriched.insert(idx + 1, "report_date", enriched.pop("report_date"))
    return enriched.drop(columns=["variant"])


def _enrich_company(findings: pd.DataFrame, data: dict[str, pd.DataFrame]) -> pd.DataFrame:
    if findings.empty or "companies" not in data:
        return findings

    meta = (
        data["companies"][["Ticker", "Company Name", "ISIN"]]
        .drop_duplicates("Ticker")
        .rename(columns={"Ticker": "ticker", "Company Name": "company_name", "ISIN": "isin"})
    )
    enriched = findings.merge(meta, on="ticker", how="left")

    # insert company_name and isin right after ticker
    ticker_idx = list(enriched.columns).index("ticker") + 1
    for col in ("isin", "company_name"):
        enriched.insert(ticker_idx, col, enriched.pop(col))

    # edgar_url last — looked up from sec_filings table (run fetch_sec_filings.py to populate)
    enriched["edgar_url"] = _lookup_edgar_urls(enriched, data)
    return enriched


def _lookup_edgar_urls(df: pd.DataFrame, data: dict[str, pd.DataFrame]) -> pd.Series:
    """Join edgar_url from the pre-built sec_filings table."""
    empty = pd.Series([None] * len(df), index=df.index, dtype=object)
    if "period" not in df.columns or "sec_filings" not in data:
        return empty

    url_map = (
        data["sec_filings"][["ticker", "period", "url"]]
        .drop_duplicates(subset=["ticker", "period"])
        .set_index(["ticker", "period"])["url"]
    )
    return df.apply(lambda r: url_map.get((r["ticker"], r["period"])), axis=1)

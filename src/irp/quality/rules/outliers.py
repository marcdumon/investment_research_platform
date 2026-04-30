from typing import ClassVar

import pandas as pd

from irp.quality.base import Finding, Rule, Severity, register

_DEFAULT_METRICS = ["Revenue", "Net Income", "Operating Income (Loss)"]


def _iqr_score(s: pd.Series, *, fence: float, min_peers: int) -> pd.Series:
    """Signed IQR distance for each element.  NaN when group is too small or IQR=0.
    Positive → upper outlier; negative → lower outlier; NaN → inlier.
    """
    if len(s) < min_peers:
        return pd.Series(float("nan"), index=s.index)
    q1, q3 = s.quantile(0.25), s.quantile(0.75)
    iqr = q3 - q1
    if iqr == 0:
        return pd.Series(float("nan"), index=s.index)

    score = pd.Series(float("nan"), index=s.index)
    upper = q3 + fence * iqr
    lower = q1 - fence * iqr
    hi = s > upper
    lo = s < lower
    score[hi] = (s[hi] - q3) / iqr
    score[lo] = (s[lo] - q1) / iqr  # negative
    return score


@register
class SectorOutliers(Rule):
    name = "sector_outlier"
    description = "IQR-based outlier vs sector peers (upper or lower fence)"
    severity = Severity.WARNING

    iqr_fence: ClassVar[float] = 3.0
    min_peers: ClassVar[int] = 5
    metrics: ClassVar[list[str]] = _DEFAULT_METRICS

    def check(self, data: dict[str, pd.DataFrame]) -> pd.DataFrame:
        required = {"income", "companies", "industries"}
        if not required.issubset(data):
            return Finding.empty_df()

        companies = data["companies"][["Ticker", "IndustryId"]].drop_duplicates(
            "Ticker"
        )
        industries = data["industries"][["IndustryId", "Sector"]].drop_duplicates(
            "IndustryId"
        )

        df = (
            data["income"]
            .merge(companies, on="Ticker", how="left")
            .merge(industries, on="IndustryId", how="left")
        )
        df["Sector"] = df["Sector"].fillna("Unknown")

        frames: list[pd.DataFrame] = []
        for metric in self.metrics:
            if metric not in df.columns:
                continue
            sub = df.dropna(subset=[metric]).copy()

            sub["_score"] = sub.groupby(["Sector", "variant"])[metric].transform(
                _iqr_score, fence=self.iqr_fence, min_peers=self.min_peers
            )
            bad = sub.dropna(subset=["_score"])
            if bad.empty:
                continue

            detail = (
                "IQR-dist="
                + bad["_score"].round(1).astype(str)
                + ", sector="
                + bad["Sector"]
                + ", value="
                + bad[metric].round(0).astype("int64").astype(str)
            )

            frames.append(
                pd.DataFrame(
                    {
                        "table": "income",
                        "ticker": bad["Ticker"].values,
                        "variant": bad["variant"].values,
                        "report_date": bad["Report Date"].astype(str).str[:10].values,
                        "column": metric,
                        "value": bad["_score"].round(3).values,
                        "rule": self.name,
                        "detail": detail.values,
                        "severity": self.severity,
                    }
                )
            )

        return pd.concat(frames, ignore_index=True) if frames else Finding.empty_df()

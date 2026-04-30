import logging

import pandas as pd

from irp.quality.base import Finding, Rule, default_registry

logger = logging.getLogger(__name__)


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

    return pd.concat(frames, ignore_index=True) if frames else Finding.empty_df()

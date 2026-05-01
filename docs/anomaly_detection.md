# Fundamental Data Quality — Anomaly Detection

## Context
SimFin fundamentals contain errors: impossible values, accounting identity violations, statistical outliers, sudden period-over-period jumps. Need a repeatable, extensible detection system. Corrections deferred to a later phase.

## Architecture

```
src/irp/quality/
├── __init__.py          # re-exports run()
├── base.py              # Finding dataclass, Rule ABC, @register, all_rules()
├── runner.py            # run(data) -> pd.DataFrame of all findings
└── rules/
    ├── __init__.py      # auto-imports all rule modules (triggers @register)
    ├── accounting.py    # AccountingIdentity
    ├── impossible.py    # ImpossibleValues
    ├── outliers.py      # SectorOutliers
    └── jumps.py         # SuddenJumps

notebooks/fundamental_quality.ipynb
```

---

## `src/irp/quality/base.py`

```python
@dataclass(frozen=True)
class Finding:
    table: str
    ticker: str
    variant: str        # 'A' | 'Q'
    report_date: str    # YYYY-MM-DD
    column: str         # affected column (or comma-sep list)
    value: object       # raw value or computed metric (e.g. rel_error, z_score)
    rule: str           # rule name
    detail: str         # human-readable explanation
    severity: str       # 'error' | 'warning'

class Rule(ABC):
    name: str
    description: str
    severity: str = "warning"

    @abstractmethod
    def check(self, data: dict[str, pd.DataFrame]) -> Sequence[Finding]: ...

_REGISTRY: list[type[Rule]] = []

def register(cls: type[Rule]) -> type[Rule]:
    _REGISTRY.append(cls)
    return cls

def all_rules() -> list[Rule]:
    return [cls() for cls in _REGISTRY]
```

## `src/irp/quality/runner.py`

```python
def run(data: dict[str, pd.DataFrame]) -> pd.DataFrame:
    findings: list[Finding] = []
    for rule in all_rules():
        findings.extend(rule.check(data))
    return pd.DataFrame([dataclasses.asdict(f) for f in findings])
```

## `src/irp/quality/rules/__init__.py`

Auto-discovery: glob `*.py` in own directory, `import_module` each → `@register` fires automatically. Adding a new rule file = zero boilerplate.

---

## Rules

### `accounting.py` — AccountingIdentity (severity: error)
`Total Assets = Total Liabilities + Total Equity` within 1% relative tolerance.
Uses `balance`.

### `impossible.py` — ImpossibleValues (severity: error)
Table of (table, column, predicate, label):
- `income` / `Revenue` / `< 0`
- `income` / `Shares (Diluted)` / `<= 0`
- `income` / `Shares (Basic)` / `<= 0`
- `balance` / `Total Assets` / `<= 0`

Single rule, config-driven list of checks — easy to extend.

### `outliers.py` — SectorOutliers (severity: warning)
Join `income` → `companies` → `industries` (on IndustryId) to get Sector.
Z-score per `(Sector, variant)` for: `Revenue`, `Net Income`, `Operating Income (Loss)`.
Flag `|z| > 4`. Requires min 5 peers in sector (skip tiny sectors).

### `jumps.py` — SuddenJumps (severity: warning)
LAG window per `(Ticker, variant)` ordered by `Report Date`.
Metrics + tables:
- `Revenue`, `Net Income` — `income`
- `Total Assets` — `balance`
Flag: `pct_change > 5.0` (>+500%) or `pct_change < -0.80` (<−80%).
Skip first period per ticker (no prior to compare).

---

## Notebook: `notebooks/fundamental_quality.ipynb`

1. **Setup** — DB connection, sys.path, imports
2. **Load data** — read income, balance, cashflow, companies, industries into dict of DataFrames
3. **Run rules** — `from irp.quality import run; findings = run(data)`
4. **Summary** — count findings by rule + severity
5. **Per-rule drill-down** — display flagged rows per rule
6. **Export** — `findings.to_csv('../data/flagged_anomalies.csv', index=False)`

---

## Files
| Path | Action |
|------|--------|
| `src/irp/quality/__init__.py` | Create |
| `src/irp/quality/base.py` | Create |
| `src/irp/quality/runner.py` | Create |
| `src/irp/quality/rules/__init__.py` | Create |
| `src/irp/quality/rules/accounting.py` | Create |
| `src/irp/quality/rules/impossible.py` | Create |
| `src/irp/quality/rules/outliers.py` | Create |
| `src/irp/quality/rules/jumps.py` | Create |
| `notebooks/fundamental_quality.ipynb` | Create |

No changes to existing `src/irp/` modules.

---

## Verification
1. `uv run pytest tests/quality/` — unit tests for each rule with synthetic data
2. Run notebook end-to-end — findings DataFrame non-empty, CSV exported
3. Add a bad synthetic row to balance in test → AccountingIdentity flags it

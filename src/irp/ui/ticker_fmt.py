import re
from datetime import date, timedelta

import pandas as pd

_PER_SHARE = re.compile(r'per.share|eps|dps|dividend.*per', re.I)
_PCT_ITEM = re.compile(r'margin|rate|ratio|yield|roe|roa|roi|roic|growth', re.I)

_PRESET_DAYS: dict[str, int] = {
    '1M': 30,
    '6M': 182,
    '1Y': 365,
    '3Y': 1095,
    '5Y': 1825,
}


def detect_scale(df: pd.DataFrame) -> tuple[float, str]:
    vals = pd.to_numeric(df.values.flatten(), errors='coerce')
    maxabs = float(pd.Series(vals).abs().max())
    if pd.isna(maxabs) or maxabs == 0:
        return 1.0, 'USD'
    if maxabs >= 1_000_000_000:
        return 1e6, 'USD millions'
    if maxabs >= 1_000_000:
        return 1e3, 'USD thousands'
    return 1.0, 'USD'


def is_per_share(item: str) -> bool:
    return bool(_PER_SHARE.search(item))


def is_pct_item(item: str, values: pd.Series) -> bool:
    if not _PCT_ITEM.search(item):
        return False
    non_null = pd.to_numeric(values, errors='coerce').dropna().abs()
    return bool(len(non_null) > 0 and (non_null < 10).all())


def fmt_cell(val: object, *, per_share: bool, pct: bool, divisor: float) -> str:
    try:
        v = float(val)  # type: ignore[arg-type]
    except (TypeError, ValueError):
        return '—'
    if pd.isna(v):
        return '—'
    if per_share:
        return f'{v:,.2f}'
    if pct:
        return f'{v * 100:.2f}%' if abs(v) <= 1 else f'{v:.2f}%'
    s = f'{v / divisor:,.1f}'
    return s[:-2] if s.endswith('.0') else s


def fmt_statement(df: pd.DataFrame) -> tuple[pd.DataFrame, str]:
    """Format a statement DataFrame (items as rows, periods as columns).

    Returns (formatted_df, scale_note).
    """
    if df.empty:
        return df, ''
    monetary_rows = [i for i in df.index if not is_per_share(str(i))]
    monetary_df = df.loc[monetary_rows] if monetary_rows else df
    divisor, scale_label = detect_scale(monetary_df)
    result = df.copy().astype(object)
    for item in df.index:
        row = df.loc[item]
        ps = is_per_share(str(item))
        pct = is_pct_item(str(item), row) if not ps else False
        for col in df.columns:
            result.loc[item, col] = fmt_cell(df.loc[item, col], per_share=ps, pct=pct, divisor=divisor)
    note = (
        f'Amounts in {scale_label} except per-share data.'
        if divisor != 1.0
        else f'Amounts in {scale_label}.'
    )
    return result, note


def fmt_price_table(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return df
    out = df.copy()
    col_lower = {c.lower(): c for c in out.columns}
    rename: dict[str, str] = {}
    for std, alts in [
        ('Open', ['open', 'o']),
        ('High', ['high', 'h']),
        ('Low', ['low', 'l']),
        ('Close', ['close', 'c']),
        ('Volume', ['volume', 'v']),
    ]:
        if std not in out.columns:
            for alt in alts:
                if alt in col_lower:
                    rename[col_lower[alt]] = std
                    break
    if rename:
        out = out.rename(columns=rename)
    if 'Date' in out.columns:
        out['Date'] = pd.to_datetime(out['Date']).dt.strftime('%Y-%m-%d')
    for col in ['Open', 'High', 'Low', 'Close']:
        if col in out.columns:
            out[col] = out[col].apply(lambda v: f'{float(v):,.2f}' if pd.notna(v) else '')
    if 'Volume' in out.columns:
        out['Volume'] = out['Volume'].apply(lambda v: f'{int(float(v)):,}' if pd.notna(v) else '')
    return out


def date_range_for_preset(preset: str) -> tuple[str, str]:
    today = date.today()
    start = today - timedelta(days=_PRESET_DAYS[preset]) if preset in _PRESET_DAYS else date(1900, 1, 1)
    return start.isoformat(), today.isoformat()

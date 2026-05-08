import re

import pandas as pd


def fmt_value(v) -> str:
    if pd.isna(v):
        return ''
    if abs(v) >= 1000:
        return f'{v:,.0f}'
    if abs(v) >= 1:
        return f'{v:,.2f}'
    return f'{v:.4f}'


def fmt_detail(s) -> str:
    if pd.isna(s):
        return s
    return re.sub(r'\b(\d{4,})\b', lambda m: f'{int(m.group()):,}', str(s))


def prep_df(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    if 'rule' in df.columns:
        df = df.drop(columns=['rule'])
    if 'company_name' in df.columns:
        loc = df.columns.get_loc('company_name')
        if isinstance(loc, int):
            df.insert(loc, 'company', df['company_name'].str[:10])
        df = df.drop(columns=['company_name'])
    if 'edgar_url' in df.columns:
        has_form = 'edgar_form' in df.columns
        def _link(r):
            u = r['edgar_url']
            form = r['edgar_form'] if has_form else None
            if not (pd.notna(u) and u and pd.notna(form) and form):
                return ''
            return f'<a href="{u}" target="_blank">{form}</a>'
        df['filings'] = df.apply(_link, axis=1)
        drop = ['edgar_url'] + (['edgar_form'] if has_form else [])
        df = df.drop(columns=drop)
    if 'value' in df.columns:
        df['value'] = df['value'].apply(fmt_value)
    return df

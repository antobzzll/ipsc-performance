import numpy as np
import pandas as pd
import streamlit as st
def q_stats(series):
    """Return the interquartile range (IQR) and median of a pandas Series or numpy array."""
    series = series.dropna()
    q75, q25 = np.percentile(series, [75, 25])
    median = np.median(series)
    return q25, median, q75

def check_iqr(q1, q3, row):
    if q1 > row['stats'][0] and q3 < row['stats'][2]:
        return True
    return False

def safe_numeric(df: pd.DataFrame, cols: list[str]) -> pd.DataFrame:
    out = df.copy()
    for col in cols:
        if col in out.columns:
            out[col] = pd.to_numeric(out[col], errors="coerce")
    return out

def init_global_select(key: str, options: list[str], default: str):
    """Initialize a global select value once, and keep it valid if options change."""
    if key not in st.session_state:
        st.session_state[key] = default if default in options else (options[0] if options else None)
    # If currently stored value is no longer in options (e.g., filters changed), fix it
    if st.session_state[key] not in options and options:
        st.session_state[key] = options[0]
    return st.session_state[key]
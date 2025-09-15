import numpy as np
import pandas as pd
import inspect
import os
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



def get_page_title():
    # Go one frame back in the call stack
    frame = inspect.stack()[1]
    caller_file = frame.filename   # full path of the caller
    filename = os.path.basename(caller_file)  # just the file name
    page_title = filename.split('_', 1)[1] if '_' in filename else filename
    return page_title.replace('_', ' ').replace('.py', '')

def init_global_select(key: str, options: list[str], default: str):
    """Initialize a global select value once, and keep it valid if options change."""
    if key not in st.session_state:
        st.session_state[key] = default if default in options else (options[0] if options else None)
    # If currently stored value is no longer in options (e.g., filters changed), fix it
    if st.session_state[key] not in options and options:
        st.session_state[key] = options[0]
    return st.session_state[key]
# app/lib/data.py
from __future__ import annotations
import pandas as pd
from pathlib import Path

# Try to import Streamlit; allow this module to work outside Streamlit too.
try:
    import streamlit as st
    _HAS_ST = True
except Exception:
    st = None
    _HAS_ST = False

from .models import (
    load_fitds_stages_core,
    class_predict,
    class_predict_per_stage
)

# ---------------- FALLBACK PATHS ----------------
MODEL_FALLBACKS: dict[str, str | None] = {
    "fitds_stages":      "../data/IPSC.csv",
    "fitds_match_stats": None,  # derived (no file)
}

# ---------------- Optional cache decorator ----------------
# Use Streamlit cache when available; else no-op.
def _cache(func):
    if _HAS_ST:
        return st.cache_data(show_spinner=False)(func)
    return func  # no caching outside Streamlit

# ---------------- Cached wrappers ----------------
@_cache
def load_fitds_stages_fallback(path: str | None = MODEL_FALLBACKS["fitds_stages"]) -> pd.DataFrame:
    if path is None:
        raise FileNotFoundError("No path provided for 'fitds_stages'.")
    if not Path(path).exists():
        raise FileNotFoundError(f"Missing dataset at: {path}")
    return load_fitds_stages_core(path)

@_cache
def load_class_predict(_ignored: str | None = None) -> pd.DataFrame:
    stages = get_data("fitds_stages")
    return class_predict(stages)

@_cache
def load_class_predict_per_stage(_ignored: str | None = None) -> pd.DataFrame:
    stages = get_data("fitds_stages")
    return class_predict_per_stage(stages)

# ---------------- Registry ----------------
MODEL_LOADERS: dict[str, callable] = {
    "fitds_stages":      load_fitds_stages_fallback,
    # FIX: use the wrapper, not the core
    "class_predict":     load_class_predict,
    "class_predict_per_stage": load_class_predict_per_stage,
}

# ---------------- Public API ----------------
def get_data(model: str) -> pd.DataFrame:
    """
    Priority:
      1) st.session_state[model]
      2) st.secrets["data_paths"][model] (if Streamlit + secrets)
      3) MODEL_FALLBACKS[model] (file path or None for derived)
    """
    # 1) session_state override (when within Streamlit)
    if _HAS_ST and (model in st.session_state) and isinstance(st.session_state[model], pd.DataFrame):
        return st.session_state[model].copy()

    # 2) optional secrets path (safe even if no secrets file)
    secrets_path = None
    if _HAS_ST:
        try:
            if "data_paths" in st.secrets and isinstance(st.secrets["data_paths"], dict):
                secrets_path = st.secrets["data_paths"].get(model)
        except Exception:
            secrets_path = None

    # 3) pick loader
    loader = MODEL_LOADERS.get(model)
    if loader is None:
        _err(f"Unknown data model: '{model}'. Available: {list(MODEL_LOADERS)}")

    # resolve path (derived models may ignore)
    path = secrets_path or MODEL_FALLBACKS.get(model)

    try:
        return loader(path)
    except FileNotFoundError as e:
        _err(str(e))

def set_data(model: str, df: pd.DataFrame) -> None:
    """Inject a DataFrame into session_state (only in Streamlit)."""
    if not _HAS_ST:
        raise RuntimeError("set_data() requires Streamlit runtime.")
    st.session_state[model] = df

def get_available_models() -> list[str]:
    return list(MODEL_LOADERS.keys())

def ensure_numeric(df: pd.DataFrame, cols: list[str]) -> pd.DataFrame:
    out = df.copy()
    for c in cols:
        if c in out.columns:
            out[c] = pd.to_numeric(out[c], errors="coerce")
    return out

# ---------------- Internal helper ----------------
def _err(msg: str):
    if _HAS_ST:
        st.error(msg); st.stop()
    raise RuntimeError(msg)
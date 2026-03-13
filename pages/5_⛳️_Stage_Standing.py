import numpy as np
import pandas as pd
import streamlit as st

from lib.data import get_data
from lib.stats import stage_standing, match_standing
from lib.utils import get_page_title

st.set_page_config(page_title="Stage Analysis", layout="wide")

# ========= I18N =========
LANG = {
    "en": {
        "select_language": "Language",
        "filters_header": "Filters",
        "data_header_help": "Affect all tables and metrics on this page",
        "match": "Match",
        "match_help": "Select the match to analyze",
        "division": "Division",
        "division_help": "Select the shooter division",
        "stage": "Stage",
        "stage_help": "Select the stage number",
        "show_match_standing": "Show match standing",
        "show_match_standing_help": "Show overall standing for the selected match and division",
        "show_raw_stage_rows": "Show raw stage rows",
        "show_raw_stage_rows_help": "Show original filtered rows for the selected stage",
        "no_data": "No data for the selected filters.",
        "stage_analysis": "Stage Standing",
        "stage_summary": "Stage Summary",
        "stage_standing_header": "Stage Standing",
        "match_standing_header": "Match Standing",
        "raw_stage_rows_header": "Raw Stage Rows",
        "n_shooters": "Shooters",
        "best_hf": "Best HF",
        "avg_hf": "Avg HF",
        "median_hf": "Median HF",
        "winner": "Winner",
        "selected_match": "Selected Match",
        "selected_division": "Selected Division",
        "selected_stage": "Selected Stage",
        "download_csv": "Download CSV",
    },
    "it": {
        "select_language": "Lingua",
        "filters_header": "Filtri",
        "data_header_help": "Influenza tutte le tabelle e metriche di questa pagina",
        "match": "Match",
        "match_help": "Seleziona il match da analizzare",
        "division": "Divisione",
        "division_help": "Seleziona la divisione del tiratore",
        "stage": "Stage",
        "stage_help": "Seleziona il numero di stage",
        "show_match_standing": "Mostra classifica match",
        "show_match_standing_help": "Mostra la classifica generale del match e divisione selezionati",
        "show_raw_stage_rows": "Mostra righe stage grezze",
        "show_raw_stage_rows_help": "Mostra le righe originali filtrate per lo stage selezionato",
        "no_data": "Nessun dato per i filtri selezionati.",
        "stage_analysis": "Classifica Stage",
        "stage_summary": "Riepilogo Stage",
        "stage_standing_header": "Classifica Stage",
        "match_standing_header": "Classifica Match",
        "raw_stage_rows_header": "Righe Grezze Stage",
        "n_shooters": "Tiratori",
        "best_hf": "Miglior HF",
        "avg_hf": "HF Medio",
        "median_hf": "HF Mediano",
        "winner": "Vincitore",
        "selected_match": "Match Selezionato",
        "selected_division": "Divisione Selezionata",
        "selected_stage": "Stage Selezionato",
        "download_csv": "Scarica CSV",
    },
}


def t(key: str, lang: str, **kwargs) -> str:
    base = LANG.get(lang, LANG["en"]).get(key, LANG["en"].get(key, key))
    return base.format(**kwargs) if kwargs else base


# ========= SIDEBAR: LANGUAGE =========
if "language" not in st.session_state:
    st.session_state.language = "it"

language_options = list(LANG.keys())
language = st.sidebar.selectbox(
    t("select_language", st.session_state.language),
    options=language_options,
    index=language_options.index(st.session_state.language),
    key="dd_language_stage_analysis",
)
st.session_state.language = language
_ = lambda k, **kw: t(k, st.session_state.language, **kw)

# ========= DATA =========
stages = get_data("fitds_stages").copy()

if "stg_n" not in stages.columns and "stg" in stages.columns:
    stages = stages.rename(columns={"stg": "stg_n"})

if "shooter_div" not in stages.columns and "div" in stages.columns:
    stages["shooter_div"] = stages["div"]

if "match_date" in stages.columns:
    stages["match_date"] = pd.to_datetime(stages["match_date"], errors="coerce")

for col in ["stg_n", "hf", "stg_match_pts"]:
    if col in stages.columns:
        stages[col] = pd.to_numeric(stages[col], errors="coerce")

st.title(f"{get_page_title()} - {_('stage_analysis')}")

# ========= FILTERS =========
st.sidebar.header(_("filters_header"), help=_("data_header_help"))

match_cols = ["match_name"]
if "match_date" in stages.columns:
    match_cols.append("match_date")

match_df = stages[match_cols].drop_duplicates().copy()

if "match_date" in match_df.columns:
    match_df = match_df.sort_values(["match_date", "match_name"])
    match_df["match_label"] = match_df.apply(
        lambda r: f"{r['match_name']} ({r['match_date'].date()})" if pd.notna(r["match_date"]) else str(r["match_name"]),
        axis=1,
    )
else:
    match_df = match_df.sort_values(["match_name"])
    match_df["match_label"] = match_df["match_name"].astype(str)

match_labels = match_df["match_label"].tolist()

if not match_labels:
    st.warning(_("no_data"))
    st.stop()

if "selected_stage_match_label" not in st.session_state:
    st.session_state.selected_stage_match_label = match_labels[0]

selected_match_label = st.sidebar.selectbox(
    _("match"),
    options=match_labels,
    index=match_labels.index(st.session_state.selected_stage_match_label)
    if st.session_state.selected_stage_match_label in match_labels else 0,
    key="dd_stage_match",
    help=_("match_help"),
)
st.session_state.selected_stage_match_label = selected_match_label

selected_match_row = match_df.loc[match_df["match_label"] == selected_match_label].iloc[0]
selected_match_name = selected_match_row["match_name"]
selected_match_date = selected_match_row["match_date"] if "match_date" in selected_match_row.index else None

filtered = stages[stages["match_name"] == selected_match_name].copy()
if "match_date" in filtered.columns and pd.notna(selected_match_date):
    filtered = filtered[filtered["match_date"] == selected_match_date].copy()

if filtered.empty:
    st.warning(_("no_data"))
    st.stop()

divisions = sorted(filtered["shooter_div"].dropna().astype(str).unique().tolist()) if "shooter_div" in filtered.columns else []
if not divisions:
    st.warning(_("no_data"))
    st.stop()

if "selected_stage_division" not in st.session_state or st.session_state.selected_stage_division not in divisions:
    st.session_state.selected_stage_division = divisions[0]

selected_division = st.sidebar.selectbox(
    _("division"),
    options=divisions,
    index=divisions.index(st.session_state.selected_stage_division),
    key="dd_stage_division",
    help=_("division_help"),
)
st.session_state.selected_stage_division = selected_division

filtered = filtered[filtered["shooter_div"].astype(str) == str(selected_division)].copy()

if filtered.empty:
    st.warning(_("no_data"))
    st.stop()

stage_values = (
    filtered["stg_n"].dropna().astype("Int64").drop_duplicates().sort_values().tolist()
    if "stg_n" in filtered.columns else []
)
if not stage_values:
    st.warning(_("no_data"))
    st.stop()

if "selected_stage_number" not in st.session_state or st.session_state.selected_stage_number not in stage_values:
    st.session_state.selected_stage_number = stage_values[0]

selected_stage = st.sidebar.selectbox(
    _("stage"),
    options=stage_values,
    index=stage_values.index(st.session_state.selected_stage_number),
    key="dd_stage_number",
    help=_("stage_help"),
)
st.session_state.selected_stage_number = selected_stage

show_match_standing = st.sidebar.checkbox(
    _("show_match_standing"),
    value=True,
    key="dd_show_match_standing",
    help=_("show_match_standing_help"),
)

show_raw_stage_rows = st.sidebar.checkbox(
    _("show_raw_stage_rows"),
    value=False,
    key="dd_show_raw_stage_rows",
    help=_("show_raw_stage_rows_help"),
)

stage_df = filtered[filtered["stg_n"] == selected_stage].copy()

if stage_df.empty:
    st.warning(_("no_data"))
    st.stop()

# ========= STANDINGS =========
stage_rank = stage_standing(
    filtered,
    match=selected_match_name,
    shooter_div=selected_division,
    stg_n=selected_stage,
).copy()

match_rank = match_standing(
    filtered,
    match=selected_match_name,
    shooter_div=selected_division,
).copy()

# ========= SUMMARY =========
winner_name = stage_rank.iloc[0]["shooter_name"] if not stage_rank.empty and "shooter_name" in stage_rank.columns else None
best_hf = float(stage_df["hf"].max()) if "hf" in stage_df.columns and stage_df["hf"].notna().any() else np.nan
avg_hf = float(stage_df["hf"].mean()) if "hf" in stage_df.columns and stage_df["hf"].notna().any() else np.nan
median_hf = float(stage_df["hf"].median()) if "hf" in stage_df.columns and stage_df["hf"].notna().any() else np.nan
n_shooters = int(stage_df["shooter_name"].nunique()) if "shooter_name" in stage_df.columns else int(len(stage_df))

st.caption(
    f"**{_('selected_match')}:** {selected_match_label}  \n"
    f"**{_('selected_division')}:** {selected_division}  \n"
    f"**{_('selected_stage')}:** {selected_stage}"
)

st.subheader(_("stage_summary"))
c1, c2, c3, c4, c5 = st.columns(5)
c1.metric(_("n_shooters"), f"{n_shooters}")
c2.metric(_("best_hf"), f"{best_hf:.4f}" if pd.notna(best_hf) else "-")
c3.metric(_("avg_hf"), f"{avg_hf:.4f}" if pd.notna(avg_hf) else "-")
c4.metric(_("median_hf"), f"{median_hf:.4f}" if pd.notna(median_hf) else "-")
c5.metric(_("winner"), str(winner_name) if winner_name is not None else "-")

st.write("---")

# ========= STAGE STANDING =========
st.subheader(_("stage_standing_header"))

st.download_button(
    label=_("download_csv"),
    data=stage_rank.to_csv(index=False).encode("utf-8"),
    file_name=f"stage_standing_{selected_match_name}_{selected_division}_stage_{selected_stage}.csv",
    mime="text/csv",
    key="dl_stage_standing",
)

st.dataframe(stage_rank, use_container_width=True, hide_index=True)

# ========= MATCH STANDING =========
if show_match_standing:
    st.subheader(_("match_standing_header"))

    st.download_button(
        label=_("download_csv"),
        data=match_rank.to_csv(index=False).encode("utf-8"),
        file_name=f"match_standing_{selected_match_name}_{selected_division}.csv",
        mime="text/csv",
        key="dl_stage_match_standing",
    )

    st.dataframe(match_rank, use_container_width=True, hide_index=True)

# ========= RAW ROWS =========
if show_raw_stage_rows:
    st.subheader(_("raw_stage_rows_header"))

    raw_df = stage_df.sort_values("hf", ascending=False) if "hf" in stage_df.columns else stage_df.copy()

    st.download_button(
        label=_("download_csv"),
        data=raw_df.to_csv(index=False).encode("utf-8"),
        file_name=f"raw_stage_rows_{selected_match_name}_{selected_division}_stage_{selected_stage}.csv",
        mime="text/csv",
        key="dl_raw_stage_rows",
    )

    st.dataframe(raw_df, use_container_width=True, hide_index=True)
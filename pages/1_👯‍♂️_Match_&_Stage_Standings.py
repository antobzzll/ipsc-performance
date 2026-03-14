import numpy as np
import pandas as pd
import streamlit as st

from lib.data import get_data
from lib.stats import match_standing, stage_standing
from lib.utils import get_page_title

st.set_page_config(page_title="Match & Stage Standings", layout="wide")

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
        "stage_help": "Select the stage to analyze",
        "show_raw_rows": "Show raw match rows",
        "show_raw_rows_help": "Show original filtered rows for the selected match and division",
        "show_stage_summary": "Show stage summary",
        "show_stage_summary_help": "Show a stage-by-stage summary for the selected match and division",
        "show_stage_standing": "Show stage standing",
        "show_stage_standing_help": "Show standing for the selected stage",
        "no_data": "No data for the selected filters.",
        "match_analysis": "Match Standing",
        "match_summary": "Match Summary",
        "match_standing_header": "Match Standing",
        "stage_summary_header": "Stage Summary",
        "stage_standing_header": "Stage Standing",
        "raw_rows_header": "Raw Match Rows",
        "n_shooters": "Shooters",
        "n_stages": "Stages",
        "winner": "Winner",
        "top_score": "Top Match Pts",
        "avg_hf": "Avg HF",
        "selected_match": "Selected Match",
        "selected_division": "Selected Division",
        "selected_stage": "Selected Stage",
        "download_csv": "Download CSV",
        "best_hf": "Best HF",
        "median_hf": "Median HF",
        "winner_stage_count": "Winner Stage Wins",
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
        "stage_help": "Seleziona lo stage da analizzare",
        "show_raw_rows": "Mostra righe match grezze",
        "show_raw_rows_help": "Mostra le righe originali filtrate per match e divisione selezionati",
        "show_stage_summary": "Mostra riepilogo stage",
        "show_stage_summary_help": "Mostra un riepilogo stage per stage del match e divisione selezionati",
        "show_stage_standing": "Mostra classifica stage",
        "show_stage_standing_help": "Mostra la classifica dello stage selezionato",
        "no_data": "Nessun dato per i filtri selezionati.",
        "match_analysis": "Classifica Match",
        "match_summary": "Riepilogo Match",
        "match_standing_header": "Classifica Match",
        "stage_summary_header": "Riepilogo Stage",
        "stage_standing_header": "Classifica Stage",
        "raw_rows_header": "Righe Grezze Match",
        "n_shooters": "Tiratori",
        "n_stages": "Stage",
        "winner": "Vincitore",
        "top_score": "Punti Match Top",
        "avg_hf": "HF Medio",
        "selected_match": "Match Selezionato",
        "selected_division": "Divisione Selezionata",
        "selected_stage": "Stage Selezionato",
        "download_csv": "Scarica CSV",
        "best_hf": "Miglior HF",
        "median_hf": "HF Mediano",
        "winner_stage_count": "Stage Vinti dal Vincitore",
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
    key="dd_language_match_analysis",
)
st.session_state.language = language
_ = lambda k, **kw: t(k, st.session_state.language, **kw)

# ========= DATA =========
df = get_data("fitds_stages").copy()

if "stg_n" not in df.columns and "stg" in df.columns:
    df = df.rename(columns={"stg": "stg_n"})

if "shooter_div" not in df.columns and "div" in df.columns:
    df["shooter_div"] = df["div"]

if "match_date" in df.columns:
    df["match_date"] = pd.to_datetime(df["match_date"], errors="coerce")

for col in ["stg_n", "hf", "stg_match_pts"]:
    if col in df.columns:
        df[col] = pd.to_numeric(df[col], errors="coerce")

st.title(f"{get_page_title()} - {_('match_analysis')}")

# ========= FILTERS =========
st.sidebar.header(_("filters_header"), help=_("data_header_help"))

match_cols = ["match_name"]
if "match_date" in df.columns:
    match_cols.append("match_date")

matches_df = df[match_cols].drop_duplicates().copy()

if "match_date" in matches_df.columns:
    matches_df = matches_df.sort_values(["match_date", "match_name"])
    matches_df["match_label"] = matches_df.apply(
        lambda r: f"{r['match_name']} ({r['match_date'].date()})" if pd.notna(r["match_date"]) else str(r["match_name"]),
        axis=1,
    )
else:
    matches_df = matches_df.sort_values(["match_name"])
    matches_df["match_label"] = matches_df["match_name"].astype(str)

match_labels = matches_df["match_label"].tolist()
if not match_labels:
    st.warning(_("no_data"))
    st.stop()

if "selected_match_analysis_label" not in st.session_state:
    st.session_state.selected_match_analysis_label = match_labels[0]

selected_match_label = st.sidebar.selectbox(
    _("match"),
    options=match_labels,
    index=match_labels.index(st.session_state.selected_match_analysis_label)
    if st.session_state.selected_match_analysis_label in match_labels else 0,
    key="dd_match_analysis_match",
    help=_("match_help"),
)
st.session_state.selected_match_analysis_label = selected_match_label

selected_match_row = matches_df.loc[matches_df["match_label"] == selected_match_label].iloc[0]
selected_match_name = selected_match_row["match_name"]
selected_match_date = selected_match_row["match_date"] if "match_date" in selected_match_row.index else None

match_df = df[df["match_name"] == selected_match_name].copy()
if "match_date" in match_df.columns and pd.notna(selected_match_date):
    match_df = match_df[match_df["match_date"] == selected_match_date].copy()

if match_df.empty:
    st.warning(_("no_data"))
    st.stop()

divisions = sorted(match_df["shooter_div"].dropna().astype(str).unique().tolist()) if "shooter_div" in match_df.columns else []
if not divisions:
    st.warning(_("no_data"))
    st.stop()

if "selected_match_analysis_division" not in st.session_state or st.session_state.selected_match_analysis_division not in divisions:
    st.session_state.selected_match_analysis_division = divisions[0]

selected_division = st.sidebar.selectbox(
    _("division"),
    options=divisions,
    index=divisions.index(st.session_state.selected_match_analysis_division),
    key="dd_match_analysis_division",
    help=_("division_help"),
)
st.session_state.selected_match_analysis_division = selected_division

match_df = match_df[match_df["shooter_div"].astype(str) == str(selected_division)].copy()

if match_df.empty:
    st.warning(_("no_data"))
    st.stop()

stage_options = []
if "stg_n" in match_df.columns:
    stage_options = (
        match_df["stg_n"]
        .dropna()
        .astype("Int64")
        .sort_values()
        .unique()
        .tolist()
    )

selected_stage = None
if stage_options:
    if "selected_match_analysis_stage" not in st.session_state or st.session_state.selected_match_analysis_stage not in stage_options:
        st.session_state.selected_match_analysis_stage = stage_options[0]

    selected_stage = st.sidebar.selectbox(
        _("stage"),
        options=stage_options,
        index=stage_options.index(st.session_state.selected_match_analysis_stage),
        key="dd_match_analysis_stage",
        help=_("stage_help"),
    )
    st.session_state.selected_match_analysis_stage = selected_stage

show_stage_summary = st.sidebar.checkbox(
    _("show_stage_summary"),
    value=True,
    key="dd_match_analysis_show_stage_summary",
    help=_("show_stage_summary_help"),
)

show_stage_standing = st.sidebar.checkbox(
    _("show_stage_standing"),
    value=True,
    key="dd_match_analysis_show_stage_standing",
    help=_("show_stage_standing_help"),
)

show_raw_rows = st.sidebar.checkbox(
    _("show_raw_rows"),
    value=False,
    key="dd_match_analysis_show_raw_rows",
    help=_("show_raw_rows_help"),
)

# ========= STANDINGS =========
standing = match_standing(
    match_df,
    match=selected_match_name,
    shooter_div=selected_division,
).copy()
standing = standing.drop(columns=["shooter_div"], errors="ignore")

stage_stand = pd.DataFrame()
if show_stage_standing and selected_stage is not None:
    stage_stand = stage_standing(
        match_df,
        match=selected_match_name,
        shooter_div=selected_division,
        stg_n=selected_stage,
    ).copy()

# ========= SUMMARY =========
winner_name = standing.iloc[0]["shooter_name"] if not standing.empty and "shooter_name" in standing.columns else None
top_score = standing["stg_match_pts"].max() if not standing.empty and "stg_match_pts" in standing.columns else np.nan
avg_hf = match_df["hf"].mean() if "hf" in match_df.columns and match_df["hf"].notna().any() else np.nan
n_shooters = match_df["shooter_name"].nunique() if "shooter_name" in match_df.columns else len(match_df)
n_stages = match_df["stg_n"].nunique() if "stg_n" in match_df.columns else np.nan

winner_stage_count = np.nan
if winner_name is not None and "stg_n" in match_df.columns and "hf" in match_df.columns and "shooter_name" in match_df.columns:
    stage_winners = (
        match_df.sort_values(["stg_n", "hf"], ascending=[True, False])
        .dropna(subset=["stg_n"])
        .groupby("stg_n", dropna=False)
        .first()
        .reset_index()
    )
    winner_stage_count = int((stage_winners["shooter_name"] == winner_name).sum())

caption = (
    f"**{_('selected_match')}:** {selected_match_label}  \n"
    f"**{_('selected_division')}:** {selected_division}"
)
if selected_stage is not None:
    caption += f"  \n**{_('selected_stage')}:** {selected_stage}"
st.caption(caption)

st.subheader(_("match_summary"))
c1, c2, c3, c4, c5, c6 = st.columns(6)
c1.metric(_("n_shooters"), f"{int(n_shooters)}" if pd.notna(n_shooters) else "-")
c2.metric(_("n_stages"), f"{int(n_stages)}" if pd.notna(n_stages) else "-")
c3.metric(_("winner"), str(winner_name) if winner_name is not None else "-")
c4.metric(_("top_score"), f"{float(top_score):.2f}" if pd.notna(top_score) else "-")
c5.metric(_("avg_hf"), f"{float(avg_hf):.4f}" if pd.notna(avg_hf) else "-")
c6.metric(_("winner_stage_count"), f"{winner_stage_count}" if pd.notna(winner_stage_count) else "-")

st.write("---")

# ========= MATCH STANDING =========
st.subheader(_("match_standing_header"))
st.dataframe(standing, use_container_width=True, hide_index=True)

# ========= STAGE SUMMARY =========
if show_stage_summary and "stg_n" in match_df.columns:
    st.subheader(_("stage_summary_header"))

    stage_summary = (
        match_df.groupby("stg_n", dropna=False)
        .agg(
            best_hf=("hf", "max") if "hf" in match_df.columns else ("stg_n", "size"),
            median_hf=("hf", "median") if "hf" in match_df.columns else ("stg_n", "size"),
            med_stage_pts=("stg_match_pts", "median") if "stg_match_pts" in match_df.columns else ("stg_n", "size"),
        )
        .reset_index()
        .sort_values("stg_n")
        .rename(columns={"stg_n": _("stage")})
    )

    st.dataframe(stage_summary, use_container_width=True, hide_index=True)
    
# ========= STAGE STANDING =========
if show_stage_standing and selected_stage is not None:
    st.subheader(_("stage_standing_header"))
    if stage_stand.empty:
        st.info(_("no_data"))
    else:
        st.dataframe(stage_stand, use_container_width=True, hide_index=True)



# ========= RAW ROWS =========
if show_raw_rows:
    st.subheader(_("raw_rows_header"))

    raw_df = (
        match_df.sort_values(["stg_n", "hf"], ascending=[True, False])
        if "stg_n" in match_df.columns and "hf" in match_df.columns
        else match_df.copy()
    )

    st.dataframe(raw_df, use_container_width=True, hide_index=True)
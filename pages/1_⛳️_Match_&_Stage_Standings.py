import numpy as np
import pandas as pd
import plotly.graph_objects as go
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
        "select_match_first": "Select a match to display the analysis.",
        "match_placeholder": "-- Select match --",
        "division": "Division",
        "division_help": "Select the shooter division",
        "stage": "Stage",
        "stage_help": "Select the stage to analyze",
        "show_raw_rows": "Show raw match rows",
        "show_raw_rows_help": "Show original filtered rows for the selected match and division",
        "show_stage_standing": "Show stage standing",
        "show_stage_standing_help": "Show standing for the selected stage",
        "no_data": "No data for the selected filters.",
        "match_analysis": "Match Standing",
        "match_summary": "Match Summary",
        "match_standing_header": "Match Standing",
        "stage_standing_header": "Stage Standing",
        "raw_rows_header": "Raw Match Rows",
        "comparison_header": "Shooter Comparison by Stage",
        "comparison_text": (
            "Compare up to three shooters stage by stage within the selected match and division. "
            "Choose points, time or hit factor as the comparison metric."
        ),
        "comparison_metric": "Comparison metric",
        "comparison_metric_help": "Metric shown on the line chart",
        "metric_pts": "Points",
        "metric_time": "Time",
        "metric_hf": "Hit Factor",
        "shooter_1": "Shooter 1",
        "shooter_2": "Shooter 2",
        "shooter_3": "Shooter 3",
        "shooter_help": "Select a shooter to compare",
        "shooter_placeholder": "-- Select shooter --",
        "select_at_least_one_shooter": "Select at least one shooter to display the comparison chart.",
        "n_shooters": "Shooters",
        "n_stages": "Stages",
        "winner": "Winner",
        "top_score": "Top Match Pts",
        "avg_hf": "Avg HF",
        "selected_match": "Selected Match",
        "selected_division": "Selected Division",
        "selected_stage": "Selected Stage",
        "download_csv": "Download CSV",
        "winner_stage_count": "Winner Stage Wins",
    },
    "it": {
        "select_language": "Lingua",
        "filters_header": "Filtri",
        "data_header_help": "Influenza tutte le tabelle e metriche di questa pagina",
        "match": "Match",
        "match_help": "Seleziona il match da analizzare",
        "select_match_first": "Seleziona un match per visualizzare l’analisi.",
        "match_placeholder": "-- Seleziona match --",
        "division": "Divisione",
        "division_help": "Seleziona la divisione del tiratore",
        "stage": "Stage",
        "stage_help": "Seleziona lo stage da analizzare",
        "show_raw_rows": "Mostra righe match grezze",
        "show_raw_rows_help": "Mostra le righe originali filtrate per match e divisione selezionati",
        "show_stage_standing": "Mostra classifica stage",
        "show_stage_standing_help": "Mostra la classifica dello stage selezionato",
        "no_data": "Nessun dato per i filtri selezionati.",
        "match_analysis": "Classifica Match",
        "match_summary": "Riepilogo Match",
        "match_standing_header": "Classifica Match",
        "stage_standing_header": "Classifica Stage",
        "raw_rows_header": "Righe Grezze Match",
        "comparison_header": "Confronto Tiratori per Stage",
        "comparison_text": (
            "Confronta fino a tre tiratori stage per stage nel match e nella divisione selezionati. "
            "Scegli punti, tempo o hit factor come metrica di confronto."
        ),
        "comparison_metric": "Metrica di confronto",
        "comparison_metric_help": "Metrica mostrata nel grafico lineare",
        "metric_pts": "Punti",
        "metric_time": "Tempo",
        "metric_hf": "Hit Factor",
        "shooter_1": "Tiratore 1",
        "shooter_2": "Tiratore 2",
        "shooter_3": "Tiratore 3",
        "shooter_help": "Seleziona un tiratore da confrontare",
        "shooter_placeholder": "-- Seleziona tiratore --",
        "select_at_least_one_shooter": "Seleziona almeno un tiratore per visualizzare il grafico di confronto.",
        "n_shooters": "Tiratori",
        "n_stages": "Stage",
        "winner": "Vincitore",
        "top_score": "Punti Match Top",
        "avg_hf": "HF Medio",
        "selected_match": "Match Selezionato",
        "selected_division": "Divisione Selezionata",
        "selected_stage": "Stage Selezionato",
        "download_csv": "Scarica CSV",
        "winner_stage_count": "Stage Vinti dal Vincitore",
    },
}


def t(key: str, lang: str, **kwargs) -> str:
    base = LANG.get(lang, LANG["en"]).get(key, LANG["en"].get(key, key))
    return base.format(**kwargs) if kwargs else base


def build_stage_comparison_chart(
    df: pd.DataFrame,
    shooters: list[str],
    metric_col: str,
    metric_label: str,
):
    if not shooters:
        st.info(_("select_at_least_one_shooter"))
        return

    need = {"shooter_name", "stg_n", metric_col}
    missing = [c for c in need if c not in df.columns]
    if missing:
        st.info(f"Missing columns for comparison chart: {missing}")
        return

    cdf = df[df["shooter_name"].isin(shooters)].copy()
    if cdf.empty:
        st.info(_("no_data"))
        return

    cdf["stg_n"] = pd.to_numeric(cdf["stg_n"], errors="coerce")
    cdf[metric_col] = pd.to_numeric(cdf[metric_col], errors="coerce")
    cdf = cdf.dropna(subset=["stg_n", metric_col])

    if cdf.empty:
        st.info(_("no_data"))
        return

    plot_df = (
        cdf.groupby(["shooter_name", "stg_n"], as_index=False)
        .agg(
            pts=("stg_match_pts", "mean") if "stg_match_pts" in cdf.columns else (metric_col, "mean"),
            time=("time", "mean") if "time" in cdf.columns else (metric_col, "mean"),
            hf=("hf", "mean") if "hf" in cdf.columns else (metric_col, "mean"),
            metric_value=(metric_col, "mean"),
        )
        .sort_values(["stg_n", "shooter_name"])
    )

    fig = go.Figure()

    for shooter in shooters:
        sdf = plot_df[plot_df["shooter_name"] == shooter].copy()
        if sdf.empty:
            continue

        customdata = np.column_stack(
            [
                sdf["pts"].to_numpy(dtype=float),
                sdf["time"].to_numpy(dtype=float),
                sdf["hf"].to_numpy(dtype=float),
            ]
        )

        fig.add_trace(
            go.Scatter(
                x=sdf["stg_n"],
                y=sdf["metric_value"],
                mode="lines+markers",
                name=shooter,
                customdata=customdata,
                hovertemplate=(
                    "Shooter: %{fullData.name}<br>"
                    "Stage: %{x}<br>"
                    "Pts: %{customdata[0]:.2f}<br>"
                    "Time: %{customdata[1]:.2f}<br>"
                    "HF: %{customdata[2]:.4f}<br>"
                    f"{metric_label}: %{{y}}<extra></extra>"
                ),
            )
        )

    fig.update_layout(
        template="plotly_white",
        margin=dict(l=10, r=10, t=10, b=10),
        hovermode="x unified",
        xaxis=dict(title=_("stage"), dtick=1),
        yaxis=dict(title=metric_label),
        legend=dict(title=""),
    )

    st.plotly_chart(fig, use_container_width=True)


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

if "hf" not in df.columns and "hit_factor" in df.columns:
    df["hf"] = df["hit_factor"]

for col in ["stg_n", "hf", "stg_match_pts", "time"]:
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

match_placeholder = _("match_placeholder")
match_labels = [match_placeholder] + matches_df["match_label"].tolist()

if len(match_labels) == 1:
    st.warning(_("no_data"))
    st.stop()

if "selected_match_analysis_label" not in st.session_state:
    st.session_state.selected_match_analysis_label = match_placeholder

if st.session_state.selected_match_analysis_label not in match_labels:
    st.session_state.selected_match_analysis_label = match_placeholder

selected_match_label = st.sidebar.selectbox(
    _("match"),
    options=match_labels,
    index=match_labels.index(st.session_state.selected_match_analysis_label),
    key="dd_match_analysis_match",
    help=_("match_help"),
)
st.session_state.selected_match_analysis_label = selected_match_label

if selected_match_label == match_placeholder:
    st.info(_("select_match_first"))
    st.stop()

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

if (
    "selected_match_analysis_division" not in st.session_state
    or st.session_state.selected_match_analysis_division not in divisions
):
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
    if (
        "selected_match_analysis_stage" not in st.session_state
        or st.session_state.selected_match_analysis_stage not in stage_options
    ):
        st.session_state.selected_match_analysis_stage = stage_options[0]

    selected_stage = st.sidebar.selectbox(
        _("stage"),
        options=stage_options,
        index=stage_options.index(st.session_state.selected_match_analysis_stage),
        key="dd_match_analysis_stage",
        help=_("stage_help"),
    )
    st.session_state.selected_match_analysis_stage = selected_stage

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

# ========= SHOOTER COMPARISON =========
available_shooters = sorted(match_df["shooter_name"].dropna().astype(str).unique().tolist())
shooter_placeholder = _("shooter_placeholder")
shooter_options = [shooter_placeholder] + available_shooters

for key in [
    "selected_match_analysis_shooter_1",
    "selected_match_analysis_shooter_2",
    "selected_match_analysis_shooter_3",
]:
    if key not in st.session_state or st.session_state[key] not in shooter_options:
        st.session_state[key] = shooter_placeholder

if "selected_match_analysis_compare_metric" not in st.session_state:
    st.session_state.selected_match_analysis_compare_metric = _("metric_hf")

st.subheader(_("comparison_header"))
st.write(_("comparison_text"))

cmp_c1, cmp_c2, cmp_c3, cmp_c4 = st.columns(4, vertical_alignment="bottom")

st.session_state.selected_match_analysis_shooter_1 = cmp_c1.selectbox(
    _("shooter_1"),
    options=shooter_options,
    index=shooter_options.index(st.session_state.selected_match_analysis_shooter_1),
    key="dd_match_analysis_shooter_1",
    help=_("shooter_help"),
)

st.session_state.selected_match_analysis_shooter_2 = cmp_c2.selectbox(
    _("shooter_2"),
    options=shooter_options,
    index=shooter_options.index(st.session_state.selected_match_analysis_shooter_2),
    key="dd_match_analysis_shooter_2",
    help=_("shooter_help"),
)

st.session_state.selected_match_analysis_shooter_3 = cmp_c3.selectbox(
    _("shooter_3"),
    options=shooter_options,
    index=shooter_options.index(st.session_state.selected_match_analysis_shooter_3),
    key="dd_match_analysis_shooter_3",
    help=_("shooter_help"),
)

metric_options = [_("metric_pts"), _("metric_time"), _("metric_hf")]
if st.session_state.selected_match_analysis_compare_metric not in metric_options:
    st.session_state.selected_match_analysis_compare_metric = _("metric_hf")

selected_metric_label = cmp_c4.selectbox(
    _("comparison_metric"),
    options=metric_options,
    index=metric_options.index(st.session_state.selected_match_analysis_compare_metric),
    key="dd_match_analysis_compare_metric",
    help=_("comparison_metric_help"),
)
st.session_state.selected_match_analysis_compare_metric = selected_metric_label

selected_shooters = []
for s in [
    st.session_state.selected_match_analysis_shooter_1,
    st.session_state.selected_match_analysis_shooter_2,
    st.session_state.selected_match_analysis_shooter_3,
]:
    if s != shooter_placeholder and s not in selected_shooters:
        selected_shooters.append(s)

if selected_metric_label == _("metric_pts"):
    metric_col = "stg_match_pts"
    metric_label = _("metric_pts")
elif selected_metric_label == _("metric_time"):
    metric_col = "time"
    metric_label = _("metric_time")
else:
    metric_col = "hf"
    metric_label = _("metric_hf")

build_stage_comparison_chart(
    match_df,
    shooters=selected_shooters,
    metric_col=metric_col,
    metric_label=metric_label,
)
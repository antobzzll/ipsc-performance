import numpy as np
import pandas as pd
import streamlit as st

from lib.data import get_data
from lib.stats import match_standing, stage_standing
from lib.utils import get_page_title
from lib.charts import stage_comparison_chart

st.set_page_config(page_title="Match & Stage Standings", layout="wide")

# ========= I18N =========
LANG = {
    "en": {
        "select_language": "Language",
        "filters_header": "Filters",
        "data_header_help": "Affect all tables and metrics on this page",
        "year": "Year",
        "year_help": "Select the year to filter matches",
        "year_placeholder": "-- Select year --",
        "match": "Match",
        "match_help": "Select the match to analyze",
        "select_match_first": "Select a match to display the analysis.",
        "match_placeholder": "-- Select match --",
        "division": "Division",
        "division_help": "Select the shooter division",
        "stage": "Stage",
        "stage_help": "Select the stage to analyze",
        "show_stage_standing": "Show stage standing",
        "show_stage_standing_help": "Show standing for the selected stage",
        "show_stage_average": "Show stage average",
        "show_stage_average_help": "Overlay the stage average of the selected metric on the chart",
        "stage_average_name": "Stage Avg",
        "no_data": "No data for the selected filters.",
        "match_summary": "Match Summary",
        "match_standing_header": "Match Standing",
        "stage_standing_header": "Stage Standing",
        "comparison_header": "Shooter Comparison by Stage",
        "comparison_text": (
            "Compare up to three shooters stage by stage within the selected match and division. "
            "Choose points, time, hit factor or rank as the comparison metric."
        ),
        "comparison_metric": "Comparison metric",
        "comparison_metric_help": "Metric shown on the line chart",
        "metric_pts": "Points",
        "metric_time": "Time",
        "metric_hf": "Hit Factor",
        "metric_rank": "Rank",
        "shooter_1": "Shooter 1",
        "shooter_2": "Shooter 2",
        "shooter_3": "Shooter 3",
        "shooter_help": "Select a shooter to compare",
        "shooter_placeholder": "-- Select shooter --",
        "select_at_least_one_shooter": "Select at least one shooter to display the comparison chart.",
        "comparison_summary_header": "Comparison Summary",
        "summary_shooter": "Shooter",
        "summary_rank": "Rank",
        "summary_pct": "Match %",
        "summary_points_pct": "Total Stage Points %",
        "summary_total_time": "Total Time",
        "n_shooters": "Shooters",
        "n_stages": "Stages",
        "winner": "Winner",
        "selected_match": "Selected Match",
        "selected_division": "Selected Division",
        "selected_stage": "Selected Stage",
        "winner_stage_count": "Winner Stage Wins",
    },
    "it": {
        "select_language": "Lingua",
        "filters_header": "Filtri",
        "data_header_help": "Influenza tutte le tabelle e metriche di questa pagina",
        "year": "Anno",
        "year_help": "Seleziona l’anno per filtrare i match",
        "year_placeholder": "-- Seleziona anno --",
        "match": "Match",
        "match_help": "Seleziona il match da analizzare",
        "select_match_first": "Seleziona un match per visualizzare l’analisi.",
        "match_placeholder": "-- Seleziona match --",
        "division": "Divisione",
        "division_help": "Seleziona la divisione del tiratore",
        "stage": "Stage",
        "stage_help": "Seleziona lo stage da analizzare",
        "show_stage_standing": "Mostra classifica stage",
        "show_stage_standing_help": "Mostra la classifica dello stage selezionato",
        "show_stage_average": "Mostra media stage",
        "show_stage_average_help": "Sovrapponi al grafico la media stage della metrica selezionata",
        "stage_average_name": "Media Stage",
        "no_data": "Nessun dato per i filtri selezionati.",
        "match_summary": "Riepilogo Match",
        "match_standing_header": "Classifica Match",
        "stage_standing_header": "Classifica Stage",
        "comparison_header": "Confronto Tiratori per Stage",
        "comparison_text": (
            "Confronta fino a tre tiratori stage per stage nel match e nella divisione selezionati. "
            "Scegli punti, tempo, hit factor o rank come metrica di confronto."
        ),
        "comparison_metric": "Metrica di confronto",
        "comparison_metric_help": "Metrica mostrata nel grafico lineare",
        "metric_pts": "Punti",
        "metric_time": "Tempo",
        "metric_hf": "Hit Factor",
        "metric_rank": "Rank",
        "shooter_1": "Tiratore 1",
        "shooter_2": "Tiratore 2",
        "shooter_3": "Tiratore 3",
        "shooter_help": "Seleziona un tiratore da confrontare",
        "shooter_placeholder": "-- Seleziona tiratore --",
        "select_at_least_one_shooter": "Seleziona almeno un tiratore per visualizzare il grafico di confronto.",
        "comparison_summary_header": "Riepilogo Confronto",
        "summary_shooter": "Tiratore",
        "summary_rank": "Rank",
        "summary_pct": "% Match",
        "summary_points_pct": "% Totale Punti Stage",
        "summary_total_time": "Tempo Totale",
        "n_shooters": "Tiratori",
        "n_stages": "Stage",
        "winner": "Vincitore",
        "selected_match": "Match Selezionato",
        "selected_division": "Divisione Selezionata",
        "selected_stage": "Stage Selezionato",
        "winner_stage_count": "Stage Vinti dal Vincitore",
    },
}


def t(key: str, lang: str, **kwargs) -> str:
    base = LANG.get(lang, LANG["en"]).get(key, LANG["en"].get(key, key))
    return base.format(**kwargs) if kwargs else base


def build_comparison_summary(
    df: pd.DataFrame,
    shooters: list[str],
    standing_df: pd.DataFrame | None = None,
) -> pd.DataFrame:
    if not shooters:
        return pd.DataFrame()

    required = {"shooter_name", "pts", "stg_max_pts", "time"}
    missing = [c for c in required if c not in df.columns]
    if missing:
        return pd.DataFrame()

    tmp = df[df["shooter_name"].astype(str).isin([str(s) for s in shooters])].copy()
    if tmp.empty:
        return pd.DataFrame()

    for col in ["pts", "stg_max_pts", "time"]:
        tmp[col] = pd.to_numeric(tmp[col], errors="coerce")

    summary = (
        tmp.groupby("shooter_name", as_index=False)
        .agg(
            total_pts=("pts", "sum"),
            total_stg_max_pts=("stg_max_pts", "sum"),
            total_time=("time", "sum"),
        )
    )

    summary["points_pct"] = np.where(
        summary["total_stg_max_pts"] > 0,
        summary["total_pts"] / summary["total_stg_max_pts"],
        np.nan,
    )

    if standing_df is not None and not standing_df.empty and "shooter_name" in standing_df.columns:
        standing_cols = ["shooter_name"]
        if "rank" in standing_df.columns:
            standing_cols.append("rank")
        if "pct" in standing_df.columns:
            standing_cols.append("pct")

        summary = summary.merge(
            standing_df[standing_cols].drop_duplicates(subset=["shooter_name"]),
            on="shooter_name",
            how="left",
        )

    order_map = {name: i for i, name in enumerate(shooters)}
    summary["sort_order"] = summary["shooter_name"].map(order_map)
    summary = summary.sort_values("sort_order").drop(columns=["sort_order"])

    return summary


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
    df["match_year"] = df["match_date"].dt.year

if "hf" not in df.columns and "hit_factor" in df.columns:
    df["hf"] = df["hit_factor"]

for col in ["stg_n", "hf", "stg_match_pts", "time", "pts", "stg_max_pts", "pct", "div_factor_standing", "hf_pct"]:
    if col in df.columns:
        df[col] = pd.to_numeric(df[col], errors="coerce")

st.title(f"{get_page_title()}")

# ========= FILTERS =========
st.sidebar.header(_("filters_header"), help=_("data_header_help"))

available_years = []
if "match_year" in df.columns:
    available_years = (
        df["match_year"]
        .dropna()
        .astype(int)
        .sort_values()
        .unique()
        .tolist()
    )

if not available_years:
    st.warning(_("no_data"))
    st.stop()

year_placeholder = _("year_placeholder")
year_options = [year_placeholder] + [str(y) for y in available_years]

if "selected_match_analysis_year" not in st.session_state:
    st.session_state.selected_match_analysis_year = str(available_years[-1])

if st.session_state.selected_match_analysis_year not in year_options:
    st.session_state.selected_match_analysis_year = str(available_years[-1])

selected_year_label = st.sidebar.selectbox(
    _("year"),
    options=year_options,
    index=year_options.index(st.session_state.selected_match_analysis_year),
    key="dd_match_analysis_year",
    help=_("year_help"),
)
st.session_state.selected_match_analysis_year = selected_year_label

if selected_year_label == year_placeholder:
    st.info(_("no_data"))
    st.stop()

selected_year = int(selected_year_label)
df = df[df["match_year"] == selected_year].copy()

match_cols = ["match_name"]
if "match_date" in df.columns:
    match_cols.append("match_date")

matches_df = df[match_cols].drop_duplicates().copy()

if "match_date" in matches_df.columns:
    matches_df = matches_df.sort_values(["match_date", "match_name"])
    matches_df["match_label"] = matches_df.apply(
        lambda r: f"{r['match_name']} ({r['match_date'].date()})"
        if pd.notna(r["match_date"])
        else str(r["match_name"]),
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

divisions = (
    sorted(match_df["shooter_div"].dropna().astype(str).unique().tolist())
    if "shooter_div" in match_df.columns
    else []
)
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

show_stage_standing = st.sidebar.checkbox(
    _("show_stage_standing"),
    value=False,
    key="dd_match_analysis_show_stage_standing",
    help=_("show_stage_standing_help"),
)

# ========= STAGE OPTIONS =========
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

    selected_stage = st.session_state.selected_match_analysis_stage

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
winner_name = (
    standing.iloc[0]["shooter_name"]
    if not standing.empty and "shooter_name" in standing.columns
    else None
)
n_shooters = (
    match_df["shooter_name"].nunique()
    if "shooter_name" in match_df.columns
    else len(match_df)
)
n_stages = match_df["stg_n"].nunique() if "stg_n" in match_df.columns else np.nan

winner_stage_count = np.nan
if (
    winner_name is not None
    and "stg_n" in match_df.columns
    and "hf" in match_df.columns
    and "shooter_name" in match_df.columns
):
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
c1, c2, c3, c4 = st.columns([0.15, 0.15, 0.5, 0.2])
c1.metric(_("n_shooters"), f"{int(n_shooters)}" if pd.notna(n_shooters) else "-")
c2.metric(_("n_stages"), f"{int(n_stages)}" if pd.notna(n_stages) else "-")
c3.metric(_("winner"), str(winner_name) if winner_name is not None else "-")
c4.metric(_("winner_stage_count"), f"{winner_stage_count}" if pd.notna(winner_stage_count) else "-")

st.write("---")

# ========= MATCH STANDING =========
st.subheader(_("match_standing_header"))
st.dataframe(standing, use_container_width=True, hide_index=True)

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

metric_options = [_("metric_pts"), _("metric_time"), _("metric_hf"), _("metric_rank")]
if "selected_match_analysis_compare_metric" not in st.session_state:
    st.session_state.selected_match_analysis_compare_metric = _("metric_hf")

if st.session_state.selected_match_analysis_compare_metric not in metric_options:
    st.session_state.selected_match_analysis_compare_metric = _("metric_hf")

if "selected_match_analysis_show_stage_average" not in st.session_state:
    st.session_state.selected_match_analysis_show_stage_average = True

st.subheader(_("comparison_header"))
st.write(_("comparison_text"))

cmp_c1, cmp_c2, cmp_c3, cmp_c4, cmp_c5 = st.columns([1, 1, 1, 1, 1])

selected_shooter_1 = cmp_c1.selectbox(
    _("shooter_1"),
    options=shooter_options,
    index=shooter_options.index(st.session_state.selected_match_analysis_shooter_1),
    key="dd_match_analysis_shooter_1",
    help=_("shooter_help"),
)

selected_shooter_2 = cmp_c2.selectbox(
    _("shooter_2"),
    options=shooter_options,
    index=shooter_options.index(st.session_state.selected_match_analysis_shooter_2),
    key="dd_match_analysis_shooter_2",
    help=_("shooter_help"),
)

selected_shooter_3 = cmp_c3.selectbox(
    _("shooter_3"),
    options=shooter_options,
    index=shooter_options.index(st.session_state.selected_match_analysis_shooter_3),
    key="dd_match_analysis_shooter_3",
    help=_("shooter_help"),
)

selected_metric_label = cmp_c4.selectbox(
    _("comparison_metric"),
    options=metric_options,
    index=metric_options.index(st.session_state.selected_match_analysis_compare_metric),
    key="dd_match_analysis_compare_metric",
    help=_("comparison_metric_help"),
)

show_stage_average = cmp_c5.checkbox(
    _("show_stage_average"),
    value=st.session_state.selected_match_analysis_show_stage_average,
    key="dd_match_analysis_show_stage_average",
    help=_("show_stage_average_help"),
)

st.session_state.selected_match_analysis_shooter_1 = selected_shooter_1
st.session_state.selected_match_analysis_shooter_2 = selected_shooter_2
st.session_state.selected_match_analysis_shooter_3 = selected_shooter_3
st.session_state.selected_match_analysis_compare_metric = selected_metric_label
st.session_state.selected_match_analysis_show_stage_average = show_stage_average

selected_shooters = []
for shooter in [selected_shooter_1, selected_shooter_2, selected_shooter_3]:
    if shooter != shooter_placeholder and shooter not in selected_shooters:
        selected_shooters.append(shooter)

use_stage_rank_as_metric = selected_metric_label == _("metric_rank")

if selected_metric_label == _("metric_pts"):
    metric_col = "pts"
    metric_label = _("metric_pts")
elif selected_metric_label == _("metric_time"):
    metric_col = "time"
    metric_label = _("metric_time")
elif selected_metric_label == _("metric_rank"):
    metric_col = "div_factor_standing"
    metric_label = _("metric_rank")
else:
    metric_col = "hf"
    metric_label = _("metric_hf")

stage_comparison_chart(
    match_df,
    shooters=selected_shooters,
    metric_col=metric_col,
    metric_label=metric_label,
    use_stage_rank_as_metric=use_stage_rank_as_metric,
    show_stage_average=show_stage_average,
    stage_average_name=_("stage_average_name"),
    empty_message=_("no_data"),
    select_message=_("select_at_least_one_shooter"),
)

comparison_summary = build_comparison_summary(
    match_df,
    selected_shooters,
    standing_df=standing,
)
if not comparison_summary.empty:
    st.subheader(_("comparison_summary_header"))

    rename_map = {
        "shooter_name": _("summary_shooter"),
        "rank": _("summary_rank"),
        "pct": _("summary_pct"),
        "points_pct": _("summary_points_pct"),
        "total_time": _("summary_total_time"),
    }

    cols = ["shooter_name", "rank", "pct", "points_pct", "total_time"]
    cols = [c for c in cols if c in comparison_summary.columns]

    display_summary = comparison_summary[cols].rename(columns=rename_map)

    format_map = {}
    if _("summary_rank") in display_summary.columns:
        format_map[_("summary_rank")] = "{:.0f}"
    if _("summary_pct") in display_summary.columns:
        format_map[_("summary_pct")] = "{:.2%}"
    if _("summary_points_pct") in display_summary.columns:
        format_map[_("summary_points_pct")] = "{:.2%}"
    if _("summary_total_time") in display_summary.columns:
        format_map[_("summary_total_time")] = "{:.2f}"

    st.dataframe(
        display_summary.style.format(format_map),
        use_container_width=True,
        hide_index=True,
    )

# ========= STAGE STANDING =========
if show_stage_standing:
    title_c1, title_c2 = st.columns([0.8, 0.2])
    title_c1.subheader(_("stage_standing_header"))

    if stage_options:
        selected_stage = title_c2.selectbox(
            _("stage"),
            options=stage_options,
            index=stage_options.index(st.session_state.selected_match_analysis_stage),
            key="dd_match_analysis_stage_top",
            help=_("stage_help"),
        )
        st.session_state.selected_match_analysis_stage = selected_stage

        stage_stand = stage_standing(
            match_df,
            match=selected_match_name,
            shooter_div=selected_division,
            stg_n=selected_stage,
        ).copy()

    if selected_stage is None or stage_stand.empty:
        st.info(_("no_data"))
    else:
        st.dataframe(stage_stand, use_container_width=True, hide_index=True)
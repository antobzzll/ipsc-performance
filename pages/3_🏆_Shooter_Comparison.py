import numpy as np
import pandas as pd
import streamlit as st

from lib.data import get_data
from lib.utils import get_page_title, safe_numeric
from lib.charts import (
    get_shooter_color_map,
    multi_shooter_match_pct,
    multi_shooter_trend,
)

st.set_page_config(page_title="Shooter Comparison", layout="wide")

# ========= I18N =========
LANG = {
    "en": {
        "select_language": "Language",
        "filters_header": "Filters",
        "data_header_help": "Affect all charts on this page",
        "division": "Division",
        "division_help": "All selected shooters must compete in this division",
        "power_factor": "Power Factor",
        "power_factor_help": "Filter by power factor",
        "match_year": "Year",
        "match_year_help": "Filter matches by year",
        "championship": "Championship",
        "championship_help": "Filter matches by championship",
        "match_levels": "Match Levels",
        "match_levels_help": "Filter matches by level",
        "matches": "Matches",
        "matches_help": "Filter by specific matches",
        "chart_options": "Chart Options",
        "ref_lines": "Ref. lines at 50%",
        "ref_lines_help": "Show reference lines at 50%",
        "no_data": "No data for the selected filters.",
        "no_division": "No division data available.",
        "shooter_1": "Shooter 1",
        "shooter_2": "Shooter 2",
        "shooter_3": "Shooter 3",
        "shooter_help": "Select a shooter",
        "shooter_placeholder": "-- Select shooter --",
        "select_shooters_first": "Select at least one shooter to display the comparison.",
        "match_results_header": "Match Results",
        "match_results_text": (
            "Division percentage over time: each shooter's final score as a fraction "
            "of the match winner's total points."
        ),
        "result_metric": "Metric",
        "result_pct": "Division %",
        "result_rank": "Rank",
        "pts_pct_header": "Stage Points %",
        "pts_pct_text": (
            "Average division stage points percentage per match "
            "(shooter pts / best pts in division for each stage)."
        ),
        "time_pct_header": "Stage Time %",
        "time_pct_text": (
            "Average division stage time percentage per match "
            "(fastest time in division / shooter time — higher means faster)."
        ),
    },
    "it": {
        "select_language": "Lingua",
        "filters_header": "Filtri",
        "data_header_help": "Influenza tutti i grafici di questa pagina",
        "division": "Divisione",
        "division_help": "Tutti i tiratori selezionati devono gareggiare in questa divisione",
        "power_factor": "Power Factor",
        "power_factor_help": "Filtra per power factor",
        "match_year": "Anno",
        "match_year_help": "Filtra i match per anno",
        "championship": "Campionato",
        "championship_help": "Filtra per campionato",
        "match_levels": "Livelli Match",
        "match_levels_help": "Filtra i match per livello",
        "matches": "Match",
        "matches_help": "Filtra per match specifici",
        "chart_options": "Opzioni Grafici",
        "ref_lines": "Linee di rif. al 50%",
        "ref_lines_help": "Mostra linee di riferimento al 50%",
        "no_data": "Nessun dato per i filtri selezionati.",
        "no_division": "Nessun dato di divisione disponibile.",
        "shooter_1": "Tiratore 1",
        "shooter_2": "Tiratore 2",
        "shooter_3": "Tiratore 3",
        "shooter_help": "Seleziona un tiratore",
        "shooter_placeholder": "-- Seleziona tiratore --",
        "select_shooters_first": "Seleziona almeno un tiratore per visualizzare il confronto.",
        "match_results_header": "Risultati Match",
        "match_results_text": (
            "Risultato di gara nel tempo: punteggio % finale di ogni tiratore "
            "rispetto al vincitore del match."
        ),
        "result_metric": "Metrica",
        "result_pct": "% Divisione",
        "result_rank": "Classifica",
        "pts_pct_header": "% Punti Stage",
        "pts_pct_text": (
            "% Punti stage raccolti"
            "(punti tiratore / punti disponibili nella gara)."
        ),
        "time_pct_header": "% Tempo Stage",
        "time_pct_text": (
            "% tempo stage sulla divisione per match "
            "(tempo più veloce in divisione / tempo tiratore — valore più alto = più veloce)."
        ),
    },
}


def t(key: str, lang: str, **kwargs) -> str:
    base = LANG.get(lang, LANG["en"]).get(key, LANG["en"].get(key, key))
    return base.format(**kwargs) if kwargs else base


def _prepare(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    if "match_date" in out.columns:
        out["match_date"] = pd.to_datetime(out["match_date"], errors="coerce")
    if "shooter_div" not in out.columns and "div" in out.columns:
        out["shooter_div"] = out["div"]
    if "stg_n" not in out.columns and "stg" in out.columns:
        out["stg_n"] = pd.to_numeric(out["stg"], errors="coerce")
    elif "stg_n" in out.columns:
        out["stg_n"] = pd.to_numeric(out["stg_n"], errors="coerce")
    if "match_year" not in out.columns and "match_date" in out.columns:
        out["match_year"] = out["match_date"].dt.year.astype("Int64")
    numeric_cols = [
        "stg_match_pts", "div_pts_perc", "div_time_perc", "div_factor_perc",
        "pts", "stg_max_pts", "time",
    ]
    for col in numeric_cols:
        if col in out.columns:
            out[col] = pd.to_numeric(out[col], errors="coerce")
    return out


# ========= SESSION STATE =========
if "language" not in st.session_state:
    st.session_state.language = "it"

# ========= SIDEBAR: LANGUAGE =========
language_options = list(LANG.keys())
language = st.sidebar.selectbox(
    t("select_language", st.session_state.language),
    options=language_options,
    index=language_options.index(st.session_state.language),
    key="dd_language_comparison",
)
st.session_state.language = language
_ = lambda k, **kw: t(k, st.session_state.language, **kw)

# ========= DATA =========
stages = _prepare(get_data("fitds_stages"))

st.title(get_page_title())

# ========= SIDEBAR FILTERS =========
st.sidebar.header(_("filters_header"), help=_("data_header_help"))

# Division (single select — required for same-division comparison)
if "shooter_div" not in stages.columns:
    st.warning(_("no_division"))
    st.stop()

divisions_available = sorted(stages["shooter_div"].dropna().astype(str).unique().tolist())
if not divisions_available:
    st.warning(_("no_division"))
    st.stop()

if (
    "selected_comparison_division" not in st.session_state
    or st.session_state.selected_comparison_division not in divisions_available
):
    st.session_state.selected_comparison_division = divisions_available[0]

selected_division = st.sidebar.selectbox(
    _("division"),
    divisions_available,
    index=divisions_available.index(st.session_state.selected_comparison_division),
    key="dd_comparison_division",
    help=_("division_help"),
)
st.session_state.selected_comparison_division = selected_division

df = stages[stages["shooter_div"].astype(str) == str(selected_division)].copy()

# Power Factor
if "power_factor" in df.columns:
    pf_available = sorted(df["power_factor"].dropna().unique().tolist())
    if pf_available:
        pf_filter = st.sidebar.selectbox(
            _("power_factor"), pf_available,
            key="dd_comparison_pf", help=_("power_factor_help"),
        )
        df = df[df["power_factor"] == pf_filter]

# Year
if "match_year" in df.columns:
    year_available = sorted(df["match_year"].dropna().unique().tolist())
    if year_available:
        year_filter = st.sidebar.multiselect(
            _("match_year"), year_available, default=year_available,
            key="dd_comparison_year", help=_("match_year_help"),
        )
        df = df[df["match_year"].isin(year_filter)]

# Championship
if "championship" in df.columns:
    champ_available = sorted(df["championship"].dropna().astype(str).unique().tolist())
    if champ_available:
        champ_filter = st.sidebar.multiselect(
            _("championship"), champ_available, default=champ_available,
            key="dd_comparison_championship", help=_("championship_help"),
        )
        df = df[df["championship"].astype(str).isin(champ_filter)]

# Match Levels
if "match_level" in df.columns:
    level_available = sorted(df["match_level"].dropna().unique().tolist())
    if level_available:
        level_filter = st.sidebar.multiselect(
            _("match_levels"), level_available, default=level_available,
            key="dd_comparison_levels", help=_("match_levels_help"),
        )
        df = df[df["match_level"].isin(level_filter)]

# Matches
if "match_date" in df.columns:
    matches_available = (
        df[["match_name", "match_date"]]
        .drop_duplicates()
        .sort_values("match_date")["match_name"]
        .tolist()
    )
else:
    matches_available = sorted(df["match_name"].dropna().unique().tolist())

match_filter = st.sidebar.multiselect(
    _("matches"), matches_available, default=matches_available,
    key="dd_comparison_matches", help=_("matches_help"),
)
df = df[df["match_name"].isin(match_filter)]

# Chart options
st.sidebar.header(_("chart_options"))
show_ref = st.sidebar.checkbox(
    _("ref_lines"), value=True,
    key="dd_comparison_ref", help=_("ref_lines_help"),
)

if df.empty:
    st.warning(_("no_data"))
    st.stop()

# ========= SHOOTER SELECTION =========
shooter_placeholder = _("shooter_placeholder")
shooters_available = sorted(df["shooter_name"].dropna().astype(str).unique().tolist())
shooter_options = [shooter_placeholder] + shooters_available

for key in [
    "selected_comparison_shooter_1",
    "selected_comparison_shooter_2",
    "selected_comparison_shooter_3",
]:
    if key not in st.session_state or st.session_state[key] not in shooter_options:
        st.session_state[key] = shooter_placeholder

sel_c1, sel_c2, sel_c3 = st.columns(3)

selected_shooter_1 = sel_c1.selectbox(
    _("shooter_1"), shooter_options,
    index=shooter_options.index(st.session_state.selected_comparison_shooter_1),
    key="dd_comparison_shooter_1", help=_("shooter_help"),
)
selected_shooter_2 = sel_c2.selectbox(
    _("shooter_2"), shooter_options,
    index=shooter_options.index(st.session_state.selected_comparison_shooter_2),
    key="dd_comparison_shooter_2", help=_("shooter_help"),
)
selected_shooter_3 = sel_c3.selectbox(
    _("shooter_3"), shooter_options,
    index=shooter_options.index(st.session_state.selected_comparison_shooter_3),
    key="dd_comparison_shooter_3", help=_("shooter_help"),
)

st.session_state.selected_comparison_shooter_1 = selected_shooter_1
st.session_state.selected_comparison_shooter_2 = selected_shooter_2
st.session_state.selected_comparison_shooter_3 = selected_shooter_3

selected_shooters = []
for s in [selected_shooter_1, selected_shooter_2, selected_shooter_3]:
    if s != shooter_placeholder and s not in selected_shooters:
        selected_shooters.append(s)

if not selected_shooters:
    st.info(_("select_shooters_first"))
    st.stop()

color_map = get_shooter_color_map(selected_shooters)

st.write("---")

# ========= CHART 1: MATCH RESULTS =========
res_c1, res_c2 = st.columns([5, 1], vertical_alignment="center")
with res_c1:
    st.subheader(_("match_results_header"))
with res_c2:
    result_metric_label = st.selectbox(
        _("result_metric"),
        [_("result_pct"), _("result_rank")],
        key="dd_comparison_result_metric",
        label_visibility="collapsed",
    )

result_metric = "rank" if result_metric_label == _("result_rank") else "pct"

st.write(_("match_results_text"))
multi_shooter_match_pct(
    df,
    shooters=selected_shooters,
    shooter_div=selected_division,
    metric=result_metric,
    show_ref=show_ref,
    color_map=color_map,
    empty_message=_("no_data"),
    select_message=_("select_shooters_first"),
)

# ========= CHARTS 2 & 3: STAGE POINTS % and TIME % (side by side) =========
pts_col, time_col = st.columns(2)

with pts_col:
    st.subheader(_("pts_pct_header"))
    st.write(_("pts_pct_text"))
    multi_shooter_trend(
        df,
        shooters=selected_shooters,
        metric_col="div_pts_perc",
        y_title="Division Points %",
        agg="mean",
        show_ref=show_ref,
        ref_val=0.5,
        tickformat=".0%",
        show_legend=False,
        color_map=color_map,
        empty_message=_("no_data"),
        select_message=_("select_shooters_first"),
    )

with time_col:
    st.subheader(_("time_pct_header"))
    st.write(_("time_pct_text"))
    multi_shooter_trend(
        df,
        shooters=selected_shooters,
        metric_col="div_time_perc",
        y_title="Division Time %",
        agg="mean",
        show_ref=show_ref,
        ref_val=0.5,
        tickformat=".0%",
        show_legend=False,
        color_map=color_map,
        empty_message=_("no_data"),
        select_message=_("select_shooters_first"),
    )

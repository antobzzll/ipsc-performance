import numpy as np
import pandas as pd
import streamlit as st
from lib.data import get_data
from lib.stats import aggregate_shooter_performance
from lib.utils import get_page_title
from lib.charts import stage_distr, stage_scatter, shooter_match_history

st.set_page_config(page_title="Shooter Performance", layout="wide")

# ========= I18N =========
LANG = {
    "en": {
        "select_language": "Language",
        "data_header": "Data",
        "data_header_help": "Affect all charts on this page",
        "normalization": "Normalization",
        "per_division": "Per Division",
        "per_class": "Per Class",
        "norm_help": "Normalize data per division or per class",
        "filters_header": "Filters",
        "shooter": "Shooter",
        "shooter_help": "Select shooter to analyze",
        "select_shooter_first": "Select a shooter to display the analysis.",
        "avg_performance": "AVG {scope} Performance",
        "division": "Division",
        "division_help": "Filter by division (e.g., Production / Open / Standard)",
        "power_factor": "Power Factor",
        "power_factor_help": "Filter by power factor (e.g., Minor / Major)",
        "match_year": "Year",
        "match_year_help": "Filter matches by year",
        "championship": "Championship",
        "championship_help": "Filter matches by championship",
        "shooter_div": "Shooter Division",
        "shooter_div_help": "Filter results by shooter division",
        "match_levels": "Match Levels",
        "match_levels_help": "Filter matches by level (if available)",
        "matches": "Matches",
        "matches_help": "Filter by specific matches (if available)",
        "chart_options": "Chart Options",
        "ref_lines": "Ref. lines at 50%",
        "ref_lines_help": "Show reference lines at 50%",
        "no_data": "No data for the selected filters.",
        "stage_perf_distr": "Stage Performance Distribution",
        "stage_perf_distr_help_text": (
            "This chart shows how your scores (or stage performance) stack up in each match. "
            "The box represents the middle range of scores for all shooters in your division, "
            "while the whiskers show the full spread. Dots outside the whiskers are unusually "
            "high or low scores (outliers). A line connects the average scores across matches, "
            "helping you see trends over time.\n\n"
            "Why it matters: You can quickly tell if your performance is consistent or if certain "
            "matches were tougher, giving you a clear picture of where you stand compared to others."
        ),
        "stage_points_time_header": "Stage Points and Time",
        "stage_points_time_text": (
            "This chart plots your individual stage results (points vs. time) for each match, "
            "with a diamond marking the average for each match. Each dot is a stage, colored by match, "
            "so you can see how your speed and accuracy vary.\n\n"
            "Why it matters: It highlights your strengths (e.g., fast and accurate stages) and weaknesses "
            "(e.g., slow or low-scoring stages), helping you focus your training on specific skills."
        ),
        "match_history_header": "Match History",
        "match_history_text": (
            "This chart shows your placement history across the selected matches. "
            "You can switch between percentage-based views and standing-based views to compare "
            "your results against the full division, your class, and the non-M/GM reference lines."
        ),
        "history_metric": "History metric",
        "percentage": "Percentage",
        "standing": "Standing",
        "show_stage_points": "Show stage points",
        "show_stage_points_help": "Show individual stage points on scatter chart",
        "show_centroid_labels": "Show centroid labels",
        "show_centroid_labels_help": "Show match name labels next to centroids on scatter chart",
        "show_regression": "Show regression line",
        "show_regression_help": "Show regression line between centroids on scatter chart",
        "stage_point_size": "Stage point size",
        "stage_point_size_help": "Size of individual stage points on scatter chart",
        "stage_point_opacity": "Stage point opacity",
        "stage_point_opacity_help": "Opacity of individual stage points on scatter chart",
        "centroid_size": "Centroid size",
        "centroid_size_help": "Size of match centroids on scatter chart",
    },
    "it": {
        "select_language": "Lingua",
        "data_header": "Dati",
        "data_header_help": "Influenza tutti i grafici di questa pagina",
        "normalization": "Normalizzazione",
        "per_division": "Per Divisione",
        "per_class": "Per Classe",
        "norm_help": "Normalizza i dati per divisione o per classe",
        "filters_header": "Filtri",
        "shooter": "Tiratore",
        "shooter_help": "Seleziona il tiratore da analizzare",
        "select_shooter_first": "Seleziona un tiratore per visualizzare l’analisi.",
        "avg_performance": "Media Prestazioni {scope}",
        "division": "Divisione",
        "division_help": "Filtra per divisione (es. Production / Open / Standard)",
        "power_factor": "Power Factor",
        "power_factor_help": "Filtra per power factor (es. Minor / Major)",
        "match_year": "Anno",
        "match_year_help": "Filtra i match per anno",
        "championship": "Campionato",
        "championship_help": "Filtra i match per campionato",
        "shooter_div": "Divisione Tiratore",
        "shooter_div_help": "Filtra i risultati per divisione del tiratore",
        "match_levels": "Livelli Match",
        "match_levels_help": "Filtra i match per livello (se disponibile)",
        "matches": "Match",
        "matches_help": "Filtra per specifici match (se disponibili)",
        "chart_options": "Opzioni Grafici",
        "ref_lines": "Linee di rif. al 50%",
        "ref_lines_help": "Mostra linee di riferimento al 50%",
        "no_data": "Nessun dato per i filtri selezionati.",
        "stage_perf_distr": "Distribuzione delle Prestazioni per Stage",
        "stage_perf_distr_help_text": (
            "Questo grafico mostra come si posizionano le tue prestazioni ottenute negli stage di gara. "
            "Ogni scatola rappresenta l'intervallo centrale delle tue prestazioni di stage rispetto alla normalizzazione scelta (divisione o classe), "
            "mentre i baffi mostrano l'intera dispersione. I punti fuori dai baffi sono valori insolitamente alti o bassi (outlier). "
            "Una linea collega i punteggi mediani tra i match per evidenziare l'andamento nel tempo.\n\n"
            "Perché è utile: ti permette rapidamente di capire quale sia stata la tua performance mediana e il livello di costanza in gara (ampiezza del box)."
        ),
        "stage_points_time_header": "Punti di Stage e Tempo",
        "stage_points_time_text": (
            "Questo grafico mostra i tuoi risultati per singolo stage (punti vs tempo) per ciascun match, "
            "con un rombo che indica la media del match. Ogni punto è uno stage, colorato per match, "
            "così da vedere come variano velocità e accuratezza.\n\n"
            "Perché è utile: mette in evidenza punti di forza (es. stage veloci e precisi) e debolezze "
            "(es. lenti o con pochi punti), aiutandoti a focalizzare l’allenamento su abilità specifiche."
        ),
        "match_history_header": "Storico Match",
        "match_history_text": (
            "Questo grafico mostra l’andamento dei tuoi piazzamenti nei match selezionati. "
            "Puoi passare da una vista percentuale a una vista per piazzamento per confrontare "
            "i risultati nella divisione, nella tua classe e rispetto alle linee di riferimento non-M/GM."
        ),
        "history_metric": "Metrica storico",
        "percentage": "Percentuale",
        "standing": "Classifica",
        "show_stage_points": "Mostra punti stage",
        "show_stage_points_help": "Mostra i punti dei singoli stage nello scatter",
        "show_centroid_labels": "Mostra etichette centroidi",
        "show_centroid_labels_help": "Mostra i nomi dei match vicino ai centroidi",
        "show_regression": "Mostra retta di regressione",
        "show_regression_help": "Mostra la retta di regressione tra i centroidi",
        "stage_point_size": "Dimensione punti stage",
        "stage_point_size_help": "Dimensione dei punti degli stage nello scatter",
        "stage_point_opacity": "Opacità punti stage",
        "stage_point_opacity_help": "Opacità dei punti degli stage nello scatter",
        "centroid_size": "Dimensione centroidi",
        "centroid_size_help": "Dimensione dei centroidi dei match nello scatter",
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
    key="dd_language"
)
st.session_state.language = language
_ = lambda k, **kw: t(k, st.session_state.language, **kw)

# ========= DATA =========
stages = get_data("fitds_stages").copy()

if "match_date" in stages.columns:
    stages["match_date"] = pd.to_datetime(stages["match_date"], errors="coerce")

if "shooter_div" not in stages.columns and "div" in stages.columns:
    stages["shooter_div"] = stages["div"]

if "stg_n" not in stages.columns and "stg" in stages.columns:
    stages["stg_n"] = pd.to_numeric(stages["stg"], errors="coerce")
elif "stg_n" in stages.columns:
    stages["stg_n"] = pd.to_numeric(stages["stg_n"], errors="coerce")

if "stg_match_pts" in stages.columns:
    stages["stg_match_pts"] = pd.to_numeric(stages["stg_match_pts"], errors="coerce")

if "championship" in stages.columns:
    stages["championship"] = stages["championship"].astype(str)

# ========= FILTERS =========
st.sidebar.header(_("data_header"), help=_("data_header_help"))
normalization = st.sidebar.selectbox(
    _("normalization"),
    [_("per_division"), _("per_class")],
    index=0,
    key="dd_normalization",
    help=_("norm_help")
)
norm = "div" if normalization == _("per_division") else "cls"
norm_header = "Division" if norm == "div" else "Class"

st.sidebar.header(_("filters_header"), help=_("data_header_help"))

shooters = ["-- Select shooter --"] + sorted(stages["shooter_name"].dropna().unique().tolist())
default_shooter = "-- Select shooter --"

if "selected_shooter" not in st.session_state:
    st.session_state.selected_shooter = default_shooter

st.title(f"{get_page_title()}")

c1, c2, c3, c4 = st.columns(4, vertical_alignment="bottom")
st.session_state.selected_shooter = c1.selectbox(
    _("shooter"),
    shooters,
    index=shooters.index(st.session_state.selected_shooter)
    if st.session_state.selected_shooter in shooters else 0,
    key="dd_shooter",
    help=_("shooter_help")
)

if st.session_state.selected_shooter == "-- Select shooter --":
    st.info(_("select_shooter_first"))
    st.stop()

# ========= SHOOTER DATA =========
sh_stages = (
    stages[stages["shooter_name"] == st.session_state.selected_shooter]
    .reset_index(drop=True)
    .copy()
)

if "shooter_div" not in sh_stages.columns and "div" in sh_stages.columns:
    sh_stages["shooter_div"] = sh_stages["div"]

if "shooter_div" in sh_stages.columns:
    divs_available = sorted(sh_stages["shooter_div"].dropna().unique().tolist())
    if divs_available:
        shooter_div_filter = st.sidebar.multiselect(
            _("shooter_div"),
            divs_available,
            default=divs_available,
            key="dd_shooter_div",
            help=_("shooter_div_help")
        )
        sh_stages = sh_stages[sh_stages["shooter_div"].isin(shooter_div_filter)]
    else:
        shooter_div_filter = []
else:
    shooter_div_filter = []

geom_perf_mean = aggregate_shooter_performance(sh_stages, stage_pct_col=f"{norm}_factor_perc")
c4.metric(_("avg_performance", scope=norm_header), f"{geom_perf_mean['G']:.0%}")
st.write("---")

c1, c2 = st.sidebar.columns(2)

if "div" in sh_stages.columns:
    div_available = sorted(sh_stages["div"].dropna().unique().tolist())
    div_filter = c1.selectbox(_("division"), div_available, key="dd_div", help=_("division_help"))
    sh_stages = sh_stages[sh_stages["div"] == div_filter]
else:
    div_filter = []

if "power_factor" in sh_stages.columns:
    pf_available = sorted(sh_stages["power_factor"].dropna().unique().tolist())
    power_factor_filter = c2.selectbox(_("power_factor"), pf_available, key="dd_power_factor", help=_("power_factor_help"))
    sh_stages = sh_stages[sh_stages["power_factor"] == power_factor_filter]
else:
    power_factor_filter = []

if "match_year" not in sh_stages.columns and "match_date" in sh_stages.columns:
    sh_stages["match_year"] = sh_stages["match_date"].dt.year.astype("Int64")

if "match_year" in sh_stages.columns:
    year_available = sorted(sh_stages["match_year"].dropna().unique().tolist())
    if year_available:
        year_filter = st.sidebar.multiselect(
            _("match_year"),
            year_available,
            default=year_available,
            key="dd_match_year",
            help=_("match_year_help")
        )
        sh_stages = sh_stages[sh_stages["match_year"].isin(year_filter)]
    else:
        year_filter = []
else:
    year_filter = []

# Championship BEFORE levels and matches
if "championship" in sh_stages.columns:
    championship_available = sorted(sh_stages["championship"].dropna().astype(str).unique().tolist())
    if championship_available:
        championship_filter = st.sidebar.multiselect(
            _("championship"),
            championship_available,
            default=championship_available,
            key="dd_championship",
            help=_("championship_help")
        )
        sh_stages = sh_stages[sh_stages["championship"].astype(str).isin(championship_filter)]
    else:
        championship_filter = []
else:
    championship_filter = []

if "match_level" in sh_stages.columns:
    levels_available = sorted(sh_stages["match_level"].dropna().unique().tolist())
    match_level_filter = st.sidebar.multiselect(
        _("match_levels"),
        levels_available,
        default=levels_available,
        key="dd_match_levels",
        help=_("match_levels_help")
    )
    df_lvl = sh_stages[sh_stages["match_level"].isin(match_level_filter)]
else:
    match_level_filter = []
    df_lvl = sh_stages.copy()

if "match_date" in df_lvl.columns:
    matches_available = (
        df_lvl[["match_name", "match_date"]]
        .drop_duplicates()
        .sort_values("match_date")["match_name"]
        .tolist()
    )
else:
    matches_available = sorted(df_lvl["match_name"].dropna().unique().tolist())

match_filter = st.sidebar.multiselect(
    _("matches"),
    matches_available,
    default=matches_available,
    key="dd_matches",
    help=_("matches_help")
)
df_lvl = df_lvl[df_lvl["match_name"].isin(match_filter)]

# ========= SIDEBAR OPTIONS =========
st.sidebar.header(_("chart_options"), help=_("data_header_help"))
show_ref = st.sidebar.checkbox(_("ref_lines"), value=True, key="dd_ref", help=_("ref_lines_help"))

# ========= MAIN DATAFRAMES =========
df = df_lvl.copy()
if match_filter:
    df = df[df["match_name"].isin(match_filter)]

if df.empty:
    st.warning(_("no_data"))
    st.stop()

stage_n_preds = df.merge(
    get_data("class_predict"),
    on=["match_name", "shooter_div", "shooter_name"],
    how="left"
)
preds = get_data("class_predict_per_stage")

if "stg_n" not in df.columns and "stg" in df.columns:
    df = df.rename(columns={"stg": "stg_n"})
if "stg_n" not in preds.columns and "stg" in preds.columns:
    preds = preds.rename(columns={"stg": "stg_n"})

if "stg_n" in df.columns and "stg_n" in preds.columns:
    df["stg_n"] = pd.to_numeric(df["stg_n"], errors="coerce").astype("Int64")
    preds["stg_n"] = pd.to_numeric(preds["stg_n"], errors="coerce").astype("Int64")

keys = ["match_name", "shooter_div", "shooter_name", "stg_n"]
cols = keys + ["pred_class", "relation", "robust_z_class", "q1", "class_median", "q3", "n_in_class"]

dup_mask = preds.duplicated(subset=keys, keep=False)
if dup_mask.any():
    st.warning(
        f"class_predict_per_stage has {dup_mask.sum()} duplicated rows on keys {keys}. "
        "Keeping the first occurrence per key."
    )
    preds = preds.sort_values(keys).drop_duplicates(subset=keys, keep="first")

stage_class_preds = df.merge(preds[cols], on=keys, how="left", validate="many_to_one")

history_df = stages.copy()

if "shooter_div" not in history_df.columns and "div" in history_df.columns:
    history_df["shooter_div"] = history_df["div"]

if shooter_div_filter and "shooter_div" in history_df.columns:
    history_df = history_df[history_df["shooter_div"].isin(shooter_div_filter)]

if div_filter and "div" in history_df.columns:
    history_df = history_df[history_df["div"] == div_filter]

if power_factor_filter and "power_factor" in history_df.columns:
    history_df = history_df[history_df["power_factor"] == power_factor_filter]

if "match_year" not in history_df.columns and "match_date" in history_df.columns:
    history_df["match_year"] = history_df["match_date"].dt.year.astype("Int64")

if year_filter and "match_year" in history_df.columns:
    history_df = history_df[history_df["match_year"].isin(year_filter)]

if championship_filter and "championship" in history_df.columns:
    history_df = history_df[history_df["championship"].astype(str).isin(championship_filter)]

if match_level_filter and "match_level" in history_df.columns:
    history_df = history_df[history_df["match_level"].isin(match_level_filter)]

if match_filter:
    history_df = history_df[history_df["match_name"].isin(match_filter)]

# ========= CHARTS =========
title_c1, title_c2 = st.columns([5, 1], vertical_alignment="center")
with title_c1:
    st.subheader(_("match_history_header"))
with title_c2:
    history_metric_label = st.selectbox(
        _("history_metric"),
        [_("percentage"), _("standing")],
        index=0,
        key="dd_history_metric_in_perf",
        label_visibility="collapsed",
    )

history_metric = "pct" if history_metric_label == _("percentage") else "rank"

st.write(_("match_history_text"))

if history_df.empty:
    st.warning(_("no_data"))
else:
    shooter_match_history(
        history_df,
        shooter_name=st.session_state.selected_shooter,
        shooter_div=div_filter if div_filter else (
            sh_stages["shooter_div"].dropna().iloc[0]
            if "shooter_div" in sh_stages.columns and not sh_stages.empty else None
        ),
        metric=history_metric,
        lock_y=True,
    )

st.subheader(_("stage_perf_distr"))
st.write(_("stage_perf_distr_help_text"))
stage_distr(stage_class_preds, norm=norm, show_ref=show_ref, lock_axes=True)

st.subheader(_("stage_points_time_header"))
st.write(_("stage_points_time_text"))
c1, c2 = st.columns([3, 1], vertical_alignment="top")
with c2:
    show_points = st.checkbox(_("show_stage_points"), value=True, key="dd_show_points", help=_("show_stage_points_help"))
    show_labels = st.checkbox(_("show_centroid_labels"), value=True, key="dd_labels", help=_("show_centroid_labels_help"))
    show_reg = st.checkbox(_("show_regression"), value=True, key="dd_show_reg", help=_("show_regression_help"))

with c1:
    point_size = st.slider(_("stage_point_size"), 20, 200, 60, 10, key="dd_pt_size", help=_("stage_point_size_help"))
    point_opacity = st.slider(_("stage_point_opacity"), 0.2, 1.0, 0.6, 0.1, key="dd_pt_opacity", help=_("stage_point_opacity_help"))
    centroid_size = st.slider(_("centroid_size"), 80, 400, 220, 20, key="dd_centroid_size", help=_("centroid_size_help"))

stage_scatter(
    stage_class_preds,
    norm=norm,
    point_size=point_size,
    point_opacity=point_opacity,
    centroid_size=centroid_size,
    show_labels=show_labels,
    show_ref=show_ref,
    lock_axes=True,
    show_points=show_points,
    show_regression=show_reg
)
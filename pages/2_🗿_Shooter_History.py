import pandas as pd
import streamlit as st
from lib.utils import get_page_title
from lib.data import get_data
from lib.charts import match_count, shooter_match_history

st.set_page_config(page_title="Shooter History", layout="wide")
st.title(get_page_title())

# ========= DATA =========
stages = get_data("fitds_stages").copy()

# Normalize common columns
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

if "match_level" in stages.columns:
    stages["match_level"] = stages["match_level"].astype(str)

# ========= SHOOTER SELECT =========
shooters = ["-- Select shooter --"] + sorted(stages["shooter_name"].dropna().unique().tolist())
default_shooter = "-- Select shooter --"

if "selected_shooter" not in st.session_state:
    st.session_state.selected_shooter = default_shooter

c1, c2, c3, c4 = st.columns(4, vertical_alignment="bottom")
st.session_state.selected_shooter = c1.selectbox(
    "Shooter",
    shooters,
    index=shooters.index(st.session_state.selected_shooter)
    if st.session_state.selected_shooter in shooters else 0,
    key="dd_shooter",
    help="Select shooter to analyze.",
)

if st.session_state.selected_shooter == "-- Select shooter --":
    st.info("Select a shooter to display the analysis.")
    st.stop()

# ========= SHOOTER DATA =========
sh_stages = (
    stages[stages["shooter_name"] == st.session_state.selected_shooter]
    .reset_index(drop=True)
    .copy()
)

if sh_stages.empty:
    st.warning("No data for the selected shooter.")
    st.stop()

# ========= TOP METRICS =========
current_class = "-"
if "match_date" in sh_stages.columns:
    latest_date = sh_stages["match_date"].max()
    latest_rows = sh_stages[sh_stages["match_date"] == latest_date].copy()
else:
    latest_rows = sh_stages.copy()

if "shooter_class" in latest_rows.columns:
    class_vals = latest_rows["shooter_class"].dropna().unique().tolist()
    if class_vals:
        current_class = class_vals[0]

total_matches = sh_stages["match_name"].nunique() if "match_name" in sh_stages.columns else 0
total_stages = int(sh_stages["stg_n"].count()) if "stg_n" in sh_stages.columns else len(sh_stages)

c2.metric("Current Class", f"{current_class}")
c3.metric("Total Matches Disputed", f"{total_matches}")
c4.metric("Total Stages Disputed", f"{total_stages}")
st.write("---")

# ========= MATCH COUNT CHART =========
st.subheader("Match Count")
c1, c2 = st.columns([3, 1], vertical_alignment="top")

x_axis_options = {
    "year": "Year",
    "match_level": "Match Level",
    "div": "Division",
    "class": "Class",
}
color_options = {
    "match_level": "Match Level",
    "div": "Division",
    "class": "Class",
    "none": "None",
}

with c2:
    st.markdown("#### Chart options")
    x_choice_label = st.selectbox("X-axis", list(x_axis_options.values()), index=0)
    x_choice = list(x_axis_options.keys())[list(x_axis_options.values()).index(x_choice_label)]

    color_choice_label = st.selectbox("Color", list(color_options.values()), index=0)
    color_choice = list(color_options.keys())[list(color_options.values()).index(color_choice_label)]

with c1:
    match_count(
        stages,
        st.session_state.selected_shooter,
        x_axis=x_choice,
        color=None if color_choice == "none" else color_choice,
    )

st.write("---")

# ========= MATCH HISTORY FILTERS =========
st.sidebar.header("Match History Filters")

# Division
standing_divs = []
if "shooter_div" in sh_stages.columns:
    standing_divs = sorted(sh_stages["shooter_div"].dropna().astype(str).unique().tolist())

selected_div = None
if standing_divs:
    if (
        "selected_history_division" not in st.session_state
        or st.session_state.selected_history_division not in standing_divs
    ):
        st.session_state.selected_history_division = standing_divs[0]

    selected_div = st.sidebar.selectbox(
        "Division",
        standing_divs,
        index=standing_divs.index(st.session_state.selected_history_division),
        key="dd_history_div",
        help="Select division for match history.",
    )
    st.session_state.selected_history_division = selected_div

# Metric
metric_options = ["Percentage", "Standing"]
if (
    "selected_history_metric_label" not in st.session_state
    or st.session_state.selected_history_metric_label not in metric_options
):
    st.session_state.selected_history_metric_label = metric_options[0]

metric_label = st.sidebar.selectbox(
    "Metric",
    metric_options,
    index=metric_options.index(st.session_state.selected_history_metric_label),
    key="dd_history_metric",
    help="Show either pct or rank.",
)
st.session_state.selected_history_metric_label = metric_label
metric = "pct" if metric_label == "Percentage" else "rank"

# Lock y axis
if "selected_history_lock_y" not in st.session_state:
    st.session_state.selected_history_lock_y = False

lock_y = st.sidebar.checkbox(
    "Lock y-axis",
    value=st.session_state.selected_history_lock_y,
    key="dd_history_lock_y",
)
st.session_state.selected_history_lock_y = lock_y

# ========= DEPENDENT FILTERS =========
history_df = stages.copy()

if selected_div is not None:
    history_df = history_df[history_df["shooter_div"].astype(str) == str(selected_div)].copy()
else:
    history_df = history_df.iloc[0:0].copy()

# Year
if "match_year" not in history_df.columns and "match_date" in history_df.columns:
    history_df["match_year"] = history_df["match_date"].dt.year.astype("Int64")

if "match_year" in history_df.columns:
    year_options = sorted(history_df["match_year"].dropna().unique().tolist())
    if year_options:
        if (
            "selected_history_years" not in st.session_state
            or not set(st.session_state.selected_history_years).issubset(set(year_options))
        ):
            st.session_state.selected_history_years = year_options

        year_filter = st.sidebar.multiselect(
            "Year",
            year_options,
            default=st.session_state.selected_history_years,
            key="dd_history_year",
            help="Filter matches by year.",
        )
        st.session_state.selected_history_years = year_filter
        history_df = history_df[history_df["match_year"].isin(year_filter)]
    else:
        year_filter = []
        st.session_state.selected_history_years = []
else:
    year_filter = []
    st.session_state.selected_history_years = []

# Championship
if "championship" in history_df.columns:
    championship_options = sorted(history_df["championship"].dropna().astype(str).unique().tolist())
    if championship_options:
        if (
            "selected_history_championships" not in st.session_state
            or not set(st.session_state.selected_history_championships).issubset(set(championship_options))
        ):
            st.session_state.selected_history_championships = championship_options

        championship_filter = st.sidebar.multiselect(
            "Championship",
            championship_options,
            default=st.session_state.selected_history_championships,
            key="dd_history_championship",
            help="Filter matches by championship.",
        )
        st.session_state.selected_history_championships = championship_filter
        history_df = history_df[history_df["championship"].astype(str).isin(championship_filter)]
    else:
        championship_filter = []
        st.session_state.selected_history_championships = []
else:
    championship_filter = []
    st.session_state.selected_history_championships = []

# Match level
if "match_level" in history_df.columns:
    match_level_options = sorted(history_df["match_level"].dropna().astype(str).unique().tolist())
    if match_level_options:
        if (
            "selected_history_match_levels" not in st.session_state
            or not set(st.session_state.selected_history_match_levels).issubset(set(match_level_options))
        ):
            st.session_state.selected_history_match_levels = match_level_options

        match_level_filter = st.sidebar.multiselect(
            "Match level",
            match_level_options,
            default=st.session_state.selected_history_match_levels,
            key="dd_history_match_level",
            help="Filter matches by match level.",
        )
        st.session_state.selected_history_match_levels = match_level_filter
        history_df = history_df[history_df["match_level"].astype(str).isin(match_level_filter)]
    else:
        match_level_filter = []
        st.session_state.selected_history_match_levels = []
else:
    match_level_filter = []
    st.session_state.selected_history_match_levels = []

# Match names
if "match_date" in history_df.columns:
    match_name_options = (
        history_df[["match_name", "match_date"]]
        .drop_duplicates()
        .sort_values("match_date")["match_name"]
        .tolist()
    )
else:
    match_name_options = (
        sorted(history_df["match_name"].dropna().astype(str).unique().tolist())
        if "match_name" in history_df.columns else []
    )

if match_name_options:
    if (
        "selected_history_match_names" not in st.session_state
        or not set(st.session_state.selected_history_match_names).issubset(set(match_name_options))
    ):
        st.session_state.selected_history_match_names = match_name_options

    match_name_filter = st.sidebar.multiselect(
        "Match name",
        match_name_options,
        default=st.session_state.selected_history_match_names,
        key="dd_history_match_name",
        help="Filter matches by match name.",
    )
    st.session_state.selected_history_match_names = match_name_filter
    history_df = history_df[history_df["match_name"].astype(str).isin(match_name_filter)]
else:
    match_name_filter = []
    st.session_state.selected_history_match_names = []

# ========= MATCH HISTORY CHART =========
st.subheader("Match History")

if selected_div is None:
    st.info("No division data available for this shooter.")
elif history_df.empty:
    st.info("No data for the selected match history filters.")
else:
    shooter_match_history(
        history_df,
        shooter_name=st.session_state.selected_shooter,
        shooter_div=selected_div,
        metric=metric,
        lock_y=lock_y,
    )
    
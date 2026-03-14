import pandas as pd
import streamlit as st
from lib.utils import get_page_title
from lib.data import get_data
from lib.charts import match_count

st.set_page_config(page_title="Home", layout="wide")
st.title(get_page_title())

# ========= DATA =========
stages = get_data("fitds_stages").copy()

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
    help="Select shooter to analyze."
)

# Stop page if no shooter is selected
if st.session_state.selected_shooter == "-- Select shooter --":
    st.info("Select a shooter to display the analysis.")
    st.stop()

# ========= FILTERED SHOOTER DATA =========
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
    sh_stages["match_date"] = pd.to_datetime(sh_stages["match_date"], errors="coerce")
    latest_date = sh_stages["match_date"].max()
    latest_rows = sh_stages[sh_stages["match_date"] == latest_date].copy()
else:
    latest_rows = sh_stages.copy()

if "shooter_class" in latest_rows.columns:
    class_vals = latest_rows["shooter_class"].dropna().unique().tolist()
    if class_vals:
        current_class = class_vals[0]

total_matches = sh_stages["match_name"].nunique() if "match_name" in sh_stages.columns else 0
total_stages = sh_stages["stg_n"].count() if "stg_n" in sh_stages.columns else (
    sh_stages["stg"].count() if "stg" in sh_stages.columns else len(sh_stages)
)

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
    st.markdown("#### Chart options:")
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
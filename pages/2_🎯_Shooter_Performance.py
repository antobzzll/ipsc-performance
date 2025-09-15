import numpy as np
import pandas as pd
import streamlit as st
from lib.data import get_data, ensure_numeric
from lib.stats import aggregate_shooter_performance
# from lib.charts import chart_stage_scatter_with_centroids, chart_distribution, chart_class_flow, chart_shooter_bubble_map  
from lib.utils import get_page_title, init_global_select
from lib.charts import stage_distr, stage_scatter, class_bubble

st.set_page_config(page_title="Shooter Performance", layout="wide")

# ========= DATA =========
stages = get_data('fitds_stages')

# ========= FILTERS =========
st.sidebar.header('Data', help="Affect all charts on this page")
normalization = st.sidebar.selectbox(
    'Normalization',
    ['Per Division', 'Per Class'],
    index=0,
    key='dd_normalization',
    help="Normalize data per division or per class"
)
norm = 'div' if normalization == 'Per Division' else 'cls'
norm_header = normalization.replace('Per ', '')

st.sidebar.header("Filters", help="Affect all charts on this page")

# Shooter
shooters = sorted(stages["shooter"].dropna().unique().tolist())
default_shooter = (
    "BUZZELLI, ANTONIO" if "BUZZELLI, ANTONIO" in shooters else (shooters[0] if shooters else None)
)
if "selected_shooter" not in st.session_state:
    st.session_state.selected_shooter = default_shooter
st.title(f"{get_page_title()}")

c1, c2, c3, c4 = st.columns(4, vertical_alignment='bottom')
st.session_state.selected_shooter = c1.selectbox(
        "Shooter",
        shooters,
        index=shooters.index(st.session_state.selected_shooter),
        key="dd_shooter",
        help="Select shooter to analyze"
    )
sh_stages = stages[(stages["shooter"] == st.session_state.selected_shooter)].reset_index().copy()
geom_perf_mean = aggregate_shooter_performance(sh_stages, stage_pct_col=f'{norm}_factor_perc')
c4.metric(f"AVG {norm_header} Performance", f"{geom_perf_mean['G']:.0%}")
st.write("---")

c1, c2 = st.sidebar.columns(2)

# Division
if "div" in sh_stages.columns:
    div_available = sorted(sh_stages["div"].dropna().unique().tolist())
    div_filter = c1.selectbox(
        "Division",
        div_available,
        # default=div_available,
        key="dd_div",
        help="Filter by division (e.g., Production / Open / Standard)"
    )
    sh_stages = sh_stages[sh_stages["div"]==div_filter]
else:
    div_filter = []

# Power Factor
if "power_factor" in sh_stages.columns:
    pf_available = sorted(sh_stages["power_factor"].dropna().unique().tolist())
    power_factor_filter = c2.selectbox(
        "Power Factor",
        pf_available,
        # default=pf_available,
        key="dd_power_factor",
        help="Filter by power factor (e.g., Minor / Major)"
    )
    sh_stages = sh_stages[sh_stages["power_factor"] == power_factor_filter]
else:
    power_factor_filter = []

# Levels
if "match_level" in sh_stages.columns:
    levels_available = sorted(sh_stages["match_level"].dropna().unique().tolist())
    match_level_filter = st.sidebar.multiselect(
        "Match Levels",
        levels_available,
        default=levels_available,
        key="dd_match_levels",
        help="Filter matches by level (if available)"
    )
    df_lvl = sh_stages[sh_stages["match_level"].isin(match_level_filter)]
else:
    match_level_filter = []
    df_lvl = sh_stages.copy()

# Matches (depends on levels + pf + div)
matches_available = (
    df_lvl[["match_name", "match_date"]]
    .drop_duplicates()
    .sort_values("match_date")["match_name"]
    .tolist()
)
match_filter = st.sidebar.multiselect(
    "Matches",
    matches_available,
    default=matches_available,
    key="dd_matches",
    help="Filter by specific matches (if available)"
)
df_lvl = df_lvl[df_lvl["match_name"].isin(match_filter)]

# ========= SIDEBAR OPTIONS =========
st.sidebar.header("Chart Options", help="Affect all charts on this page")
# lock_axes = st.sidebar.checkbox("Lock axes to 0-100%", value=True, key="dd_lock", help="Force both axes to always span 0% to 100%")
show_ref = st.sidebar.checkbox("Ref. lines at 50%", value=True, key="dd_ref", help="Show reference lines at 50%")

# ========= MAIN DATAFRAMES =========
df = df_lvl.copy()
if match_filter:
    df = df[df["match_name"].isin(match_filter)]
if df.empty:
    st.warning("No data for the selected filters.")
    st.stop()
stage_n_preds = df.merge(get_data('class_predict'), on=['match_name','div','shooter'], how='left')
preds = get_data('class_predict_per_stage')

keys = ['match_name', 'div', 'shooter', 'stg']  # include stg!
cols = keys + ['pred_class','relation','robust_z_class','q1','class_median','q3','n_in_class']

stage_class_preds = (
    df.merge(preds[cols], on=keys, how='left', validate='one_to_one')
)# ========= CHARTS =========

st.subheader("Stage Performance Distribution")
c1, c2 = st.columns([2, 1], vertical_alignment='top')
with c1:
    stage_distr(
        stage_class_preds,
        norm=norm,
        show_ref=show_ref,
        lock_axes=True
    )
with c2:
    st.write('''
             This chart shows how your scores (or stage performance) stack up in each match. 
             The "box" represents the middle range of scores for all shooters in your division, while the "whiskers" show the full spread. Dots outside the whiskers are unusually high or low scores (outliers). 
             A line connects the average scores across matches, helping you see trends over time. 
             Why it matters: You can quickly tell if your performance is consistent or if certain matches were tougher, giving you a clear picture of where you stand compared to others.
             ''')

st.subheader("Class Prediction Based on Performance")
c1, c2 = st.columns([1, 2], vertical_alignment='top')
with c1:
    st.write('''This chart focuses on your personal performance across matches. 
             Each bubble represents a match, placed by match name and your median score. 
             The bubble size shows how consistent you were (bigger means more consistent), and colors indicate your predicted skill class (e.g., A, B). 
             A line connects your scores to show progress over time. 
             Why it matters: You can see how your consistency and skill level evolve, helping you set goals to climb to a higher class or stay steady in tough matches.''')
with c2:
    class_bubble(
        stage_n_preds,
        shooter=st.session_state.selected_shooter,
        show_ref=show_ref
    )
    
    
st.subheader("Stage Points and Time")
st.write('''
         This chart plots your individual stage results (points vs. time) for each match, with a diamond marking the average for each match. 
         Each dot is a stage, colored by match, so you can see how your speed and accuracy vary. 
         Why it matters: It highlights your strengths (e.g., fast and accurate stages) and weaknesses (e.g., slow or low-scoring stages), helping you focus your training on specific skills.
         ''')
c1, c2 = st.columns([3,1], vertical_alignment='top')
with c2:
    show_points = st.checkbox("Show stage points", value=True, key="dd_show_points", help="Show individual stage points on scatter chart")
    show_labels = st.checkbox("Show centroid labels", value=True, key="dd_labels", help="Show match name labels next to centroids on scatter chart")
    show_reg = st.checkbox("Show regression line", value=True, key="dd_show_reg", help="Show regression line between centroids on scatter chart")
    point_size = st.slider("Stage point size", 20, 200, 60, 10, key="dd_pt_size", help="Size of individual stage points on scatter chart")
    point_opacity = st.slider("Stage point opacity", 0.2, 1.0, 0.6, 0.1, key="dd_pt_opacity", help="Opacity of individual stage points on scatter chart")
    centroid_size = st.slider("Centroid size", 80, 400, 220, 20, key="dd_centroid_size", help="Size of match centroids on scatter chart")

with c1:
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

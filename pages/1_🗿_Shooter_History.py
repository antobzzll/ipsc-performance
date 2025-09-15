import streamlit as st
from lib.utils import get_page_title, init_global_select
from lib.data import get_data
from lib.charts import match_count

st.set_page_config(page_title="Home", layout="wide")
st.title(get_page_title())

stages = get_data('fitds_stages')

shooters = sorted(stages["shooter"].dropna().unique().tolist())
default_shooter = "BUZZELLI, ANTONIO" if "BUZZELLI, ANTONIO" in shooters else shooters[0]
if "selected_shooter" not in st.session_state:
    st.session_state.selected_shooter = default_shooter
    
c1, c2, c3, c4 = st.columns(4, vertical_alignment='bottom')
st.session_state.selected_shooter = c1.selectbox(
        "Shooter",
        shooters,
        index=shooters.index(st.session_state.selected_shooter),
        key="dd_shooter",
        help="Select shooter to analyze."
    )
sh_stages = stages[stages["shooter"] == st.session_state.selected_shooter].reset_index().copy()
c2.metric("Current Class", f"{sh_stages[(sh_stages['match_date'] == sh_stages['match_date'].max())]['class'].unique()[0]}")
c3.metric("Total Matches Disputed", f"{sh_stages['match_name'].nunique()}")
c4.metric("Total Stages Disputed", f"{sh_stages['stg'].count()}")
st.write("---")

st.subheader('Match Count')
c1, c2 = st.columns([3, 1], vertical_alignment='top')
x_choice_options = ['year', 'match_level', 'div', 'class',]
color_choice_options = ['match_level', 'div', 'class', 'none']
# Map choices to readable options
x_axis_options = {
    'year': 'Year',
    'match_level': 'Match Level',
    'div': 'Division',
    'class': 'Class'
}
color_options = {
    'match_level': 'Match Level',
    'div': 'Division',
    'class': 'Class',
    'none': 'None'
}
# Show mapped options in selectboxes

with c2:
    st.markdown("#### Chart options:")
    x_choice_label = st.selectbox('X-axis', list(x_axis_options.values()), index=0)
    x_choice = list(x_axis_options.keys())[list(x_axis_options.values()).index(x_choice_label)]
    color_choice_label = st.selectbox('Color', list(color_options.values()), index=0)
    color_choice = list(color_options.keys())[list(color_options.values()).index(color_choice_label)]
with c1:
    match_count(stages, st.session_state.selected_shooter, x_axis=x_choice, color=None if color_choice=='none' else color_choice)
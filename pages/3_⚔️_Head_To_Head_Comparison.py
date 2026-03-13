# import numpy as np
# import pandas as pd
# import streamlit as st
# from lib.data import get_data
# from lib.stats import aggregate_shooter_performance
# from lib.utils import get_page_title
# from lib.charts import stage_distr, stage_scatter, class_bubble

# st.set_page_config(page_title="Shooter Performance", layout="wide")

# # ========= DATA =========
# stages = get_data('fitds_stages')

# # ========= FILTERS =========
# st.sidebar.header('Data', help="Affect all charts on this page")
# normalization = st.sidebar.selectbox(
#     'Normalization',
#     ['Per Division', 'Per Class'],
#     index=0,
#     key='dd_normalization',
#     help="Normalize data per division or per class"
# )
# norm = 'div' if normalization == 'Per Division' else 'cls'
# norm_header = normalization.replace('Per ', '')

# st.sidebar.header("Filters", help="Affect all charts on this page")

# # Shooter
# shooters = sorted(stages["shooter"].dropna().unique().tolist())
# default_shooter = (
#     "BUZZELLI, ANTONIO" if "BUZZELLI, ANTONIO" in shooters else (shooters[0] if shooters else None)
# )
# if "selected_shooter" not in st.session_state:
#     st.session_state.selected_shooter = default_shooter
# st.title(f"{get_page_title()}")

# c1, c2, c3, c4 = st.columns(4, vertical_alignment='bottom')
# st.session_state.selected_shooter = c1.selectbox(
#         "Shooter",
#         shooters,
#         index=shooters.index(st.session_state.selected_shooter),
#         key="dd_shooter",
#         help="Select shooter to analyze"
#     )
# sh_stages = stages[(stages["shooter"] == st.session_state.selected_shooter)].reset_index().copy()
# geom_perf_mean = aggregate_shooter_performance(sh_stages, stage_pct_col=f'{norm}_factor_perc')
# c4.metric(f"AVG {norm_header} Performance", f"{geom_perf_mean['G']:.0%}")
# st.write("---")
import streamlit as st
import pandas as pd
import plotly.graph_objects as go
import plotly.colors as pc
import numpy as np

# def stage_distr(df: pd.DataFrame, norm='div', show_ref=True, lock_axes=True):
#     # Validate required columns
#     y = f"{norm}_factor_perc"
#     need = {"match_name", y, "match_date"}
#     if not need.issubset(df.columns):
#         st.info(f"Columns `match_name`, `match_date`, `{y}` required — skipping distribution chart.")
#         return

#     # Sort matches by date
#     match_order = (
#         df[["match_name", "match_date"]]
#         .drop_duplicates()
#         .sort_values("match_date")["match_name"]
#         .tolist()
#     )

#     # Create figure
#     fig = go.Figure()

#     # Add boxplots with centered outliers
#     for match in match_order:
#         match_df = df[df["match_name"] == match]

#         fig.add_trace(
#             go.Box(
#                 y=match_df[y],
#                 name=match,
#                 boxpoints="outliers",  # Show outliers
#                 jitter=0,              # Disable jitter to center outliers
#                 pointpos=0,            # Center outliers on the box (0 = middle of box)
#                 marker=dict(size=5, opacity=0.6),
#                 line=dict(width=1),
#                 hoverinfo="y+text",
#                 # Build hover text: include Stage + Predicted Class
#                 hovertemplate="Stage: %{customdata[0]}<br>"
#                             "Predicted Class: %{customdata[1]}<br>"
#                             "Result: %{y:.2%}<extra></extra>",
#                 text=None,  # we now use customdata instead of text
#                 customdata=np.stack([
#                     match_df["stg"] if "stg" in match_df.columns else [""] * len(match_df),
#                     match_df["pred_class"] if "pred_class" in match_df.columns else [""] * len(match_df),
#                 ], axis=-1),
#             )
#         )

#     # Add median line
#     med = (
#         df.groupby("match_name", as_index=False)[y]
#         .median()
#         .rename(columns={y: "median"})
#         .merge(df[["match_name", "match_date"]].drop_duplicates(), on="match_name")
#         .sort_values("match_date")
#     )
#     fig.add_trace(
#         go.Scatter(
#             x=med["match_name"],
#             y=med["median"],
#             mode="lines+markers",
#             name="Median",
#             line=dict(color="grey", dash="dashdot", width=2),
#             marker=dict(size=8),
#             hovertemplate="Match: %{x}<br>Median: %{y:.2%}<extra></extra>",
#         )
#     )

#     # Add reference line at 50% if requested
#     if show_ref:
#         fig.add_hline(
#             y=0.5,
#             line_dash="dash",
#             line_color="gray",
#             line_width=2,
#             annotation_text="50%",
#             annotation_position="top left",
#         )

#     # Update layout
#     y_label_prefix = "Division" if norm == "div" else "Class"
#     y_axis = dict(
#         title=f"{y_label_prefix} Stage Result",
#         tickformat=".0%",
#         range=[0, 1] if lock_axes else [df[y].min(), df[y].max()],
#     )
#     fig.update_layout(
#         xaxis=dict(
#             # title="Match",
#             categoryorder="array",
#             categoryarray=match_order,
#             tickangle=45,  # Rotate labels for better readability if needed
#         ),
#         yaxis=y_axis,
#         showlegend=False,  # Hide legend as in original
#         template="plotly_white",
#         hovermode="closest",
#         margin=dict(l=10, r=10, t=10, b=10)
#     )

#     # Display in Streamlit
#     st.plotly_chart(fig, use_container_width=True)
    
def stage_distr(df: pd.DataFrame, norm='div', show_ref=True, lock_axes=True):
    import numpy as np
    import pandas as pd
    import plotly.graph_objects as go
    import streamlit as st

    # Validate required columns
    y = f"{norm}_factor_perc"
    need = {"match_name", y, "match_date"}
    if not need.issubset(df.columns):
        st.info(f"Columns `match_name`, `match_date`, `{y}` required — skipping distribution chart.")
        return

    # Prefer stage-level predicted class for outliers
    stage_pred_col = None
    for candidate in ["pred_class_stage", "pred_class_per_stage", "pred_class"]:
        if candidate in df.columns:
            stage_pred_col = candidate
            break

    # Stage id column
    stage_col = "stg_n" if "stg_n" in df.columns else ("stg" if "stg" in df.columns else None)

    # Sort matches by date
    match_order = (
        df[["match_name", "match_date"]]
        .drop_duplicates()
        .sort_values("match_date")["match_name"]
        .tolist()
    )

    # Create figure
    fig = go.Figure()

    # --- 1) Distribution layer (boxplots) ---
    for match in match_order:
        match_df = df[df["match_name"] == match].copy()

        stage_vals = (
            match_df[stage_col].astype(str).to_numpy()
            if stage_col is not None
            else np.array([""] * len(match_df), dtype=object)
        )

        pred_vals = (
            match_df[stage_pred_col].fillna("").astype(str).to_numpy()
            if stage_pred_col is not None
            else np.array([""] * len(match_df), dtype=object)
        )

        customdata = np.column_stack([stage_vals, pred_vals])

        fig.add_trace(
            go.Box(
                y=match_df[y],
                name=match,
                boxpoints="outliers",
                jitter=0,
                pointpos=0,
                marker=dict(size=5, opacity=0.6),
                line=dict(width=1),
                hovertemplate=(
                    "Stage: %{customdata[0]}<br>"
                    "Predicted Class: %{customdata[1]}<br>"
                    "Result: %{y:.2%}<extra></extra>"
                ),
                customdata=customdata,
                showlegend=False,
            )
        )

    # --- 2) Median line across matches ---
    med = (
        df.groupby("match_name", as_index=False)[y]
        .median()
        .rename(columns={y: "median"})
        .merge(df[["match_name", "match_date"]].drop_duplicates(), on="match_name")
        .sort_values("match_date")
    )

    fig.add_trace(
        go.Scatter(
            x=med["match_name"],
            y=med["median"],
            mode="lines+markers",
            name="Median",
            line=dict(color="grey", dash="dashdot", width=2),
            marker=dict(size=8),
            hovertemplate="Match: %{x}<br>Median: %{y:.2%}<extra></extra>",
            showlegend=False,
        )
    )

    # --- 3) Bubble overlay: ONE bubble per match ---
    # Keep match-level mode if available; otherwise fallback to stage-level mode
    bubble_pred_col = None
    for candidate in ["pred_class", "pred_class_stage", "pred_class_per_stage"]:
        if candidate in df.columns:
            bubble_pred_col = candidate
            break

    if bubble_pred_col is not None:
        cls = (
            df[["match_name", bubble_pred_col]]
            .dropna(subset=[bubble_pred_col])
            .groupby("match_name")[bubble_pred_col]
            .agg(lambda s: s.mode().iat[0] if not s.mode().empty else s.iloc[0])
            .reset_index()
            .rename(columns={bubble_pred_col: "pred_class_mode"})
        )

        bubbles = med[["match_name", "median"]].merge(cls, on="match_name", how="left")

        fig.add_trace(
            go.Scatter(
                x=bubbles["match_name"],
                y=bubbles["median"],
                mode="markers+text",
                text=bubbles["pred_class_mode"].fillna(""),
                textposition="middle center",
                textfont=dict(size=22, color="#1f1f1f"),
                showlegend=False,
                hovertemplate=(
                    "Match: %{x}<br>"
                    "Predicted Class: %{text}<br>"
                    "Median: %{y:.2%}<extra></extra>"
                ),
            )
        )

    # --- 4) Reference line at 50% ---
    if show_ref:
        fig.add_hline(
            y=0.5,
            line_dash="dash",
            line_color="gray",
            line_width=2,
            annotation_text="50%",
            annotation_position="top left",
        )

    # --- 5) Layout ---
    y_label_prefix = "Division" if norm == "div" else "Class"
    y_min = pd.to_numeric(df[y], errors="coerce").min()
    y_max = pd.to_numeric(df[y], errors="coerce").max()

    y_axis = dict(
        title=f"{y_label_prefix} Stage Result",
        tickformat=".0%",
        range=[0, 1] if lock_axes else [y_min, y_max],
    )

    fig.update_layout(
        xaxis=dict(
            categoryorder="array",
            categoryarray=match_order,
            tickangle=45,
        ),
        yaxis=y_axis,
        showlegend=False,
        template="plotly_white",
        hovermode="closest",
        margin=dict(l=10, r=10, t=10, b=10),
    )

    st.plotly_chart(fig, use_container_width=True)  

def stage_scatter(
    df: pd.DataFrame,
    norm: str = "div",
    point_size: int = 60,
    point_opacity: float = 0.6,
    centroid_size: int = 220,
    show_labels: bool = True,
    show_ref: bool = True,
    lock_axes: bool = True,
    show_points: bool = True,
    show_regression: bool = False,
    true_pts: bool = False
):

    def first_nonempty(series: pd.Series) -> str:
        s = series.dropna().astype(str).str.strip()
        s = s[s != ""]
        return s.iloc[0] if not s.empty else ""

    def mode_nonempty(series: pd.Series) -> str:
        s = series.dropna().astype(str).str.strip()
        s = s[s != ""]
        if s.empty:
            return ""
        m = s.mode()
        return m.iloc[0] if not m.empty else s.iloc[0]


    if true_pts:
        y_label_prefix = ""
        x_label_prefix = "Division" if norm == "div" else "Class"
        y = 'pts_pct'
    else:
        y_label_prefix = "Division" if norm == "div" else "Class"
        x_label_prefix = "Division" if norm == "div" else "Class"
        y = f"{norm}_pts_perc"
    
    x = f"{norm}_time_perc"
    f = f"{norm}_factor_perc"

    required = ["match_date", y, x, "match_name", "stg_n", f]
    missing = [c for c in required if c not in df.columns]
    if missing:
        st.info(f"Missing required columns for scatter: {missing}")
        return

    # Optional prediction columns:
    # - stage-level prediction: pred_class
    # - match-level prediction: try common names first
    match_pred_candidates = [
        "match_pred_class",
        "pred_class_match",
        "overall_pred_class",
        "pred_class_overall",
        "match_class_pred",
        "match_predicted_class",
    ]
    match_pred_col = next((c for c in match_pred_candidates if c in df.columns), None)
    has_stage_pred = "pred_class" in df.columns

    keep_cols = required.copy()
    if has_stage_pred:
        keep_cols.append("pred_class")
    if match_pred_col is not None:
        keep_cols.append(match_pred_col)

    # Prepare working dataframe
    sdf = df[keep_cols].copy()

    # Numeric conversion
    for col in [x, y, f, "stg_n"]:
        if col in sdf.columns:
            sdf[col] = pd.to_numeric(sdf[col], errors="coerce")

    if "match_date" in sdf.columns:
        sdf["match_date"] = pd.to_datetime(sdf["match_date"], errors="coerce")

    # Remove invalid/problematic stages from the chart entirely
    sdf = sdf.replace([np.inf, -np.inf], np.nan)
    sdf = sdf.dropna(subset=[x, y, f])
    sdf = sdf[(sdf[x] != 0) & (sdf[y] != 0) & (sdf[f] != 0)]

    if sdf.empty:
        st.info("No valid stage data to plot.")
        return

    sdf = sdf.sort_values("match_date")
    sdf = sdf.rename(columns={x: "Time (%)", y: "Points (%)", f: "Result (%)"})

    # Detect scale
    tmax = sdf["Time (%)"].max(skipna=True)
    pmax = sdf["Points (%)"].max(skipna=True)
    is_fraction = max(tmax, pmax) <= 1.5 if pd.notna(tmax) and pd.notna(pmax) else True
    ref_val = 0.5 if is_fraction else 50
    dom = [0, 1] if is_fraction else [0, 100]
    format_str = ".1%" if is_fraction else ".1f"

    # Match order
    match_order = (
        sdf[["match_name", "match_date"]]
        .drop_duplicates()
        .sort_values("match_date")["match_name"]
        .tolist()
    )

    # Compute centroids on cleaned data only
    cent = (
        sdf.groupby("match_name", as_index=False)
        .agg(
            mean_time=("Time (%)", "mean"),
            mean_points=("Points (%)", "mean"),
        )
        .rename(columns={"mean_time": "Time (%)", "mean_points": "Points (%)"})
    )

    # Add one class prediction per match for centroid hover
    if match_pred_col is not None:
        match_pred_df = (
            sdf.groupby("match_name", as_index=False)[match_pred_col]
            .agg(first_nonempty)
            .rename(columns={match_pred_col: "match_pred_class"})
        )
    elif has_stage_pred:
        match_pred_df = (
            sdf.groupby("match_name", as_index=False)["pred_class"]
            .agg(mode_nonempty)
            .rename(columns={"pred_class": "match_pred_class"})
        )
    else:
        match_pred_df = cent[["match_name"]].copy()
        match_pred_df["match_pred_class"] = ""

    cent = cent.merge(match_pred_df, on="match_name", how="left")
    cent["match_pred_class"] = cent["match_pred_class"].fillna("").astype(str)
    cent["label"] = cent["match_name"]

    if cent.empty:
        st.info("No valid centroid data to plot.")
        return

    # Determine axis ranges
    if show_points:
        axis_df = sdf[["Time (%)", "Points (%)"]].copy()
    else:
        axis_df = cent[["Time (%)", "Points (%)"]].copy()

    axis_df = axis_df.replace([np.inf, -np.inf], np.nan).dropna(subset=["Time (%)", "Points (%)"])

    if axis_df.empty:
        st.info("No valid data to plot.")
        return

    x_min, x_max = axis_df["Time (%)"].min(), axis_df["Time (%)"].max()
    y_min, y_max = axis_df["Points (%)"].min(), axis_df["Points (%)"].max()
    x_range = dom if lock_axes else [x_min, x_max]
    y_range = dom if lock_axes else [y_min, y_max]

    # Color map
    colors = pc.qualitative.Plotly
    color_map = {match: colors[i % len(colors)] for i, match in enumerate(match_order)}

    fig = go.Figure()

    # Stage points
    if show_points:
        for match in match_order:
            match_df = sdf[sdf["match_name"] == match].copy()
            if match_df.empty:
                continue

            point_plotly_size = int((point_size / np.pi) ** 0.5 * 2)

            if has_stage_pred:
                pred_vals = match_df["pred_class"].fillna("").astype(str).to_numpy()
            else:
                pred_vals = np.array([""] * len(match_df), dtype=object)

            result_vals = pd.to_numeric(match_df["Result (%)"], errors="coerce").to_numpy()
            customdata = np.column_stack([result_vals, pred_vals])

            fig.add_trace(
                go.Scatter(
                    x=match_df["Time (%)"],
                    y=match_df["Points (%)"],
                    mode="markers",
                    marker=dict(
                        size=point_plotly_size,
                        opacity=point_opacity,
                        color=color_map[match],
                    ),
                    name=match,
                    text=match_df["stg_n"].astype("Int64").astype(str),
                    customdata=customdata,
                    hovertemplate=(
                        "Match: %{fullData.name}<br>"
                        "Stage: %{text}<br>"
                        "Predicted Class: %{customdata[1]}<br>"
                        f"Result: %{{customdata[0]:{format_str}}}<br>"
                        f"Time: %{{x:{format_str}}}<br>"
                        f"Points: %{{y:{format_str}}}<extra></extra>"
                    ),
                    showlegend=False,
                )
            )

    # Centroids
    for match in match_order:
        match_cent = cent[cent["match_name"] == match].copy()
        if match_cent.empty:
            continue

        centroid_plotly_size = int((centroid_size / np.pi) ** 0.5 * 2)
        centroid_customdata = np.column_stack(
            [match_cent["match_pred_class"].replace("", "—").to_numpy()]
        )

        fig.add_trace(
            go.Scatter(
                x=match_cent["Time (%)"],
                y=match_cent["Points (%)"],
                mode="markers",
                marker=dict(
                    symbol="diamond",
                    size=centroid_plotly_size,
                    color=color_map[match],
                    line=dict(color="black", width=1),
                ),
                name=match,
                customdata=centroid_customdata,
                hovertemplate=(
                    "Match: %{fullData.name}<br>"
                    "Predicted Class: %{customdata[0]}<br>"
                    f"Time: %{{x:{format_str}}}<br>"
                    f"Points: %{{y:{format_str}}}<extra></extra>"
                ),
                showlegend=False,
            )
        )

    # Labels
    if show_labels:
        for match in match_order:
            match_cent = cent[cent["match_name"] == match].copy()
            if match_cent.empty:
                continue

            fig.add_trace(
                go.Scatter(
                    x=match_cent["Time (%)"],
                    y=match_cent["Points (%)"],
                    mode="text",
                    text=match_cent["label"],
                    textposition="top right",
                    textfont=dict(size=11, color=color_map[match]),
                    showlegend=False,
                    hoverinfo="skip",
                )
            )

    # Regression line through centroids
    if show_regression and len(cent) >= 2:
        reg_df = cent[["Time (%)", "Points (%)"]].copy()
        reg_df = reg_df.replace([np.inf, -np.inf], np.nan).dropna()
        reg_df = reg_df[(reg_df["Time (%)"] != 0) & (reg_df["Points (%)"] != 0)]

        cx = reg_df["Time (%)"].to_numpy(dtype=float)
        cy = reg_df["Points (%)"].to_numpy(dtype=float)

        if len(cx) >= 2 and len(np.unique(cx)) >= 2:
            try:
                a, b = np.polyfit(cx, cy, 1)

                y_hat = a * cx + b
                ss_res = np.sum((cy - y_hat) ** 2)
                ss_tot = np.sum((cy - np.mean(cy)) ** 2)
                r2 = 1 - ss_res / ss_tot if ss_tot > 0 else np.nan

                x0, x1 = float(np.min(cx)), float(np.max(cx))
                line_x = np.linspace(x0, x1, 50)
                line_y = a * line_x + b

                fig.add_trace(
                    go.Scatter(
                        x=line_x,
                        y=line_y,
                        mode="lines",
                        line=dict(color="grey", width=2, dash="dashdot"),
                        name="Centroid trend",
                        hovertemplate=(
                            f"Trend: y = {a:.3f}·x + {b:.3f}<br>"
                            f"R² = {r2:.3f}<extra></extra>"
                        ),
                        showlegend=False,
                    )
                )

                fig.add_annotation(
                    xref="paper",
                    yref="paper",
                    x=0.01,
                    y=0.99,
                    xanchor="left",
                    yanchor="top",
                    text=f"Trend (centroids): y = {a:.3f}·x + {b:.3f} &nbsp; | &nbsp; R² = {r2:.3f}",
                    showarrow=False,
                    font=dict(size=12, color="#333"),
                    bgcolor="rgba(255,255,255,0.6)",
                    bordercolor="#999",
                    borderwidth=1,
                    borderpad=4,
                )
            except (np.linalg.LinAlgError, ValueError, TypeError):
                pass

    # Reference lines
    if show_ref:
        fig.add_vline(x=ref_val, line_dash="dash", line_color="gray", line_width=2)
        fig.add_hline(y=ref_val, line_dash="dash", line_color="gray", line_width=2)

    # Layout
    x_axis = dict(
        title=f"{x_label_prefix} Time",
        tickformat=".0%" if is_fraction else ".0f",
        range=x_range,
        showgrid=True,
        gridcolor="lightgray",
        dtick=0.125 if is_fraction else 12.5,
    )
    y_axis = dict(
        title=f"{y_label_prefix} Points",
        tickformat=".0%" if is_fraction else ".0f",
        range=y_range,
        showgrid=True,
        gridcolor="lightgray",
        dtick=0.125 if is_fraction else 12.5,
    )

    fig.update_layout(
        xaxis=x_axis,
        yaxis=y_axis,
        showlegend=False,
        template="plotly_white",
        hovermode="closest",
        margin=dict(l=10, r=10, t=10, b=10),
    )

    st.plotly_chart(fig, use_container_width=True)
    

def match_count(
    df: pd.DataFrame,
    shooter: str,
    *,
    x_axis: str = "year",
    color: str | None = "match_level",
):
    
    # Columns required for the most flexible combinations; we validate again later
    needed = {
        "shooter_name", "match_name", "match_date", "match_level", "shooter_div", "shooter_class",
    }
    miss = needed - set(df.columns)
    if miss:
        st.info(f"Match count needs columns: {sorted(miss)}")
        return

    # Filter and prepare data
    d = df.loc[df["shooter_name"] == shooter].copy()
    if d.empty:
        st.info("No data for the specified shooter.")
        return

    d["match_date"] = pd.to_datetime(d["match_date"], errors="coerce")

    # Sort match_name by match_date
    match_order = (
        d[["match_name", "match_date"]]
        .drop_duplicates()
        .sort_values("match_date")["match_name"]
        .tolist()
    )
    d = d.sort_values("match_date")

    # Normalize options
    x_axis = (x_axis or "year").lower()
    color = (color or "match_level").lower() if color is not None else None
    # If color equals x, disable color stacking
    if color == x_axis:
        color = None

    # Work at match-level (1 row per match for this shooter)
    matches = (
        d[["match_name", "match_date", "match_level", "shooter_div", "shooter_class"]]
        .drop_duplicates(subset=["match_name"])  # avoid counting stages
        .copy()
    )
    # Prepare time and labels
    matches["match_date"] = pd.to_datetime(matches["match_date"], errors="coerce")
    matches["year"] = matches["match_date"].dt.year.astype("Int64")
    # Robust level label (e.g., L1, L2); fall back to 'Unknown'
    lvl_num = pd.to_numeric(matches["match_level"], errors="coerce").astype("Int64")
    matches["lvl_label"] = np.where(
        lvl_num.notna(),
        "L" + lvl_num.astype(str),
        "Unknown",
    )
    # Decide x field, labels, and title
    def _labels_with_unknown(values: list[str], has_unknown: bool) -> list[str]:
        out = list(values)
        if has_unknown and "Unknown" not in out:
            out.append("Unknown")
        return out

    x_field = None
    x_title = ""
    x_labels: list[str] = []

    if x_axis == "year":
        x_field = "year"
        # Build labels and order
        years = matches["year"].dropna().astype(int).sort_values().unique().tolist()
        x_labels = _labels_with_unknown([str(y) for y in years], matches["year"].isna().any())
        x_title = "Year"
        # Use a display label column for consistent string x-values
        matches["x_label"] = matches["year"].apply(lambda v: str(int(v)) if pd.notna(v) else "Unknown")
    elif x_axis in {"match_level", "level"}:
        x_field = "lvl_label"
        # Order by numeric level when possible (L1 < L2 < ...), Unknown last
        lvl_order = (
            matches.dropna(subset=["lvl_label"]).assign(_n=matches["lvl_label"].str.extract(r"(\d+)").astype(float))
        )
        # compute ordered unique labels by numeric token if present
        known_lvls = (
            matches.loc[matches["lvl_label"].notna(), "lvl_label"].unique().tolist()
        )
        # try to sort by numeric part; fallback alphabetical
        def _lvl_key(s: str):
            import re
            m = re.search(r"(\d+)", str(s))
            return (float(m.group(1)) if m else float("inf"), str(s))
        known_lvls_sorted = sorted(known_lvls, key=_lvl_key)
        x_labels = _labels_with_unknown(known_lvls_sorted, matches["lvl_label"].isna().any())
        x_title = "Match Level"
        matches["x_label"] = matches["lvl_label"].fillna("Unknown")
    elif x_axis in {"div", "division"}:
        x_field = "div"
        cats = sorted(matches["shooter_div"].dropna().astype(str).unique().tolist())
        x_labels = _labels_with_unknown(cats, matches["shooter_div"].isna().any())
        x_title = "Division"
        matches["x_label"] = matches["shooter_div"].fillna("Unknown").astype(str)
    elif x_axis in {"class"}:
        x_field = "class"
        cats = sorted(matches["shooter_class"].dropna().astype(str).unique().tolist())
        x_labels = _labels_with_unknown(cats, matches["shooter_class"].isna().any())
        x_title = "Class"
        matches["x_label"] = matches["shooter_class"].fillna("Unknown").astype(str)
    else:
        st.info(f"Unknown x_axis '{x_axis}'. Use 'year', 'match_level', 'div', or 'class'.")
        return

    # Decide color field and legend title
    color_field = None
    legend_title = None
    if color in {None, "none"}:
        color_field = None
    elif color in {"match_level", "level"}:
        color_field = "lvl_label"
        legend_title = "Match Level"
    elif color in {"div", "division"}:
        color_field = "shooter_div"
        legend_title = "Division"
    elif color in {"class"}:
        color_field = "shooter_class"
        legend_title = "Class"
    else:
        st.info(f"Unknown color '{color}'. Use 'match_level', 'div', 'class', or None.")
        return

    # Aggregate counts
    group_cols = ["x_label"] + ([color_field] if color_field else [])
    counts = matches.groupby(group_cols, dropna=False).size().reset_index(name="n")
    if counts.empty:
        st.info("No matches found to plot.")
        return

    # Build color categories and mapping
    fig = go.Figure()
    palette = pc.qualitative.Plotly
    if color_field:
        series_vals = counts[color_field].fillna("Unknown").astype(str).unique().tolist()
        series_vals_sorted = sorted([s for s in series_vals if s != "Unknown"]) + (["Unknown"] if "Unknown" in series_vals else [])
        color_map = {val: palette[i % len(palette)] for i, val in enumerate(series_vals_sorted)}
        for val in series_vals_sorted:
            sub = counts[counts[color_field].fillna("Unknown").astype(str) == val]
            fig.add_trace(
                go.Bar(
                    x=sub["x_label"],
                    y=sub["n"],
                    name=str(val),
                    marker_color=color_map[val],
                    hovertemplate=f"{x_title}: %{{x}}<br>{val}: %{{y}}<extra></extra>",
                )
            )
    else:
        # Single series
        fig.add_trace(
            go.Bar(
                x=counts["x_label"],
                y=counts["n"],
                name="Matches",
                marker_color=palette[0],
                hovertemplate=f"{x_title}: %{{x}}<br>Matches: %{{y}}<extra></extra>",
            )
        )

    # Layout
    fig.update_layout(
        barmode="stack" if color_field else "group",
        xaxis=dict(
            title=x_title,
            categoryorder="array",
            categoryarray=x_labels,
            type="category",
        ),
        yaxis=dict(title="Matches"),
        legend=dict(title=legend_title) if color_field else None,
        template="plotly_white",
        margin=dict(l=10, r=10, t=10, b=10)
    )

    st.plotly_chart(fig, use_container_width=True)


def shooter_match_history(
    df: pd.DataFrame,
    shooter_name: str,
    shooter_div: str,
    metric: str = "pct",   # "pct" or "rank"
    show_ref: bool = True,
    lock_y: bool = False,
):
    import numpy as np
    import pandas as pd
    import plotly.graph_objects as go
    import streamlit as st

    from lib.stats import match_standing

    needed = {
        "match_name",
        "match_date",
        "shooter_name",
        "shooter_div",
        "shooter_class",
        "stg_match_pts",
    }
    missing = [c for c in needed if c not in df.columns]
    if missing:
        st.info(f"Missing required columns for match history chart: {missing}")
        return

    work = df.copy()
    work["match_date"] = pd.to_datetime(work["match_date"], errors="coerce")
    work["stg_match_pts"] = pd.to_numeric(work["stg_match_pts"], errors="coerce")

    work = work[work["shooter_div"].astype(str) == str(shooter_div)].copy()
    if work.empty:
        st.info("No data for selected division.")
        return

    matches = (
        work[["match_name", "match_date"]]
        .drop_duplicates()
        .sort_values("match_date")
        .reset_index(drop=True)
    )

    rows = []
    for _, r in matches.iterrows():
        match_name = r["match_name"]
        match_date = r["match_date"]

        standing = match_standing(
            work,
            match=match_name,
            shooter_div=shooter_div,
        )

        if standing.empty:
            continue

        shooter_row = standing[standing["shooter_name"] == shooter_name].copy()
        if shooter_row.empty:
            continue

        shooter_row = shooter_row.iloc[0]

        rows.append(
            {
                "match_name": match_name,
                "match_date": match_date,
                "championship": shooter_row["championship"],
                "shooter_name": shooter_name,
                "shooter_div": shooter_div,
                "shooter_class": shooter_row["shooter_class"],
                "stg_match_pts": pd.to_numeric(shooter_row["stg_match_pts"], errors="coerce"),
                "div_rank": pd.to_numeric(shooter_row["rank"], errors="coerce"),
                "div_pct": pd.to_numeric(shooter_row["pct"], errors="coerce"),
                "class_rank": pd.to_numeric(shooter_row["class_rank"], errors="coerce"),
                "class_pct": pd.to_numeric(shooter_row["class_pct"], errors="coerce"),
                "pct_abcd": pd.to_numeric(shooter_row["pct_abcd"], errors="coerce"),
                "pct_abcd_minus_16": pd.to_numeric(shooter_row["pct_abcd_minus_16"], errors="coerce"),
            }
        )

    if not rows:
        st.info("No match history data available for selected shooter.")
        return

    plot_df = pd.DataFrame(rows).sort_values("match_date")
    # plot_df["match_label"] = plot_df.apply(
    #     lambda r: f"{r['match_name']} ({r['match_date'].date()})"
    #     if pd.notna(r["match_date"]) else str(r["match_name"]),
    #     axis=1,
    # )
    plot_df["match_label"] = plot_df["match_name"]  # simpler label; date is in hover

    if metric not in {"pct", "rank"}:
        st.info("metric must be 'pct' or 'rank'.")
        return

    # Colors
    div_color = "rgba(31, 119, 180, 1.0)"
    faded_1 = "rgba(255, 127, 14, 0.45)"
    faded_2 = "rgba(44, 160, 44, 0.45)"
    faded_3 = "rgba(214, 39, 40, 0.45)"

    if metric == "pct":
        y_div = "div_pct"
        y_cls = "class_pct"
        y_non_m_gm = "pct_abcd"
        y_non_m_gm_m16 = "pct_abcd_minus_16"

        ymax = plot_df[[y_div, y_cls, y_non_m_gm, y_non_m_gm_m16]].max(skipna=True).max()
        yaxis = dict(
            title="Percentage",
            tickformat=".0%",
            range=[0, max(1, float(ymax))] if lock_y and pd.notna(ymax) else None,
        )

        text_div = None
        text_cls = None
        text_non_m_gm = None
        text_non_m_gm_m16 = None
        textpos_div = None
        textpos_cls = None
        textpos_non_m_gm = None
        textpos_non_m_gm_m16 = None

        hover_div = "Division %: %{y:.1%}<extra></extra>"
        hover_cls = "Class %: %{y:.1%}<extra></extra>"
        hover_non_m_gm = "Pct vs first non-M/GM: %{y:.1%}<extra></extra>"
        hover_non_m_gm_m16 = "Pct vs first non-M/GM - 16 pts: %{y:.1%}<extra></extra>"
    else:
        y_div = "div_rank"
        y_cls = "class_rank"

        max_rank = pd.concat([plot_df[y_div], plot_df[y_cls]], axis=0).max(skipna=True)
        yaxis = dict(
            title="Standing",
            autorange="reversed",
            range=[float(max_rank) + 0.5, 0.5] if lock_y and pd.notna(max_rank) else None,
            dtick=1,
            showticklabels=False,
            ticks="",
            showgrid=False,
            zeroline=False,
        )

        text_div = plot_df[y_div].astype("Int64").astype(str)
        text_cls = plot_df[y_cls].astype("Int64").astype(str)
        textpos_div = "top center"
        textpos_cls = "bottom center"

        hover_div = "Division rank: %{y}<extra></extra>"
        hover_cls = "Class rank: %{y}<extra></extra>"

    fig = go.Figure()

    # Division = solid/plain and fully opaque
    fig.add_trace(
        go.Scatter(
            x=plot_df["match_label"],
            y=plot_df[y_div],
            mode="lines+markers+text",
            name="Division",
            text=text_div,
            textposition=textpos_div,
            line=dict(dash="solid", color=div_color, width=2.5),
            marker=dict(color=div_color),
            textfont=dict(color=div_color),
            customdata=np.column_stack([
                plot_df["shooter_class"].fillna("").astype(str),
                plot_df["stg_match_pts"].astype(float),
            ]),
            hovertemplate=(
                "Match: %{x}<br>"
                "Class: %{customdata[0]}<br>"
                "Match pts: %{customdata[1]:.2f}<br>"
                + hover_div
            ),
        )
    )

    # Class = faded
    fig.add_trace(
        go.Scatter(
            x=plot_df["match_label"],
            y=plot_df[y_cls],
            mode="lines+markers+text",
            name="Class",
            text=text_cls,
            textposition=textpos_cls,
            line=dict(dash="dash", color=faded_1, width=2),
            marker=dict(color=faded_1),
            textfont=dict(color=faded_1),
            customdata=np.column_stack([
                plot_df["shooter_class"].fillna("").astype(str),
                plot_df["stg_match_pts"].astype(float),
            ]),
            hovertemplate=(
                "Match: %{x}<br>"
                "Class: %{customdata[0]}<br>"
                "Match pts: %{customdata[1]:.2f}<br>"
                + hover_cls
            ),
        )
    )

    if metric == "pct":
        fig.add_trace(
            go.Scatter(
                x=plot_df["match_label"],
                y=plot_df[y_non_m_gm],
                mode="lines+markers+text",
                name="ABCD Recalculated",
                text=text_non_m_gm,
                textposition=textpos_non_m_gm,
                line=dict(dash="dot", color=faded_2, width=2),
                marker=dict(color=faded_2),
                textfont=dict(color=faded_2),
                customdata=np.column_stack([
                    plot_df["shooter_class"].fillna("").astype(str),
                    plot_df["stg_match_pts"].astype(float),
                ]),
                hovertemplate=(
                    "Match: %{x}<br>"
                    "Class: %{customdata[0]}<br>"
                    "Match pts: %{customdata[1]:.2f}<br>"
                    + hover_non_m_gm
                ),
            )
        )

        fig.add_trace(
            go.Scatter(
                x=plot_df["match_label"],
                y=plot_df[y_non_m_gm_m16],
                mode="lines+markers+text",
                name="ABCD Recalculated - 16 pts",
                text=text_non_m_gm_m16,
                textposition=textpos_non_m_gm_m16,
                line=dict(dash="dashdot", color=faded_3, width=2),
                marker=dict(color=faded_3),
                textfont=dict(color=faded_3),
                customdata=np.column_stack([
                    plot_df["shooter_class"].fillna("").astype(str),
                    plot_df["stg_match_pts"].astype(float),
                ]),
                hovertemplate=(
                    "Match: %{x}<br>"
                    "Class: %{customdata[0]}<br>"
                    "Match pts: %{customdata[1]:.2f}<br>"
                    + hover_non_m_gm_m16
                ),
            )
        )
    if show_ref:
        fig.add_hline(
            y=0.5,
            line_dash="dash",
            line_color="gray",
            line_width=2,
            annotation_text="50%",
            annotation_position="top left",
        )
        
    fig.update_layout(
        template="plotly_white",
        margin=dict(l=10, r=10, t=10, b=10),
        hovermode="closest",
        xaxis=dict(
            title="Match",
            tickangle=45,
            categoryorder="array",
            categoryarray=plot_df["match_label"].tolist(),
        ),
        yaxis=yaxis,
        legend=dict(title="Scope"),
        showlegend=True,
    )

    st.plotly_chart(fig, use_container_width=True)
    # print(plot_df[["match_name", "championship", y_div, y_cls, y_non_m_gm, y_non_m_gm_m16]])


def compare_shooters_line(
    df: pd.DataFrame,
    shooter_1: str,
    shooter_2: str,
    metric_col: str,
    y_title: str,
    *,
    show_ref: bool = True,
    show_markers: bool = True,
    ref_val: float = 0.5,
):
    import numpy as np
    import pandas as pd
    import plotly.graph_objects as go
    import streamlit as st

    needed = {"shooter_name", "match_name", metric_col}
    missing = [c for c in needed if c not in df.columns]
    if missing:
        st.info(f"Missing required columns for comparison chart: {missing}")
        return

    sdf = df[df["shooter_name"].isin([shooter_1, shooter_2])].copy()

    if sdf.empty:
        st.info("No data to plot.")
        return

    group_cols = ["shooter_name", "match_name"]
    if "match_date" in sdf.columns:
        group_cols.append("match_date")

    plot_df = (
        sdf.groupby(group_cols, dropna=False)[metric_col]
        .mean()
        .reset_index()
        .rename(columns={metric_col: "metric_value"})
    )

    if plot_df.empty:
        st.info("No data to plot.")
        return

    if "match_date" in plot_df.columns:
        plot_df["match_date"] = pd.to_datetime(plot_df["match_date"], errors="coerce")
        plot_df = plot_df.sort_values(["match_date", "match_name", "shooter_name"])
        plot_df["match_label"] = plot_df.apply(
            lambda r: f"{r['match_name']} ({r['match_date'].date()})"
            if pd.notna(r["match_date"]) else str(r["match_name"]),
            axis=1,
        )
        match_order = (
            plot_df[["match_name", "match_date", "match_label"]]
            .drop_duplicates()
            .sort_values(["match_date", "match_name"])["match_label"]
            .tolist()
        )
    else:
        plot_df = plot_df.sort_values(["match_name", "shooter_name"])
        plot_df["match_label"] = plot_df["match_name"].astype(str)
        match_order = (
            plot_df[["match_name", "match_label"]]
            .drop_duplicates()
            .sort_values(["match_name"])["match_label"]
            .tolist()
        )

    fig = go.Figure()

    for shooter in [shooter_1, shooter_2]:
        shooter_df = plot_df[plot_df["shooter_name"] == shooter].copy()
        if shooter_df.empty:
            continue

        if "match_date" in shooter_df.columns:
            date_vals = shooter_df["match_date"].dt.strftime("%Y-%m-%d").fillna("").to_numpy()
        else:
            date_vals = np.array([""] * len(shooter_df), dtype=object)

        customdata = np.column_stack([
            shooter_df["match_name"].astype(str).to_numpy(),
            date_vals,
        ])

        fig.add_trace(
            go.Scatter(
                x=shooter_df["match_label"],
                y=shooter_df["metric_value"],
                mode="lines+markers" if show_markers else "lines",
                name=shooter,
                customdata=customdata,
                hovertemplate=(
                    "Shooter: %{fullData.name}<br>"
                    "Match: %{customdata[0]}<br>"
                    "Date: %{customdata[1]}<br>"
                    "Value: %{y:.1%}<extra></extra>"
                ),
            )
        )

    if len(fig.data) == 0:
        st.info("No data to plot.")
        return

    if show_ref:
        fig.add_hline(y=ref_val, line_dash="dash", line_color="gray", line_width=2)

    fig.update_layout(
        template="plotly_white",
        hovermode="x unified",
        margin=dict(l=10, r=10, t=10, b=10),
        legend=dict(title=""),
        xaxis=dict(
            title="Match",
            categoryorder="array",
            categoryarray=match_order,
        ),
        yaxis=dict(
            title=y_title,
            tickformat=".0%",
            showgrid=True,
            gridcolor="lightgray",
        ),
    )

    st.plotly_chart(fig, use_container_width=True)

def stage_comparison_chart(
    df: pd.DataFrame,
    shooters: list[str],
    metric_col: str = "hf",
    metric_label: str = "Hit Factor",
    *,
    stage_col: str = "stg_n",
    shooter_col: str = "shooter_name",
    pts_col: str = "stg_match_pts",
    time_col: str = "time",
    hf_col: str = "hf",
    stage_rank_col: str = "div_factor_standing",
    show_stage_average: bool = True,
    stage_average_name: str = "Stage Avg",
    empty_message: str = "No data for the selected filters.",
    select_message: str = "Select at least one shooter to display the comparison chart.",
):
    """
    Plot a stage-by-stage comparison for up to three shooters.
    Optionally overlays the stage average of the selected metric.
    """
    if not shooters:
        st.info(select_message)
        return

    required_cols = {shooter_col, stage_col, metric_col}
    missing_cols = [c for c in required_cols if c not in df.columns]
    if missing_cols:
        st.info(f"Missing columns for comparison chart: {missing_cols}")
        return

    # full stage data for stage-average calculation
    full_df = df.copy()
    for col in [stage_col, metric_col, pts_col, time_col, hf_col, stage_rank_col]:
        if col in full_df.columns:
            full_df[col] = pd.to_numeric(full_df[col], errors="coerce")
    full_df = full_df.dropna(subset=[stage_col, metric_col])

    # selected shooters only
    chart_df = full_df[full_df[shooter_col].astype(str).isin([str(s) for s in shooters])].copy()
    if chart_df.empty:
        st.info(empty_message)
        return

    agg_map = {
        "metric_value": (metric_col, "mean"),
        "pts": (pts_col, "mean") if pts_col in chart_df.columns else (metric_col, "mean"),
        "time": (time_col, "mean") if time_col in chart_df.columns else (metric_col, "mean"),
        "hf": (hf_col, "mean") if hf_col in chart_df.columns else (metric_col, "mean"),
        "stage_rank": (stage_rank_col, "min") if stage_rank_col in chart_df.columns else (metric_col, "mean"),
    }

    plot_df = (
        chart_df.groupby([shooter_col, stage_col], as_index=False)
        .agg(**agg_map)
        .sort_values([stage_col, shooter_col])
    )

    stage_avg_df = pd.DataFrame()
    if show_stage_average:
        stage_avg_df = (
            full_df.groupby(stage_col, as_index=False)[metric_col]
            .mean()
            .rename(columns={metric_col: "stage_avg"})
            .sort_values(stage_col)
        )

    fig = go.Figure()

    if show_stage_average and not stage_avg_df.empty:
        fig.add_trace(
            go.Scatter(
                x=stage_avg_df[stage_col],
                y=stage_avg_df["stage_avg"],
                mode="lines+markers",
                name=stage_average_name,
                line=dict(color="lightgrey", dash="dash", width=2),
                hovertemplate=(
                    # "Stage: %{x}<br>"
                    f"{stage_average_name}: %{{y:.4f}}<extra></extra><br>"
                ),
            )
        )

    for shooter in shooters:
        shooter_df = plot_df[plot_df[shooter_col].astype(str) == str(shooter)].copy()
        if shooter_df.empty:
            continue

        customdata = np.column_stack(
            [
                shooter_df["pts"].to_numpy(dtype=float),
                shooter_df["time"].to_numpy(dtype=float),
                shooter_df["hf"].to_numpy(dtype=float),
                shooter_df["stage_rank"].to_numpy(dtype=float),
            ]
        )

        fig.add_trace(
            go.Scatter(
                x=shooter_df[stage_col],
                y=shooter_df["metric_value"],
                mode="lines+markers",
                name=str(shooter),
                customdata=customdata,
                hovertemplate=(
                    # "Shooter: %{fullData.name}<br>"
                    "<br>Standing: %{customdata[3]:.0f}<br>"
                    "Pts: %{customdata[0]:.2f}<br>"
                    "Time: %{customdata[1]:.2f}<br>"
                    "HF: %{customdata[2]:.4f}<br>"
                ),
            )
        )

    fig.update_layout(
        template="plotly_white",
        margin=dict(l=10, r=10, t=10, b=10),
        hovermode="x unified",
        xaxis=dict(title="Stage", dtick=1),
        yaxis=dict(title=metric_label),
        legend=dict(title=""),
    )

    st.plotly_chart(fig, use_container_width=True)
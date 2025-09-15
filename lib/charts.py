import streamlit as st
import pandas as pd
import plotly.graph_objects as go
import plotly.colors as pc
import numpy as np

def stage_distr(df: pd.DataFrame, norm='div', show_ref=True, lock_axes=True):
    # Validate required columns
    y = f"{norm}_factor_perc"
    need = {"match_name", y, "match_date"}
    if not need.issubset(df.columns):
        st.info(f"Columns `match_name`, `match_date`, `{y}` required — skipping distribution chart.")
        return

    # Sort matches by date
    match_order = (
        df[["match_name", "match_date"]]
        .drop_duplicates()
        .sort_values("match_date")["match_name"]
        .tolist()
    )

    # Create figure
    fig = go.Figure()

    # Add boxplots with centered outliers
    for match in match_order:
        match_df = df[df["match_name"] == match]

        fig.add_trace(
            go.Box(
                y=match_df[y],
                name=match,
                boxpoints="outliers",  # Show outliers
                jitter=0,              # Disable jitter to center outliers
                pointpos=0,            # Center outliers on the box (0 = middle of box)
                marker=dict(size=5, opacity=0.6),
                line=dict(width=1),
                hoverinfo="y+text",
                # Build hover text: include Stage + Predicted Class
                hovertemplate="Stage: %{customdata[0]}<br>"
                            "Predicted Class: %{customdata[1]}<br>"
                            "Result: %{y:.2%}<extra></extra>",
                text=None,  # we now use customdata instead of text
                customdata=np.stack([
                    match_df["stg"] if "stg" in match_df.columns else [""] * len(match_df),
                    match_df["pred_class"] if "pred_class" in match_df.columns else [""] * len(match_df),
                ], axis=-1),
            )
        )

    # Add median line
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
        )
    )

    # Add reference line at 50% if requested
    if show_ref:
        fig.add_hline(
            y=0.5,
            line_dash="dash",
            line_color="gray",
            line_width=2,
            annotation_text="50%",
            annotation_position="top left",
        )

    # Update layout
    y_label_prefix = "Division" if norm == "div" else "Class"
    y_axis = dict(
        title=f"{y_label_prefix} Stage Result",
        tickformat=".0%",
        range=[0, 1] if lock_axes else [df[y].min(), df[y].max()],
    )
    fig.update_layout(
        xaxis=dict(
            # title="Match",
            categoryorder="array",
            categoryarray=match_order,
            tickangle=45,  # Rotate labels for better readability if needed
        ),
        yaxis=y_axis,
        showlegend=False,  # Hide legend as in original
        template="plotly_white",
        hovermode="closest",
        margin=dict(l=10, r=10, t=10, b=10)
    )

    # Display in Streamlit
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
    show_regression: bool = False,   # 🆕 add regression line through centroids
):
    y_label_prefix = "Division" if norm == "div" else "Class"
    x_label_prefix = "Division" if norm == "div" else "Class"
    y = f"{norm}_pts_perc"
    x = f"{norm}_time_perc"

    needed = ["match_date", y, x, "match_name", "stg", f"{norm}_factor_perc", "pred_class"]
    missing = [c for c in needed if c not in df.columns]
    if missing:
        st.info(f"Missing required columns for scatter: {missing}")
        return

    # Detect scale
    tmax = df[x].max(skipna=True)
    pmax = df[y].max(skipna=True)
    is_fraction = max(tmax, pmax) <= 1.5
    ref_val = 0.5 if is_fraction else 50
    dom = [0, 1] if is_fraction else [0, 100]
    format_str = ".1%" if is_fraction else ".1f"

    # Match order
    match_order = (
        df[["match_name", "match_date"]]
        .drop_duplicates()
        .sort_values("match_date")["match_name"]
        .tolist()
    )

    # Prepare data
    sdf = df[needed].copy().sort_values("match_date")
    sdf.rename(columns={x: "Time (%)", y: "Points (%)"}, inplace=True)

    # Compute centroids
    cent = (
        sdf.groupby("match_name", as_index=False)
           .agg(mean_time=("Time (%)", "mean"), mean_points=("Points (%)", "mean"))
           .rename(columns={"mean_time": "Time (%)", "mean_points": "Points (%)"})
    )
    cent["label"] = cent["match_name"]

    # Determine axis ranges
    if show_points:
        x_min, x_max = sdf["Time (%)"].min(), sdf["Time (%)"].max()
        y_min, y_max = sdf["Points (%)"].min(), sdf["Points (%)"].max()
    else:
        x_min, x_max = cent["Time (%)"].min(), cent["Time (%)"].max()
        y_min, y_max = cent["Points (%)"].min(), cent["Points (%)"].max()
    x_range = dom if lock_axes else [x_min, x_max]
    y_range = dom if lock_axes else [y_min, y_max]

    # Color map per match
    import plotly.colors as pc
    colors = pc.qualitative.Plotly
    color_map = {match: colors[i % len(colors)] for i, match in enumerate(match_order)}

    fig = go.Figure()

    # Stage points
    if show_points:
        for match in match_order:
            match_df = sdf[sdf["match_name"] == match]
            if match_df.empty:
                continue
            point_plotly_size = int((point_size / np.pi) ** 0.5 * 2)  # approx diameter from area

            # Build customdata with [result, pred_class]
            customdata = np.stack([
                match_df[f"{norm}_factor_perc"],
                match_df["pred_class"] if "pred_class" in match_df.columns else [""] * len(match_df),
            ], axis=-1)

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
                    text=match_df["stg"],
                    customdata=customdata,
                    hovertemplate=(
                        "Match: %{data.name}<br>"
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
        match_cent = cent[cent["match_name"] == match]
        if match_cent.empty:
            continue
        centroid_plotly_size = int((centroid_size / np.pi) ** 0.5 * 2)
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
                hovertemplate=(
                    "Match: %{data.name}<br>"
                    f"Time: %{{x:{format_str}}}<br>"
                    f"Points: %{{y:{format_str}}}<extra></extra>"
                ),
                showlegend=False,
            )
        )

    # Labels
    if show_labels:
        for match in match_order:
            match_cent = cent[cent["match_name"] == match]
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

    # 🆕 Regression line through centroids
    if show_regression and len(cent) >= 2:
        cx = cent["Time (%)"].to_numpy(dtype=float)
        cy = cent["Points (%)"].to_numpy(dtype=float)
        # Fit y = a*x + b
        a, b = np.polyfit(cx, cy, 1)
        # R^2
        y_hat = a * cx + b
        ss_res = np.sum((cy - y_hat) ** 2)
        ss_tot = np.sum((cy - np.mean(cy)) ** 2)
        r2 = 1 - ss_res / ss_tot if ss_tot > 0 else np.nan

        # Draw line across centroid x-range (bounded to visible range)
        x0, x1 = float(np.min(cx)), float(np.max(cx))
        line_x = np.linspace(x0, x1, 50)
        line_y = a * line_x + b

        fig.add_trace(
            go.Scatter(
                x=line_x, y=line_y,
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

        # Annotate equation (top-left inside plot area)
        fig.add_annotation(
            xref="paper", yref="paper",
            x=0.01, y=0.99,
            xanchor="left", yanchor="top",
            text=f"Trend (centroids): y = {a:.3f}·x + {b:.3f} &nbsp; | &nbsp; R² = {r2:.3f}",
            showarrow=False,
            font=dict(size=12, color="#333"),
            bgcolor="rgba(255,255,255,0.6)",
            bordercolor="#999",
            borderwidth=1,
            borderpad=4,
        )

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
    
def class_bubble(
    df: pd.DataFrame,
    shooter: str,
    *,
    x_domain: tuple[str, str] | None = None,
    y_domain: tuple[float, float] | None = None,
    size_range: tuple[int, int] = (60, 900),   # area-like scale
    show_legend: bool = True,
    show_ref: bool = True,
):
    needed = {
        "shooter", "match_name", "match_date", "div",
        "pred_class", "relation", "sh_median", "consistency_cover"
    }
    miss = needed - set(df.columns)
    if miss:
        st.info(f"Bubble map needs columns: {sorted(miss)}")
        return

    d = df.loc[df["shooter"] == shooter].copy()
    if d.empty:
        st.info("No data for the specified shooter.")
        return

    # --- Prep & ordering ---
    d["match_date"] = pd.to_datetime(d["match_date"], errors="coerce")
    d["consistency_cover"] = pd.to_numeric(d["consistency_cover"], errors="coerce")

    match_order = (
        d[["match_name", "match_date"]]
        .drop_duplicates()
        .sort_values("match_date")["match_name"]
        .tolist()
    )
    d = d.sort_values("match_date")

    # --- Y formatting (percent vs absolute) ---
    frac = pd.to_numeric(d["sh_median"], errors="coerce").max(skipna=True) <= 1.5
    fmt  = ".0%" if frac else ".0f"

    # --- Axes domains (optional external control) ---
    if x_domain is None and match_order:
        x_domain = (match_order[0], match_order[-1])

    if y_domain is None:
        vals = pd.to_numeric(d["sh_median"], errors="coerce")
        if vals.notna().any():
            pad = 0.02 * (vals.max() - vals.min() + 1e-12)
            y_domain = (float(vals.min() - pad), float(vals.max() + pad))

    # --- Normalize relation & colors ---
    def _norm_rel(x: str) -> str:
        x = str(x).strip().lower()
        if "within" in x: return "within"
        if "above"  in x: return "above"
        if "below"  in x: return "below"
        return x

    def _label_rel(n: str) -> str:
        return {
            "within": "Within Class IQR",
            "above":  "Above Class IQR",
            "below":  "Below Class IQR",
        }.get(n, n.title())

    d["rel_norm"]   = d["relation"].map(_norm_rel)
    d["rel_label"]  = d["rel_norm"].map(_label_rel)

    stroke_colors = {"within": "#1565c0", "above": "#2e7d32", "below": "#ef6c00"}  # green/blue/orange

    # --- Figure ---
    fig = go.Figure()

    # Guide line for trajectory
    guide_data = (
        d[["match_name", "match_date", "sh_median"]]
        .drop_duplicates()
        .sort_values("match_date")
    )
    fig.add_trace(
        go.Scatter(
            x=guide_data["match_name"],
            y=guide_data["sh_median"],
            mode="lines",
            line=dict(color="#9e9e9e", width=3, dash="dot"),
            opacity=0.25,
            name="Guide",
            showlegend=False,
            hoverinfo="skip",
        )
    )

    # --- One trace per relation (so legend shows relation only) ---
    for rel in ["within", "above", "below"]:
        subset = d[d["rel_norm"] == rel]
        if subset.empty:
            continue

        # Convert consistency (0..1) to Plotly marker sizes (diameter-ish)
        sizes = subset["consistency_cover"].fillna(0).apply(
            lambda x: float(np.sqrt(np.interp(x, [0, 1], [size_range[0]/np.pi, size_range[1]/np.pi])) * 2.0)
        )

        # Use neutral fill + colored stroke; put pred_class TEXT inside
        fig.add_trace(
            go.Scatter(
                x=subset["match_name"],
                y=subset["sh_median"],
                mode="markers+text",
                text=subset["pred_class"],
                textposition="middle center",
                textfont=dict(size=12, color="#1f1f1f"),
                marker=dict(
                    size=sizes,
                    color="rgba(0,0,0,0.00)",  # neutral, semi-transparent fill
                    line=dict(color=stroke_colors.get(rel, "black"), width=3.0),
                ),
                name=_label_rel(rel),
                showlegend=show_legend,
                customdata=subset[["match_name", "div", "pred_class", "relation", "sh_median", "consistency_cover"]],
                hovertemplate=(
                    "Match: %{customdata[0]}<br>"
                    "Division: %{customdata[1]}<br>"
                    "Predicted Class: %{customdata[2]}<br>"
                    "Relation: %{customdata[3]}<br>"
                    f"Shooter Median: %{{customdata[4]:{fmt}}}<br>"
                    "Consistency: %{customdata[5]:.0%}<extra></extra>"
                ),
            )
        )

    # Reference line at 50% (optional)
    if show_ref:
        ref_val = 0.5 if frac else 50
        fig.add_hline(
            y=ref_val,
            line_dash="dash",
            line_color="gray",
            line_width=2,
            annotation_text=f"{ref_val:.0%}" if frac else f"{ref_val}",
            annotation_position="top left",
        )

    # Layout / axes
    x_axis = dict(
        categoryorder="array",
        categoryarray=match_order,
        tickangle=45,
        showgrid=False,
        autorange=True,
        # title="Match",
    )
    y_axis = dict(
        title="Shooter Median",
        tickformat=fmt,
        range=y_domain,
        showgrid=True,
        autorange=True,
    )

    fig.update_layout(
        xaxis=x_axis,
        yaxis=y_axis,
        showlegend=show_legend,
        legend=dict(
            title="Relation",
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="center",
            x=0.5,
        ) if show_legend else None,
        template="plotly_white",
        hovermode="closest",
        autosize=True,
        margin=dict(l=10, r=10, t=10, b=10),
    )

    # Render
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
        "shooter", "match_name", "match_date", "match_level", "div", "class",
    }
    miss = needed - set(df.columns)
    if miss:
        st.info(f"Match count needs columns: {sorted(miss)}")
        return

    # Filter and prepare data
    d = df.loc[df["shooter"] == shooter].copy()
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
        d[["match_name", "match_date", "match_level", "div", "class"]]
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
        cats = sorted(matches["div"].dropna().astype(str).unique().tolist())
        x_labels = _labels_with_unknown(cats, matches["div"].isna().any())
        x_title = "Division"
        matches["x_label"] = matches["div"].fillna("Unknown").astype(str)
    elif x_axis in {"class"}:
        x_field = "class"
        cats = sorted(matches["class"].dropna().astype(str).unique().tolist())
        x_labels = _labels_with_unknown(cats, matches["class"].isna().any())
        x_title = "Class"
        matches["x_label"] = matches["class"].fillna("Unknown").astype(str)
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
        color_field = "div"
        legend_title = "Division"
    elif color in {"class"}:
        color_field = "class"
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

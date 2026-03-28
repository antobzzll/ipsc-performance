import numpy as np
import pandas as pd

def aggregate_shooter_performance(
    df: pd.DataFrame,
    *,
    stage_pct_col: str = "div_factor_perc",   # stage % in [0,1], already division-normalized
    match_name_col: str = "match_name",
    match_date_col: str = "match_date",       # can exist or not; used only for output/sorting
    weight_mode: str = "stages",              # "stages" or "equal"
    min_stages_per_match: int | None = None,  # e.g., 6 to ignore tiny matches
    epsilon: float = 1e-6                     # floor to avoid zeroing the geo-mean
) -> dict:
    """
    Aggregate a single shooter's IPSC performance across matches:
      1) For each match, compute the median of stage percentages (stage_pct_col).
      2) Weight matches (by #stages or equally).
      3) Return the weighted geometric mean (in [0,1]) plus a per-match table.

    Parameters
    ----------
    df : DataFrame
        Single-shooter stage-level rows. Must include stage_pct_col and match_name_col.
        Optionally includes match_date_col (datetime64) for nicer output.
    stage_pct_col : str
        Column with division-normalized stage performance in [0,1].
    match_name_col : str
        Column identifying the match.
    match_date_col : str
        Optional datetime column. If missing, it's ignored.
    weight_mode : {"stages","equal"}
        "stages": weight each match by the number of stages the shooter has in that match.
        "equal":  equal weight per match.
    min_stages_per_match : int | None
        If set, drop matches with fewer than this number of shooter stages.
    epsilon : float
        Small floor to guard the geometric mean against zeros.

    Returns
    -------
    dict with keys:
        - G : float   -> weighted geometric mean in [0,1] (np.nan if no valid matches)
        - n_matches : int
        - per_match : DataFrame with columns:
            ["match", "match_date", "median_stage", "n_stages", "weight", "weight_norm"]
    """
    if df.empty:
        return {"G": np.nan, "n_matches": 0, "per_match": pd.DataFrame()}

    # Keep only required columns + optional date
    cols = [match_name_col, stage_pct_col]
    if match_date_col in df.columns:
        cols.append(match_date_col)
    work = df[cols].copy()

    # Ensure numeric percentages
    work[stage_pct_col] = pd.to_numeric(work[stage_pct_col], errors="coerce")

    # Per-match median and stage count
    agg = (
        work.groupby(match_name_col, dropna=False)
            .agg(
                median_stage=(stage_pct_col, "median"),
                n_stages=(stage_pct_col, "count"),
                match_date=(match_date_col, "max") if match_date_col in work.columns else ("median_stage", "size")
            )
            .reset_index()
    )
    # If date not present, drop the placeholder
    if match_date_col not in work.columns:
        agg = agg.drop(columns=["match_date"])

    # Optional filter: minimum stages per match
    if min_stages_per_match is not None:
        agg = agg.loc[agg["n_stages"] >= int(min_stages_per_match)].copy()

    if agg.empty:
        return {"G": np.nan, "n_matches": 0, "per_match": pd.DataFrame()}

    # Weights
    if weight_mode not in {"stages", "equal"}:
        raise ValueError("weight_mode must be 'stages' or 'equal'.")

    if weight_mode == "stages":
        agg["weight"] = agg["n_stages"].astype(float)
    else:  # equal
        agg["weight"] = 1.0

    # Normalize weights
    total_w = agg["weight"].sum()
    if total_w <= 0:
        return {"G": np.nan, "n_matches": int(agg.shape[0]), "per_match": agg.rename(columns={match_name_col: "match"})}

    agg["weight_norm"] = agg["weight"] / total_w

    # Weighted geometric mean (with epsilon floor)
    s = np.maximum(agg["median_stage"].astype(float), epsilon)
    logG = np.sum(agg["weight_norm"] * np.log(s))
    G = float(np.exp(logG))

    # Tidy per-match output
    out_cols = [match_name_col, "median_stage", "n_stages", "weight", "weight_norm"]
    if "match_date" in agg.columns:
        out_cols.insert(1, "match_date")
    per_match = agg[out_cols].rename(columns={match_name_col: "match"})

    return {"G": G, "n_matches": int(per_match.shape[0]), "per_match": per_match}


def stage_standing(
    df: pd.DataFrame,
    match: str,
    shooter_div: str,
    stg_n,
    *,
    match_name_col: str = "match_name",
    shooter_div_col: str = "shooter_div",
    stage_col: str = "stg_n",
    hf_col: str = "hf",
) -> pd.DataFrame:
    """
    Stage ranking for one match / division / stage.
    """
    stage_df = df.loc[
        (df[match_name_col] == match) &
        (df[shooter_div_col] == shooter_div) &
        (df[stage_col] == stg_n)
    ].copy()

    if stage_df.empty:
        return pd.DataFrame(
            columns=[
                "rank", "shooter_name", "shooter_class",
                "pts", "time", "hf", "stg_match_pts", "hf_pct"
            ]
        )

    stage_df[hf_col] = pd.to_numeric(stage_df[hf_col], errors="coerce")
    stage_df = stage_df.sort_values(hf_col, ascending=False)

    stage_df["rank"] = (
        stage_df[hf_col]
        .rank(method="min", ascending=False)
        .astype("Int64")
    )

    cols = [
        "rank", "shooter_name", "shooter_class",
        "pts", "time", "hf", "stg_match_pts", "hf_pct"
    ]
    cols = [c for c in cols if c in stage_df.columns]

    return stage_df[cols].reset_index(drop=True)


def match_standing(
    df: pd.DataFrame,
    match: str,
    shooter_div: str,
    *,
    match_name_col: str = "match_name",
    shooter_div_col: str = "shooter_div",
    shooter_name_col: str = "shooter_name",
    shooter_class_col: str = "shooter_class",
    match_pts_col: str = "stg_match_pts",
) -> pd.DataFrame:
    """
    Match ranking for one match / division, summing stage match points.

    Returns:
    - overall division standing:
        - rank
        - pct (vs division winner)
    - class standing:
        - class_rank
        - class_pct (vs class winner)
    - pct_vs_non_m_gm:
        - pct vs first shooter in A/B/C/D
    - pct_vs_non_m_gm_minus_16:
        - pct_vs_non_m_gm minus 16 percentage points
    - championship:
        - championship label from source df
    """
    match_df = df.loc[
        (df[match_name_col] == match) &
        (df[shooter_div_col] == shooter_div)
    ].copy()

    if match_df.empty:
        return pd.DataFrame(
            columns=[
                "championship",
                "shooter_div",
                "rank",
                "class_rank",
                "shooter_name",
                "shooter_class",
                "stg_match_pts",
                "pct",
                "class_pct",
                "pct_abcd",
                "pct_abcd_minus_16",
            ]
        )

    match_df[match_pts_col] = pd.to_numeric(match_df[match_pts_col], errors="coerce")

    championship_value = np.nan
    if "championship" in match_df.columns:
        ch_vals = match_df["championship"].dropna().unique().tolist()
        if ch_vals:
            championship_value = ch_vals[0]

    standing = (
        match_df.groupby([shooter_name_col, shooter_class_col], dropna=False)[match_pts_col]
        .sum()
        .reset_index()
        .sort_values(match_pts_col, ascending=False)
        .reset_index(drop=True)
    )

    # Overall division rank / pct
    standing["rank"] = (
        standing[match_pts_col]
        .rank(method="min", ascending=False)
        .astype("Int64")
    )

    max_pts = standing[match_pts_col].max()
    if pd.isna(max_pts) or max_pts == 0:
        standing["pct"] = np.nan
    else:
        standing["pct"] = (standing[match_pts_col] / max_pts).round(4)

    # Class rank / pct
    standing["class_rank"] = (
        standing.groupby(shooter_class_col, dropna=False)[match_pts_col]
        .rank(method="min", ascending=False)
        .astype("Int64")
    )

    class_max = standing.groupby(shooter_class_col, dropna=False)[match_pts_col].transform("max")
    standing["class_pct"] = np.where(
        class_max.isna() | (class_max == 0),
        np.nan,
        (standing[match_pts_col] / class_max).round(4),
    )

    # Pct vs first shooter in A/B/C/D
    non_m_gm = standing[
        standing[shooter_class_col].astype(str).str.upper().isin(["A", "B", "C", "D"])
    ].copy()

    if non_m_gm.empty:
        ref_non_m_gm_pts = np.nan
    else:
        ref_non_m_gm_pts = pd.to_numeric(non_m_gm.iloc[0][match_pts_col], errors="coerce")

    if pd.isna(ref_non_m_gm_pts) or ref_non_m_gm_pts == 0:
        standing["pct_vs_non_m_gm"] = np.nan
    else:
        standing["pct_vs_non_m_gm"] = (standing[match_pts_col] / ref_non_m_gm_pts).round(4)

    # Deduct 16 percentage points
    standing["pct_vs_non_m_gm_minus_16"] = (
        standing["pct_vs_non_m_gm"] - 0.16
    ).round(4)

    standing["shooter_div"] = shooter_div
    standing["championship"] = championship_value

    return standing[
        [
            "championship",
            "shooter_div",
            "rank",
            shooter_name_col,
            match_pts_col,
            "pct",
            shooter_class_col,
            "pct_vs_non_m_gm",
            "pct_vs_non_m_gm_minus_16",
            "class_rank",
            "class_pct",
        ]
    ].rename(
        columns={
            shooter_name_col: "shooter_name",
            shooter_class_col: "shooter_class",
            match_pts_col: "stg_match_pts",
            "pct_vs_non_m_gm": "pct_abcd",
            "pct_vs_non_m_gm_minus_16": "pct_abcd_minus_16",
        }
    )
    
def performance_class_from_pct(values):
        """
        Map percentage values in [0, 1] to merit classes:
        GM 95-100
        M  85-94.99
        A  75-84.99
        B  60-74.99
        C  40-59.99
        D   0-39.99
        """
        s = pd.to_numeric(pd.Series(values), errors="coerce")
        out = pd.Series(index=s.index, dtype="object")

        out[(s >= 0.95) & (s <= 1.00)] = "GM"
        out[(s >= 0.85) & (s < 0.95)] = "M"
        out[(s >= 0.75) & (s < 0.85)] = "A"
        out[(s >= 0.60) & (s < 0.75)] = "B"
        out[(s >= 0.40) & (s < 0.60)] = "C"
        out[(s >= 0.00) & (s < 0.40)] = "D"

        return out
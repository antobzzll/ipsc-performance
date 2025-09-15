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
# app/lib/models_core.py
from __future__ import annotations
import pandas as pd
import numpy as np

# ---------- Base model: stages ----------
def load_fitds_stages_core(path: str) -> pd.DataFrame:
    """
    PURE loader: no Streamlit. Safe to import/use from notebooks.
    """
    df = pd.read_csv(path)
    # if "cat" in df.columns:
    #     df = df.drop(columns="cat")
    
    df['hit_factor'] = round(df['pts'] / df['time'], 4)
    # df['hf_check'] = df['hit_factor'] - df['factor']
    
    df['div_pts_perc'] = df.groupby(['match_name', 'shooter_div', 'stg_n'])['pts'].transform(lambda x: (x / x.max()))
    df['div_first_time'] = df.groupby(['match_name', 'shooter_div', 'stg_n'])['time'].transform(lambda x: x[df['hit_factor'] > 0].min() if any(df['hit_factor'] > 0) else None)
    df['div_time_perc'] = df['div_first_time'] / df['time']
    df['div_factor_perc'] = df.groupby(['match_name', 'shooter_div', 'stg_n'])['hit_factor'].transform(lambda x: (x / x.max()))
    df['div_factor_standing'] = df.groupby(['match_name', 'shooter_div', 'stg_n'])['hit_factor'].transform(lambda x: x.rank(method='first', ascending=False))
    
    df['cls_pts_perc'] = df.groupby(['match_name', 'shooter_div', 'shooter_class', 'stg_n'])['pts'].transform(lambda x: (x / x.max()))
    df['cls_first_time'] = df.groupby(['match_name', 'shooter_div', 'shooter_class', 'stg_n'])['time'].transform(lambda x: x[df['hit_factor'] > 0].min() if any(df['hit_factor'] > 0) else None)
    df['cls_time_perc'] = df['cls_first_time'] / df['time']
    df['cls_factor_perc'] = df.groupby(['match_name', 'shooter_div', 'shooter_class', 'stg_n'])['hit_factor'].transform(lambda x: (x / x.max()))
    df['cls_factor_standing'] = df.groupby(['match_name', 'shooter_div', 'shooter_class', 'stg_n'])['hit_factor'].transform(lambda x: x.rank(method='first', ascending=False))
    
    match_abcd = df[df['shooter_class'].isin(['A', 'B', 'C', 'D'])].copy()
    df['div_pts_perc_abcd'] = match_abcd.groupby(['match_name', 'shooter_div', 'stg_n'])['pts'].transform(lambda x: (x / x.max()))
    df['div_first_time_abcd'] = match_abcd.groupby(['match_name', 'shooter_div', 'stg_n'])['time'].transform(lambda x: x[match_abcd['hit_factor'] > 0].min() if any(match_abcd['hit_factor'] > 0) else None)
    df['div_time_perc_abcd'] = match_abcd['div_first_time'] / match_abcd['time']
    df['div_factor_perc_abcd'] = match_abcd.groupby(['match_name', 'shooter_div', 'stg_n'])['hit_factor'].transform(lambda x: (x / x.max()))
    df['div_factor_standing_abcd'] = match_abcd.groupby(['match_name', 'shooter_div', 'stg_n'])['hit_factor'].transform(lambda x: x.rank(method='first', ascending=False))
    
    df['cls_pts_perc_abcd'] = match_abcd.groupby(['match_name', 'shooter_div', 'shooter_class', 'stg_n'])['pts'].transform(lambda x: (x / x.max()))
    df['cls_first_time_abcd'] = match_abcd.groupby(['match_name', 'shooter_div', 'shooter_class', 'stg_n'])['time'].transform(lambda x: x[match_abcd['hit_factor'] > 0].min() if any(match_abcd['hit_factor'] > 0) else None)
    df['cls_time_perc_abcd'] = match_abcd['cls_first_time'] / match_abcd['time']
    df['cls_factor_perc_abcd'] = match_abcd.groupby(['match_name', 'shooter_div', 'shooter_class', 'stg_n'])['hit_factor'].transform(lambda x: (x / x.max()))
    df['cls_factor_standing_abcd'] = match_abcd.groupby(['match_name', 'shooter_div', 'shooter_class', 'stg_n'])['hit_factor'].transform(lambda x: x.rank(method='first', ascending=False))

    # # ----- FIELD (match-level) CLASS BANDS -----
    # match_stats = (
    #     df.groupby(["match_name", "div", "class"])["div_factor_perc"]
    #       .agg(q25=lambda s: s.quantile(0.25),
    #            median="median",
    #            q75=lambda s: s.quantile(0.75))
    #       .reset_index()
    #       .rename(columns={"q25":"q25_match","median":"median_match","q75":"q75_match"})
    # )

    # # ----- SHOOTER (per match, per division) -----
    # shooter_div_match_stats = (
    #     df.groupby(["match_name","div","shooter"])["div_factor_perc"]
    #       .agg(q25=lambda s: s.quantile(0.25),
    #            median="median",
    #            q75=lambda s: s.quantile(0.75))
    #       .reset_index()
    #       .rename(columns={
    #           "q25":"q25_shooter_div_match",
    #           "median":"median_shooter_div_match",
    #           "q75":"q75_shooter_div_match",
    #       })
    # )

    # # Merge
    # df = df.merge(match_stats, on=["match_name","div","class"], how="left")
    # df = df.merge(shooter_div_match_stats, on=["match_name","div","shooter"], how="left")

    # # Flags
    # df["within_shooter_div_iqr"] = df["div_factor_perc"].between(
    #     df["q25_shooter_div_match"], df["q75_shooter_div_match"], inclusive="both"
    # )

    # # Which class band contains the value (per match/div)?
    # bands = (
    #     match_stats.groupby(["match_name","div"])
    #                .apply(lambda g: g[["class","q25_match","q75_match","median_match"]].to_dict("records"))
    #                .to_dict()
    # )

    # def class_band_for_value(row):
    #     key = (row["match_name"], row["div"])
    #     recs = bands.get(key, [])
    #     v = row["div_factor_perc"]
    #     if pd.isna(v) or not recs:
    #         return np.nan
    #     in_band = [r for r in recs if r["q25_match"] <= v <= r["q75_match"]]
    #     if len(in_band) == 1:
    #         return in_band[0]["class"]
    #     if len(in_band) > 1:
    #         best = min(in_band, key=lambda r: abs(v - r["median_match"]))
    #         return best["class"]
    #     return np.nan

    # df["within_class_band"] = df.apply(class_band_for_value, axis=1)
    return df

def class_predict(
    stages: pd.DataFrame,
    *,
    min_class_n: int = 8,                 # min shooters to define a class band in a match/div
    min_stage_n: int = 3,                 # min stages to compute a shooter median
    min_stage_n_consistency: int = 5,     # min stages to trust shooter IQR for consistency
    consistency_threshold: float = 0.7,   # ≥70% of shooter IQR inside class IQR → "consistent"
    robust_scale_floor: float = 1e-6,     # floor for robust z denominator
) -> pd.DataFrame:
    """
    Required columns in `stages`:
      ['match_name','shooter_div','shooter_class','shooter_name','div_factor_perc']

    Logic:
    - Compute one row per shooter / match / division using the shooter's stage median.
    - Build class bands from the distribution of SHOOTER medians (not raw stage rows).
    - Predict the closest class band in the same match/division.
    - Compute consistency by comparing the shooter's IQR with the chosen class IQR.

    Returns one row per (shooter, match_name, shooter_div) with:
      ['shooter_name','match_name','shooter_div','n_stages','sh_median',
       'pred_class','relation','dist_to_class_median',
       'q1','class_median','q3','n_in_class',
       'robust_z_class',
       'sh_q1','sh_q3','sh_iqr','consistency_cover','consistent']
    """
    needed = {"match_name", "shooter_div", "shooter_class", "shooter_name", "div_factor_perc"}
    miss = needed - set(stages.columns)
    if miss:
        raise KeyError(f"Missing columns: {sorted(miss)}")

    df = stages.copy()
    df["div_factor_perc"] = pd.to_numeric(df["div_factor_perc"], errors="coerce")

    # ------------------------------------------------------------
    # 1) Shooter summary per (shooter, match, div, actual class)
    # ------------------------------------------------------------
    shooter_base = (
        df.groupby(
            ["shooter_name", "match_name", "shooter_div", "shooter_class"],
            dropna=False
        )["div_factor_perc"]
        .agg(n_stages="count", sh_median="median")
        .reset_index()
    )
    shooter_base = shooter_base[shooter_base["n_stages"] >= min_stage_n].copy()

    empty_cols = [
        "shooter_name", "match_name", "shooter_div", "n_stages", "sh_median",
        "pred_class", "relation", "dist_to_class_median",
        "q1", "class_median", "q3", "n_in_class",
        "robust_z_class", "sh_q1", "sh_q3", "sh_iqr",
        "consistency_cover", "consistent"
    ]

    if shooter_base.empty:
        return pd.DataFrame(columns=empty_cols)

    shooter_sum = shooter_base[
        ["shooter_name", "match_name", "shooter_div", "n_stages", "sh_median"]
    ].copy()

    # ------------------------------------------------------------
    # 2) Class bands from SHOOTER medians, not raw stage rows
    # ------------------------------------------------------------
    class_bands = (
        shooter_base.groupby(
            ["match_name", "shooter_div", "shooter_class"],
            dropna=False
        )["sh_median"]
        .agg(
            n_in_class="count",               # shooters in class band
            q1=lambda s: s.quantile(0.25),
            class_median="median",
            q3=lambda s: s.quantile(0.75),
        )
        .reset_index()
    )
    class_bands = class_bands[class_bands["n_in_class"] >= min_class_n].copy()

    if class_bands.empty:
        out = shooter_sum.copy()
        out[[
            "pred_class", "relation", "dist_to_class_median",
            "q1", "class_median", "q3", "n_in_class",
            "robust_z_class", "sh_q1", "sh_q3", "sh_iqr",
            "consistency_cover", "consistent"
        ]] = np.nan
        return out[empty_cols]

    # ------------------------------------------------------------
    # 3) Join shooter summaries to all candidate class bands
    # ------------------------------------------------------------
    cand = shooter_sum.merge(class_bands, on=["match_name", "shooter_div"], how="left")
    cand = cand[cand["q1"].notna()].copy()

    if cand.empty:
        out = shooter_sum.copy()
        out[[
            "pred_class", "relation", "dist_to_class_median",
            "q1", "class_median", "q3", "n_in_class",
            "robust_z_class", "sh_q1", "sh_q3", "sh_iqr",
            "consistency_cover", "consistent"
        ]] = np.nan
        return out[empty_cols]

    cand["within_iqr"] = (cand["sh_median"] >= cand["q1"]) & (cand["sh_median"] <= cand["q3"])
    cand["dist_to_class_median"] = (cand["sh_median"] - cand["class_median"]).abs()
    cand["_within_rank"] = np.where(cand["within_iqr"], 0, 1)

    # Tie-breaks:
    # 1) inside class IQR
    # 2) closest class median
    # 3) larger class sample
    # 4) alphabetical class as deterministic fallback
    picked = (
        cand.sort_values(
            [
                "shooter_name", "match_name", "shooter_div",
                "_within_rank", "dist_to_class_median", "n_in_class", "shooter_class"
            ],
            ascending=[True, True, True, True, True, False, True]
        )
        .groupby(["shooter_name", "match_name", "shooter_div"], as_index=False)
        .first()
    )

    # ------------------------------------------------------------
    # 4) Relation vs chosen class band
    # ------------------------------------------------------------
    def _relation(row):
        if row["sh_median"] < row["q1"]:
            return "Below Class IQR"
        if row["sh_median"] > row["q3"]:
            return "Above Class IQR"
        return "Within Class IQR"

    picked["relation"] = picked.apply(_relation, axis=1)

    # Robust z vs chosen class band
    class_iqr = (picked["q3"] - picked["q1"]).replace(0, np.nan)
    robust_scale = (class_iqr / 1.349).clip(lower=robust_scale_floor)
    picked["robust_z_class"] = (picked["sh_median"] - picked["class_median"]) / robust_scale

    # ------------------------------------------------------------
    # 5) Consistency: shooter IQR, only if enough stages
    # ------------------------------------------------------------
    shooter_iqr = (
        df.groupby(["shooter_name", "match_name", "shooter_div"], dropna=False)["div_factor_perc"]
        .agg(
            n="count",
            sh_q1=lambda s: s.quantile(0.25),
            sh_q3=lambda s: s.quantile(0.75),
        )
        .reset_index()
    )
    shooter_iqr = shooter_iqr[shooter_iqr["n"] >= min_stage_n_consistency].copy()
    shooter_iqr["sh_iqr"] = shooter_iqr["sh_q3"] - shooter_iqr["sh_q1"]
    shooter_iqr = shooter_iqr.drop(columns="n")

    out = picked.merge(
        shooter_iqr,
        on=["shooter_name", "match_name", "shooter_div"],
        how="left"
    )

    # proportion of shooter IQR inside chosen class IQR
    eps = 1e-12
    inter_len = np.maximum(
        0.0,
        np.minimum(out["sh_q3"], out["q3"]) - np.maximum(out["sh_q1"], out["q1"])
    )
    out["consistency_cover"] = inter_len / (out["sh_iqr"].replace(0, np.nan) + eps)

    # Distinguish "not consistent" from "not enough data"
    out["consistent"] = pd.Series(pd.NA, index=out.index, dtype="boolean")
    has_consistency_data = out["sh_iqr"].notna()
    out.loc[has_consistency_data, "consistent"] = (
        out.loc[has_consistency_data, "consistency_cover"] >= consistency_threshold
    ).astype("boolean")

    # Final tidy
    out = out.rename(columns={"shooter_class": "pred_class"})

    cols = [
        "shooter_name", "match_name", "shooter_div", "n_stages", "sh_median",
        "pred_class", "relation", "dist_to_class_median",
        "q1", "class_median", "q3", "n_in_class",
        "robust_z_class", "sh_q1", "sh_q3", "sh_iqr",
        "consistency_cover", "consistent"
    ]
    return out[cols]

def class_predict_per_stage(
    stages: pd.DataFrame,
    *,
    min_class_n: int = 8  # min observations to define a class band in a match/div
) -> pd.DataFrame:
    """
    Stage-level class prediction.

    Required columns in `stages`:
      ['match_name','shooter_div','shooter_class','shooter_name','div_factor_perc']
    Optional: 'stg_n' (stage id/number). If present, it will be kept in the output.

    Returns one row per input stage with:
      ['shooter_name','match_name','shooter_div',('stg_n' if present),'div_factor_perc',
       'orig_class','pred_class','relation','dist_to_class_median',
       'q1','class_median','q3','n_in_class','robust_z_class']
    """
    needed = {'match_name','shooter_div','shooter_class','shooter_name','div_factor_perc'}
    miss = needed - set(stages.columns)
    if miss:
        raise KeyError(f"Missing columns: {sorted(miss)}")

    df = stages.copy()
    df['div_factor_perc'] = pd.to_numeric(df['div_factor_perc'], errors='coerce')

    # ---- Base columns in the final output (order matters) ----
    base_cols = ['shooter_name', 'match_name', 'shooter_div']
    if 'stg_n' in df.columns:
        base_cols.append('stg_n')
    base_cols += ['div_factor_perc', 'orig_class']

    # ---- Build class bands per (match, div, class) ----
    class_bands = (
        df.groupby(['match_name', 'shooter_div', 'shooter_class'])['div_factor_perc']
          .agg(
              n_in_class='count',
              q1=lambda s: s.quantile(0.25),
              class_median='median',
              q3=lambda s: s.quantile(0.75)
          )
          .reset_index()
    )
    class_bands = class_bands[class_bands['n_in_class'] >= min_class_n].copy()

    # If no valid class bands, return df with NaNs for prediction fields
    if class_bands.empty:
        out = df.copy()
        out = out.rename(columns={'shooter_class': 'orig_class'})
        out = out[ [c for c in base_cols if c in out.columns] ]
        out[['pred_class','relation','dist_to_class_median',
             'q1','class_median','q3','n_in_class','robust_z_class']] = np.nan
        return out

    # ---- Add stable row id (so we can pick best band per original stage row) ----
    df = df.reset_index(drop=False).rename(columns={'index': '__rowid'})
    df = df.rename(columns={'shooter_class': 'orig_class'})

    # ---- Merge candidates: each stage row gets all class bands for its match/div ----
    # Keep band class under a dedicated name to avoid confusion
    bands = class_bands.rename(columns={'shooter_class': 'band_class'})

    cand = df.merge(
        bands,
        on=['match_name', 'shooter_div'],
        how='left'
    )

    # Drop rows where no band stats exist (should be rare but safe)
    cand = cand[cand['q1'].notna()].copy()
    if cand.empty:
        out = df.copy()
        out = out[ [c for c in base_cols + ['__rowid'] if c in out.columns] ]
        out[['pred_class','relation','dist_to_class_median',
             'q1','class_median','q3','n_in_class','robust_z_class']] = np.nan
        return out.drop(columns='__rowid')

    # ---- Signals vs each candidate band ----
    perf = cand['div_factor_perc']
    cand['within_iqr'] = (perf >= cand['q1']) & (perf <= cand['q3'])
    cand['dist_to_class_median'] = (perf - cand['class_median']).abs()
    cand['_within_rank'] = np.where(cand['within_iqr'], 0, 1)

    # ---- Pick best band per stage: inside-IQR first, then closest median ----
    picked = (
        cand.sort_values(['__rowid', '_within_rank', 'dist_to_class_median'])
            .groupby('__rowid', as_index=False)
            .first()
    )

    # ---- Rename chosen band class to pred_class ----
    picked = picked.rename(columns={'band_class': 'pred_class'})

    # ---- Relation for the stage value vs chosen band ----
    def _relation(row):
        val = row['div_factor_perc']
        if pd.isna(val) or pd.isna(row['q1']) or pd.isna(row['q3']):
            return np.nan
        if val < row['q1']:
            return 'below'
        if val > row['q3']:
            return 'above'
        return 'within'

    picked['relation'] = picked.apply(_relation, axis=1)

    # ---- Robust z for the stage value vs chosen band ----
    iqr = (picked['q3'] - picked['q1']).replace(0, np.nan)
    robust_scale = iqr / 1.349
    picked['robust_z_class'] = (picked['div_factor_perc'] - picked['class_median']) / robust_scale

    # ---- Final tidy + strict keep ----
    keep = base_cols + [
        'pred_class', 'relation', 'dist_to_class_median',
        'q1', 'class_median', 'q3', 'n_in_class', 'robust_z_class'
    ]

    missing = [c for c in keep if c not in picked.columns]
    if missing:
        raise KeyError(
            f"class_predict_per_stage: missing columns in output: {missing}. "
            f"Available: {list(picked.columns)}"
        )

    out = picked[keep].copy()
    return out
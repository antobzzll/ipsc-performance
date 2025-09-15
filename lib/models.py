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
    
    df['div_pts_perc'] = df.groupby(['match_name', 'div', 'stg'])['pts'].transform(lambda x: (x / x.max()))
    df['div_first_time'] = df.groupby(['match_name', 'div', 'stg'])['time'].transform(lambda x: x[df['hit_factor'] > 0].min() if any(df['hit_factor'] > 0) else None)
    df['div_time_perc'] = df['div_first_time'] / df['time']
    df['div_factor_perc'] = df.groupby(['match_name', 'div', 'stg'])['hit_factor'].transform(lambda x: (x / x.max()))
    df['div_factor_standing'] = df.groupby(['match_name', 'div', 'stg'])['hit_factor'].transform(lambda x: x.rank(method='first', ascending=False))
    
    df['cls_pts_perc'] = df.groupby(['match_name', 'div', 'class', 'stg'])['pts'].transform(lambda x: (x / x.max()))
    df['cls_first_time'] = df.groupby(['match_name', 'div', 'class', 'stg'])['time'].transform(lambda x: x[df['hit_factor'] > 0].min() if any(df['hit_factor'] > 0) else None)
    df['cls_time_perc'] = df['cls_first_time'] / df['time']
    df['cls_factor_perc'] = df.groupby(['match_name', 'div', 'class', 'stg'])['hit_factor'].transform(lambda x: (x / x.max()))
    df['cls_factor_standing'] = df.groupby(['match_name', 'div', 'class', 'stg'])['hit_factor'].transform(lambda x: x.rank(method='first', ascending=False))
    
    match_abcd = df[df['class'].isin(['A', 'B', 'C', 'D'])].copy()
    df['div_pts_perc_abcd'] = match_abcd.groupby(['match_name', 'div', 'stg'])['pts'].transform(lambda x: (x / x.max()))
    df['div_first_time_abcd'] = match_abcd.groupby(['match_name', 'div', 'stg'])['time'].transform(lambda x: x[match_abcd['hit_factor'] > 0].min() if any(match_abcd['hit_factor'] > 0) else None)
    df['div_time_perc_abcd'] = match_abcd['div_first_time'] / match_abcd['time']
    df['div_factor_perc_abcd'] = match_abcd.groupby(['match_name', 'div', 'stg'])['hit_factor'].transform(lambda x: (x / x.max()))
    df['div_factor_standing_abcd'] = match_abcd.groupby(['match_name', 'div', 'stg'])['hit_factor'].transform(lambda x: x.rank(method='first', ascending=False))
    
    df['cls_pts_perc_abcd'] = match_abcd.groupby(['match_name', 'div', 'class', 'stg'])['pts'].transform(lambda x: (x / x.max()))
    df['cls_first_time_abcd'] = match_abcd.groupby(['match_name', 'div', 'class', 'stg'])['time'].transform(lambda x: x[match_abcd['hit_factor'] > 0].min() if any(match_abcd['hit_factor'] > 0) else None)
    df['cls_time_perc_abcd'] = match_abcd['cls_first_time'] / match_abcd['time']
    df['cls_factor_perc_abcd'] = match_abcd.groupby(['match_name', 'div', 'class', 'stg'])['hit_factor'].transform(lambda x: (x / x.max()))
    df['cls_factor_standing_abcd'] = match_abcd.groupby(['match_name', 'div', 'class', 'stg'])['hit_factor'].transform(lambda x: x.rank(method='first', ascending=False))

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
    min_class_n: int = 8,       # min observations to define a class band in a match/div
    min_stage_n: int = 3,       # min stages to compute a shooter median
    min_stage_n_consistency: int = 5,  # min stages to trust shooter IQR for consistency
    consistency_threshold: float = 0.7 # ≥70% of shooter IQR inside class IQR → "consistent"
) -> pd.DataFrame:
    """
    Required columns in `stages`:
      ['match_name','div','class','shooter','div_factor_perc']

    Returns one row per (shooter, match_name, div) with:
      ['shooter','match_name','div','n_stages','sh_median',
       'pred_class','relation','dist_to_class_median',
       'q1','class_median','q3','n_in_class',
       'robust_z_class',
       'sh_q1','sh_q3','sh_iqr','consistency_cover','consistent']
    """
    needed = {'match_name','div','class','shooter','div_factor_perc'}
    miss = needed - set(stages.columns)
    if miss:
        raise KeyError(f"Missing columns: {sorted(miss)}")

    df = stages.copy()
    df['div_factor_perc'] = pd.to_numeric(df['div_factor_perc'], errors='coerce')

    # --- Shooter summary per (shooter, match, div)
    shooter_sum = (
        df.groupby(['shooter','match_name','div'])['div_factor_perc']
          .agg(n_stages='count', sh_median='median')
          .reset_index()
    )
    shooter_sum = shooter_sum[shooter_sum['n_stages'] >= min_stage_n].copy()
    if shooter_sum.empty:
        return pd.DataFrame(columns=[
            'shooter','match_name','div','n_stages','sh_median',
            'pred_class','relation','dist_to_class_median',
            'q1','class_median','q3','n_in_class',
            'robust_z_class','sh_q1','sh_q3','sh_iqr','consistency_cover','consistent'
        ])

    # --- Class bands per (match, div, class)
    class_bands = (
        df.groupby(['match_name','div','class'])['div_factor_perc']
          .agg(n_in_class='count',
               q1=lambda s: s.quantile(0.25),
               class_median='median',
               q3=lambda s: s.quantile(0.75))
          .reset_index()
    )
    class_bands = class_bands[class_bands['n_in_class'] >= min_class_n].copy()
    if class_bands.empty:
        out = shooter_sum.copy()
        out[['pred_class','relation','dist_to_class_median',
             'q1','class_median','q3','n_in_class',
             'robust_z_class','sh_q1','sh_q3','sh_iqr','consistency_cover','consistent']] = np.nan
        return out

    # --- Join candidates (all classes in same match/div for each shooter summary)
    cand = shooter_sum.merge(class_bands, on=['match_name','div'], how='left')
    cand = cand[cand['q1'].notna()].copy()
    if cand.empty:
        out = shooter_sum.copy()
        out[['pred_class','relation','dist_to_class_median',
             'q1','class_median','q3','n_in_class',
             'robust_z_class','sh_q1','sh_q3','sh_iqr','consistency_cover','consistent']] = np.nan
        return out

    # Selection signals
    cand['within_iqr'] = (cand['sh_median'] >= cand['q1']) & (cand['sh_median'] <= cand['q3'])
    cand['dist_to_class_median'] = (cand['sh_median'] - cand['class_median']).abs()

    # Rank: within_IQR first, then closest median
    cand['_within_rank'] = np.where(cand['within_iqr'], 0, 1)

    picked = (
        cand.sort_values(['shooter','match_name','div','_within_rank','dist_to_class_median'])
            .groupby(['shooter','match_name','div'], as_index=False)
            .first()
    )

    # Relation vs chosen class band
    def _relation(row):
        if row['sh_median'] < row['q1']:
            return 'Below Class IQR'
        if row['sh_median'] > row['q3']:
            return 'Above Class IQR'
        return 'Within Class IQR'

    picked['relation'] = picked.apply(_relation, axis=1)

    # Robust z vs chosen class band
    iqr = (picked['q3'] - picked['q1']).replace(0, np.nan)
    robust_scale = iqr / 1.349
    picked['robust_z_class'] = (picked['sh_median'] - picked['class_median']) / robust_scale

    # --- Consistency: shooter IQR (only if enough stages)
    # compute per (shooter, match, div)
    shooter_iqr = (
        df.groupby(['shooter','match_name','div'])['div_factor_perc']
          .agg(n='count',
               sh_q1=lambda s: s.quantile(0.25),
               sh_q3=lambda s: s.quantile(0.75))
          .reset_index()
    )
    shooter_iqr = shooter_iqr[shooter_iqr['n'] >= min_stage_n_consistency].copy()
    shooter_iqr['sh_iqr'] = shooter_iqr['sh_q3'] - shooter_iqr['sh_q1']
    shooter_iqr = shooter_iqr.drop(columns='n')

    out = picked.merge(shooter_iqr, on=['shooter','match_name','div'], how='left')

    # consistency_cover = proportion of shooter IQR inside chosen class IQR
    # = length( intersection( [sh_q1, sh_q3], [q1, q3] ) ) / max(sh_iqr, eps)
    eps = 1e-12
    inter_len = np.maximum(0.0, np.minimum(out['sh_q3'], out['q3']) - np.maximum(out['sh_q1'], out['q1']))
    out['consistency_cover'] = inter_len / (out['sh_iqr'].replace(0, np.nan) + eps)
    # consistent if shooter has IQR and ≥ threshold
    out['consistent'] = np.where(out['sh_iqr'].notna() & (out['consistency_cover'] >= consistency_threshold),
                                 True, False)

    # Final tidy
    out = out.rename(columns={'class': 'pred_class'})
    cols = ['shooter','match_name','div','n_stages','sh_median',
            'pred_class','relation','dist_to_class_median',
            'q1','class_median','q3','n_in_class',
            'robust_z_class','sh_q1','sh_q3','sh_iqr','consistency_cover','consistent']
    return out[cols]

def class_predict_per_stage(
    stages: pd.DataFrame,
    *,
    min_class_n: int = 8  # min observations to define a class band in a match/div
) -> pd.DataFrame:
    """
    Stage-level class prediction.

    Required columns in `stages`:
      ['match_name','div','class','shooter','div_factor_perc']
    Optional: 'stg' (stage id/number). If present, it will be kept in the output.

    Returns one row per input stage with:
      ['shooter','match_name','div',('stg' if present),'div_factor_perc',
       'orig_class','pred_class','relation','dist_to_class_median',
       'q1','class_median','q3','n_in_class','robust_z_class']
    """
    needed = {'match_name','div','class','shooter','div_factor_perc'}
    miss = needed - set(stages.columns)
    if miss:
        raise KeyError(f"Missing columns: {sorted(miss)}")

    df = stages.copy()
    df['div_factor_perc'] = pd.to_numeric(df['div_factor_perc'], errors='coerce')

    # Build class bands per (match, div, class)
    class_bands = (
        df.groupby(['match_name','div','class'])['div_factor_perc']
          .agg(n_in_class='count',
               q1=lambda s: s.quantile(0.25),
               class_median='median',
               q3=lambda s: s.quantile(0.75))
          .reset_index()
    )
    class_bands = class_bands[class_bands['n_in_class'] >= min_class_n].copy()

    # Base columns to return
    base_cols = ['shooter','match_name','div','div_factor_perc','orig_class']
    if 'stg' in df.columns:
        base_cols.insert(3, 'stg')  # ... div, stg, div_factor_perc, orig_class

    if class_bands.empty:
        out = df.copy()
        out = out.rename(columns={'class': 'orig_class'})
        out = out[ [c for c in base_cols if c in out.columns] ]
        out[['pred_class','relation','dist_to_class_median',
             'q1','class_median','q3','n_in_class','robust_z_class']] = np.nan
        return out

    # Row id to pick best candidate per stage row
    df = df.reset_index(drop=False).rename(columns={'index': '__rowid'})

    # Merge with explicit suffixes to avoid class_x/class_y confusion
    cand = df.merge(class_bands,
                    on=['match_name','div'],
                    how='left',
                    suffixes=('_row','_band'))

    # Rename the original and band class columns
    if 'class_row' in cand.columns:
        cand = cand.rename(columns={'class_row': 'orig_class'})
    else:
        cand = cand.rename(columns={'class': 'orig_class'})  # fallback if suffixing didn't occur

    if 'class_band' in cand.columns:
        cand = cand.rename(columns={'class_band': 'pred_class'})
    else:
        # If suffixing didn’t occur for some reason, assume band class is the remaining 'class'
        cand = cand.rename(columns={'class': 'pred_class'})

    # Drop rows where we have no valid band stats
    cand = cand[cand['q1'].notna()].copy()
    if cand.empty:
        out = df.rename(columns={'class': 'orig_class'})
        out = out[ [c for c in base_cols + ['__rowid'] if c in out.columns] ]
        out[['pred_class','relation','dist_to_class_median',
             'q1','class_median','q3','n_in_class','robust_z_class']] = np.nan
        return out.drop(columns='__rowid')

    # Signals vs each candidate band
    perf = cand['div_factor_perc']
    cand['within_iqr'] = (perf >= cand['q1']) & (perf <= cand['q3'])
    cand['dist_to_class_median'] = (perf - cand['class_median']).abs()
    cand['_within_rank'] = np.where(cand['within_iqr'], 0, 1)

    # Pick best band per stage: inside-IQR first, then closest median
    picked = (
        cand.sort_values(['__rowid','_within_rank','dist_to_class_median'])
            .groupby('__rowid', as_index=False)
            .first()
    )

    # Relation for the stage value vs chosen band
    def _relation(row):
        val = row['div_factor_perc']
        if val < row['q1']:
            return 'below'
        if val > row['q3']:
            return 'above'
        return 'within'
    picked['relation'] = picked.apply(_relation, axis=1)

    # Robust z for the stage value vs chosen band
    iqr = (picked['q3'] - picked['q1']).replace(0, np.nan)
    robust_scale = iqr / 1.349
    picked['robust_z_class'] = (picked['div_factor_perc'] - picked['class_median']) / robust_scale

    # Final tidy
    keep = [c for c in base_cols if c in picked.columns] + [
        'pred_class','relation','dist_to_class_median',
        'q1','class_median','q3','n_in_class','robust_z_class'
    ]
    out = picked[keep].copy()

    return out
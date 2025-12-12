import matplotlib.pyplot as plt
import seaborn as sns
import scipy.stats as stats
import numpy as np
import pandas as pd
from itertools import combinations as comb

def performance_ci_overlapping(cis: pd.DataFrame):
    pairs = list(comb(cis.index, 2))
    combinations = []
    ious = []
    overlapab = []
    overlapba = []
    dice = []
    for a, b in pairs:
        ci = cis.loc[[a, b], ['ci_low', 'ci_high']].astype(float)

        # sanity: ensure low <= high
        ci[['ci_low','ci_high']] = pd.DataFrame({
            'ci_low':  ci[['ci_low','ci_high']].min(axis=1),
            'ci_high': ci[['ci_low','ci_high']].max(axis=1),
        }, index=ci.index)

        lows  = ci['ci_low'].values
        highs = ci['ci_high'].values

        # lengths
        len_a = highs[0] - lows[0]
        len_b = highs[1] - lows[1]

        # union and intersection
        union = max(highs) - min(lows)
        inter = max(0.0, min(highs) - max(lows))   # clamp to 0 (no negative overlap)

        # metrics
        iou_pct    = (inter / union * 100) if union > 0 else 0.0               # your current notion
        over_a_pct = (inter / len_a * 100) if len_a > 0 else 0.0               # overlap as % of A
        over_b_pct = (inter / len_b * 100) if len_b > 0 else 0.0               # overlap as % of B
        dice_pct   = (2*inter / (len_a + len_b) * 100) if (len_a+len_b)>0 else 0.0  # Sørensen–Dice

        # print(
        #     f"Comparing {a} vs {b} | IoU %: {iou_pct:.2f} | "
        #     f"Overlap%AinB: {over_a_pct:.2f} | Overlap%BinA: {over_b_pct:.2f} | Dice %: {dice_pct:.2f}"
        # )
        combinations.append((a, b))
        ious.append(iou_pct)
        overlapab.append(over_a_pct)
        overlapba.append(over_b_pct)
        dice.append(dice_pct)
    cis_comparisons = pd.DataFrame({
        
        'iou_pct': ious,
        'overlap_a_in_b_pct': overlapab,
        'overlap_b_in_a_pct': overlapba,
        'dice_pct': dice
    }, index=pd.MultiIndex.from_tuples(combinations, names=['shooter_a', 'shooter_b']))
    return cis_comparisons

def comparative_match_analysis(data, match, shooters: list):
    data = data[(data['match_name'] == match) & (data['shooter'].isin(shooters))]

    # Create figure with two subplots: 3 parts width for line plot, 1 part for scatter
    fig, (ax1, ax2) = plt.subplots(
        1, 2,
        figsize=(16, 6),
        gridspec_kw={'width_ratios': [2.5, 1.5]},
        dpi=300
    )

    # --- Line Plot ---
    y = 'div_factor_perc'
    sns.lineplot(
        data=data,
        x='stg',
        y=y,
        hue='shooter',
        style='shooter',
        markers=True,
        dashes=False,
        legend='full',
        ax=ax1
    )

    # Add mean lines and CI boxes
    palette = sns.color_palette()
    shooters_ = []
    ci_lows = []
    ci_highs = []
    for i, shooter in enumerate(data['shooter'].unique()):
        vals = data.loc[data['shooter'] == shooter, y]
        mean_val = vals.mean()
        
        # 95% CI for the mean
        sem = stats.sem(vals)
        ci_low, ci_high = stats.t.interval(0.95, len(vals)-1, loc=mean_val, scale=sem)

        # CI box shading
        ax1.axhspan(ci_low, ci_high, color=palette[i], alpha=0.15)
        
        # Mean line
        ax1.axhline(mean_val, color=palette[i], linestyle='--', alpha=0.7)
        
        # Store for legend
        shooters_.append(shooter)
        ci_lows.append(ci_low)
        ci_highs.append(ci_high)

    # Rotate x-axis labels without warning
    plt.setp(ax1.get_xticklabels(), rotation=45)

    ax1.grid(True, linestyle='--', alpha=0.7, linewidth=0.5)
    ax1.set_xlabel('Stage')
    ax1.set_ylabel('Division Performance (%)')
    ax1.set_title('Performance by Stage (%)')
    ax1.xaxis.label.set_visible(False)  # hides it

    # --- Scatter Plot ---
    hue_order = list(data['shooter'].unique())  # keep current order
    palette = sns.color_palette(n_colors=len(hue_order))
    color_map = {s: palette[i] for i, s in enumerate(hue_order)}

    sns.scatterplot(
        data=data,
        x='div_time_perc',
        y='div_pts_perc',
        hue='shooter',
        style='shooter',
        hue_order=hue_order,
        palette=color_map,
        markers=True,
        alpha=0.5,
        s=100,
        ax=ax2
    )

    # 🔹 Annotate with stage ("stg")
    for _, row in data.iterrows():
        ax2.annotate(
            row['stg'].replace('Stage ', ''),                           # text
            (row['div_time_perc'], row['div_pts_perc']),  # xy position
            textcoords="offset points",
            xytext=(4, 4),                        # small offset so text isn’t on top of point
            fontsize=7,
            color=color_map[row['shooter']]       # match shooter color
        )

    # --- Centroids + 95% CI crosses ---
    time_centroids = []
    pts_centroids = []
    for shooter in hue_order:
        g = data.loc[data['shooter'] == shooter, ['div_time_perc', 'div_pts_perc']].dropna()
        if g.empty:
            continue

        mean_x = g['div_time_perc'].mean()
        time_centroids.append(mean_x)
        mean_y = g['div_pts_perc'].mean()
        pts_centroids.append(mean_y)

        # standard errors (protect against n=1)
        n = len(g)
        sem_x = stats.sem(g['div_time_perc']) if n > 1 else 0.0
        sem_y = stats.sem(g['div_pts_perc'])  if n > 1 else 0.0

        # 95% t-intervals (fall back to point if n=1)
        if n > 1 and np.isfinite(sem_x):
            ci_x_low, ci_x_high = stats.t.interval(0.95, n-1, loc=mean_x, scale=sem_x)
        else:
            ci_x_low = ci_x_high = mean_x

        if n > 1 and np.isfinite(sem_y):
            ci_y_low, ci_y_high = stats.t.interval(0.95, n-1, loc=mean_y, scale=sem_y)
        else:
            ci_y_low = ci_y_high = mean_y

        # plot centroid with CI error bars
        c = color_map[shooter]
        ax2.errorbar(
            mean_x, mean_y,
            xerr=[[mean_x - ci_x_low], [ci_x_high - mean_x]],
            yerr=[[mean_y - ci_y_low], [ci_y_high - mean_y]],
            fmt='o',
            markersize=15,
            color=c,          # marker color
            ecolor=c,         # error bar color
            elinewidth=1.2,
            capsize=4,
            markeredgecolor='white',
            zorder=6
        )
        # optional label next to centroid
        ax2.annotate(shooter, (mean_x, mean_y), textcoords='offset points',
                    xytext=(6, 6), fontsize=8, color=c)

    ax2.grid(True, linestyle='--', alpha=0.7, linewidth=0.5)
    ax2.set_xlabel('Time (%)')
    ax2.set_ylabel('Points (%)')
    ax2.set_title('Points vs Time (%) by Stage')
    
    # # keep legend only on ax1 as you had
    # handles, labels = ax1.get_legend_handles_labels()
    # ax1.legend(handles, labels, loc='best')
    ax2.legend().set_visible(False)
    plt.tight_layout()
    plt.suptitle(f'Comparative Performance Analysis\nMatch: {match}', fontsize=14, y=1.05)

    cis = pd.DataFrame({
        'ci_low': ci_lows,
        'ci_high': ci_highs
    }, index=shooters_)
    
    centroids = pd.DataFrame({
        'time_centroid': time_centroids,
        'pts_centroid': pts_centroids
    }, index=shooters_)
    centroids['overall'] = centroids.mean(axis=1)
    return cis, centroids

def relative_performance(data: pd.DataFrame,
                         match,
                         shooter_a: str,
                         shooter_b: str,
                         stage_points_col: str | None = None,
                         metric: str = "geom",          # "geom" | "arith" | "both"
                         include_total_points: bool = True):
    df = data[(data['match_name'] == match) &
              (data['shooter'].isin([shooter_a, shooter_b]))] \
             .pivot(index='stg', columns='shooter', values='div_factor_perc') \
             .reset_index().copy()

    for c in [shooter_a, shooter_b]:
        df[c] = pd.to_numeric(df[c], errors="coerce")

    eps = 1e-12
    df["ratio_A_over_B"] = df[shooter_a] / (df[shooter_b] + eps)
    df["ratio_B_over_A"] = df[shooter_b] / (df[shooter_a] + eps)

    w = None
    if stage_points_col and stage_points_col in df.columns:
        w = pd.to_numeric(df[stage_points_col], errors="coerce").fillna(0).clip(lower=0).to_numpy()

    a_over_b = df["ratio_A_over_B"].to_numpy()
    b_over_a = df["ratio_B_over_A"].to_numpy()

    def arith_mean(x, w=None):
        return np.nanmean(x) if w is None else np.average(x, weights=w)

    def geom_mean(x, w=None):
        m = (~np.isnan(x)) & (x > 0)
        if not np.any(m): return np.nan
        return np.exp(np.nanmean(np.log(x[m]))) if w is None else np.exp(np.average(np.log(x[m]), weights=w[m]))

    summary = {"shooter_A": shooter_a, "shooter_B": shooter_b}

    if metric in ("geom", "both"):
        summary["A_over_B_mean_geom"] = geom_mean(a_over_b)
        summary["B_over_A_mean_geom"] = geom_mean(b_over_a)
        if w is not None:
            summary["A_over_B_weighted_geom"] = geom_mean(a_over_b, w)
            summary["B_over_A_weighted_geom"] = geom_mean(b_over_a, w)

    if metric in ("arith", "both"):
        summary["A_over_B_mean_arith"] = arith_mean(a_over_b)
        summary["B_over_A_mean_arith"] = arith_mean(b_over_a)
        if w is not None:
            summary["A_over_B_weighted_arith"] = arith_mean(a_over_b, w)
            summary["B_over_A_weighted_arith"] = arith_mean(b_over_a, w)

    if include_total_points and (w is not None):
        tot_A = np.nansum(df[shooter_a].to_numpy() * w)
        tot_B = np.nansum(df[shooter_b].to_numpy() * w)
        summary["A_vs_B_total_points_ratio"] = (tot_A / tot_B) if tot_B > 0 else np.nan
        summary["B_vs_A_total_points_ratio"] = (tot_B / tot_A) if tot_A > 0 else np.nan

    stage_table = df[["stg", shooter_a, shooter_b, "ratio_A_over_B", "ratio_B_over_A"]]
    return stage_table, pd.Series(summary)

def stage_performance_across_matches(data, shooter_name, stage_alpha=0.5, matches: list = None):
    df = data[data['shooter'] == shooter_name].copy()
    
    if matches is not None:
        if type(matches) is list and len(matches) != 0:
            df = df[df['match_name'].isin(matches)]
    else:
        matches = df['match_name'].unique()
        
    fig, (ax1, ax2) = plt.subplots(
        1, 2,
        figsize=(16, 6),
        gridspec_kw={'width_ratios': [2, 2]},
        dpi=300
    )
    
    # --- Scatter Plot ---
    hue_order = list(df['match_name'].unique())
    palette = sns.color_palette(n_colors=len(hue_order))
    color_map = {s: palette[i] for i, s in enumerate(hue_order)}

    sns.scatterplot(
        data=df,
        x='div_time_perc',
        y='div_pts_perc',
        hue='match_name',
        hue_order=hue_order,
        palette=color_map,
        markers=True,
        alpha=stage_alpha,
        s=100,
        ax=ax1
    )
    # # 🔹 Annotate each point with its stage ("stg")
    # for _, row in df.iterrows():
    #     ax1.annotate(
    #         row['stg'],                                # text
    #         (row['div_time_perc'], row['div_pts_perc']),  # xy position
    #         textcoords="offset points",
    #         xytext=(4, 4),                             # small offset
    #         fontsize=7,
    #         color=color_map[row['match_name']]         # match match_name color
        # )
    # --- Centroids + CI bars ---
    for match in hue_order:
        g = df.loc[df['match_name'] == match, ['div_time_perc', 'div_pts_perc']].dropna()
        if g.empty:
            continue

        mean_x = g['div_time_perc'].mean()
        mean_y = g['div_pts_perc'].mean()

        n = len(g)
        sem_x = stats.sem(g['div_time_perc']) if n > 1 else 0.0
        sem_y = stats.sem(g['div_pts_perc']) if n > 1 else 0.0

        if n > 1 and np.isfinite(sem_x):
            ci_x_low, ci_x_high = stats.t.interval(0.95, n-1, loc=mean_x, scale=sem_x)
        else:
            ci_x_low = ci_x_high = mean_x

        if n > 1 and np.isfinite(sem_y):
            ci_y_low, ci_y_high = stats.t.interval(0.95, n-1, loc=mean_y, scale=sem_y)
        else:
            ci_y_low = ci_y_high = mean_y

        c = color_map[match]
        ax1.errorbar(
            mean_x, mean_y,
            xerr=[[mean_x - ci_x_low], [ci_x_high - mean_x]],
            yerr=[[mean_y - ci_y_low], [ci_y_high - mean_y]],
            fmt='o',
            markersize=15,
            color=c,
            ecolor=c,
            elinewidth=1.2,
            capsize=4,
            markeredgecolor='white',
            zorder=6
        )
        # Optional label next to centroid
        ax1.annotate(match, (mean_x, mean_y), textcoords='offset points',
                    xytext=(6, 6), fontsize=8, color=c)

    # --- Reference lines ---
    ax1.axhline(0.5, color='gray', linestyle='--', linewidth=1, alpha=0.7)
    ax1.axvline(0.5, color='gray', linestyle='--', linewidth=1, alpha=0.7)

    # --- Axis limits & labels ---
    ax1.set_xlim(0, 1)
    ax1.set_ylim(0, 1)
    ax1.grid(True, linestyle='--', alpha=0.7, linewidth=0.5)
    ax1.set_xlabel('Time (%)')
    ax1.set_ylabel('Points (%)')
    ax1.set_title('Stage Performance Points vs Time (%)')
    ax1.legend(loc='upper left')
    
    # --- Box Plot ---
    sns.boxplot(
        data=df,
        x='match_name',
        y='div_factor_perc',
        hue='match_name',
        ax=ax2
    )

    plt.setp(ax2.get_xticklabels(), rotation=45)
    ax2.set_ylim(0, 1)
    ax2.axhline(0.5, color='gray', linestyle='--', linewidth=1, alpha=0.7)
    ax2.grid(True, axis='y', linestyle='--', alpha=0.7, linewidth=0.5)
    ax2.set_title('Stage Performance Distribution')
    # ax2.set_xlabel('das')
    ax2.xaxis.label.set_visible(False)  # hides it
    ax2.set_ylabel('Division Performance (%)')
    
    plt.tight_layout()
    plt.suptitle(f'Stage Performance Across Matches\n{df['shooter'].unique()[0]}', fontsize=14, y=1.05)

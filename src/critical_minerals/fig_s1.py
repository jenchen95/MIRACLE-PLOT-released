import pandas as pd
import matplotlib
matplotlib.use('Agg') # 适用于无界面环境
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.gridspec as gridspec
import numpy as np
import string

# ---- 1. 配置与路径 ----
FILE_PATH = r"E:\Pythoncode\Plot\data\data_output\emissions_contributions_add_detailed.parquet"
OUT_PATH = r"E:\Pythoncode\Plot\data\data_output\emissions_bar_sorted.png"

STRAT_ORDER = ['bau_bau_bau_recycl_nosub', 'bau_llt_mir_recych_nosub',
               'bau_bau_bau_recycl_sub', 'allo_bau_bau_recycl_nosub',
               'allo_llt_mir_recych_sub']
STRAT_SHORT = ['Ref', 'Circ.', 'Subst.', 'Alloc.', 'All-in']
STRAT_FULL  = ['Ref', 'Circular', 'Substitution', 'Allocation', 'All-in']

REGIONS = ['R10AFRICA','R10CHINA+','R10EUROPE','R10INDIA+','R10LATIN_AM',
           'R10MIDDLE_EAST','R10NORTH_AM','R10PAC_OECD','R10REF_ECON','R10REST_ASIA']

REG_LABELS = {
    'R10AFRICA': 'Africa', 'R10CHINA+': 'China+', 'R10EUROPE': 'Europe',
    'R10INDIA+': 'India+', 'R10LATIN_AM': 'Latin Am', 'R10MIDDLE_EAST': 'Middle East',
    'R10NORTH_AM': 'North Am', 'R10PAC_OECD': 'Pac OECD', 'R10REF_ECON': 'Ref Econ',
    'R10REST_ASIA': 'Rest Asia',
}

REG_COLORS = {
    'R10AFRICA': '#e6194b', 'R10CHINA+': '#f58231', 'R10EUROPE': '#3cb44b',
    'R10INDIA+': '#4363d8', 'R10LATIN_AM': '#42d4f4', 'R10MIDDLE_EAST': '#f032e6',
    'R10NORTH_AM': '#a9a9ff', 'R10PAC_OECD': '#9A6324', 'R10REF_ECON': '#800000',
    'R10REST_ASIA': '#469990',
}

GLOBAL_COLOR = '#555555'

# 图题与轴标签（数据列为 add_contribution_gt：相对基准的额外量，用语与 emissions gap 一致）
YLABEL_GLOBAL_LONG = "Cumulative CO₂ emissions gap (Gt CO₂)"
YLABEL_REGION_SHORT = "Emissions gap (Gt CO₂)"
FIG_SUPTITLE_MAIN = "Cumulative CO₂ emissions gap by mitigation strategy and region — {cat}"
FIG_SUPTITLE_SUB = (
    "Additional CO₂ vs. baseline pathways · bar = median · "
    "error bars = 25th–75th percentile · regions sorted by Ref median"
)

# 字号：只改 FS_SCALE 即可整体放大/缩小子图与图例文字（1.0 ≈ 原先比例）
FS_SCALE = 1.2
FONT = {
    'xtick_global': 10 * FS_SCALE,
    'xtick_region': 10 * FS_SCALE,
    'title_global': 10 * FS_SCALE,
    'title_region': 10 * FS_SCALE,
    'ylabel_long': 10 * FS_SCALE,
    'ylabel_short': 10 * FS_SCALE,
    'ytick': 10 * FS_SCALE,
    'legend': 12 * FS_SCALE,
    'suptitle': 10 * FS_SCALE,
    'legend_marker': 10 * FS_SCALE,
    'panel_label': 11 * FS_SCALE,
}

# ---- 2. 数据加载与统计处理 ----
df = pd.read_parquet(FILE_PATH)

# 过滤策略
df_f = df[df['strategy'].isin(STRAT_ORDER)].copy()

# category 顺序（将输出 8 张图：C1..C8；若数据缺失则自动跳过）
CAT_ORDER = [f"C{i}" for i in range(1, 9)]
cats_present = set(df_f["category"].dropna().unique().tolist()) if "category" in df_f.columns else set()
CATS_TO_PLOT = [c for c in CAT_ORDER if c in cats_present] or CAT_ORDER

# ---- 3. 绘图函数 ----
def draw_panel(ax, stats_list, title, bar_color, ylabel=None, ref=None, 
               x_labels=None, is_global=False, ylim=None):
    n = len(stats_list)
    x = np.arange(n)
    bw = 0.58
    cap_size = 4 if is_global else 3

    for i, (med, p25, p75) in enumerate(stats_list):
        c = '#aaaaaa' if i == 0 else bar_color # 第一个柱子灰色，其余彩色
        ax.bar(x[i], med, width=bw, color=c, alpha=0.85, zorder=3, linewidth=0.6, edgecolor='white')
        
        # 误差棒 (25th-75th)
        ax.errorbar(x[i], med, yerr=[[med - p25], [p75 - med]], 
                    fmt='none', color='#222222', linewidth=1.4, capsize=cap_size, capthick=1.4, zorder=5)

    if ref is not None:
        ax.axhline(ref, color='#333333', lw=1.1, ls='--', dashes=(5, 4), zorder=6, alpha=0.85)

    ax.axhline(0, color='#999999', lw=0.5, zorder=1)
    ax.set_xlim(-0.55, n - 0.45)
    if ylim: ax.set_ylim(ylim)

    ax.set_xticks(x)
    if x_labels:
        ax.set_xticklabels(
            x_labels,
            fontsize=FONT['xtick_global'] if is_global else FONT['xtick_region'],
            rotation=45, ha='right')
    
    ax.set_title(title, fontsize=FONT['title_global'] if is_global else FONT['title_region'],
                 fontweight='bold', pad=3)
    if ylabel: ax.set_ylabel(ylabel, fontsize=FONT['ylabel_long'])
    ax.tick_params(axis='y', labelsize=FONT['ytick'])
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.grid(axis='y', lw=0.4, alpha=0.4, color='#888888', zorder=0)

def _build_out_path_for_category(base_out_path: str, cat: str) -> str:
    if not isinstance(base_out_path, str) or "." not in base_out_path:
        return base_out_path
    stem, ext = base_out_path.rsplit(".", 1)
    return f"{stem}_{cat}.{ext}"


for cat in CATS_TO_PLOT:
    df_cat = df_f[df_f["category"] == cat].copy() if "category" in df_f.columns else df_f.copy()

    # A. 区域统计（按 category 单独算）
    reg_grouped = (
        df_cat.groupby(["region", "strategy"])["add_contribution_gt"]
        .quantile([0.5, 0.25, 0.75])
        .unstack()
    )

    reg_stats = {}
    for reg in REGIONS:
        stats_list = []
        for s in STRAT_ORDER:
            try:
                row = reg_grouped.loc[(reg, s)]
                stats_list.append((row[0.5], row[0.25], row[0.75]))
            except KeyError:
                stats_list.append((0, 0, 0))
        reg_stats[reg] = stats_list

    # B. 全球统计：每个 path_abb 内部把 10 区域加总，再在 strategy 内对 path_abb 取分位数（不混入其它 category）
    global_path_sums = (
        df_cat.groupby(["strategy", "path_abb"])["add_contribution_gt"]
        .sum()
    )
    global_qs = global_path_sums.groupby("strategy").quantile([0.5, 0.25, 0.75]).unstack()

    global_stat_list = []
    for s in STRAT_ORDER:
        if s in global_qs.index:
            row = global_qs.loc[s]
            global_stat_list.append((row[0.5], row[0.25], row[0.75]))
        else:
            global_stat_list.append((0, 0, 0))

    # C. 区域排序：按 Ref 的中位数降序排（按当前 category）
    region_sorted = sorted(REGIONS, key=lambda r: reg_stats[r][0][0], reverse=True)

    # ---- 4. 画布布局 ----
    fig = plt.figure(figsize=(16, 9))
    fig.patch.set_facecolor("white")

    gs = gridspec.GridSpec(
        2, 6, figure=fig, width_ratios=[1.8, 1, 1, 1, 1, 1],
        hspace=0.45, wspace=0.28, left=0.06, right=0.98, top=0.88, bottom=0.14
    )

    # 全球面板
    ax_g = fig.add_subplot(gs[:, 0])
    draw_panel(
        ax_g, global_stat_list, "Global", bar_color=GLOBAL_COLOR,
        ylabel=YLABEL_GLOBAL_LONG, ref=None,
        x_labels=STRAT_FULL, is_global=True,
    )

    # 区域面板
    reg_axes = [fig.add_subplot(gs[i // 5, (i % 5) + 1]) for i in range(10)]
    all_p75 = [reg_stats[r][i][2] for r in REGIONS for i in range(5)]
    all_p25 = [reg_stats[r][i][1] for r in REGIONS for i in range(5)]
    ylim_reg = (min(0, min(all_p25)) * 1.05, max(all_p75) * 1.12)

    for i, (ax, reg) in enumerate(zip(reg_axes, region_sorted)):
        draw_panel(ax, reg_stats[reg], REG_LABELS[reg], bar_color=REG_COLORS[reg],
                   x_labels=STRAT_SHORT, ylim=ylim_reg)
        if i in (0, 5):
            ax.set_ylabel(YLABEL_REGION_SHORT, fontsize=FONT["ylabel_short"])

    # 子图序号：a,b,c,...（Global + 10 regions）
    for li, ax in enumerate([ax_g] + reg_axes):
        lab = string.ascii_lowercase[li]
        ax.text(
            -0.1, 1.07, lab,
            transform=ax.transAxes, va="top", ha="left",
            fontsize=FONT["panel_label"], fontweight="bold",
        )

    # ---- 5. 图例与标题 ----
    legend_handles = [mpatches.Patch(color="#aaaaaa", alpha=0.9, label="Ref — all regions")]
    for reg in region_sorted:
        legend_handles.append(mpatches.Patch(color=REG_COLORS[reg], alpha=0.85, label=REG_LABELS[reg]))

    legend_handles += [
        plt.Line2D([0], [0], color="#222", lw=1.8, label="Median bar"),
        plt.Line2D([0], [0], color="#222", lw=1.4, marker="_", markersize=FONT["legend_marker"], label="25th–75th pct"),
    ]

    # fig.legend(
    #     handles=legend_handles, loc="lower center", ncol=7, fontsize=FONT["legend"],
    #     frameon=False, bbox_to_anchor=(0.5, -0.05),
    # )
    fig.suptitle(
        f"{FIG_SUPTITLE_MAIN.format(cat=cat)}\n{FIG_SUPTITLE_SUB}",
        fontsize=FONT["suptitle"], fontweight="bold", y=0.955,
    )

    # 保存（每个 category 一张图）
    out_path_cat = _build_out_path_for_category(OUT_PATH, cat)
    plt.savefig(out_path_cat, dpi=160, bbox_inches="tight", facecolor="white")
    plt.close()
    print(f"Success! Plot saved to: {out_path_cat}")
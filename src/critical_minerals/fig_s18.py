import polars as pl
import ultraplot as uplt
from funcs import fit_linear
from matplotlib.patches import Patch
from matplotlib.lines import Line2D
import numpy as np

# Strategy configuration: colors and display names
strategy_config = {
    "bau_bau_bau_recycl_nosub": {"color": "#c33b34", "name": "Ref"},
    "bau_llt_mir_recych_nosub": {"color": "#ef8641", "name": "Circular"},
    "bau_bau_bau_recycl_sub": {"color": "#f09878", "name": "Substitution"},
    "bau_llt_mir_recych_sub": {"color": "#f7ce7c", "name": "Circular\nSubstitution"},
    "allo_bau_bau_recycl_nosub": {"color": "#f8d653", "name": "Allocation"},
    "allo_llt_mir_recych_nosub": {"color": "#7cc4d2", "name": "Allocation\nCircular"},
    "allo_bau_bau_recycl_sub": {"color": "#93d2ad", "name": "Allocation\nSubstitution"},
    "allo_llt_mir_recych_sub": {"color": "#c0eb96", "name": "All-in"},
}

# Element color mapping for line plots
element_colors = {
    "Li": "#1f77b4",  # Blue
    "Co": "#ff7f0e",  # Orange
    "Ni": "#2ca02c",  # Green
    "Mn": "#d62728",  # Red
    "Nd": "#9467bd",  # Purple
    "Dy": "#8c564b",  # Brown
    "Ga": "#e377c2",  # Pink
    "In": "#7f7f7f",  # Gray
    "Ge": "#bcbd22",  # Yellow-green
    "Te": "#17becf",  # Cyan
    "Mg": "#ff9896",  # Light red
    "Cr": "#e7298a",  # Magenta
    "Cu": "#66a61e",  # Olive green
    "Ag": "#e6ab02",  # Gold
    "Se": "#a6761d",  # Brown-orange
}

# Region color mapping for line plots
region_colors = {
    "R10AFRICA": "#e41a1c",
    "R10CHINA+": "#377eb8",
    "R10EUROPE": "#4daf4a",
    "R10INDIA+": "#984ea3",
    "R10LATIN_AM": "#ff7f00",
    "R10MIDDLE_EAST": "#ffff33",
    "R10NORTH_AM": "#a65628",
    "R10PAC_OECD": "#f781bf",
    "R10REF_ECON": "#999999",
    "R10REST_ASIA": "#66c2a5",
}

strategies = list(strategy_config.keys())

# Load data
print("Loading data...")
abapath_n = (
    pl.scan_parquet("../../data/data_raw/miracle_oscar/abapath_n.parquet")
    .group_by(['path_abb', 'year'])
    .agg(pl.sum('abatement').alias('ideal_abate'))
)

category = (
    pl.scan_csv("../../data/data_raw/miracle_oscar/category.csv")
    .with_columns(
        path_abb=pl.concat_str(pl.col('model'), pl.col('scenario'), separator='|')
    )
    .select(['path_abb', 'category'])
)

# Select 8 strategies for heatmap
selected_strategies = [
    "bau_bau_bau_recycl_nosub",
    "bau_llt_mir_recych_nosub",
    "bau_bau_bau_recycl_sub",
    "bau_llt_mir_recych_sub",
    "allo_bau_bau_recycl_nosub",
    "allo_llt_mir_recych_nosub",
    "allo_bau_bau_recycl_sub",
    "allo_llt_mir_recych_sub",
]

# Load aggregated elasticity for heatmaps
# 选项1：先按年份取中值，再对中值取均值
# 第一步：按 (strategy, element, region, year) 分组，对每个年份的不同 path_abb 取中值
elasticity_by_year = (
    pl.scan_parquet("../../data/data_raw/miracle_oscar/elasticity.parquet")
    .filter(pl.col('strategy').is_in(selected_strategies))
    .join(category, on=['path_abb'], how='left')
    .filter(pl.col('category') == 'C2')
    .filter(pl.col('year') >= 2026)
    # ❌ 这里不要写筛选，保留 0 和微小值参与统计
    .group_by(['strategy','element','region','year'])
    .agg(
        pl.median('semi_elasticity').alias('semi_elasticity_median_by_year'),
        pl.median('dual_value').alias('dual_value_median_by_year'),
        pl.median('abate').alias('abate_median_by_year')
    )
    .collect()
)

# 第二步：按 (strategy, element, region) 分组，对这些年份的中值取均值
elasticity = (
    elasticity_by_year
    .group_by(['strategy','element','region'])
    .agg(
        pl.mean('semi_elasticity_median_by_year').alias('semi_elasticity'),
        pl.mean('dual_value_median_by_year').alias('dual_value'),
        pl.mean('abate_median_by_year').alias('abate')
    )
    # ✅ 聚合完之后再筛选，目的是：如果不具备统计意义（整体太小），就不在热图显示
    .filter(pl.col('semi_elasticity') > 0.01)
)

# Load raw elasticity data for line plots (keep path_abb, strategy, region separate)
elasticity_raw = (
    pl.scan_parquet("../../data/data_raw/miracle_oscar/elasticity.parquet")
    .filter(pl.col('strategy').is_in(strategies))
    .join(category, on=['path_abb'], how='left')
    .filter(pl.col('category') == 'C2')
    .select(['path_abb', 'strategy', 'element', 'region', 'year', 'semi_elasticity'])
    .collect()
)

frame = (
    pl.scan_parquet("../../data/data_raw/miracle_oscar/r10_final_abate_agg.parquet")
    .group_by(['path_abb', 'strategy', 'year'])
    .agg(pl.sum('final_deployment').alias('final_deployment'), pl.sum('abate').alias('abate'), pl.sum('baseline_deployment').alias('baseline_deployment'))
    .join(category, on=["path_abb"], how="left")
    .filter(pl.col("category") == "C2")  # Only C2 category
    .join(abapath_n, on=['path_abb', 'year'], how='left')
    .filter(pl.col("strategy").is_in(strategies))
    .select('category', 'path_abb', 'strategy', 'year', 'final_deployment', 'abate', 'baseline_deployment', 'ideal_abate')
    .collect()
)

frame_list = []
for pairs, df in frame.group_by(['category', 'path_abb', 'strategy']):
    df = df.sort('year')
    df = df.select(
        pl.arange(pl.min('year'), pl.max('year') + 1).alias('year'),
        pl.lit(pairs[0]).alias('category'),
        pl.lit(pairs[1]).alias('path_abb'),
        pl.lit(pairs[2]).alias('strategy'),
        pl.exclude('category', 'path_abb', 'strategy', 'year').map_batches(
            lambda y: fit_linear(y, df['year'])).explode()
        )
    frame_list.append(df)
frame = pl.concat(frame_list)

frame = (
    frame.with_columns([
        pl.col("final_deployment").cum_sum().over(["path_abb", "strategy"]).alias("cum_final_deployment"),
        pl.col('abate').cum_sum().over(['path_abb', 'strategy']).alias('cum_abate'),
        pl.col("baseline_deployment").cum_sum().over(["path_abb", "strategy"]).alias("cum_baseline_deployment"),
        pl.col('ideal_abate').cum_sum().over(['path_abb', 'strategy']).cast(pl.Float64).alias('cum_ideal_abate')
    ])
    .select('category', 'path_abb', 'strategy', 'year', 'cum_final_deployment', 'cum_abate', 'cum_baseline_deployment', 'cum_ideal_abate')
)


strategies_list = sorted(frame['strategy'].unique().to_list())

# Remove single path_abb filter - plot all path_abb
print("Category: C2")
print(f"Total path_abb: {frame['path_abb'].n_unique()}")
print(f"Strategies: {strategies_list}")

uplt.rc["grid"] = False
uplt.rc["abc.size"] = 9
uplt.rc["font.family"] = "Arial"
uplt.rc["legend.fontsize"] = 8

# Prepare data for achievement rate plot
data_2100 = frame.filter(pl.col('year') == 2100)

# Calculate abatement achievement rate
data_2100 = data_2100.with_columns(
    (pl.col('cum_abate') / pl.col('cum_ideal_abate') * 100).alias('achievement_rate')
)

# Sort by achievement rate
data_sorted = data_2100.sort('achievement_rate')


# Prepare bar data
bar_indices = list(range(len(data_sorted)))
bar_heights = data_sorted['achievement_rate'].to_list()
bar_colors = [strategy_config.get(s, {}).get("color", "#999999") for s in data_sorted['strategy'].to_list()]

# ============================================================================
# FIGURE 1: HEATMAPS
# ============================================================================

print("\n" + "="*80)
print("Creating Figure 1: Heatmaps...")


print(f"Filtered elasticity data shape: {elasticity.shape}")
print(f"Strategies: {elasticity['strategy'].unique().to_list()}")
print(f"Elements: {sorted(elasticity['element'].unique().to_list())}")
print(f"Regions: {sorted(elasticity['region'].unique().to_list())}")

# Get union of all elements across the 3 strategies (sorted alphabetically, reversed)
all_elements = sorted(elasticity['element'].unique().to_list(), reverse=True)
print(f"\nUnion of elements (y-axis): {all_elements}")

# Get all unique regions (sorted alphabetically)
all_regions = sorted(elasticity['region'].unique().to_list())
print(f"All regions (x-axis): {all_regions}")

# Find global min/max for shared colormap scale
vmin = elasticity['semi_elasticity'].min()
vmax = elasticity['semi_elasticity'].max()
print(f"\nElasticity range: [{vmin:.4f}, {vmax:.4f}]")

# Create Figure 1: Heatmaps + Bar plot
uplt.rc["grid"] = False
fig1 = uplt.figure(refwidth=8, refheight=1.35, facecolor=(0, 0, 0, 0), share=False, wspace=1)

# Layout: 3 rows - Row 1: bar plot, Row 2-3: 4 heatmaps each
layout1 = [
    [1, 1, 1, 1],      # Bar plot spanning all 4 columns
    [2, 3, 4, 5],      # Heatmaps 1-4
    [6, 7, 8, 9],      # Heatmaps 5-8
]

axes1 = fig1.subplots(layout1, hratios=[0.7, 1, 1], wratios=[1, 1, 1, 1])

# Plot each strategy heatmap
for idx, strategy in enumerate(selected_strategies):
    # Heatmaps are at axes1[1] through axes1[6] (indices 1-6)
    ax = axes1[idx + 1]

    # Filter data for this strategy
    strategy_data = elasticity.filter(pl.col('strategy') == strategy)

    # Create matrix: rows = elements, cols = regions
    matrix = np.full((len(all_elements), len(all_regions)), np.nan)

    for row in strategy_data.iter_rows(named=True):
        element_idx = all_elements.index(row['element'])
        region_idx = all_regions.index(row['region'])
        matrix[element_idx, region_idx] = row['semi_elasticity']

    # Plot heatmap
    im = ax.heatmap(
        matrix,
        aspect='auto',
        labels=True,
        labels_kw={'fontsize': 6, 'formatter': 'simple'},
        precision=2,
        cmap='YlOrRd',  # Yellow to Orange to Red sequential colormap
        vmin=vmin,
        vmax=vmax,
    )

    # Set title with strategy color and display name
    strategy_color = strategy_config[strategy]["color"]
    display_name = strategy_config[strategy]["name"].replace('\n', '+')
    ax.set_title(
        display_name,
        fontsize=9,
        pad=8
    )

    # Set ticks
    ax.set_xticks(range(len(all_regions)))
    ax.set_yticks(range(len(all_elements)))

    # Set labels - only leftmost plots in each row show y-labels
    if idx % 4 == 0:  # First column (indices 0, 4)
        ax.set_yticklabels(all_elements, fontsize=8)
        ax.set_ylabel('Metal', fontsize=8)
    else:
        ax.set_yticklabels([])

    # Show x-labels only on the last row (indices 4-7)
    if idx >= 4:  # Last row (second row of heatmaps)
        ax.set_xticklabels([r.replace('R10', '') for r in all_regions],
                           rotation=45, ha='right', fontsize=8)
    else:
        ax.set_xticklabels([])


    # Add grid lines between cells
    ax.set_xticks(np.arange(len(all_regions)) - 0.5, minor=True)
    ax.set_yticks(np.arange(len(all_elements)) - 0.5, minor=True)
    ax.grid(which='minor', color='white', linestyle='-', linewidth=0.5)
    ax.tick_params(which='minor', size=0)

    # Keep spines visible for axis boundaries
    for spine in ax.spines.values():
        spine.set_edgecolor('black')
        spine.set_linewidth(0.5)

# Add shared colorbar between the two rows
# Position it on the right side, spanning vertically between the two rows
cbar = fig1.colorbar(
    im,
    loc='r',  # Right side
    row=(2, 3),  # Span rows 2 and 3 (the two heatmap rows)
    label='Mean Mt Mitigation per % Supply',
    labelsize=8,
    ticklabelsize=7,
    length=0.5,  # Make it shorter (50% of the available height)
)

# ============================================================================
# BAR PLOT in Figure 1
# ============================================================================

# Bar plot is at axes1[0] (subplot 1, in Row 1)
ax_bar1 = axes1[0]

# Plot bars
ax_bar1.bar(
    bar_indices,
    bar_heights,
    width=1.0,
    color=bar_colors,
    alpha=0.99,
    edgecolors='none'
)

# Format bar plot
ax_bar1.format(
    titleweight='bold',
    xlabelsize=8,
    ylabelsize=8,
    xticklabelsize=7,
    yticklabelsize=7,
    xticklabels=[],
    xlocator=False,
    xlabel="Scenarios",
    ylabel="Abatement Achievement Rate (%)",
    yloc='left',
    xlim=(0, len(data_sorted))
)
ax_bar1.spines[['right', 'top']].set_visible(False)

# Add legend with display names for Figure 1 - below the bar plot
handles1 = [Patch(facecolor=cfg["color"], edgecolor='none') for cfg in strategy_config.values()]
labels1 = [cfg["name"].replace('\n', '+') for cfg in strategy_config.values()]

axes1.tick_params(axis='both', which='major', length=1)

ax_bar1.legend(
    handles1,
    labels1,
    ncols=2,
    fontsize=8,
    handlelength=0.6,
    handleheight=0.6,
    handletextpad=0.3,
    columnspacing=0.8,
    labelspacing=0.6,
    loc="ul",
    center=True,
    frame=False,
)

# Apply ABC labels to bar plot (a) and heatmaps (b-i)
for i, ax_idx in enumerate([0, 1, 2, 3, 4, 5, 6, 7, 8], start=1):
    axes1[ax_idx].number = i
    axes1[ax_idx].format(abc='a', abcloc='l')

# Overall formatting for Figure 1
fig1.format(tickminor=False)

# Save Figure 1
output_path1 = "../../testpic/heatmaps_elasticity.pdf"
fig1.save(output_path1)
print(f"\nFigure 1 (Heatmaps) saved to {output_path1}")

# ============================================================================
# FIGURE 2: LINE PLOTS + BAR PLOT
# ============================================================================

print("\n" + "="*80)
print("Creating Figure 2: Line plots...")

# Create Figure 2: Line plots only (5 rows, 2 regions per row, 8 strategies per region)
fig2 = uplt.figure(refwidth=0.3, refheight=1.1, facecolor=(0, 0, 0, 0), share=False, spany=True, wspace=1)

# Layout: 5 rows × 16 columns (2 regions per row, 8 strategies per region)
# 10 regions × 8 strategies = 80 subplots
# Row 1: Regions 1-2 (subplots 1-16)
# Row 2: Regions 3-4 (subplots 17-32)
# Row 3: Regions 5-6 (subplots 33-48)
# Row 4: Regions 7-8 (subplots 49-64)
# Row 5: Regions 9-10 (subplots 65-80)
layout2 = [
    # Row 1: Regions 1-2 (subplots 1-16)
    [1, 2, 3, 4, 5, 6, 7, 8,   9,10,11,12,13,14,15,16],
    # Row 2: Regions 3-4 (subplots 17-32)
    [17,18,19,20,21,22,23,24,  25,26,27,28,29,30,31,32],
    # Row 3: Regions 5-6 (subplots 33-48)
    [33,34,35,36,37,38,39,40,  41,42,43,44,45,46,47,48],
    # Row 4: Regions 7-8 (subplots 49-64)
    [49,50,51,52,53,54,55,56,  57,58,59,60,61,62,63,64],
    # Row 5: Regions 9-10 (subplots 65-80)
    [65,66,67,68,69,70,71,72,  73,74,75,76,77,78,79,80],
]

axes2 = fig2.subplots(layout2, hspace=[1.9, 1.9, 1.9, 1.9])

# ============================================================================
# LINE PLOTS
# ============================================================================


# Use all 8 strategies from strategy_config for line plots
line_strategies = list(strategy_config.keys())

# 1. 数据准备：计算平滑后的数据 (这部分保持不变)
# Use raw elasticity data (path_abb, strategy, region level - NOT aggregated)
elasticity_line = (
    elasticity_raw
    .filter(pl.col('strategy').is_in(line_strategies))
)

# Calculate 10-year moving average using polars for all data
window_size = 1
elasticity_line_smoothed = (
    elasticity_line
    .fill_null(0.0)
    .sort(['strategy', 'path_abb', 'region', 'element', 'year'])
    .with_columns([
        pl.col('semi_elasticity')
        .rolling_mean(window_size=window_size, min_samples=1)
        .over(['strategy', 'path_abb', 'region', 'element'])
        .alias('elasticity_smoothed')
    ])
)

print("Aggregating data for Ribbon Charts...")
# 【核心修改步骤 1】数据聚合
# 不再区分 path_abb，而是计算每个时间点的统计分布
ribbon_data_agg = (
    elasticity_line_smoothed
    .group_by(['region', 'strategy', 'element', 'year'])
    .agg([
        # 计算中位数作为中心线
        pl.col('elasticity_smoothed').median().alias('median'),
        # 计算 25% 和 75% 分位数作为丝带的边界 (即 50% 的核心分布区间)
        pl.col('elasticity_smoothed').quantile(0.25).alias('q25'),
        pl.col('elasticity_smoothed').quantile(0.75).alias('q75'),
        # 如果想看更宽的范围 (如90%区间)，可以解开下面两行的注释
        # pl.col('elasticity_smoothed').quantile(0.05).alias('q05'),
        # pl.col('elasticity_smoothed').quantile(0.95).alias('q95'),
    ])
    .sort(['region', 'strategy', 'element', 'year'])
)

all_line_regions = sorted(elasticity_line['region'].unique().to_list())
print(f"Plotting {len(all_line_regions)} regions for {len(line_strategies)} strategies using Ribbon Charts")

# Plot region subplots - one subplot per strategy per region
for region_idx, region in enumerate(all_line_regions):
    # For each region, plot 8 strategy subplots
    for strat_idx, strategy in enumerate(line_strategies):
        # Calculate subplot index
        ax_idx = region_idx * 8 + strat_idx
        ax = axes2[ax_idx]

        # Filter aggregated data for this region and strategy
        region_strat_data = ribbon_data_agg.filter(
            (pl.col('region') == region) & (pl.col('strategy') == strategy)
        )

        # Get unique elements present in this filtered data
        unique_elements = region_strat_data['element'].unique().to_list()

        # 【核心修改步骤 2】绘制丝带图
        # 不再循环 path_abb，而是循环 element 绘制聚合后的结果
        for element in unique_elements:
            if element not in element_colors:
                continue

            # 提取当前元素的聚合数据
            element_data = region_strat_data.filter(pl.col('element') == element).sort('year')

            # 过滤数据：只保留2026年及以后的数据
            element_data = element_data.filter(
                (pl.col('year') != 2100) & (pl.col('year') >= 2026) &
                ((pl.col('year') % 10 == 0))
            )

            # 准备绘图数据 (转换为 numpy 数组)
            years = element_data['year'].to_numpy()
            medians = element_data['median'].to_numpy()
            q25s = element_data['q25'].to_numpy()
            q75s = element_data['q75'].to_numpy()

            line_color = element_colors[element]

            # 1. 绘制半透明的丝带填充区域 (表示不确定性范围)
            # zorder设置得比较小，确保它在实线下方
            ax.fill_between(
                years,
                q25s,
                q75s,
                color=line_color,
                alpha=0.3,       # 设置透明度，0.3-0.5 比较合适
                edgecolor='none',# 不显示填充区域的边框
                zorder=2
            )

            # 2. 绘制加粗的中位数实线 (表示中心趋势)
            ax.plot(
                years,
                medians,
                linewidth=1.5,   # 比之前的 0.5 粗一些，突出显示
                alpha=1.0,       # 实线不透明
                color=line_color,
                zorder=3         # 确保线在丝带上方
            )

        # Format subplot (这部分格式化代码基本保持不变，做了一点点优化)
        region_name = region.replace('R10', '')
        strategy_name = strategy_config[strategy]["name"]

        # 优化 Y 轴标签显示逻辑：在每一行的最左侧显示（每行第一个region的第一个策略）
        # Row starts: 0, 16, 32, 48, 64
        ylabel = 'Mt Mitigation per % Supply' if ax_idx in [0, 16, 32, 48, 64] else ''
        is_leftmost_col = (strat_idx == 0)

        # 优化 X 轴标签显示逻辑
        # Only show xlabel and xticklabels for the last row (Region 9-10: axes 64-79)
        show_strategy_label = (64 <= ax_idx <= 79)
        xlabel = strategy_name if show_strategy_label else ''

        # Show xticklabels for last row only
        show_xticklabels = (64 <= ax_idx <= 79)

        ax.format(
            xlabel=xlabel,
            xlabelsize=6,
            xlocator=[2050, 2100],
            xrotation=45,
            xticklabels=None if show_xticklabels else [], # Proplot 中 None 表示使用默认刻度
            ylabel=ylabel,
            ylabelsize=7,
            xticklabelsize=7,
            yticklabelsize=7,
            # 强制 Y 轴刻度只在最左侧显示
            ytickloc='left' if is_leftmost_col else 'none',
            yticklabelloc='left' if is_leftmost_col else 'none',
            xlim=(2026, 2100)  # 从2026年开始显示
        )

        # Add region name as annotation
        if strat_idx == 3:  # Middle of 8 strategies
            ax.annotate(
                region_name,
                xy=(1.15, 1.05),
                xycoords='axes fraction',
                ha='center',
                va='bottom',
                fontsize=7,
                annotation_clip=False,
            )

        # Hide left spine for strategies 2-8 in each region
        if strat_idx > 0:
            ax.spines['left'].set_visible(False)
            ax.tick_params(left=False, labelleft=False)

        # 优化脊柱显示：只保留左侧和底部
        ax.spines[['right', 'top']].set_visible(False)


# Share y-axis limits (保持不变，这对于丝带图比较也很重要)
# Collect y-limits for each region across all strategies
region_ylims = {}
for region_idx, region in enumerate(all_line_regions):
    y_maxs = []
    for strat_idx in range(len(line_strategies)):
        ax_idx = region_idx * 8 + strat_idx
        # 获取当前包含了丝带和实线的最大Y值
        ylim = axes2[ax_idx].get_ylim()
        y_maxs.append(ylim[1])
    # Store region-specific max with fixed lower bound
    region_ylims[region] = (0, max(y_maxs))

# Apply fixed lower bound (0) with region-specific upper bounds
for region_idx, region in enumerate(all_line_regions):
    ymin, ymax = region_ylims[region]
    for strat_idx in range(len(line_strategies)):
        ax_idx = region_idx * 8 + strat_idx
        axes2[ax_idx].set_ylim(ymin, ymax)

# Remove old region labels code since we still have region annotations
# Add region labels as spanning titles for each group of 8 subplots
for region_idx, region in enumerate(all_line_regions):
    region_name = region.replace('R10', '')

    # Get the 8 subplot indices for this region (now sequential)
    first_ax_idx = region_idx * 8
    last_ax_idx = first_ax_idx + 7


# ============================================================================
# METAL LEGEND (on the right side)
# ============================================================================

# Add element legend using fig2.legend instead of a subplot
element_legend_handles = []
element_legend_labels = []

unique_elements = sorted(elasticity_line['element'].unique().to_list())
for element in unique_elements:
    if element in element_colors:
        h_line = Line2D([0], [0], color=element_colors[element], linewidth=2)
        element_legend_handles.append(h_line)
        element_legend_labels.append(element)

# Place metal legend on the right side of the figure
fig2.legend(
    element_legend_handles,
    element_legend_labels,
    loc='right',
    ncols=1,
    fontsize=7,
    handlelength=1.0,
    handleheight=0.5,
    handletextpad=0.3,
    frameon=False,
    title='Metal',
    title_fontsize=8,
)

axes2.tick_params(axis='both', which='major', length=1)

# Apply ABC labels for Figure 2
# Line plots: first strategy of each region (10 regions)
# Regions 1-10: axes 0, 8, 16, 24, 32, 40, 48, 56, 64, 72 (every 8th starting from 0)
abc_axes2 = [0, 8, 16, 24, 32, 40, 48, 56, 64, 72]

# Set the number attribute for axes that should have ABC labels
for i, ax_idx in enumerate(abc_axes2, start=1):
    axes2[ax_idx].number = i
    axes2[ax_idx].format(abc='a', abcloc='ul')

# Overall formatting for Figure 2
fig2.format(tickminor=False)

# Save Figure 2
output_path2 = "../../testpic/line_bar_elasticity.svg"
fig2.save(output_path2)
print(f"\nFigure 2 (Line plots) saved to {output_path2}")

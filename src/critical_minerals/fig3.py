import ultraplot as uplt
import polars as pl
from matplotlib.patches import Patch
import numpy as np
from pathlib import Path

# Resolve paths relative to repo root (so CWD doesn't matter)
REPO_ROOT = Path(__file__).resolve().parents[2]
DATA_RAW = REPO_ROOT / "data" / "data_raw" / "miracle_oscar"
TESTPIC = REPO_ROOT / "testpic"

# Define category colors (C1-C8)
category_colors = {
    "C1": ("#97CEE4", 0.95),  # Blue
    "C2": ("#778663", 0.35),  # Green
    "C3": ("#6F7899", 0.45),  # Red-Orange
    "C4": ("#A7C682", 0.45),  # Orange
    "C5": ("#8CA7D0", 0.45),  # Light Blue
    "C6": ("#FAC182", 0.45),  # Yellow
    "C7": ("#F18872", 0.45),  # Pink
    "C8": ("#BD7161", 0.45),  # Gray
}

# Configuration for error bars - choose between percentile ranges
USE_IQR_PERCENTILES = True  # True for 25th-75th percentiles (IQR), False for 5th-95th percentiles

# Six main-text strategies (drop Circular+Substitution and Allocation+Circular)
strategies = [
    'bau_bau_bau_recycl_nosub',
    'bau_llt_mir_recych_nosub',
    'bau_bau_bau_recycl_sub',
    'allo_bau_bau_recycl_nosub',
    'allo_bau_bau_recycl_sub',
    'allo_llt_mir_recych_sub',
]

# Leftover strategies for SI roses only
si_strategies = [
    'bau_llt_mir_recych_sub',      # Circular + Substitution
    'allo_llt_mir_recych_nosub',   # Allocation + Circular
]

strategy_labels = {
    'bau_bau_bau_recycl_nosub': 'Ref',
    'bau_llt_mir_recych_nosub': 'Circular',
    'bau_bau_bau_recycl_sub': 'Substitution',
    'bau_llt_mir_recych_sub': 'Circular +\nSubstitution',
    'allo_bau_bau_recycl_nosub': 'Allocation',
    'allo_llt_mir_recych_nosub': 'Allocation +\nCircular',
    'allo_bau_bau_recycl_sub': 'Allocation +\nSubstitution',
    'allo_llt_mir_recych_sub': 'All-in',
}

NROWS = 2
NCOLS = 3

# Toggle CO₂ KDE panels below each strategy panel
SHOW_CO2_KDE = False


def plot_regional_rose(ax, strategy_reg_data, categories, regions, label_fontsize=7):
    """Draw a regional spider/rose plot onto ax for one strategy."""
    if strategy_reg_data.is_empty():
        ax.axis('off')
        return

    angles = np.linspace(0, 2 * np.pi, len(regions), endpoint=False)

    for i in range(len(regions)):
        ax.plot([0, np.cos(angles[i])], [0, np.sin(angles[i])],
                'k-', alpha=0.2, lw=0.25)

    for r in [0.25, 0.5, 0.75, 1.0]:
        circle_x = r * np.cos(np.linspace(0, 2 * np.pi, 100))
        circle_y = r * np.sin(np.linspace(0, 2 * np.pi, 100))
        ax.plot(circle_x, circle_y, 'k-', alpha=0.1, lw=0.25)

    all_category_values = []
    category_data_dict = {}

    for category in categories:
        cat_reg_data = strategy_reg_data.filter(pl.col("category") == category)
        if cat_reg_data.is_empty():
            continue
        spider_data = (
            cat_reg_data
            .group_by("region")
            .agg(pl.median("diff_reg").alias("median_diff"))
            .sort("region")
        )
        values = []
        for region in regions:
            region_value = spider_data.filter(pl.col("region") == region)
            if not region_value.is_empty():
                val = region_value["median_diff"][0]
                values.append(val)
                all_category_values.append(val)
            else:
                values.append(0)
        category_data_dict[category] = values

    global_max = max(all_category_values) if all_category_values else 1.0

    for category in categories:
        if category not in category_data_dict:
            continue
        values = category_data_dict[category]
        norm_values = [v / global_max for v in values]
        x_coords = [nv * np.cos(ang) for ang, nv in zip(angles, norm_values)]
        y_coords = [nv * np.sin(ang) for ang, nv in zip(angles, norm_values)]
        x_coords.append(x_coords[0])
        y_coords.append(y_coords[0])
        color, _ = category_colors.get(category, ("#999999", 0.45))
        ax.plot(x_coords, y_coords, 'o-', linewidth=1.2, markersize=2,
                color=color, alpha=0.75, label=category)
        ax.fill(x_coords, y_coords, alpha=0.1, color=color)

    for r in [0.25, 0.5, 0.75]:
        actual_value = r * global_max
        if r == 0.75 and actual_value < 0.01:
            value_text = f'{actual_value:.4f}'
        elif actual_value < 0.01:
            value_text = f'{actual_value:.3f}'
        else:
            value_text = f'{actual_value:.2f}'
        ax.text(0, r + 0.05, value_text,
                fontsize=label_fontsize - 1, fontweight='bold',
                ha='center', va='bottom', color='gray', alpha=0.8, zorder=10)

    for i, region in enumerate(regions):
        angle = angles[i]
        label_x = 1.15 * np.cos(angle)
        label_y = 1.15 * np.sin(angle)
        if 3.9 <= angle <= 5.5:
            label_x = 1.0 * np.cos(angle)
            label_y = 1.0 * np.sin(angle)
            ha, va = ('right', 'top') if angle < 4.7 else ('center', 'top')
        else:
            ha, va = 'center', 'center'
        ax.text(label_x, label_y, region.replace('R10', ''),
                fontsize=label_fontsize, fontweight='bold', ha=ha, va=va)

    ax.set_xlim(-1.35, 1.35)
    ax.set_ylim(-1.35, 1.35)
    ax.set_aspect('equal')
    ax.axis('off')


# Read co2 emission data and additional co2 emission data
if SHOW_CO2_KDE:
    co2_base = (
        pl.read_parquet(DATA_RAW / "r10_ghg.parquet")
        .filter(pl.col('scope').is_in(['co2_industry', 'co2_energy']))
        .filter(pl.col('year').is_between(2021, 2100))
        .with_columns(
            path_abb=pl.concat_str(pl.col('model'), pl.col('scenario'), separator='|')
        )
        .group_by(['path_abb'])
        .agg((pl.sum('value') / 1000).alias('base'))  # Mt CO2 to Gt CO2
    )

    co2_add = (
        pl.read_parquet(DATA_RAW / "co2_add_reg.parquet")
        .group_by(['path_abb', 'strategy'])
        .agg((pl.sum('Emissions') / 1000).alias('add'))  # Mt CO2 to Gt CO2
        .sort('path_abb', 'strategy')
    )

    category = pl.read_csv(DATA_RAW / "category.csv")

    co2 = (
        co2_add.join(co2_base, on=['path_abb'], how='left')
        .join(category, on=['path_abb'], how='left')
        .with_columns(
            co2=pl.col('add') + pl.col('base')
        )
        .select(pl.col('category', 'path_abb', 'strategy', 'co2'))
        .sort('path_abb', 'strategy')
    )

# Read data
print("Loading data from parquet...")
df = pl.read_parquet(DATA_RAW / "temp_diff_2100.parquet")

print("Loading regional data from parquet...")
df_reg = pl.read_parquet(DATA_RAW / "temp_diff_2100_by_reg.parquet")

keep_strategies = strategies + si_strategies
df = df.filter(pl.col("strategy").is_in(keep_strategies))
df_reg = df_reg.filter(pl.col("strategy").is_in(keep_strategies))

categories = sorted(
    df.filter(pl.col("strategy").is_in(strategies))["category"].unique().to_list()
)
regions = sorted(df_reg["region"].unique().to_list())

print(f"Found {len(strategies)} main strategies (+ {len(si_strategies)} SI-only)")
print(f"Found {len(categories)} categories")
print(f"Found {len(regions)} regions")
print(f"Total data points: {len(df)}")

# ---------------------------------------------------------------------------
# Main figure: 2×3 scatter + bar (BR) + rose (UL) insets
# ---------------------------------------------------------------------------
uplt.rc["grid"] = False
fig = uplt.figure(
    refwidth=3.6,
    facecolor=(0, 0, 0, 0),
    sharex=0,
    spanx=False,
    sharey=1,
    spany=False,
)
axes = fig.subplots(nrows=NROWS, ncols=NCOLS, hspace=3.5, wspace=2.8)

print("Calculating global y-limits for inset plots...")
all_bar_heights = []
for strategy in strategies:
    strategy_reg_data = df_reg.filter(pl.col("strategy") == strategy)
    if not strategy_reg_data.is_empty():
        bar_data = (
            strategy_reg_data
            .group_by(["model", "scenario", "category"])
            .agg(pl.sum("diff_reg").alias("global_diff"))
            .group_by("category")
            .agg([
                pl.median("global_diff").alias("median_diff"),
                pl.quantile("global_diff", 0.75).alias("p75_diff"),
                pl.quantile("global_diff", 0.95).alias("p95_diff"),
            ])
        )
        if not bar_data.is_empty():
            if USE_IQR_PERCENTILES:
                all_bar_heights.extend(bar_data["p75_diff"].to_list())
            else:
                all_bar_heights.extend(bar_data["p95_diff"].to_list())

if all_bar_heights:
    global_ylim = (0, max(all_bar_heights) * 1.05)
    print(f"Global y-limits for insets: {global_ylim}")
else:
    global_ylim = (0, 1)

co2_panels = []
n_grid = NROWS * NCOLS

if SHOW_CO2_KDE:
    co2_axes = axes.panel_axes('b', width=0.28, space=2.8, share=False)
    for flat_idx in range(n_grid):
        pax = co2_axes.flat[flat_idx] if hasattr(co2_axes, 'flat') else co2_axes[flat_idx]
        parent_ax = axes.flat[flat_idx] if hasattr(axes, 'flat') else axes[flat_idx]
        try:
            pax.set_zorder(parent_ax.get_zorder() - 1)
        except Exception:
            pax.set_zorder(-1)
        try:
            pax.get_shared_x_axes().remove(pax)
            pax.get_shared_y_axes().remove(pax)
        except Exception:
            pass

for idx, strategy in enumerate(strategies):
    row = idx // NCOLS
    col = idx % NCOLS
    ax = axes[row, col]

    print(f"\nPlotting strategy: {strategy}")

    strategy_data = df.filter(pl.col("strategy") == strategy)
    if strategy_data.is_empty():
        print(f"  No data for {strategy}")
        continue

    for category in categories:
        cat_data = strategy_data.filter(pl.col("category") == category)
        if cat_data.is_empty():
            continue

        print(f"  Category {category}: {len(cat_data)} points")
        x_data = cat_data["value_scen"].to_numpy()
        y_data = cat_data["value_cf"].to_numpy()
        color, alpha = category_colors.get(category, ("#999999", 0.45))

        try:
            ax.scatter(
                x_data,
                y_data,
                s=6,
                c=color,
                alpha=alpha,
                edgecolors='none',
                label=category if idx == 0 else None,
            )
        except Exception as e:
            print(f"  Error plotting {category}: {e}")

    ax.format(xlim=(0.75, 5.25), ylim=(0.75, 5.25))
    xlim = ax.get_xlim()
    ylim = ax.get_ylim()
    lim_min = min(xlim[0], ylim[0])
    lim_max = max(xlim[1], ylim[1])
    ax.plot([lim_min, lim_max], [lim_min, lim_max],
            'k--', lw=0.5, alpha=0.5, zorder=1)

    ax.format(
        urtitle=strategy_labels.get(strategy, strategy),
        titleweight='bold',
        titlesize=14,
        xlabelsize=12,
        ylabelsize=12,
        xticklabelsize=11,
        yticklabelsize=11,
    )

    print(f"  Adding inset bar plot for {strategy}")
    inset_ax = ax.inset(
        [0.52, 0.05, 0.45, 0.42],
        transform="axes",
        zoom=False,
    )

    strategy_reg_data = df_reg.filter(pl.col("strategy") == strategy)

    if not strategy_reg_data.is_empty():
        bar_data = (
            strategy_reg_data
            .group_by(["model", "scenario", "category"])
            .agg(pl.sum("diff_reg").alias("global_diff"))
            .group_by("category")
            .agg([
                pl.median("global_diff").alias("median_diff"),
                pl.quantile("global_diff", 0.25).alias("p25_diff"),
                pl.quantile("global_diff", 0.75).alias("p75_diff"),
                pl.quantile("global_diff", 0.05).alias("p05_diff"),
                pl.quantile("global_diff", 0.95).alias("p95_diff"),
            ])
            .sort("category")
        )

        for cat_idx, category in enumerate(categories):
            value = bar_data.filter(pl.col("category") == category)
            if value.is_empty():
                continue

            height = value["median_diff"][0]
            p25 = value["p25_diff"][0]
            p75 = value["p75_diff"][0]
            p05 = value["p05_diff"][0]
            p95 = value["p95_diff"][0]
            base_color, base_alpha = category_colors.get(category, ("#999999", 0.45))

            inset_ax.bar(
                np.array([cat_idx]),
                np.array([height]),
                0.95,
                color=base_color,
                edgecolor='none',
                alpha=base_alpha,
            )

            if USE_IQR_PERCENTILES:
                lower_error = height - p25
                upper_error = p75 - height
            else:
                lower_error = height - p05
                upper_error = p95 - height

            inset_ax.errorbar(
                cat_idx, height,
                yerr=[[lower_error], [upper_error]],
                fmt='none',
                color='black',
                capsize=2,
                capthick=0.5,
                elinewidth=0.5,
                alpha=0.8,
            )

        inset_ax.format(
            xticks=np.arange(len(categories)),
            xticklabels=categories,
            xticklabelsize=10,
            yticklabelsize=10,
            grid=False,
            tickminor=False,
            ylim=global_ylim,
            ylabel="°C",
            ylabelsize=10,
            ylabelweight='bold',
        )
        inset_ax.set_xlabel('')
        inset_ax.spines['top'].set_visible(False)
        inset_ax.spines['right'].set_visible(False)
        inset_ax.tick_params(axis='both', length=1, labelsize=10)
        for label in inset_ax.get_xticklabels() + inset_ax.get_yticklabels():
            label.set_fontweight('bold')

    print(f"  Adding rose inset for {strategy}")
    spider_ax = ax.inset(
        [0.10, 0.56, 0.42, 0.42],
        transform="axes",
        zoom=False,
    )
    plot_regional_rose(
        spider_ax, strategy_reg_data, categories, regions, label_fontsize=7,
    )

    if SHOW_CO2_KDE:
        panel_idx = row * NCOLS + col
        co2_ax = co2_axes.flat[panel_idx] if hasattr(co2_axes, 'flat') else co2_axes[panel_idx]
        co2_panels.append(co2_ax)
        strategy_co2_data = co2.filter(pl.col("strategy") == strategy)

        if not strategy_co2_data.is_empty():
            strategy_co2_data = strategy_co2_data.filter(pl.col("category").is_not_null())
            all_co2_data = []
            category_data = {}

            for category in categories:
                cat_co2_data = strategy_co2_data.filter(pl.col("category") == category)
                if not cat_co2_data.is_empty():
                    co2_values = cat_co2_data["co2"].to_numpy()
                    if len(co2_values) >= 2:
                        category_data[category] = co2_values
                        all_co2_data.extend(co2_values)

            if all_co2_data and category_data:
                min_co2, max_co2 = min(all_co2_data), max(all_co2_data)
                bin_range = max_co2 - min_co2
                n_bins = min(30, max(10, int(bin_range / 50)))
                bins = np.linspace(min_co2 - bin_range * 0.05,
                                   max_co2 + bin_range * 0.05, n_bins)

                for category in categories:
                    if category not in category_data:
                        continue
                    co2_values = category_data[category]
                    color, alpha = category_colors.get(category, ("#999999", 0.45))
                    try:
                        from scipy.stats import gaussian_kde
                        kde = gaussian_kde(co2_values)
                        x_min, x_max = co2_values.min(), co2_values.max()
                        x_range = np.linspace(x_min - (x_max - x_min) * 0.1,
                                              x_max + (x_max - x_min) * 0.1, 200)
                        density = kde(x_range)
                        approx_bin_width = (x_max - x_min) / n_bins
                        scaled_density = density * len(co2_values) * approx_bin_width
                        co2_ax.fill_between(
                            x_range, scaled_density,
                            alpha=alpha, color=color, edgecolor='none',
                        )
                        co2_ax.plot(
                            x_range, scaled_density,
                            color=color, alpha=min(alpha + 0.3, 1.0), linewidth=0.8,
                        )
                    except Exception:
                        co2_ax.hist(
                            co2_values, bins=bins, alpha=alpha,
                            color=color, edgecolor='none',
                        )

            co2_ax.format(
                xlabel="Cumulative CO₂ (2021–2100, Gt)",
                ylabel="Density",
                xlabelsize=9,
                ylabelsize=8,
                xticklabelsize=8,
                yticklabelsize=8,
                grid=False,
                tickminor=False,
            )
            co2_ax.spines['top'].set_visible(False)
            co2_ax.spines['right'].set_visible(False)

if SHOW_CO2_KDE and co2_panels:
    print("Setting shared x and y-limits for CO2 panels...")
    all_co2_xlims = []
    all_co2_ylims = []
    for panel in co2_panels:
        all_co2_xlims.extend(panel.get_xlim())
        all_co2_ylims.extend(panel.get_ylim())
    if all_co2_xlims and all_co2_ylims:
        co2_xmin, co2_xmax = min(all_co2_xlims), max(all_co2_xlims)
        co2_ymin, co2_ymax = min(all_co2_ylims), max(all_co2_ylims)
        for panel in co2_panels:
            panel.set_xlim(co2_xmin, co2_xmax)
            panel.set_ylim(co2_ymin, co2_ymax)

axes.format(
    abc=True,
    abcloc="ul",
    abcsize=14,
    tickminor=False,
    ylabel="Total warming (°C)",
    ylabelsize=12,
)

for col in range(NCOLS):
    axes[NROWS - 1, col].format(xlabel="Targeted warming (°C)", xlabelsize=12)
    if NROWS > 1:
        axes[0, col].set_xlabel("")

legend_handles = []
legend_labels = []
for category in categories:
    color, _ = category_colors.get(category, ("#999999", 0.45))
    patch = Patch(facecolor=color, edgecolor='none', alpha=0.6, label=category)
    legend_handles.append(patch)
    legend_labels.append(category)

legend = fig.legend(
    legend_handles,
    legend_labels,
    loc="bottom",
    ncols=len(categories),
    fontsize=12,
    handlelength=2.2,
    handleheight=1.1,
    handletextpad=0.5,
    columnspacing=1.4,
    frame=False,
    center=True,
)
for text in legend.get_texts():
    text.set_fontsize(12)

output_path = TESTPIC / "density_scen_vs_cf_by_strategy.svg"
fig.save(output_path)
print(f"\nFigure saved to {output_path}")

# ---------------------------------------------------------------------------
# SI figure: roses for the two strategies omitted from the main 2×3
# ---------------------------------------------------------------------------
print("\nBuilding SI regional-rose figure (leftover strategies)...")
si_nrows, si_ncols = 1, len(si_strategies)
fig_si = uplt.figure(
    refwidth=3.6,
    facecolor=(0, 0, 0, 0),
    sharex=0,
    spanx=False,
    sharey=0,
    spany=False,
)
axes_si = fig_si.subplots(nrows=si_nrows, ncols=si_ncols, hspace=3.5, wspace=2.8)

for idx, strategy in enumerate(si_strategies):
    ax = axes_si.flat[idx] if hasattr(axes_si, "flat") else axes_si[idx]
    print(f"  SI rose: {strategy}")
    strategy_reg_data = df_reg.filter(pl.col("strategy") == strategy)
    plot_regional_rose(ax, strategy_reg_data, categories, regions, label_fontsize=8)
    ax.format(
        urtitle=strategy_labels.get(strategy, strategy),
        titleweight='bold',
        titlesize=14,
    )

axes_si.format(abc=True, abcloc="ul", abcsize=14, grid=False)

si_legend = fig_si.legend(
    legend_handles,
    legend_labels,
    loc="bottom",
    ncols=len(categories),
    fontsize=12,
    handlelength=2.2,
    handleheight=1.1,
    handletextpad=0.5,
    columnspacing=1.4,
    frame=False,
    center=True,
)
for text in si_legend.get_texts():
    text.set_fontsize(12)

si_output_path = TESTPIC / "density_regional_roses_by_strategy.svg"
fig_si.save(si_output_path)
print(f"SI figure saved to {si_output_path}")

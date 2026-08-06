import polars as pl
import ultraplot as uplt
import numpy as np

# Define category colors (from fig3.py)
category_colors = {
    "C1": ("#97CEE4", 1),  # Blue
    "C2": ("#778663", 1),  # Green
    "C3": ("#6F7899", 1),  # Red-Orange
    "C4": ("#A7C682", 1),  # Orange
    "C5": ("#8CA7D0", 1),  # Light Blue
    "C6": ("#FAC182", 1),  # Yellow
    "C7": ("#F18872", 1),  # Pink
    "C8": ("#BD7161", 1),  # Gray
}

# Load the summary data
print("Loading emissions contribution summary data...")
data = pl.read_parquet("../../data/data_output/emissions_contributions_add_summary.parquet")

print(f"Loaded {len(data)} rows")
print(f"Categories: {sorted(data['category'].unique().to_list())}")
print(f"Strategies: {sorted(data['strategy'].unique().to_list())}")
print(f"Regions: {sorted(data['region'].unique().to_list())}")

# Get unique values
categories = sorted(data['category'].unique().to_list())
strategies = [
    'bau_bau_bau_recycl_nosub',
    'bau_llt_mir_recych_nosub',
    'bau_bau_bau_recycl_sub',
    'bau_llt_mir_recych_sub',
    'allo_bau_bau_recycl_nosub',
    'allo_llt_mir_recych_nosub',
    'allo_bau_bau_recycl_sub',
    'allo_llt_mir_recych_sub',
]
regions = sorted(data['region'].unique().to_list())

# Strategy labels (shortened for space)
# Strategy display names (consistent with Figs. 1â€?)
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

print(f"\nCreating figure with {len(categories)} rows x {len(strategies)} columns = {len(categories) * len(strategies)} subplots")

# Create figure with 8 rows (categories) x 8 columns (strategies)
uplt.rc["grid"] = False
uplt.rc["axes.linewidth"] = 0.25
uplt.rc["tick.width"] = 0.15
uplt.rc["suptitle.size"] = 7
# sharex=1: x tick labels only on bottom row (sharex=True repeats them on every row)
fig = uplt.figure(refwidth=0.65, refaspect=0.65, sharey=1, spany=0, sharex=1, spanx=1)
axes = fig.subplots(nrows=len(categories), ncols=len(strategies), hspace=1.7, wspace=0.8)
last_row = len(categories) - 1

# Calculate global x-axis limits across all subplots
print("\nCalculating global x-axis limits...")
all_values = []
for cat_idx, category in enumerate(categories):
    for strat_idx, strategy in enumerate(strategies):
        subset = data.filter(
            (pl.col('category') == category) &
            (pl.col('strategy') == strategy)
        )
        if not subset.is_empty():
            all_values.extend(subset['median_add_contribution_pct'].to_list())
            all_values.extend(subset['p25_add_contribution_pct'].to_list())
            all_values.extend(subset['p75_add_contribution_pct'].to_list())

if all_values:
    global_xlim = (0, max(all_values) * 1.1)  # Start from 0, add 10% padding
    print(f"Global x-limits: {global_xlim}")
else:
    global_xlim = (0, 100)

region_half = 0.4
global_ylim = ((len(regions) - 1) + region_half + 0.2, -region_half - 0.2)

# Calculate global additional emissions with the same method as
# tab_emissions_add_global_stats.py (quantiles of scenario-level global totals).
print("\nCalculating global emissions per category-strategy...")
category_map = pl.read_csv('../../data/data_raw/miracle_oscar/category.csv')
global_emissions = (
    pl.read_parquet("../../data/data_raw/miracle_oscar/co2_add_reg.parquet")
    .filter(pl.col('strategy').is_in(strategies))
    .group_by(['path_abb', 'strategy'])
    .agg((pl.sum('Emissions') / 1000).alias('add'))  # Mt CO2 to Gt CO2
    .join(category_map, on=['path_abb'], how='left')
    .group_by(['category', 'strategy'])
    .agg([
        pl.col('add').quantile(0.25).alias('global_p25_gt'),
        pl.col('add').quantile(0.50).alias('global_median_gt'),
        pl.col('add').quantile(0.75).alias('global_p75_gt'),
    ])
)
print(f"Calculated global emissions for {len(global_emissions)} category-strategy combinations")

# Plot forest plots
for cat_idx, category in enumerate(categories):
    for strat_idx, strategy in enumerate(strategies):
        ax = axes[cat_idx, strat_idx]

        print(f"Plotting {category} x {strategy}")

        # Filter data for this category and strategy
        subset = data.filter(
            (pl.col('category') == category) &
            (pl.col('strategy') == strategy)
        ).sort('region')

        if subset.is_empty():
            ax.text(0.5, 0.5, 'No data', transform=ax.transAxes,
                   ha='center', va='center', fontsize=5, color='gray')
            ax.set_xlim(global_xlim)
            ax.set_ylim(global_ylim)
            ax.format(
                xlim=global_xlim,
                ylim=global_ylim,
                urtitle="No data",
                xticklabels=cat_idx == last_row,
            )
            continue

        # Get data for plotting
        y_positions = np.arange(len(regions))

        # Get category color
        color, alpha = category_colors.get(category, ("#999999", 1))

        # For each region, plot median with error bars (p25 to p75)
        for region_idx, region in enumerate(regions):
            region_data = subset.filter(pl.col('region') == region)

            if not region_data.is_empty():
                median = region_data['median_add_contribution_pct'][0]
                p25 = region_data['p25_add_contribution_pct'][0]
                p75 = region_data['p75_add_contribution_pct'][0]

                # Calculate error bar lengths
                lower_error = median - p25
                upper_error = p75 - median

                # Plot median as point using category color
                ax.plot(median, region_idx, 'o', markersize=1.1,
                       color=color, alpha=min(alpha + 0.3, 1.0), zorder=3)

                # Plot error bars (p25 to p75) using category color
                ax.errorbar(
                    median, region_idx,
                    xerr=[[lower_error], [upper_error]],
                    fmt='none',
                    color=color,
                    capsize=1.5,
                    capthick=0.5,
                    elinewidth=0.5,
                    alpha=alpha,
                    zorder=2
                )

        ax.set_xlim(global_xlim)
        ax.set_ylim(global_ylim)

        # Get global emissions for this category-strategy
        global_data = global_emissions.filter(
            (pl.col('category') == category) &
            (pl.col('strategy') == strategy)
        )

        if not global_data.is_empty():
            global_median = global_data['global_median_gt'][0]
            global_p25 = global_data['global_p25_gt'][0]
            global_p75 = global_data['global_p75_gt'][0]
            # Format the urtitle text
            urtitle_text = f"{global_median:.1f} Gt\n({global_p25:.1f}-{global_p75:.1f})"
        else:
            urtitle_text = "No data"

        # Format subplot
        ax.format(
            yticks=y_positions,
            yticklabels=[r.replace('R10', '') for r in regions] if strat_idx == 0 else False,
            yticklabelsize=4.5,
            xticklabelsize=4.5,
            xlim=global_xlim,
            ylim=global_ylim,
            xticklabels=cat_idx == last_row,
            urtitle=urtitle_text,
            titlesize=5,
            ylabel=f'{category}' if strat_idx == 0 else '',
            ylabelsize=4.5,
        )

        # Add strategy label on top row
        if cat_idx == 0:
            ax.format(
                title=strategy_labels.get(strategy, strategy),
                titlesize=5,
                titleweight='normal'
            )

# Overall formatting
axes.format(
    abc=True,
    abcsize=6,
    titlepad=2.5,
    xlabel='Contribution (%)',
    xlabelsize=4.5,
    suptitle='Regional contribution to additional CO2 emissions by category and strategy',
    xtickminor=True,
    ticklen=2,
)


# Hide right and top spines for all axes
for ax in axes:
    ax.spines[['right', 'top']].set_visible(False)


# Save figure
output_path = "../../testpic/emissions_contribution_forest.svg"
fig.save(output_path)
print(f"\nFigure saved to {output_path}")

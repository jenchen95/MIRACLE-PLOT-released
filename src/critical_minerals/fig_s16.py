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

# Strategy display names (consistent with Figs. 1â€?)
strategy_labels = {
    'bau_bau_bau_recycl_nosub': 'Ref',
    'bau_llt_mir_recych_nosub': 'Circular',
    'bau_bau_bau_recycl_sub': 'Substitution',
    'bau_llt_mir_recych_sub': 'Circular + Substitution',
    'allo_bau_bau_recycl_nosub': 'Allocation',
    'allo_llt_mir_recych_nosub': 'Allocation + Circular',
    'allo_bau_bau_recycl_sub': 'Allocation + Substitution',
    'allo_llt_mir_recych_sub': 'All-in',
}

print(f"\nCreating figure with {len(categories)} rows x {len(regions)} columns = {len(categories) * len(regions)} subplots for waterfall plots")

# Create figure with 8 rows (categories) x 10 columns (regions)
uplt.rc["grid"] = False
uplt.rc["axes.linewidth"] = 0.25
uplt.rc["tick.width"] = 0.15
uplt.rc["suptitle.size"] = 7
# sharey=1: per-row strategy labels on the left column (sharey=True only labels bottom-left)
fig = uplt.figure(refwidth=0.4, refaspect=0.8, sharey=1, spany=False, sharex=True)
axes = fig.subplots(nrows=len(categories), ncols=len(regions), hspace=1.3, wspace=0.3)

# Calculate global x-axis limits for horizontal bar plots (using emissions in Gt)
print("\nCalculating global x-axis limits...")
all_emissions = []
for cat_idx, category in enumerate(categories):
    for strat_idx, strategy in enumerate(strategies):
        subset = data.filter(
            (pl.col('category') == category) &
            (pl.col('strategy') == strategy)
        )
        if not subset.is_empty():
            all_emissions.extend(subset['median_add_contribution_gt'].to_list())
            all_emissions.extend(subset['p75_add_contribution_gt'].to_list())

if all_emissions:
    max_emission = max(all_emissions)
    global_xlim = (-1, max_emission * 1.2)  # Start from -1 to show negative values, add 20% padding
    print(f"Global x-limits: {global_xlim}")
else:
    global_xlim = (-1, 10)

bar_height = 0.8
bar_half = bar_height / 2
global_ylim = ((len(strategies) - 1) + bar_half + 0.2, -bar_half - 0.2)

# Plot waterfall plots (8 categories Ã— 10 regions = 80 subplots)
for cat_idx, category in enumerate(categories):
    for reg_idx, region in enumerate(regions):
        ax = axes[cat_idx, reg_idx]

        print(f"Plotting {category} x {region}")

        # Get category color
        color, alpha = category_colors.get(category, ("#999999", 1))

        # Y positions for strategies (horizontal bars)
        y_positions = np.arange(len(strategies))
        emissions_values = []
        p25_values = []
        p75_values = []

        # Get emissions for each strategy in this category-region combination
        for strategy in strategies:
            # Filter data for this category, region, and strategy
            subset = data.filter(
                (pl.col('category') == category) &
                (pl.col('region') == region) &
                (pl.col('strategy') == strategy)
            )

            if not subset.is_empty():
                emission = subset['median_add_contribution_gt'][0]
                p25 = subset['p25_add_contribution_gt'][0]
                p75 = subset['p75_add_contribution_gt'][0]
                emissions_values.append(emission)
                p25_values.append(p25)
                p75_values.append(p75)
            else:
                emissions_values.append(0)
                p25_values.append(0)
                p75_values.append(0)

        # Create horizontal bars for this category-region combination
        bars = ax.barh(y_positions, emissions_values,
                     color=color, alpha=alpha,
                     edgecolor='white', linewidth=0.3)
        ax.set_xlim(global_xlim)
        ax.set_ylim(global_ylim)

        # Add error bars (p25 to p75)
        for i, (median, p25, p75) in enumerate(zip(emissions_values, p25_values, p75_values)):
            if median > 0:  # Only add error bars where there's data
                lower_error = median - p25
                upper_error = p75 - median
                ax.errorbar(
                    median, i,
                    xerr=[[lower_error], [upper_error]],
                    fmt='none',
                    color=color,
                    capsize=1.5,
                    capthick=0.5,
                    elinewidth=0.5,
                    alpha=min(alpha + 0.2, 1.0),
                    zorder=3
                )

        # Format subplot
        ax.format(
            ylabel=f'{category}' if reg_idx == 0 else '',
            ylabelsize=4,
            yticks=y_positions,
            yticklabels=[strategy_labels.get(s, s) for s in strategies] if reg_idx == 0 else False,
            yticklabelsize=3,
            xticklabelsize=3.5,
            xlim=global_xlim,
            ylim=global_ylim,
            yreverse=True,
        )

       

        # Add region labels on top row
        if cat_idx == 0:
            ax.format(
                title=region.replace('R10', ''),
                titlesize=4,
                titleweight='normal'
            )

# Overall formatting
axes.format(
    abc=True,
    abcsize=4,
    titlepad=2.5,
    xlabel='Gt CO2',
    xlabelsize=5,
    suptitle='Additional emissions by category-region combinations across strategies',
    xtickminor=False,
    ticklen=1.5,
    xlocator=[0, 200, 400, 600],
)

# Hide right and top spines for all axes
for ax in axes:
    ax.spines[['right', 'top']].set_visible(False)

# Save figure
output_path = "../../testpic/emissions_contribution_bar.svg"
fig.save(output_path)
print(f"\nFigure saved to {output_path}")

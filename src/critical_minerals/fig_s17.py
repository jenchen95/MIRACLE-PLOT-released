import polars as pl
import ultraplot as uplt
import numpy as np
import pandas as pd
from statannotations.Annotator import Annotator

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

# Load the summary data for plotting
print("Loading emissions contribution summary data...")
data = pl.read_parquet("../../data/data_output/emissions_contributions_add_summary.parquet")

# Load the detailed raw data for statistical testing
print("Loading detailed raw data for statistical testing...")
raw_data = pl.read_parquet("../../data/data_output/emissions_contributions_add_detailed.parquet")

print(f"Loaded {len(data)} rows")
print(f"Categories: {sorted(data['category'].unique().to_list())}")
print(f"Strategies: {sorted(data['strategy'].unique().to_list())}")
print(f"Regions: {sorted(data['region'].unique().to_list())}")

# Get unique values - focus on just the 4 strategies for comparison
categories = sorted(data['category'].unique().to_list())
strategies = [
    'bau_bau_bau_recycl_nosub',
    'bau_llt_mir_recych_nosub',
    'allo_bau_bau_recycl_nosub',
    'allo_llt_mir_recych_nosub',
]
regions = sorted(data['region'].unique().to_list())

# Strategy display names (consistent with Figs. 1â€?; 4-strategy subset for circular comparisons)
strategy_labels = {
    'bau_bau_bau_recycl_nosub': 'Ref',
    'bau_llt_mir_recych_nosub': 'Circular',
    'allo_bau_bau_recycl_nosub': 'Allocation',
    'allo_llt_mir_recych_nosub': 'Allocation +\nCircular',
}
group_order = [strategy_labels[s] for s in strategies]
ref_label = strategy_labels['bau_bau_bau_recycl_nosub']
circular_label = strategy_labels['bau_llt_mir_recych_nosub']
allocation_label = strategy_labels['allo_bau_bau_recycl_nosub']
alloc_circular_label = strategy_labels['allo_llt_mir_recych_nosub']

print(f"\nCreating figure with {len(categories)} rows x {len(regions)} columns = {len(categories) * len(regions)} subplots")

# Create figure with 8 rows (categories) x 10 columns (regions)
uplt.rc["grid"] = False
uplt.rc["axes.linewidth"] = 0.25
uplt.rc["tick.width"] = 0.15
uplt.rc["suptitle.size"] = 7
fig = uplt.figure(refwidth=0.4, refaspect=0.8, sharey=True, spany=False, sharex=True)
axes = fig.subplots(nrows=len(categories), ncols=len(regions), hspace=1.3, wspace=0.3)

# Calculate global y-axis limits for vertical bar plots (using emissions in Gt)
print("\nCalculating global y-axis limits...")
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
    global_ylim = (0, max_emission * 1.2)  # Start from 0, add 20% padding
    print(f"Global y-limits: {global_ylim}")
else:
    global_ylim = (0, 10)

bar_width = 0.8
bar_half = bar_width / 2
global_xlim = (-bar_half - 0.2, (len(strategies) - 1) + bar_half + 0.2)

# Plot vertical bar plots (8 categories Ã— 10 regions = 80 subplots)
for cat_idx, category in enumerate(categories):
    for reg_idx, region in enumerate(regions):
        ax = axes[cat_idx, reg_idx]

        print(f"Plotting {category} x {region}")

        # Get category color
        color, alpha = category_colors.get(category, ("#999999", 1))

        # X positions for strategies (vertical bars)
        x_positions = np.arange(len(strategies))
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

        # Create vertical bars for this category-region combination
        bars = ax.bar(x_positions, emissions_values, width=bar_width,
                     color=color, alpha=alpha,
                     edgecolor='white', linewidth=0.3)
        ax.set_xlim(global_xlim)

        # Add error bars (p25 to p75)
        for i, (median, p25, p75) in enumerate(zip(emissions_values, p25_values, p75_values)):
            if median > 0:  # Only add error bars where there's data
                lower_error = median - p25
                upper_error = p75 - median
                ax.errorbar(
                    i, median,
                    yerr=[[lower_error], [upper_error]],
                    fmt='none',
                    color=color,
                    capsize=1.0,        # Smaller caps
                    capthick=0.3,       # Thinner caps
                    elinewidth=0.3,     # Thinner error lines
                    alpha=min(alpha + 0.2, 1.0),
                    zorder=3
                )

        # statannotations loc='outside' maps y via axes coords; shared global
        # ylim on cols 2+ places brackets at the panel top instead of bar tops
        panel_tops = [p75 for p75 in p75_values if p75 > 0]
        panel_ylim_top = max(panel_tops) * 1.08 if panel_tops else 1
        ax.set_ylim(0, panel_ylim_top)

        # Add statistical annotations comparing BAU vs OPT groups
        try:
            # Get raw data for this category-region combination
            category_region_data = raw_data.filter(
                (pl.col('category') == category) &
                (pl.col('region') == region)
            )

            if not category_region_data.is_empty():
                # Prepare data for statistical testing
                test_data = []

                for strategy in strategies:
                    strategy_data = category_region_data.filter(pl.col('strategy') == strategy)
                    if not strategy_data.is_empty():
                        if 'add_contribution_gt' in strategy_data.columns:
                            values = strategy_data['add_contribution_gt'].to_list()
                            group_name = strategy_labels.get(strategy)
                            if group_name is None:
                                continue

                            for value in values:
                                test_data.append({
                                    'strategy': strategy,
                                    'group': group_name,
                                    'value': value
                                })

                if len(test_data) > 0:
                    # Create DataFrame for statannotations
                    test_df = pd.DataFrame(test_data)

                    # Check available groups
                    available_groups = test_df['group'].unique()
                    print(f"Available groups for {category} x {region}: {available_groups}")

                    # Compare Ref vs Circular, and Allocation vs Allocation + Circular
                    annotations_added = 0

                    if ref_label in available_groups and circular_label in available_groups:
                        ref_data = test_df[test_df['group'] == ref_label]
                        circular_data = test_df[test_df['group'] == circular_label]

                        if len(ref_data) >= 3 and len(circular_data) >= 3:
                            try:
                                ref_pairs = [(ref_label, circular_label)]
                                annotator_ref = Annotator(
                                    ax, ref_pairs, data=test_df, x='group', y='value', order=group_order,
                                )
                                annotator_ref.configure(
                                    test='Mann-Whitney',
                                    text_format='star',
                                    loc='outside',
                                    verbose=True,
                                    line_width=0.5,      # Thinner lines
                                    fontsize=5,          # Smaller text
                                    line_height=0.02,    # Smaller line height
                                )
                                annotator_ref.apply_and_annotate()
                                annotations_added += 1
                                print(f"Added Ref vs Circular comparison for {category} x {region}")
                            except Exception as inner_e:
                                print(f"Ref vs Circular comparison failed for {category} x {region}: {inner_e}")

                    if allocation_label in available_groups and alloc_circular_label in available_groups:
                        alloc_data = test_df[test_df['group'] == allocation_label]
                        alloc_circ_data = test_df[test_df['group'] == alloc_circular_label]

                        if len(alloc_data) >= 3 and len(alloc_circ_data) >= 3:
                            try:
                                alloc_pairs = [(allocation_label, alloc_circular_label)]
                                annotator_alloc = Annotator(
                                    ax, alloc_pairs, data=test_df, x='group', y='value', order=group_order,
                                )
                                annotator_alloc.configure(
                                    test='Mann-Whitney',
                                    text_format='star',
                                    loc='outside',
                                    verbose=True,
                                    line_width=0.5,      # Thinner lines
                                    fontsize=5,          # Smaller text
                                    line_height=0.02,    # Smaller line height
                                )
                                annotator_alloc.apply_and_annotate()
                                annotations_added += 1
                                print(f"Added Allocation vs Allocation + Circular comparison for {category} x {region}")
                            except Exception as inner_e:
                                print(f"Allocation comparison failed for {category} x {region}: {inner_e}")

                    if annotations_added == 0:
                        print(f"No annotations added for {category} x {region} - insufficient data or groups missing")

        except Exception as e:
            print(f"Warning: Could not add statistical annotation for {category} x {region}: {e}")
            pass

        # Format subplot
        ax.format(
            xticks=x_positions,
            xticklabels=[strategy_labels.get(s, s) for s in strategies] if cat_idx == len(categories) - 1 else [],
            xticklabelsize=3,
            yticklabelsize=3.5,
            xlim=global_xlim,
            ylim=global_ylim,
            ylabel=f'{category}', ylabelsize=4,
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
    xlabel='Strategies',
    xlabelsize=5,
    suptitle='Emissions comparison: different circular economy strategies with statistical annotations',
    xtickminor=False,
    ticklen=1.5,
    xrotation=90,
)

# Hide right and top spines for all axes
for ax in axes:
    ax.spines[['right', 'top']].set_visible(False)

# Add super y-axis label as text annotation
fig.text(0.03, 0.5, 'Gt', ha='center', va='center', rotation=90, fontsize=5)

# Save figure
output_path = "../../testpic/emissions_contribution_bar_anno.svg"
fig.save(output_path)
print(f"\nFigure saved to {output_path}")
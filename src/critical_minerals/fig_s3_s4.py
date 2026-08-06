import ultraplot as uplt
import polars as pl
import numpy as np
from matplotlib.patches import Patch


# Read and aggregate deployment data
print("Loading data from parquet...")
df = pl.read_parquet("../../data/data_raw/miracle_oscar/deployment_gaps_interp.parquet").filter(pl.col('year') == 2100).filter(~pl.col('subtech').str.starts_with('Motor_'))

print(f"Original data shape: {df.shape}")

# Sum across years - group by path_abb, strategy, sector, subtech, region
print("Aggregating across years...")
df_agg = (
    df.group_by(['path_abb', 'strategy', 'sector', 'subtech', 'region'])
    .agg([
        pl.sum('baseline_deployment').alias('baseline_deployment'),
        pl.sum('final_deployment').alias('final_deployment')
    ])
    .sort('region', 'strategy', 'subtech')
)

print(f"Aggregated data shape: {df_agg.shape}")

# Calculate deployment gap: baseline - final
# Positive gap = shortfall (final < baseline)
# Negative gap = surplus (final > baseline)
print("Calculating deployment gap...")
df_with_gap = df_agg.with_columns([
    (pl.col('baseline_deployment') - pl.col('final_deployment')).alias('gap')
])

# Join with category data
print("Loading and joining category data...")
category = pl.read_csv('../../data/data_raw/miracle_oscar/category.csv')
df_with_category = df_with_gap.join(category, on=['path_abb'], how='left')

# Aggregate across path_abb using MEAN (within each category).
# NOTE: deployment is zero-inflated and bimodal — for many subtechs the median
# is 0 because <50% of scenarios deploy, which would erase entire strategies
# (e.g. allo_bau_bau_recycl_sub shows ~0). Mean represents the expected
# deployment across the scenario ensemble and is additively consistent for the
# normalized stacked bars.
print("Aggregating across paths using mean...")
df_mean = (
    df_with_category
    .group_by(['category', 'strategy', 'sector', 'subtech', 'region'])
    .agg([
        pl.mean('baseline_deployment').alias('baseline_deployment'),
        pl.mean('final_deployment').alias('final_deployment'),
        pl.mean('gap').alias('gap')
    ])
    .sort('category', 'sector', 'strategy', 'region', 'subtech')
)

print(f"Mean aggregated data shape: {df_mean.shape}")

# Add tech_type column based on subtech prefix
print("Adding tech_type grouping...")
df_mean = df_mean.with_columns([
    pl.when(pl.col('subtech').str.starts_with('PV_')).then(pl.lit('PV'))
    .when(pl.col('subtech').str.starts_with('CSP_')).then(pl.lit('CSP'))
    .when(pl.col('subtech').str.starts_with('Onshore_')).then(pl.lit('Onshore'))
    .when(pl.col('subtech').str.starts_with('Offshore_')).then(pl.lit('Offshore'))
    .when(pl.col('subtech').str.starts_with('BEV_')).then(pl.lit('BEV'))
    .when(pl.col('subtech').str.starts_with('PHEV_')).then(pl.lit('PHEV'))
    .when(pl.col('subtech').str.starts_with('Motor_')).then(pl.lit('Motor'))
    .otherwise(pl.lit('Other'))
    .alias('tech_type')
])

# Strategy labels (shortened) - ORDER MATTERS! This defines the bar order
strategy_labels = {
    'bau_bau_bau_recycl_nosub': 'Reference',
    'bau_bau_bau_recycl_sub': 'Substitution',
    'allo_bau_bau_recycl_nosub': 'Global Allocation',
    'allo_bau_bau_recycl_sub': 'Global Allocation+Substitution',
}

# Get unique values
categories = sorted([cat for cat in df_mean['category'].unique().to_list() if cat is not None])
regions = sorted(df_mean['region'].unique().to_list())
# Use strategy order from strategy_labels (not alphabetical)
strategies = [s for s in strategy_labels.keys() if s in df_mean['strategy'].unique().to_list()]
sectors = sorted(df_mean['sector'].unique().to_list())

print(f"Categories: {len(categories)} - {categories}")
print(f"Regions: {len(regions)}")
print(f"Strategies: {len(strategies)}")
print(f"Sectors: {len(sectors)}")

# Define base colors and create gradients for each tech type
def create_gradient_colors(base_color, n_colors):
    """Create gradient colors with adaptive brightness range based on subtech count"""
    if n_colors == 1:
        return [base_color]

    # Convert hex to RGB
    base_color = base_color.lstrip('#')
    r, g, b = tuple(int(base_color[i:i+2], 16) / 255.0 for i in (0, 2, 4))

    # Determine max lightness factor based on number of colors
    if n_colors == 2:
        max_lightness = 0.5
    elif n_colors == 3:
        max_lightness = 0.6
    elif n_colors == 4:
        max_lightness = 0.7
    else:  # 5 or more
        max_lightness = 0.9

    colors = []
    for i in range(n_colors):
        # Create gradient from base color to brighter
        lightness_factor = (i / (n_colors - 1)) * max_lightness if n_colors > 1 else 0
        new_r = r + (1.0 - r) * lightness_factor
        new_g = g + (1.0 - g) * lightness_factor
        new_b = b + (1.0 - b) * lightness_factor
        colors.append((new_r, new_g, new_b))

    return colors

# Base colors for each tech type
tech_type_base_colors = {
    'PV': '##D39222',       # Orange
    'CSP': '#F5353C',      # Red
    'Onshore': '#18ADE7',  # Light Blue
    'Offshore': '#6A4C93', # Teal
    'BEV': '#AF7905',      # Purple
    'PHEV': '#4CC9F0',     # Yellow
}

# Create color mapping for all subtechs (will be used across all figures)
print("Creating color mapping for all tech types...")
global_color_map = {}
global_tech_types = []

for sector in sectors:
    sector_data = df_mean.filter(pl.col('sector') == sector)
    tech_types = sorted(sector_data['tech_type'].unique().to_list())

    for tech_type in tech_types:
        if tech_type not in global_tech_types:
            global_tech_types.append(tech_type)

        subtechs_in_type = sorted(
            sector_data.filter(pl.col('tech_type') == tech_type)['subtech'].unique().to_list()
        )
        n_subtechs = len(subtechs_in_type)
        base_color = tech_type_base_colors.get(tech_type, '#999999')
        gradient_colors = create_gradient_colors(base_color, n_subtechs)

        for i, subtech in enumerate(subtechs_in_type):
            global_color_map[subtech] = gradient_colors[i]

print(f"Global tech types: {global_tech_types}")

# Focus on C2 category only and plot by region
TARGET_CATEGORY = 'C2'

# Loop through sectors and create separate figures
for sector in sectors:
    print(f"\nCreating figure for {sector} sector (Category {TARGET_CATEGORY})...")
    uplt.rc["grid"] = False

    # 2×5 grid for 10 regions
    fig = uplt.figure(refwidth=2.2, facecolor=(0, 0, 0, 0), sharey=True, spany=True)
    axes = fig.subplots(nrows=2, ncols=5, wspace=1.5, hspace=3)

    # Bars are normalized to 100% per (region, strategy), so the ylabel is a
    # share regardless of sector.
    ylabel = 'Share of deployment (%)'

    # Filter data for this sector and C2 category
    sector_cat_data = df_mean.filter(
        (pl.col('sector') == sector) & (pl.col('category') == TARGET_CATEGORY)
    )

    if sector_cat_data.is_empty():
        print(f"  No data for {sector}-{TARGET_CATEGORY}")
        continue

    # Get tech types and subtechs for this sector
    tech_types = sorted(sector_cat_data['tech_type'].unique().to_list())
    all_subtechs = []
    for tech_type in tech_types:
        subtechs_in_type = sorted(
            sector_cat_data.filter(pl.col('tech_type') == tech_type)['subtech'].unique().to_list()
        )
        all_subtechs.extend(subtechs_in_type)

    # Aggregate across paths (mean within each region, strategy, subtech) for the
    # bar segments. (After filtering to one sector + C2 this is effectively a
    # no-op pass-through of df_mean, but kept for an explicit per-region table.)
    plot_data = (
        sector_cat_data
        .group_by(['region', 'strategy', 'subtech', 'tech_type'])
        .agg([
            pl.mean('baseline_deployment').alias('baseline_deployment'),
            pl.mean('final_deployment').alias('final_deployment'),
            pl.mean('gap').alias('gap')
        ])
    )

    # Bars are normalized to 100% per (region, strategy), so every bar tops
    # out at 120 — no need to compute a data-driven ylim.
    ylim_max = 120.0
    print(f"Setting ylim for {sector} sector: (0, {ylim_max:.2f})")

    # Plot for each region
    for reg_idx, region in enumerate(regions):
        row = reg_idx // 5
        col = reg_idx % 5
        ax = axes[row, col]

        print(f"\nPlotting {sector} sector, {region} region")

        # Filter data for this region
        region_data = plot_data.filter(pl.col('region') == region)

        if region_data.is_empty():
            print(f"  No data for {sector}-{region}")
            ax.format(title=region.replace('R10', ''))
            continue

        # Setup bar positions for strategies
        bar_width = 0.6  # Wider bars since we have only 4 strategies
        x_positions = np.arange(len(strategies))

        # Plot bars for each strategy
        for strat_idx, strategy in enumerate(strategies):
            # First pass: collect this strategy's subtech segments and the
            # total used for normalization (sum of positive final + positive
            # gap), so each bar sums to 100%.
            strat_rows = region_data.filter(pl.col('strategy') == strategy)
            segments = []  # (subtech, height)
            sum_final = 0.0
            total_gap = 0.0
            for subtech in all_subtechs:
                subtech_data = strat_rows.filter(pl.col('subtech') == subtech)
                if not subtech_data.is_empty():
                    height = subtech_data['final_deployment'][0]
                    gap = subtech_data['gap'][0]
                    total_gap += gap
                    if height > 0:
                        segments.append((subtech, height))
                        sum_final += height

            total = sum_final + max(total_gap, 0.0)
            if total <= 0:
                continue

            # Second pass: draw normalized segments (each / total * 100)
            bottom = 0.0
            for subtech, height in segments:
                height_norm = height / total * 100
                ax.bar(
                    np.array([x_positions[strat_idx]]),
                    np.array([height_norm]),
                    bar_width,
                    bottom=np.array([bottom]),
                    color=global_color_map.get(subtech, '#999999'),
                    edgecolor='none',
                    label=subtech if reg_idx == 0 and strat_idx == 0 else None
                )
                bottom += height_norm

            # Gap segment on top, normalized, if positive (baseline > final)
            if total_gap > 0:
                gap_norm = total_gap / total * 100
                ax.bar(
                    np.array([x_positions[strat_idx]]),
                    np.array([gap_norm]),
                    bar_width,
                    bottom=np.array([bottom]),
                    color='#CCCCCC',
                    linewidth=0.5,
                    alpha=0.6,
                    label='Gap (baseline > final)' if reg_idx == 0 and strat_idx == 0 else None
                )

        # Format subplot
        ax.format(
            title=region.replace('R10', ''),
            titlesize=8,
            xlabel='',
            ylabel='',
            xlabelsize=7,
            ylabelsize=7,
            xticks=x_positions,
            xticklabels=[strategy_labels.get(s, s) for s in strategies],
            xticklabelsize=7,
            yticklabelsize=7,
            xrotation=45,
        )

    # Overall formatting for this sector's figure
    axes.format(
        abc=True,
        abcloc="ul",
        ylim=(0, ylim_max),
        ylabel=ylabel,
        ylabelsize=10,
    )

    # Create legend organized by tech type in rows with variable columns
    print(f"\nCreating organized legend for {sector} sector...")

    legend_elements = []

    # Define the tech type order for legend rows
    tech_type_order = ['PV', 'CSP', 'Onshore', 'Offshore', 'BEV', 'PHEV', 'Motor']

    # Find max subtechs to determine row width (filter by this sector and C2)
    max_subtechs = 0
    for tech_type in tech_type_order:
        if tech_type in global_tech_types:
            count = len([s for s in global_color_map.keys()
                        if sector_cat_data.filter((pl.col('subtech') == s) & (pl.col('tech_type') == tech_type)).height > 0])
            max_subtechs = max(max_subtechs, count)

    # Use max subtechs as ncols (should be 7 for BEV/PHEV)
    ncols = max(max_subtechs, 7)

    # For each tech type, add all its subtechs to legend with blank padding to fill the row
    for tech_type in tech_type_order:
        if tech_type not in global_tech_types:
            continue

        # Get all subtechs for this tech type from the color map (filter by sector and C2)
        subtechs_for_type = []
        for subtech in global_color_map.keys():
            # Check if this subtech belongs to this tech type, sector, and C2
            subtech_rows = sector_cat_data.filter(
                (pl.col('subtech') == subtech) & (pl.col('tech_type') == tech_type)
            )
            if subtech_rows.height > 0:
                subtechs_for_type.append(subtech)

        if not subtechs_for_type:
            continue

        # Add patches for this tech type's subtechs
        for subtech in sorted(subtechs_for_type):
            color = global_color_map[subtech]
            patch = Patch(facecolor=color, edgecolor='none', alpha=0.8,
                         label=subtech.replace('_', ' '))
            legend_elements.append(patch)

        # Add blank padding to complete the row
        n_blanks = ncols - len(subtechs_for_type)
        for _ in range(n_blanks):
            legend_elements.append(Patch(facecolor='none', edgecolor='none', label=''))

    # Add gap legend with blank padding to fill the row
    gap_patch = Patch(facecolor='#CCCCCC', alpha=0.6, label='Gap (baseline > final)')
    legend_elements.append(gap_patch)
    for _ in range(ncols - 1):
        legend_elements.append(Patch(facecolor='none', edgecolor='none', label=''))

    fig.legend(
        legend_elements,
        [elem.get_label() for elem in legend_elements],
        ncols=ncols,
        fontsize=7,
        handlelength=1.0,
        handleheight=0.7,
        handletextpad=0.3,
        columnspacing=0.6,
        labelspacing=0.4,
        frame=False,
        loc="bottom",
    )

    # Save figure
    output_path = f"../../testpic/deploy_stacked_bar_C2_by_region_{sector}.pdf"
    fig.save(output_path)
    print(f"\nFigure saved to {output_path}")

print("\n" + "="*60)
print("Stacked bar plots with categories created successfully!")
print("="*60)

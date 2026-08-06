import polars as pl

# Load the same data as in fig_emissons_accum.py
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

print("Loading data...")
category = pl.read_csv('../../data/data_raw/miracle_oscar/category.csv')

co2_base = (
    pl.scan_parquet('../../data/data_raw/miracle_oscar/r10_ghg.parquet')
    .filter(pl.col('year').is_between(2021, 2100))
    .filter(pl.col('scope').is_in(['co2_industry', 'co2_energy']))
    .with_columns(path_abb = pl.concat_str(pl.col('model'), pl.col('scenario'), separator='|'))
    .group_by(['path_abb','region'])
    .agg((pl.sum('value') / 1000).alias('base'))  # Mt CO2 to Gt CO2
    .sort('path_abb','region')
    .collect()
)

co2_add_reg = (
    pl.scan_parquet("../../data/data_raw/miracle_oscar/co2_add_reg.parquet")
    .filter(pl.col('strategy').is_in(strategies))
    .with_columns(Emissions = pl.col('Emissions') / 1000)  # Mt CO2 to Gt CO2
    .group_by(['path_abb','region', 'strategy'])
    .agg(pl.col('Emissions').sum().alias('add'))
    .sort('path_abb','strategy','region')
    .collect()
)

co2 = (
    co2_add_reg.join(co2_base, on=['path_abb','region'])
    .join(category, on =['path_abb'])
    .select(pl.col('category','path_abb','region','strategy','add','base'))
)

print(f"Loaded data with {len(co2)} rows")
print(f"Categories: {sorted(co2['category'].unique().to_list())}")
print(f"Regions: {sorted(co2['region'].unique().to_list())}")
print(f"Strategies: {sorted(co2['strategy'].unique().to_list())}")

# Calculate regional contributions
print("\nCalculating regional contributions...")

# Method 1: Calculate contributions for additional emissions (add field)
emissions_contributions_add = (
    co2
    .filter(pl.col('category').is_not_null())  # Remove null categories
    .group_by(['category', 'strategy', 'path_abb'])
    .agg([
        pl.col('add').sum().alias('total_add_per_scenario'),
        pl.struct(['region', 'add']).alias('regional_data')
    ])
    .explode('regional_data')
    .unnest('regional_data')
    .with_columns([
        # Calculate percentage contribution of each region
        (pl.col('add') / pl.col('total_add_per_scenario') * 100).alias('add_contribution_pct'),
        # Calculate absolute contribution in Gt
        pl.col('add').alias('add_contribution_gt')
    ])
    .select([
        'category', 'strategy', 'path_abb', 'region',
        'add_contribution_pct', 'add_contribution_gt',
        'total_add_per_scenario'
    ])
    .sort(['category', 'strategy', 'path_abb', 'region'])
)

# Method 2: Calculate contributions for total emissions (base + add)
emissions_contributions_total = (
    co2
    .filter(pl.col('category').is_not_null())  # Remove null categories
    .with_columns(total_emissions = pl.col('base') + pl.col('add'))
    .group_by(['category', 'strategy', 'path_abb'])
    .agg([
        pl.col('total_emissions').sum().alias('total_emissions_per_scenario'),
        pl.struct(['region', 'total_emissions', 'base', 'add']).alias('regional_data')
    ])
    .explode('regional_data')
    .unnest('regional_data')
    .with_columns([
        # Calculate percentage contribution of each region for total emissions
        (pl.col('total_emissions') / pl.col('total_emissions_per_scenario') * 100).alias('total_contribution_pct'),
        # Keep absolute values
        pl.col('total_emissions').alias('total_contribution_gt'),
        pl.col('base').alias('base_contribution_gt'),
        pl.col('add').alias('add_contribution_gt')
    ])
    .select([
        'category', 'strategy', 'path_abb', 'region',
        'total_contribution_pct', 'total_contribution_gt',
        'base_contribution_gt', 'add_contribution_gt',
        'total_emissions_per_scenario'
    ])
    .sort(['category', 'strategy', 'path_abb', 'region'])
)

# Summary statistics across scenarios (median, percentiles)
print("Calculating summary statistics...")

# Summary for additional emissions contributions
summary_add_contributions = (
    emissions_contributions_add
    .group_by(['category', 'strategy', 'region'])
    .agg([
        pl.col('add_contribution_pct').median().alias('median_add_contribution_pct'),
        pl.col('add_contribution_pct').quantile(0.25).alias('p25_add_contribution_pct'),
        pl.col('add_contribution_pct').quantile(0.75).alias('p75_add_contribution_pct'),
        pl.col('add_contribution_gt').median().alias('median_add_contribution_gt'),
        pl.col('add_contribution_gt').quantile(0.25).alias('p25_add_contribution_gt'),
        pl.col('add_contribution_gt').quantile(0.75).alias('p75_add_contribution_gt'),
        pl.col('add_contribution_pct').count().alias('n_scenarios')
    ])
    .sort(['category', 'strategy', 'region'])
)

# Summary for total emissions contributions
summary_total_contributions = (
    emissions_contributions_total
    .group_by(['category', 'strategy', 'region'])
    .agg([
        pl.col('total_contribution_pct').median().alias('median_total_contribution_pct'),
        pl.col('total_contribution_pct').quantile(0.25).alias('p25_total_contribution_pct'),
        pl.col('total_contribution_pct').quantile(0.75).alias('p75_total_contribution_pct'),
        pl.col('total_contribution_gt').median().alias('median_total_contribution_gt'),
        pl.col('total_contribution_gt').quantile(0.25).alias('p25_total_contribution_gt'),
        pl.col('total_contribution_gt').quantile(0.75).alias('p75_total_contribution_gt'),
        pl.col('base_contribution_gt').median().alias('median_base_contribution_gt'),
        pl.col('add_contribution_gt').median().alias('median_add_contribution_gt'),
        pl.col('total_contribution_pct').count().alias('n_scenarios')
    ])
    .sort(['category', 'strategy', 'region'])
)

# Cut of additional emissions vs the baseline strategy (bau_bau_bau_recycl_nosub)
# For every path_abb × region × strategy, the "cut" is how much the strategy's
# additional emissions are reduced relative to the baseline strategy (Gt CO2).
# Baseline rows are kept with cut = 0 as a reference.
print("\nCalculating add-emissions cut vs baseline strategy...")
baseline_strategy = 'bau_bau_bau_recycl_nosub'

add_baseline = (
    co2
    .filter(pl.col('strategy') == baseline_strategy)
    .select(['path_abb', 'region', pl.col('add').alias('add_baseline')])
)

emissions_add_cut_detailed = (
    co2
    .filter(pl.col('category').is_not_null())
    .join(add_baseline, on=['path_abb', 'region'], how='left')
    .with_columns((pl.col('add_baseline') - pl.col('add')).alias('add_cut_gt'))
    .select([
        'category', 'path_abb', 'region', 'strategy',
        'add_baseline', pl.col('add').alias('add_strategy'), 'add_cut_gt',
    ])
    .sort(['category', 'strategy', 'path_abb', 'region'])
)

# Category-median summary: median cut across all path_abb within each
# (category, strategy, region), with IQR for uncertainty.
emissions_add_cut_summary = (
    emissions_add_cut_detailed
    .group_by(['category', 'strategy', 'region'])
    .agg([
        pl.col('add_cut_gt').median().alias('median_add_cut_gt'),
        pl.col('add_cut_gt').quantile(0.25).alias('p25_add_cut_gt'),
        pl.col('add_cut_gt').quantile(0.75).alias('p75_add_cut_gt'),
        pl.col('add_baseline').median().alias('median_add_baseline_gt'),
        pl.col('add_strategy').median().alias('median_add_strategy_gt'),
        pl.col('add_cut_gt').count().alias('n_path_abb'),
    ])
    .sort(['category', 'strategy', 'region'])
)

# Export results
print("Exporting results...")

# Export detailed scenario-level data (with contribution % per scenario)
emissions_contributions_add.write_parquet("../../data/data_output/emissions_contributions_add_detailed.parquet")
emissions_contributions_total.write_parquet("../../data/data_output/emissions_contributions_total_detailed.parquet")

# Export summary statistics (median of percentages across scenarios - the CORRECT approach)
summary_add_contributions.write_parquet("../../data/data_output/emissions_contributions_add_summary.parquet")
summary_total_contributions.write_parquet("../../data/data_output/emissions_contributions_total_summary.parquet")

# Export cut-vs-baseline tables
emissions_add_cut_detailed.write_parquet("../../data/data_output/emissions_add_cut_detailed.parquet")
emissions_add_cut_summary.write_parquet("../../data/data_output/emissions_add_cut_summary.parquet")

print("✅ Files exported:")
print("  - ../../data/data_output/emissions_contributions_add_detailed.parquet")
print("      (Contribution % for each scenario/path_abb)")
print("  - ../../data/data_output/emissions_contributions_total_detailed.parquet")
print("      (Contribution % for each scenario/path_abb, total emissions)")
print("  - ../../data/data_output/emissions_contributions_add_summary.parquet")
print("      (Median contribution % across scenarios, with IQR)")
print("  - ../../data/data_output/emissions_contributions_total_summary.parquet")
print("      (Median contribution % across scenarios, total emissions)")
print("  - ../../data/data_output/emissions_add_cut_detailed.parquet")
print("      (Add-emissions cut vs baseline strategy, per path_abb × region × strategy)")
print("  - ../../data/data_output/emissions_add_cut_summary.parquet")
print("      (Median cut vs baseline across path_abb, per category × strategy × region)")

# Display sample results
print("\n" + "="*80)
print("REGIONAL CONTRIBUTION ANALYSIS")
print("="*80)
print("\n📊 Additional emissions contributions (median across scenarios):")
print(summary_add_contributions.head(15))

print("\n" + "="*80)
print("ADD-EMISSIONS CUT vs BASELINE (bau_bau_bau_recycl_nosub)")
print("="*80)
print("\n🔪 Median cut (Gt CO2) across path_abb, per category × strategy × region:")
print(emissions_add_cut_summary.head(15))

print("\n" + "="*80)
print("METHODOLOGY")
print("="*80)
print("""
For each category-strategy-region combination:

1. Calculate contribution % for EACH scenario (path_abb):
   contribution_pct = (region_emissions / total_emissions_in_scenario) * 100

   Within each scenario, all regional contributions sum to 100%

2. Take the median of contribution_pct across all scenarios
   Also calculate 25th and 75th percentiles (IQR) for uncertainty

This is the ONLY correct approach because:
- Different path_abb represent alternative scenarios (not additive)
- Different categories represent different climate pathways (not additive)
- Strategies represent different policy options applied to same scenarios

Example for C1, bau_bau_bau_recycl_nosub, CHINA+:
- Scenario 1: CHINA+ emits 300 Gt out of 1000 Gt total = 30%
- Scenario 2: CHINA+ emits 250 Gt out of 900 Gt total = 27.8%
- Scenario 3: CHINA+ emits 280 Gt out of 950 Gt total = 29.5%
- Median contribution: median([30%, 27.8%, 29.5%]) = 29.5%
- IQR: [27.8%, 30%]
""")
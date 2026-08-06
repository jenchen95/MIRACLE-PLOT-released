import polars as pl

# Use the same scenario name as in fig1.py
scen_name = "edrive_singweight"

# Load the same strategy list as in tab_emissions_add_global_stats.py
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

print("Loading historical data for base temperature calculation...")
# Load historical data to calculate pre-industrial base (1850-1900 mean)
hist = pl.read_parquet(
    f"../../data/data_raw/miracle_oscar/{scen_name}/dtg_weighted_hist_2000mc.parquet"
).sort("year")

# Calculate pre-industrial base temperature (1850-1900 mean)
base_temp = (
    hist.filter(pl.col("year").is_between(1850, 1900))
    .filter(pl.col("measure") == "mean")
    .select(pl.mean(["value"]))
    .to_numpy()
    .ravel()
    .item()
)

print(f"Pre-industrial base temperature: {base_temp:.4f}°C")

print("Loading scenario data (baseline/scen)...")
# Load scenario data (treated as baseline for comparison)
scen = (
    pl.scan_parquet(
        f"../../data/data_raw/miracle_oscar/{scen_name}/dtg_weighted_scen_2000mc.parquet"
    )
    .with_columns(
        pl.col("scen")
        .str.splitn("|", 2)
        .struct.rename_fields(["model", "scenario"])
    )
    .unnest("scen")
    .filter(pl.col("measure") == "mean")
    .filter(pl.col("year") == 2100)  # Focus on 2100 only
    .select(pl.col('model', 'scenario', 'category', 'year', 'value'))
    .with_columns((pl.col('value') - base_temp).alias('warming_scen'))  # Offset by base temp
    .collect()
)

print("Loading counterfactual data (cf)...")
# Load counterfactual data with specified strategies
cf = (
    pl.scan_parquet(
        f"../../data/data_raw/miracle_oscar/{scen_name}/dtg_weighted_cf_2000mc_category*_scen_all.parquet"
    )
    .with_columns(
        pl.col("scen")
        .str.splitn("|", 3)
        .struct.rename_fields(["model", "scenario", "strategy"])
    )
    .unnest("scen")
    .filter(pl.col("strategy").is_in(strategies))
    .filter(pl.col("measure") == "mean")
    .filter(pl.col("year") == 2100)  # Focus on 2100 only
    .select(pl.col('model', 'scenario', 'category', 'strategy', 'year', 'value'))
    .with_columns((pl.col('value') - base_temp).alias('warming_cf'))  # Offset by base temp
    .collect()
)

print("Calculating additional warming (cf - scen) at raw data level...")
# Join cf and scen data to calculate additional warming for each model/scenario run
# (This is kept for backward compatibility with existing analysis, but the comparison matrix contains all the info)
warming_add = (
    cf.join(
        scen.select('model', 'scenario', 'category', 'warming_scen'),
        on=['model', 'scenario', 'category'],
        how='inner'
    )
    .with_columns(
        (pl.col('warming_cf') - pl.col('warming_scen')).alias('add_warming')
    )
    .select('model', 'scenario', 'category', 'strategy', 'add_warming')
)

# Note: The comparison matrix below contains all strategy vs ideal comparisons
# plus all strategy vs strategy comparisons in a unified format

# Prepare all strategies including 'ideal' for Cartesian product comparison
all_strategies = ['ideal'] + strategies  # Include ideal as a strategy
print("Creating Cartesian product of all strategy comparisons...")
print(f"All strategies: {all_strategies}")

# Prepare unified data structure: combine scen (ideal) and cf data
print("Preparing unified strategy data...")
# Convert scen data to same format as cf data
scen_unified = (
    scen
    .with_columns(pl.lit('ideal').alias('strategy'))
    .select(['model', 'scenario', 'category', 'strategy', 'warming_scen'])
    .rename({'warming_scen': 'warming'})
)

# Convert cf data to same format
cf_unified = (
    cf
    .select(['model', 'scenario', 'category', 'strategy', 'warming_cf'])
    .rename({'warming_cf': 'warming'})
)

# Combine all strategy data
all_strategy_data = pl.concat([scen_unified, cf_unified])

# Create Cartesian product comparison matrix
comparison_results = []

for category in all_strategy_data['category'].unique().sort():
    print(f"\nProcessing category: {category}")
    category_data = all_strategy_data.filter(pl.col('category') == category)

    # Create all possible strategy pairs (Cartesian product)
    for strategy1 in all_strategies:
        for strategy2 in all_strategies:
            print(f"  Comparing {strategy1} vs {strategy2}", end="")

            if strategy1 == strategy2:
                # Same strategy compared to itself = 0
                comparison_results.append({
                    'category': category,
                    'strategy': strategy1,
                    'base_strategy': strategy2,
                    'diff_p05': 0.0,
                    'diff_p25': 0.0,
                    'diff_median': 0.0,
                    'diff_p75': 0.0,
                    'diff_p95': 0.0
                })
                print(": 0.000°C (same strategy)")
            else:
                # Get data for both strategies
                strategy1_data = (
                    category_data
                    .filter(pl.col('strategy') == strategy1)
                    .select(['model', 'scenario', 'warming'])
                    .rename({'warming': 'strategy1_warming'})
                )

                strategy2_data = (
                    category_data
                    .filter(pl.col('strategy') == strategy2)
                    .select(['model', 'scenario', 'warming'])
                    .rename({'warming': 'strategy2_warming'})
                )

                # Check if both strategies have data
                if not strategy1_data.is_empty() and not strategy2_data.is_empty():
                    # Join and calculate paired differences (strategy1 - strategy2)
                    paired_differences = (
                        strategy1_data.join(strategy2_data, on=['model', 'scenario'], how='inner')
                        .with_columns(
                            (pl.col('strategy1_warming') - pl.col('strategy2_warming')).alias('temp_diff')
                        )
                    )

                    if not paired_differences.is_empty():
                        # Calculate statistics of the differences
                        diff_stats = paired_differences.select([
                            pl.col('temp_diff').quantile(0.05).alias('diff_p05'),
                            pl.col('temp_diff').quantile(0.25).alias('diff_p25'),
                            pl.col('temp_diff').quantile(0.50).alias('diff_median'),
                            pl.col('temp_diff').quantile(0.75).alias('diff_p75'),
                            pl.col('temp_diff').quantile(0.95).alias('diff_p95')
                        ]).to_dict(as_series=False)

                        comparison_results.append({
                            'category': category,
                            'strategy': strategy1,
                            'base_strategy': strategy2,
                            'diff_p05': diff_stats['diff_p05'][0],
                            'diff_p25': diff_stats['diff_p25'][0],
                            'diff_median': diff_stats['diff_median'][0],
                            'diff_p75': diff_stats['diff_p75'][0],
                            'diff_p95': diff_stats['diff_p95'][0]
                        })

                        print(f": {diff_stats['diff_median'][0]:.3f}°C")
                    else:
                        print(": No paired data")
                        comparison_results.append({
                            'category': category,
                            'strategy': strategy1,
                            'base_strategy': strategy2,
                            'diff_p05': None,
                            'diff_p25': None,
                            'diff_median': None,
                            'diff_p75': None,
                            'diff_p95': None
                        })
                else:
                    print(": Missing data")
                    comparison_results.append({
                        'category': category,
                        'strategy': strategy1,
                        'base_strategy': strategy2,
                        'diff_p05': None,
                        'diff_p25': None,
                        'diff_median': None,
                        'diff_p75': None,
                        'diff_p95': None
                    })

# Create final comparison matrix DataFrame
final_comparison_matrix = pl.DataFrame(comparison_results).sort('category', 'strategy', 'base_strategy')

# Save results - only need the comparison matrix since it contains everything
output_file = '../../data/data_output/warming_comparison_matrix.csv'

# Save full comparison matrix (contains all strategies vs all strategies, including vs ideal)
final_comparison_matrix.write_csv(output_file)

print(f"Complete comparison matrix saved to: {output_file}")
print(f"Matrix shape: {final_comparison_matrix.shape}")
print(f"Number of strategy pairs per category: {len(all_strategies)}² = {len(all_strategies)**2}")
print("Includes all strategies vs ideal, plus all strategy-vs-strategy comparisons")

print("\nSample of comparison matrix:")
print(final_comparison_matrix.head(10))

print("\nTo get specific comparisons:")
print("- All strategies vs ideal: filter where base_strategy='ideal'")
print("- Strategy A vs Strategy B: filter where strategy='A' and base_strategy='B'")
print("- Strategy performance ranking: sort by diff_median for any base_strategy")
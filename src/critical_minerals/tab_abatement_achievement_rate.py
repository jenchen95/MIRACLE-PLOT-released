import polars as pl
from funcs import fit_linear

# Strategy configuration
strategies = [
    "bau_bau_bau_recycl_nosub",
    "bau_llt_mir_recych_nosub",
    "bau_bau_bau_recycl_sub",
    "bau_llt_mir_recych_sub",
    "allo_bau_bau_recycl_nosub",
    "allo_llt_mir_recych_nosub",
    "allo_bau_bau_recycl_sub",
    "allo_llt_mir_recych_sub",
]

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

frame = (
    pl.scan_parquet("../../data/data_raw/miracle_oscar/r10_final_abate_agg.parquet")
    .group_by(['path_abb', 'strategy', 'year'])
    .agg(
        pl.sum('final_deployment').alias('final_deployment'),
        pl.sum('abate').alias('abate'),
        pl.sum('baseline_deployment').alias('baseline_deployment')
    )
    .join(category, on=["path_abb"], how="left")
    .filter(pl.col("category") == "C2")
    .join(abapath_n, on=['path_abb', 'year'], how='left')
    .filter(pl.col("strategy").is_in(strategies))
    .select('category', 'path_abb', 'strategy', 'year', 'final_deployment', 'abate', 'baseline_deployment', 'ideal_abate')
    .collect()
)

# Interpolate missing years using fit_linear
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

# Calculate cumulative values
frame = (
    frame.with_columns([
        pl.col("final_deployment").cum_sum().over(["path_abb", "strategy"]).alias("cum_final_deployment"),
        pl.col('abate').cum_sum().over(['path_abb', 'strategy']).alias('cum_abate'),
        pl.col("baseline_deployment").cum_sum().over(["path_abb", "strategy"]).alias("cum_baseline_deployment"),
        pl.col('ideal_abate').cum_sum().over(['path_abb', 'strategy']).cast(pl.Float64).alias('cum_ideal_abate')
    ])
    .select('category', 'path_abb', 'strategy', 'year', 'cum_final_deployment', 'cum_abate', 'cum_baseline_deployment', 'cum_ideal_abate')
)

# Filter for year 2100
data_2100 = frame.filter(pl.col('year') == 2100)

# Calculate achievement rate and window statistics over strategy
data_2100 = data_2100.with_columns(
    (pl.col('cum_abate') / pl.col('cum_ideal_abate') * 100).alias('achievement_rate')
).with_columns([
    pl.col('achievement_rate').median().over('strategy').alias('strategy_median_rate'),
    pl.col('achievement_rate').quantile(0.05).over('strategy').alias('strategy_p5_rate'),
    pl.col('achievement_rate').quantile(0.95).over('strategy').alias('strategy_p95_rate'),
])

# Export to CSV
output_path = "../../data/data_output/abatement_achievement_rate.csv"
data_2100.write_csv(output_path)
print(f"Data exported to {output_path}")
print(f"Shape: {data_2100.shape}")
print(f"Columns: {data_2100.columns}")
print(f"\nSample data:")
print(data_2100.head(10))

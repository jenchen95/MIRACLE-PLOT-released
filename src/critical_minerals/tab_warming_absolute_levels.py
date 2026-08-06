from pathlib import Path

import polars as pl

# Use the same scenario name as in fig1.py
scen_name = "edrive_singweight"

# Resolve paths relative to repo root (so CWD doesn't matter)
REPO_ROOT = Path(__file__).resolve().parents[2]
DATA_RAW = REPO_ROOT / "data" / "data_raw" / "miracle_oscar" / scen_name
DATA_OUTPUT = REPO_ROOT / "data" / "data_output"

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
    DATA_RAW / "dtg_weighted_hist_2000mc.parquet"
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

print("Loading scenario data (ideal strategy)...")
# Load scenario data (treated as 'ideal' strategy)
scen = (
    pl.scan_parquet(
        DATA_RAW / "dtg_weighted_scen_2000mc.parquet"
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
    .with_columns((pl.col('value') - base_temp).alias('warming'))  # Offset by base temp
    .with_columns(pl.lit('ideal').alias('strategy'))  # Add strategy column
    .select(['model', 'scenario', 'category', 'strategy', 'warming'])
    .collect()
)

print("Loading counterfactual data (cf strategies)...")
# Load counterfactual data with specified strategies
cf = (
    pl.scan_parquet(
        str(DATA_RAW / "dtg_weighted_cf_2000mc_category*_scen_all.parquet")
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
    .with_columns((pl.col('value') - base_temp).alias('warming'))  # Offset by base temp
    .select(['model', 'scenario', 'category', 'strategy', 'warming'])
    .collect()
)

print("Combining all strategy data...")
# Combine ideal (scen) and cf strategies
all_strategy_data = pl.concat([scen, cf])

print("Preparing absolute warming details (year 2100)...")
warming_absolute_details = (
    all_strategy_data
    .select(["category", "model", "scenario", "strategy", "warming"])
    .sort(["category", "model", "scenario", "strategy"])
)

# Save results
output_file = DATA_OUTPUT / "warming_absolute_levels_details.csv"
DATA_OUTPUT.mkdir(parents=True, exist_ok=True)
warming_absolute_details.write_csv(output_file)

print(f"\nAbsolute warming levels saved to: {output_file}")
print(f"Table shape: {warming_absolute_details.shape}")

print("\nSample of results:")
print(warming_absolute_details.head(10))

print("\nSummary statistics:")
print(f"- All values are in °C relative to pre-industrial baseline ({base_temp:.4f}°C)")
print(f"- Year: 2100")
print(f"- Includes 'ideal' strategy + {len(strategies)} counterfactual strategies")

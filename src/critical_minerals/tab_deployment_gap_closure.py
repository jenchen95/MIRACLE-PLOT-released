import polars as pl

# Regional-level deployment gap closure.
#
# gap_closure_pct = final_deployment / baseline_deployment * 100
#   100%  -> baseline target met (gap closed)
#   <100% -> shortfall remains (positive gap)
#   >100% -> over-deployment (surplus)
#   baseline == 0 -> null (undefined, excluded from stats)
#
# Regional level = subtechs summed within each (path_abb, strategy, sector,
# region) BEFORE taking the ratio, giving a value-weighted closure rate.

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

print("Loading deployment data...")
category = pl.read_csv("../../data/data_raw/miracle_oscar/category.csv")

# Year 2100, exclude Motor_* subtechs (consistent with fig_s3_s4).
df = (
    pl.read_parquet("../../data/data_raw/miracle_oscar/deployment_gaps_interp.parquet")
    .filter(pl.col("year") == 2100)
    .filter(~pl.col("subtech").str.starts_with("Motor_"))
    .filter(pl.col("strategy").is_in(strategies))
)

# Regional aggregation: sum subtechs (and run_id replicates) within each
# (path_abb, strategy, sector, region). gap is derived as baseline - final.
regional = (
    df
    .group_by(["path_abb", "strategy", "sector", "region"])
    .agg([
        pl.sum("baseline_deployment").alias("baseline_deployment"),
        pl.sum("final_deployment").alias("final_deployment"),
    ])
    .with_columns((pl.col("baseline_deployment") - pl.col("final_deployment")).alias("gap"))
    .join(category, on=["path_abb"], how="left")
    .with_columns(
        pl.when(pl.col("baseline_deployment") > 0)
        .then(pl.col("final_deployment") / pl.col("baseline_deployment") * 100)
        .otherwise(None)
        .alias("gap_closure_pct")
    )
    .select([
        "category", "path_abb", "strategy", "sector", "region",
        "baseline_deployment", "final_deployment", "gap", "gap_closure_pct",
    ])
    .sort(["category", "sector", "strategy", "region", "path_abb"])
)

# Global rows: sum baseline and final across ALL regions per scenario
# (path_abb x strategy x sector), recompute gap_closure_pct, and label
# region = "Global". Appended to the regional tables so a single parquet holds
# both per-region and global rows (region column distinguishes them).
global_rows = (
    regional
    .filter(pl.col("category").is_not_null())
    .group_by(["category", "path_abb", "strategy", "sector"])
    .agg([
        pl.sum("baseline_deployment").alias("baseline_deployment"),
        pl.sum("final_deployment").alias("final_deployment"),
    ])
    .with_columns((pl.col("baseline_deployment") - pl.col("final_deployment")).alias("gap"))
    .with_columns(
        pl.when(pl.col("baseline_deployment") > 0)
        .then(pl.col("final_deployment") / pl.col("baseline_deployment") * 100)
        .otherwise(None)
        .alias("gap_closure_pct")
    )
    .with_columns(pl.lit("Global").alias("region"))
    .select([
        "category", "path_abb", "strategy", "sector", "region",
        "baseline_deployment", "final_deployment", "gap", "gap_closure_pct",
    ])
)

regional = pl.concat([regional, global_rows]).sort(["category", "sector", "strategy", "region", "path_abb"])

# Summary across path_abb within each (category, strategy, sector, region).
# Includes the "Global" rows. Both mean and median are reported: the deployment
# data is zero-inflated, so median is 0 for combos where <50% of scenarios
# deploy, while mean captures the deploying mass — read whichever fits.
summary = (
    regional
    .filter(pl.col("category").is_not_null())
    .group_by(["category", "strategy", "sector", "region"])
    .agg([
        pl.col("gap_closure_pct").mean().alias("mean_gap_closure_pct"),
        pl.col("gap_closure_pct").median().alias("median_gap_closure_pct"),
        pl.col("gap_closure_pct").quantile(0.25).alias("p25_gap_closure_pct"),
        pl.col("gap_closure_pct").quantile(0.75).alias("p75_gap_closure_pct"),
        pl.col("baseline_deployment").mean().alias("mean_baseline_deployment"),
        pl.col("final_deployment").mean().alias("mean_final_deployment"),
        pl.col("gap_closure_pct").count().alias("n_path_abb"),
    ])
    .sort(["category", "sector", "strategy", "region"])
)

# Export
regional.write_parquet("../../data/data_output/deployment_gap_closure_regional.parquet")
summary.write_parquet("../../data/data_output/deployment_gap_closure_summary.parquet")

print("✅ Files exported:")
print("  - ../../data/data_output/deployment_gap_closure_regional.parquet")
print(f"      (per path_abb × strategy × sector × region, incl. region='Global'; {regional.height} rows)")
print("  - ../../data/data_output/deployment_gap_closure_summary.parquet")
print(f"      (mean/median + IQR across path_abb, per region incl. 'Global'; {summary.height} rows)")

print(f"\nCategories: {sorted([c for c in regional['category'].unique().to_list() if c is not None])}")
print(f"Sectors: {sorted(regional['sector'].unique().to_list())}")
print(f"Regions: {sorted(regional['region'].unique().to_list())}")

print("\nSample regional summary (C2, Power):")
print(
    summary
    .filter((pl.col("category") == "C2") & (pl.col("sector") == "Power"))
    .head(12)
)

print("\nGlobal summary rows (all categories, Power):")
print(
    summary
    .filter((pl.col("sector") == "Power") & (pl.col("region") == "Global"))
    .select(["category", "strategy", "mean_gap_closure_pct", "median_gap_closure_pct",
             "p25_gap_closure_pct", "p75_gap_closure_pct", "n_path_abb"])
)

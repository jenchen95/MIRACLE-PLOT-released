import polars as pl

print("Loading carbon sequestration data...")

# 1) Load and filter sequestration data to year 2100 and target variable
sequestration = (
    pl.scan_parquet("../../data/data_raw/miracle_oscar/r10_carbon_sequestration_interp_c18.parquet")
    .filter(pl.col("year") == 2100)
    .filter(pl.col("variable") == "Carbon Sequestration|CCS")
    .group_by(["model", "scenario", "unit"])
    .agg(pl.first("sum_2025_2100_world").alias("sum_2025_2100_world"))
    .with_columns(
        path_abb=pl.concat_str(pl.col("model"), pl.col("scenario"), separator="|")
    )
)

# 2) Keep only path_abb that appear in abapath_n.parquet
valid_paths = (
    pl.scan_parquet("../../data/data_raw/miracle_oscar/abapath_n.parquet")
    .select("path_abb")
    .unique()
)

# 3) Build path_abb in category.csv and join category info
category = (
    pl.scan_csv("../../data/data_raw/miracle_oscar/category.csv")
    .with_columns(
        path_abb=pl.concat_str(pl.col("model"), pl.col("scenario"), separator="|")
    )
    .select(["path_abb", "category"])
    .unique()
)

frame = (
    sequestration
    .join(valid_paths, on="path_abb", how="inner")
    .join(category, on="path_abb", how="left")
    .collect()
)

# 4) Category-level summary table only (no path/model/scenario in output)
frame = (
    frame
    .filter(pl.col("category").is_not_null())
    .filter(pl.col("category").str.contains(r"^C[1-8]$"))
    .group_by(["category", "unit"])
    .agg([
        pl.col("sum_2025_2100_world").median().alias("median"),
        pl.col("sum_2025_2100_world").quantile(0.05).alias("p5"),
        pl.col("sum_2025_2100_world").quantile(0.25).alias("p25"),
        pl.col("sum_2025_2100_world").quantile(0.75).alias("p75"),
        pl.col("sum_2025_2100_world").quantile(0.95).alias("p95"),
    ])
    .sort(["category", "unit"])
)

output_path = "../../data/data_output/tab_carbon_sequestration.parquet"
frame.write_parquet(output_path)

print(f"Data exported to {output_path}")
print(f"Shape: {frame.shape}")
print(f"Columns: {frame.columns}")
print("\nSample data:")
print(frame.head(10))

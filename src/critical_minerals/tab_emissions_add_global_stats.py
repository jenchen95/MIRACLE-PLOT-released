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

co2_add = (
    pl.read_parquet("../../data/data_raw/miracle_oscar/co2_add_reg.parquet")
    .filter(pl.col('strategy').is_in(strategies))
    .group_by(['path_abb', 'strategy'])
    .agg((pl.sum('Emissions') / 1000).alias('add'))  # Mt CO2 to Gt CO2
    .join(category, on=['path_abb'], how='left')
    .group_by(['category', 'strategy'])
    .agg(
        p05 = pl.col('add').quantile(0.05),
        p25 = pl.col('add').quantile(0.25),
        median = pl.col('add').quantile(0.50),
        p75 = pl.col('add').quantile(0.75),
        p95 = pl.col('add').quantile(0.95)
    )
    .sort('category', 'strategy')
)

co2_add.write_csv('../../data/data_output/co2_add_global_stats.csv')

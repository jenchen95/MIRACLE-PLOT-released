import polars as pl

temp_scen = (
    pl.scan_parquet('../../data/data_raw/miracle_oscar/edrive_singweight/dtg_weighted_scen_2000mc.parquet')
    .filter(pl.col('measure') == 'mean')
    .select(pl.col('scen').str.splitn('|', 2).struct.rename_fields(['model', 'scenario']), pl.col('category'), pl.col('year'), pl.col('value').alias('value_scen'))
    .unnest('scen')
)

temp_cf = (
    pl.scan_parquet('../../data/data_raw/miracle_oscar/edrive_singweight/dtg_weighted_cf_2000mc_category*_scen_all.parquet')
    .filter(pl.col('measure') == 'mean')
    .select(pl.col('scen').str.splitn('|', 3).struct.rename_fields(['model', 'scenario', 'strategy']), pl.col('category'), pl.col('year'), pl.col('value').alias('value_cf'))
    .unnest('scen')
)

temp_diff_2100 = (
    temp_cf.join(temp_scen, on=['model', 'scenario', 'category', 'year'], how='left')
    .with_columns(
        pl.col('value_cf').sub(pl.col('value_scen')).alias('diff_agg')
    )
    .filter(pl.col('year')==2100)
    .select(pl.col('model', 'scenario', 'strategy', 'category', 'diff_agg', 'value_scen', 'value_cf'))
)

temp_diff_2100.sink_parquet('../../data/data_raw/miracle_oscar/temp_diff_2100.parquet')

co2_add_reg_ratio = (
    pl.scan_parquet("../../data/data_raw/miracle_oscar/co2_add_reg.parquet")
    .select(pl.col('path_abb').str.splitn('|', 2).struct.rename_fields(['model', 'scenario']), pl.col('strategy'), pl.col('region'), pl.col('Emissions'))
    .unnest('path_abb')
    .group_by('model','scenario','strategy','region')
    .agg(pl.sum('Emissions'))
    .select(pl.col('model', 'scenario', 'strategy', 'region', 'Emissions'))
    .with_columns(
        ratio = (pl.col('Emissions') / (pl.col('Emissions').sum()).over('model','scenario','strategy')),
        Emissions_sum = pl.col('Emissions').sum().over('model','scenario','strategy')
    )
    .select(pl.col('model', 'scenario', 'strategy', 'region', 'ratio', 'Emissions_sum'))
    .sort('model','scenario','strategy','region')
)

temp_diff_2100_by_reg = (
    temp_diff_2100.join(co2_add_reg_ratio, on=['model', 'scenario', 'strategy'], how='inner')
    .with_columns(
        diff_reg = pl.when(
            (pl.col('diff_agg') > 0) & (pl.col('Emissions_sum') > 0) |
            (pl.col('diff_agg') < 0) & (pl.col('Emissions_sum') < 0)
        )
        .then(pl.col('diff_agg') * pl.col('ratio'))
        .otherwise(0)
    )
    .select(pl.exclude('diff_agg', 'Emissions_sum', 'value_scen', 'value_cf'))
    .collect()
)

temp_diff_2100_by_reg.write_parquet('../../data/data_raw/miracle_oscar/temp_diff_2100_by_reg.parquet')

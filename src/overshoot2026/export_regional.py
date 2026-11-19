"""Slide 7, right: regional.json, extra warming in 2100 attributed to each R10 region.

    uv run python src/overshoot2026/export_regional.py [DATA_DIR] [--out DECK_DATA_DIR]

The regional rose of fig3.py: temp_diff_2100_by_reg.parquet (written by
temp_diff_by_reg.py: the 2100 warming of a strategy minus Ideal, split over the regions
by their share of the strategy's additional CO2), then per region the median of
`diff_reg` over the pathways of category C2, in °C.
Also kept (not drawn): the median share of additional CO2 per region
(tab_emissions_contribution.py, median_add_contribution_pct).
"""

import polars as pl

from _shared import CATEGORY, CODE_TO_ID, STRATEGIES, parse_args, r, write_json

REGIONS = [
    ("R10AFRICA", "Africa"), ("R10CHINA+", "China+"), ("R10EUROPE", "Europe"), ("R10INDIA+", "India+"),
    ("R10LATIN_AM", "Latin America"), ("R10MIDDLE_EAST", "Middle East"), ("R10NORTH_AM", "North America"),
    ("R10PAC_OECD", "Pacific OECD"), ("R10REF_ECON", "Reforming Economies"), ("R10REST_ASIA", "Rest of Asia"),
]


def main() -> None:
    args = parse_args(__doc__.splitlines()[0])
    d = args.data_dir
    codes = list(STRATEGIES.values())

    # fig3.py plot_regional_rose: median diff_reg per region, per category and strategy
    df_reg = pl.read_parquet(d / "temp_diff_2100_by_reg.parquet").filter(
        pl.col("strategy").is_in(codes), pl.col("category") == CATEGORY
    )
    rose = df_reg.group_by("strategy", "region").agg(pl.median("diff_reg").alias("median_diff"))
    # fig3.py inset bar: sum over regions per pathway, then median
    glob = (
        df_reg.group_by("model", "scenario", "strategy").agg(pl.sum("diff_reg").alias("global_diff"))
        .group_by("strategy").agg(pl.median("global_diff"), pl.len().alias("n"))
    )

    # tab_emissions_contribution.py, method 1: share of additional CO2 per region
    category = pl.read_csv(d / "category.csv").with_columns(
        path_abb=pl.concat_str("model", "scenario", separator="|")
    ).select("path_abb", "category")
    add = (
        pl.scan_parquet(d / "co2_add_reg.parquet").filter(pl.col("strategy").is_in(codes))
        .with_columns(Emissions=pl.col("Emissions") / 1000)
        .group_by("path_abb", "region", "strategy").agg(pl.col("Emissions").sum().alias("add"))
        .collect()
        .join(category, on="path_abb")
        .filter(pl.col("category") == CATEGORY)
        .with_columns(pct=pl.col("add") / pl.col("add").sum().over("strategy", "path_abb") * 100)
        .group_by("strategy", "region").agg(pl.col("pct").median())
    )

    ids = [c for c, _ in REGIONS]
    values, shares, totals = {}, {}, {}
    for code, sid in CODE_TO_ID.items():
        v = {row["region"]: row["median_diff"] for row in rose.filter(pl.col("strategy") == code).iter_rows(named=True)}
        values[sid] = {reg: r(v.get(reg, 0.0), 4) for reg in ids}
        s = {row["region"]: row["pct"] for row in add.filter(pl.col("strategy") == code).iter_rows(named=True)}
        shares[sid] = {reg: r(s.get(reg), 1) for reg in ids}
        g = glob.filter(pl.col("strategy") == code)
        totals[sid] = r(g["global_diff"][0], 3) if g.height else None

    data = {
        "metric_label": "Extra warming in 2100 by region of extra CO₂",
        "unit": "°C",
        "decimals": 3,
        "sort_by": "bau",
        "source": f"MIRACLE-OSCAR temp_diff_2100_by_reg, category {CATEGORY}, median over pathways "
                  f"(n={glob['n'].max()}); logic of MIRACLE-PLOT fig3.py (regional rose)",
        "category": CATEGORY,
        "regions": [{"id": c, "label": l} for c, l in REGIONS],
        "values": values,
        "global_median": totals,
        "share_of_additional_co2_pct": shares,
    }
    write_json(args.out, "regional", data)
    for sid, v in values.items():
        top = sorted(v.items(), key=lambda kv: -kv[1])[:3]
        print(f"  {sid:13s} global {totals[sid]}  top: " + ", ".join(f"{k} {x:.3f}" for k, x in top))


if __name__ == "__main__":
    main()

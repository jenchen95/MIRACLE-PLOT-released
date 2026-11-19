"""Slide 7, left: deployment_gap.json, deployment by technology as % of the scenario baseline.

    uv run python src/overshoot2026/export_deployment_gap.py [DATA_DIR] [--out DECK_DATA_DIR]

Reuses fig2.py's global deployment aggregation (aggregate_medians, category C2):
deduplicate the element rows, then per sector the median over pathways of the
achievement ratio (final / baseline), the median technology composition of the final
deployment, and the median baseline per technology. fig2.py stacks these as a share of
the sector baseline; here each technology's final over its own baseline gives the
"% of baseline" bar the deck draws.
"""

import polars as pl

from _shared import CATEGORY, CODE_TO_ID, STRATEGIES, parse_args, r, write_json

TECHS = [("PV", "Solar PV"), ("CSP", "CSP"), ("Onshore", "Onshore wind"), ("Offshore", "Offshore wind"),
         ("BEV", "BEV"), ("PHEV", "PHEV")]


def aggregate_medians_deployment(base: pl.LazyFrame) -> pl.DataFrame:
    """fig2.py aggregate_medians(), deployment half, extra_group_cols=[] (global)."""
    dep_level2 = ["category", "strategy", "sector", "tech"]
    dep_dedupe_keys = ["category", "path_abb", "strategy", "region", "sector", "tech", "subtech"]
    deployment_base = base.group_by(dep_dedupe_keys).agg(
        pl.col("baseline_deployment").first().alias("baseline_deployment"),
        pl.col("final_deployment").first().alias("final_deployment"),
    )
    path_keys = ["category", "path_abb", "strategy", "sector"]
    strat_keys = ["category", "strategy", "sector"]
    tech_path = deployment_base.group_by([*path_keys, "tech"]).agg(
        pl.col("baseline_deployment").sum().alias("b_tech"),
        pl.col("final_deployment").sum().alias("f_tech"),
    )
    sector_path = (
        tech_path.group_by(path_keys)
        .agg(pl.col("b_tech").sum().alias("b_sec"), pl.col("f_tech").sum().alias("f_sec"))
        .with_columns((pl.col("f_sec") / pl.col("b_sec")).alias("r_sec"))
    )
    tech_with_sector = tech_path.join(sector_path, on=path_keys, how="left").with_columns(
        pl.when(pl.col("f_sec") > 0).then(pl.col("f_tech") / pl.col("f_sec")).otherwise(0.0).alias("w_tech")
    )
    r_med = sector_path.group_by(strat_keys).agg(pl.col("r_sec").median().alias("r_med"))
    baseline_med = tech_path.group_by(dep_level2).agg(pl.col("b_tech").median().alias("baseline_deployment_med"))
    w_med = tech_with_sector.group_by(dep_level2).agg(pl.col("w_tech").median().alias("w_med"))
    return (
        baseline_med.join(w_med, on=dep_level2, how="left")
        .join(r_med, on=strat_keys, how="left")
        .with_columns(
            pl.col("w_med").sum().over(strat_keys).alias("w_sum"),
            pl.col("baseline_deployment_med").sum().over(strat_keys).alias("b_total"),
        )
        .with_columns(
            pl.when(pl.col("w_sum") > 0).then(pl.col("w_med") / pl.col("w_sum")).otherwise(0.0).alias("w_norm")
        )
        .with_columns((pl.col("w_norm") * pl.col("r_med") * pl.col("b_total")).alias("final_deployment_med"))
        .select(dep_level2 + ["baseline_deployment_med", "final_deployment_med", "r_med"])
        .collect()
    )


def main() -> None:
    args = parse_args(__doc__.splitlines()[0])
    base = (
        pl.scan_parquet(args.data_dir / "deploy_metal_use.parquet")
        .filter(pl.col("strategy").is_in(list(STRATEGIES.values())))
        .filter(pl.col("category") == CATEGORY)
    )
    dep = aggregate_medians_deployment(base)
    n_paths = (
        base.select("strategy", "path_abb").unique().group_by("strategy").len().collect()
    )
    values, sectors = {}, {}
    for code, sid in CODE_TO_ID.items():
        rows = dep.filter(pl.col("strategy") == code)
        values[sid] = {
            t: r(100 * row["final_deployment_med"] / row["baseline_deployment_med"], 1)
            for t, _ in TECHS
            for row in rows.filter(pl.col("tech") == t).iter_rows(named=True)
        }
        sectors[sid] = {row["sector"]: r(100 * row["r_med"], 1)
                        for row in rows.select("sector", "r_med").unique().iter_rows(named=True)}

    data = {
        "metric_label": "Deployment reached, % of baseline",
        "unit": "%",
        "baseline": 100,
        # substitution moves deployment into CSP and offshore wind (> 400% of their baseline under
        # All-in); the axis stops here and longer bars are cut with their value written beside
        "axis_max": 200,
        "source": f"MIRACLE deploy_metal_use, category {CATEGORY}, global, medians over pathways "
                  f"(n={n_paths['len'].max()}); logic of MIRACLE-PLOT fig2.py",
        "category": CATEGORY,
        "technologies": [{"id": t, "label": l} for t, l in TECHS],
        "values": values,
        "sector_totals": sectors,
    }
    write_json(args.out, "deployment_gap", data)
    for sid, v in values.items():
        print(f"  {sid:13s} " + "  ".join(f"{t} {x:5.1f}" for t, x in v.items()) + f"   sectors {sectors[sid]}")


if __name__ == "__main__":
    main()

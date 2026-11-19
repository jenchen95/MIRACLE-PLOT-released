"""Slide 8: supply_expansion.json, the succession of binding minerals along fig4.py's relaxation path.

    uv run python src/overshoot2026/export_supply_expansion.py [DATA_DIR] [--out DECK_DATA_DIR]
        [--pathway "COFFEE 1.1|EN_NPi2020_400f_lowBECCS"] [--strategy bau] [--steps 12]

fig4_pipeline() is fig4.py's main() for one pathway, line for line: elasticity_evolve
(iter_element == 1) spread onto a common grid of `n_uniform` optimisation steps; abatement
gain over the emissions gap (co2_add_reg, 5-year snapshots) gives the cumulative gap
closure (%, and in Gt against the annual gap); supply increments over the 2025-2100 basis
supply of COFFEE 1.1|EN_NPi2020_400f_lowBECCS give the required annual supply growth
rate. Its output is fig4.py's companion_metals_gap_closure.csv for that pathway.

The deck draws 10-12 steps, while fig4.py has one per optimisation step (about 300).
So the optimisation steps up to the last one fig4.py plots (cumulative abatement < 100%)
are cut into `--steps` equal blocks; per block:
- binding: the mineral with the largest abatement gain in the block (the one whose
  relaxation closes most of the gap there);
- gap_closure_cum: fig4.py's "Cumulative gap closure (%)" at the block's last step
  (gap_closure_gt_cum: the same in Gt CO2);
- supply_growth: fig4.py's "Required annual supply growth rate (%)" of that mineral at
  the block's last step (basis_growth: the same before any relaxation, step 0).
"""

import argparse
from pathlib import Path

import polars as pl

from _shared import STRATEGIES, r, write_json, DEFAULT_DATA_DIR, DEFAULT_OUT_DIR

BAU_STRATEGIES = ["bau_bau_bau_recycl_nosub", "bau_llt_mir_recych_nosub"]
SUB_STRATEGY = ["bau_bau_bau_recycl_sub", "bau_llt_mir_recych_sub"]
ALLO_STRATEGIES = ["allo_bau_bau_recycl_nosub", "allo_llt_mir_recych_nosub",
                   "allo_bau_bau_recycl_sub", "allo_llt_mir_recych_sub"]
ALL_STRATEGIES = BAU_STRATEGIES + SUB_STRATEGY + ALLO_STRATEGIES
BASIS_PATH = "COFFEE 1.1|EN_NPi2020_400f_lowBECCS"   # fig4.py: one path_abb with all supply information

ELEMENT_NAMES = {
    "Ag": "Silver", "Al": "Aluminium", "As": "Arsenic", "B": "Boron", "Cd": "Cadmium", "Co": "Cobalt",
    "Cr": "Chromium", "Cu": "Copper", "Dy": "Dysprosium", "Fe": "Iron", "Ga": "Gallium", "Ge": "Germanium",
    "In": "Indium", "Li": "Lithium", "Mn": "Manganese", "Mo": "Molybdenum", "Nb": "Niobium", "Nd": "Neodymium",
    "Ni": "Nickel", "P": "Phosphorus", "Pb": "Lead", "Pr": "Praseodymium", "Se": "Selenium", "Si": "Silicon",
    "Sn": "Tin", "Tb": "Terbium", "Te": "Tellurium", "Ti": "Titanium", "V": "Vanadium", "Zn": "Zinc",
    "Zr": "Zirconium", "Graphite": "Graphite",
}


def fig4_pipeline(data_dir, path_abb):
    """fig4.py main() for one path_abb: returns (export_df incl. step 0, evolve_sample, baseline_gap)."""
    ev_file = data_dir / "elasticity_evolve.parquet"
    co2_file = data_dir / "co2_add_reg.parquet"
    evolve = (
        pl.scan_parquet(ev_file)
        .filter(pl.col("path_abb") == path_abb)
        .filter(pl.col("strategy").is_in(ALL_STRATEGIES))
        .filter(pl.col("iter_element") == 1)
        .select(pl.exclude(["run_id"]))
        .collect()
    )
    if evolve.is_empty():
        raise SystemExit(f"no relaxation data for {path_abb}")
    elements = evolve.select("element").unique().to_series().sort().to_list()

    global_basis_supply = (
        pl.scan_parquet(ev_file)
        .filter(pl.col("iter_id") == 0)
        .filter(pl.col("path_abb").is_in([BASIS_PATH]))
        .group_by(["path_abb", "strategy", "element"])
        .agg(pl.col("supply").sum().alias("global_basis_supply"))
        .with_columns(path_abb=pl.lit(path_abb))
        .collect()
    )
    emis_gap = (
        pl.scan_parquet(co2_file)
        .filter(pl.col("path_abb") == path_abb)
        .filter(pl.col("strategy").is_in(ALL_STRATEGIES))
        .filter(pl.col("year") % 5 == 0)
        .group_by(["path_abb", "strategy"])
        .agg(pl.sum("Emissions").alias("emis_gap"))
        .collect()
    )
    emis_gap_annual = (
        pl.scan_parquet(co2_file)
        .filter(pl.col("path_abb") == path_abb)
        .filter(pl.col("strategy").is_in(ALL_STRATEGIES))
        .group_by(["path_abb", "strategy"])
        .agg(pl.sum("Emissions").alias("emis_gap_annual"))
        .collect()
    )
    basis_endpoints = (
        pl.scan_parquet(ev_file)
        .filter(pl.col("iter_id") == 0)
        .filter(pl.col("path_abb").is_in([BASIS_PATH]))
        .with_columns(path_abb=pl.lit(path_abb))
        .filter(pl.col("year").is_in([2025, 2100]))
        .group_by(["path_abb", "strategy", "element", "year"])
        .agg(pl.col("supply").sum())
        .sort("year")
        .group_by(["path_abb", "strategy", "element"])
        .agg(
            pl.col("supply").filter(pl.col("year") == 2025).first().alias("supply_2025"),
            pl.col("supply").filter(pl.col("year") == 2100).first().alias("supply_2100"),
        )
        .with_columns(basis_supply_rate=(pl.col("supply_2100") / pl.col("supply_2025")).pow(1.0 / 75) - 1)
        .collect()
    )
    n_uniform = (
        evolve.group_by(["path_abb", "strategy", "year"])
        .agg(pl.max("iter_id").alias("iter_id_max"))
        .group_by(["path_abb", "strategy"])
        .agg(pl.col("iter_id_max").max().alias("n_uniform"))
    )
    evolve_sample = (
        evolve.join(n_uniform, on=["path_abb", "strategy"], how="left")
        .with_columns(iter_id_max=pl.col("iter_id").max().over(["path_abb", "strategy", "year"]))
        .with_columns(
            block_start=((pl.col("iter_id") - 1) * pl.col("n_uniform") / pl.col("iter_id_max")).floor().cast(pl.Int64) + 1,
            block_end=(pl.col("iter_id") * pl.col("n_uniform") / pl.col("iter_id_max")).floor().cast(pl.Int64),
        )
        .with_columns(
            block_size=pl.col("block_end") - pl.col("block_start") + 1,
            uniform_ids=pl.int_ranges(pl.col("block_start"), pl.col("block_end") + 1),
        )
        .explode("uniform_ids")
        .with_columns(
            uniform_id=pl.col("uniform_ids"),
            supply_increment=pl.col("supply_increment") / pl.col("block_size"),
            abate_gain=pl.col("abate_gain") / pl.col("block_size"),
        )
        .drop(["iter_id_max", "block_start", "block_end", "block_size", "uniform_ids"])
        .group_by(["path_abb", "strategy", "uniform_id", "element", "year"])
        .agg(
            pl.col("supply_increment").sum().alias("uniform_supply_increment"),
            pl.col("abate_gain").sum().alias("abate_gain"),
        )
    )
    evolve_sample = (
        evolve_sample.group_by(["path_abb", "strategy", "uniform_id", "element"])
        .agg(pl.sum("uniform_supply_increment").alias("uniform_supply_increment"), pl.sum("abate_gain").alias("abate_gain"))
        .with_columns(uniform_abate_gain=pl.sum("abate_gain").over(["path_abb", "strategy", "uniform_id"]))
        .join(emis_gap, on=["path_abb", "strategy"], how="left")
        .join(global_basis_supply, on=["path_abb", "strategy", "element"], how="left")
        .join(basis_endpoints, on=["path_abb", "strategy", "element"], how="left")
        .with_columns(
            ratio_increment=pl.when(
                (pl.col("uniform_supply_increment") / pl.col("global_basis_supply").fill_null(0)) > 1e-3
            )
            .then(pl.col("uniform_supply_increment") / pl.col("global_basis_supply").fill_null(0))
            .otherwise(0),
            ratio_abate=pl.col("uniform_abate_gain") / pl.col("emis_gap").fill_null(0),
            ratio_abate_gain=pl.col("abate_gain") / pl.col("emis_gap").fill_null(0),
        )
        .sort("path_abb", "strategy", "uniform_id", "element")
        .with_columns(
            ratio_cumsum_increment=pl.col("uniform_supply_increment").cum_sum().over(["path_abb", "strategy", "element"])
            / pl.col("global_basis_supply") * 100,
        )
        .with_columns(
            cumsum_supply_increment=pl.col("uniform_supply_increment").cum_sum().over(["path_abb", "strategy", "element"]),
        )
        .with_columns(
            increment_supply_rate=(
                ((pl.col("supply_2100") + pl.col("cumsum_supply_increment")) / pl.col("supply_2025")).pow(1.0 / 75) - 1
            ) * 100
        )
    )
    abate_cum = (
        evolve_sample.select("path_abb", "strategy", "uniform_id", "ratio_abate")
        .unique()
        .sort("path_abb", "strategy", "uniform_id")
        .with_columns(ratio_cumsum_abate=pl.col("ratio_abate").cum_sum().over(["path_abb", "strategy"]) * 100)
    )
    evolve_sample = (
        evolve_sample.join(emis_gap_annual, on=["path_abb", "strategy"], how="left")
        .join(abate_cum.select("path_abb", "strategy", "uniform_id", "ratio_cumsum_abate"),
              on=["path_abb", "strategy", "uniform_id"], how="left")
        .with_columns(cumsum_abate=pl.col("ratio_cumsum_abate") / 100 * pl.col("emis_gap_annual"))
        .sort("path_abb", "strategy", "uniform_id", "element")
    )
    evolve_cumulative = (
        evolve_sample.with_columns(
            [pl.when(pl.col("element") == el).then(pl.col("ratio_cumsum_increment")).otherwise(None).alias(el)
             for el in elements]
        )
        .with_columns(pl.col(elements).forward_fill().over(["path_abb", "strategy"]))
        .unique(subset=["path_abb", "strategy", "uniform_id"], keep="last")
        .unpivot(index=["path_abb", "strategy", "uniform_id", "ratio_cumsum_abate", "cumsum_abate"],
                 on=elements, variable_name="element", value_name="ratio_cumsum_increment")
        .filter(pl.col("ratio_cumsum_increment").is_not_null())
        .sort(["path_abb", "strategy", "uniform_id"])
        .filter(pl.col("ratio_cumsum_abate") < 100)
    )
    evolve_filled = (
        evolve_cumulative.join(
            evolve_sample.select("path_abb", "strategy", "uniform_id", "element", "increment_supply_rate",
                                 "basis_supply_rate", "emis_gap_annual"),
            on=["path_abb", "strategy", "uniform_id", "element"], how="left",
        )
        .sort("path_abb", "strategy", "element", "uniform_id")
        .with_columns(
            pl.col("increment_supply_rate").forward_fill().over(["path_abb", "strategy", "element"]),
            pl.col("emis_gap_annual").forward_fill().over(["path_abb", "strategy", "element"]),
        )
    )
    baseline_gap = emis_gap_annual.filter(pl.col("strategy") == BAU_STRATEGIES[0]).select("emis_gap_annual").item()

    export_df = evolve_filled.select(
        pl.col("path_abb").alias("Pathway"),
        pl.col("strategy").alias("Strategy"),
        pl.col("uniform_id").alias("Optimization step"),
        pl.col("element").alias("Element"),
        pl.col("ratio_cumsum_increment").alias("Cumulative supply increment (%)"),
        pl.col("ratio_cumsum_abate").alias("Cumulative abatement (%)"),
        pl.col("increment_supply_rate").alias("Required annual supply growth rate (%)"),
        (pl.col("cumsum_abate") / 1000).alias("Cumulative gap closure (Gt CO2)"),
        ((pl.col("cumsum_abate") + baseline_gap - pl.col("emis_gap_annual")) / baseline_gap * 100)
        .alias("Cumulative gap closure (%)"),
        (pl.col("emis_gap_annual") / 1000).alias("Emissions gap (Gt CO2)"),
    )
    step_zero = (
        evolve_sample.select("path_abb", "strategy", "element", "basis_supply_rate", "emis_gap_annual")
        .unique()
        .select(
            pl.col("path_abb").alias("Pathway"),
            pl.col("strategy").alias("Strategy"),
            pl.lit(0, dtype=pl.Int64).alias("Optimization step"),
            pl.col("element").alias("Element"),
            pl.lit(0.0).alias("Cumulative supply increment (%)"),
            pl.lit(0.0).alias("Cumulative abatement (%)"),
            (pl.col("basis_supply_rate") * 100).alias("Required annual supply growth rate (%)"),
            pl.lit(0.0).alias("Cumulative gap closure (Gt CO2)"),
            ((baseline_gap - pl.col("emis_gap_annual")) / baseline_gap * 100).alias("Cumulative gap closure (%)"),
            (pl.col("emis_gap_annual") / 1000).alias("Emissions gap (Gt CO2)"),
        )
    )
    export = pl.concat([step_zero, export_df]).sort(["Pathway", "Strategy", "Optimization step", "Element"])
    return export, evolve_sample, baseline_gap


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("data_dir", nargs="?", type=Path, default=DEFAULT_DATA_DIR)
    p.add_argument("--out", type=Path, default=DEFAULT_OUT_DIR)
    p.add_argument("--pathway", default=BASIS_PATH, help="path_abb (default: fig4.py's C2 reference pathway)")
    p.add_argument("--strategy", default="bau", choices=list(STRATEGIES), help="deck strategy id")
    p.add_argument("--steps", type=int, default=12, help="number of steps drawn on the slide")
    args = p.parse_args()
    code = STRATEGIES[args.strategy]

    export, sample, baseline_gap = fig4_pipeline(args.data_dir, args.pathway)
    ex = export.filter(pl.col("Strategy") == code)
    sm = sample.filter(pl.col("strategy") == code)
    last = int(ex["Optimization step"].max())            # last step fig4.py plots (abatement < 100%)
    n = min(args.steps, last)
    bounds = [round(last * (i + 1) / n) for i in range(n)]
    starts = [1] + [b + 1 for b in bounds[:-1]]

    per_step = ex.group_by("Optimization step").agg(
        pl.col("Cumulative gap closure (%)").first(), pl.col("Cumulative gap closure (Gt CO2)").first(),
        pl.col("Emissions gap (Gt CO2)").first(),
    )
    rate = {(row["Optimization step"], row["Element"]): row["Required annual supply growth rate (%)"]
            for row in ex.iter_rows(named=True)}
    steps, used = [], []
    for i, (a, b) in enumerate(zip(starts, bounds)):
        blk = (sm.filter(pl.col("uniform_id").is_between(a, b))
               .group_by("element").agg(pl.col("abate_gain").sum(), pl.col("uniform_supply_increment").sum())
               .sort("abate_gain", descending=True))
        el = blk["element"][0]
        row = per_step.filter(pl.col("Optimization step") == b).row(0, named=True)
        steps.append({
            "step": i + 1,
            "opt_steps": [a, b],
            "binding": el,
            "binding_share": r(blk["abate_gain"][0] / blk["abate_gain"].sum() * 100, 1),
            "gap_closure_cum": r(row["Cumulative gap closure (%)"], 2),
            "gap_closure_gt_cum": r(row["Cumulative gap closure (Gt CO2)"], 2),
            "supply_growth": r(rate.get((b, el)), 2),
            "basis_growth": r(rate.get((0, el)), 2),
        })
        if el not in used:
            used.append(el)

    gap_gt = ex["Emissions gap (Gt CO2)"].drop_nulls()[0]
    data = {
        "pathway": args.pathway,
        "strategy": args.strategy,
        "units": {"gap_closure_cum": "% of the emissions gap", "gap_closure_gt_cum": "Gt CO₂",
                  "supply_growth": "% per year"},
        "emissions_gap_gt": r(gap_gt, 1),
        "optimisation_steps": last,
        "source": "MIRACLE elasticity_evolve and co2_add_reg; logic of MIRACLE-PLOT fig4.py, "
                  f"{last} optimisation steps grouped into {n}",
        "minerals": [{"id": e, "label": ELEMENT_NAMES.get(e, e)} for e in used],
        "steps": steps,
    }
    write_json(args.out, "supply_expansion", data)
    print(f"  {args.pathway} / {code}: gap {gap_gt:.1f} Gt CO2, {last} optimisation steps -> {n}")
    for s in steps:
        print(f"  step {s['step']:2d} [{s['opt_steps'][0]:3d}-{s['opt_steps'][1]:3d}] {s['binding']:3s} "
              f"({s['binding_share']:5.1f}% of gain)  closed {s['gap_closure_cum']:6.2f}% = {s['gap_closure_gt_cum']:7.2f} Gt"
              f"  growth {s['supply_growth']} %/yr (basis {s['basis_growth']})")


if __name__ == "__main__":
    main()

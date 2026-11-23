"""Slide 8: supply_expansion.json, fig4.py's panels (co-limiting metals along the supply-optimisation sequence).

    uv run python src/overshoot2026/export_supply_expansion.py [DATA_DIR] [--out DECK_DATA_DIR]
        [--pathway "COFFEE 1.1|EN_NPi2020_400f_lowBECCS"]

fig4_pipeline() is fig4.py's main() for one pathway, line for line: elasticity_evolve
(iter_element == 1) spread onto a common grid of `n_uniform` optimisation steps; abatement
gain over the emissions gap (co2_add_reg, 5-year snapshots) gives the cumulative gap
closure (%, and in Gt against the annual gap); supply increments over the 2025-2100 basis
supply of COFFEE 1.1|EN_NPi2020_400f_lowBECCS give the cumulative supply increase (% of
baseline) and the required annual supply growth rate.

Per strategy of fig4.py's main-text grid (MAIN_TEXT_STRATEGIES), as plot_stacked() draws it:
- x: cumulative gap closure, Gt CO2 (cumsum_abate / 1000), one value per optimisation step
  that fig4.py keeps (cumulative abatement < 100%), with the origin prepended;
- y: per element, the cumulative supply increase in % of the baseline supply
  (ratio_cumsum_increment, forward-filled), stacked in fig4.py's order (first step at
  which the element is relaxed, then name); colours are fig4.py's elem_color;
- inset: gap closure in % of the Reference gap against the optimisation step:
  (cumsum_abate + Reference gap - strategy gap) / Reference gap;
- growth: fig4.py's "Required annual supply growth rate (%)" per element, at the origin
  (basis rate) and at every step (the basis rate until the element is first relaxed), for
  the slide's readout, table and text;
- entered: per element, the optimisation step at which its cumulative supply increase first
  exceeds zero (before rounding; some increases are far below the 0.001% that `y` keeps).
  The bundle size the slide counts, "metals expanded so far", is the number of elements
  entered by the step shown: the same series as the stack, the same rule for every strategy;
- binding / binding_gain: per step, the elements whose supply is relaxed at that step and
  the abatement each of them gains (Gt CO2), largest gain first. fig4.py's step grid puts
  the relaxations of every year at one step, so a step can have several elements; the
  first is the one the slide's readout names as binding.
The summary printed at the end gives the numbers behind the slide's three points.
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
    """fig4.py main() for one path_abb.

    Returns (export_df incl. step 0, evolve_sample, baseline_gap, evolve_cumulative with rank, emis_gap by strategy):
    export_df is companion_metals_gap_closure.csv, evolve_cumulative is fig4.py's cum_data.
    """
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
    )
    element_rank = (
        evolve_cumulative.group_by(["path_abb", "strategy", "element"])
        .agg(pl.min(["uniform_id"]))
        .sort(["path_abb", "strategy", "uniform_id", "element"])
        .with_columns(rank=pl.col(["uniform_id"]).rank("ordinal").over(["path_abb", "strategy"], order_by=pl.col("element")))
    )
    evolve_cumulative = (
        evolve_cumulative.join(element_rank, on=["path_abb", "strategy", "element", "uniform_id"], how="left")
        .sort(["path_abb", "strategy", "element", "uniform_id"])
        .with_columns(rank=pl.col("rank").forward_fill().over(["path_abb", "strategy", "element"]))
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
    emis_gap_dict = {row["strategy"]: row["emis_gap_annual"] for row in emis_gap_annual.iter_rows(named=True)}
    return export, evolve_sample, baseline_gap, evolve_cumulative, emis_gap_dict


# fig4.py NATURE_PALETTE and MAIN_TEXT_STRATEGIES; PANELS in the order slide 8 steps through them,
# which is the order of fig4.py's 2 x 3 grid (Circular right after Reference)
NATURE_PALETTE = ["#332288", "#117733", "#44AA99", "#88CCEE", "#DDCC77", "#CC6677", "#AA4499", "#882255", "#6699CC",
                  "#999933", "#AA7744", "#4477AA", "#228833", "#EE6677", "#6655AA", "#994455", "#557744", "#77AADD",
                  "#CC8866", "#8866AA"]
PANELS = [  # (deck id, model code, fig4.py title)
    ("bau", "bau_bau_bau_recycl_nosub", "Reference"),
    ("circular", "bau_llt_mir_recych_nosub", "Circular"),
    ("substitution", "bau_bau_bau_recycl_sub", "Substitution"),
    ("allocation", "allo_bau_bau_recycl_nosub", "Global Allocation"),
    ("allosub", "allo_bau_bau_recycl_sub", "Global Allocation+Substitution"),
    ("allin", "allo_llt_mir_recych_sub", "All-in"),
]


def element_colors(data_dir) -> dict:
    """fig4.py elem_color: palette by the sorted elements of COFFEE 1.1, four strategies."""
    ref = (
        pl.scan_parquet(data_dir / "elasticity_evolve.parquet")
        .filter(pl.col("path_abb") == BASIS_PATH)
        .filter(pl.col("strategy").is_in(BAU_STRATEGIES + ["allo_bau_bau_recycl_nosub", "allo_llt_mir_recych_nosub"]))
        .filter(pl.col("iter_element") == 1)
        .select("element").unique().collect().to_series().sort().to_list()
    )
    return {e: NATURE_PALETTE[i % len(NATURE_PALETTE)] for i, e in enumerate(ref)}


def panel(cum, export, sample, code, baseline_gap, strategy_gap):
    """One fig4.py panel: plot_stacked() data, the inset curve, the growth rates and the binding elements."""
    data = cum.filter(pl.col("strategy") == code)
    order = (data.group_by("element").agg(pl.col("rank").min()).sort("rank", "element")["element"].to_list())
    wide = (
        data.sort(["uniform_id", "rank", "element"])
        .pivot(values="ratio_cumsum_increment", index=["uniform_id", "cumsum_abate"], on="element", maintain_order=True)
        .fill_null(0)
        .sort("uniform_id")
    )
    steps = wide["uniform_id"].to_list()
    delta = baseline_gap - strategy_gap
    ex = export.filter(pl.col("Strategy") == code)
    rate = {(row["Optimization step"], row["Element"]): row["Required annual supply growth rate (%)"]
            for row in ex.iter_rows(named=True)}
    relaxed = {}                                         # step -> [(element, abatement gain in Mt)], largest gain first
    for row in (sample.filter(pl.col("strategy") == code, pl.col("uniform_supply_increment") > 0)
                .sort("abate_gain", "element", descending=[True, False]).iter_rows(named=True)):
        relaxed.setdefault(row["uniform_id"], []).append((row["element"], row["abate_gain"]))
    entered = {}                                         # element -> first step with a cumulative increase > 0
    for e in order:
        first = wide.filter(pl.col(e) > 0)["uniform_id"]
        entered[e] = int(first[0]) if len(first) else None

    def growth(e):
        """The required growth rate at every step; until an element is first relaxed it keeps its basis rate (step 0)."""
        out, last = [], None
        for st in [0] + steps:
            last = rate.get((st, e), last) if rate.get((st, e)) is not None else last
            out.append(r(last, 2))
        return out

    return {
        "steps": [0] + steps,
        "x": [0.0] + [r(v / 1000, 2) for v in wide["cumsum_abate"]],
        "closure": [r(delta / baseline_gap * 100, 2)]
                   + [r((v + delta) / baseline_gap * 100, 2) for v in wide["cumsum_abate"]],
        "elements": order,
        "y": {e: [0.0] + [r(v, 3) for v in wide[e]] for e in order},
        "growth": {e: growth(e) for e in order},
        "entered": entered,
        "binding": [[]] + [[e for e, _ in relaxed.get(st, [])] for st in steps],
        "binding_gain": [[]] + [[r(g / 1000, 3) for _, g in relaxed.get(st, [])] for st in steps],
        "gap_gt": r(strategy_gap / 1000, 1),
    }


def summary(sid, title, p):
    """The numbers behind the slide's points (printed, for checking against the manuscript)."""
    n = len(p["x"]) - 1
    first = p["elements"][0]
    alone = max(i for i in range(n + 1)                              # last step at which only the first element has grown
                if all(p["y"][e][i] == 0 for e in p["elements"][1:]))
    top = sorted(((p["growth"][e][-1], e) for e in p["elements"]), reverse=True)[:4]
    print(f"  {title:31s} {len(p['elements']):2d} elements {' '.join(p['elements'])}")
    print(f"    {n} optimisation steps; closure {p['closure'][0]:.1f}% -> {p['closure'][-1]:.1f}% "
          f"(+{p['x'][-1]:.0f} Gt of its own {p['gap_gt']:.0f} Gt gap)")
    print(f"    {first} alone up to step {p['steps'][alone]}: {p['x'][alone]:.1f} Gt, closure {p['closure'][alone]:.1f}%, "
          f"{first} growth {p['growth'][first][alone]}%/yr (basis {p['growth'][first][0]})")
    size = lambda i: sum(1 for e in p["elements"] if p["entered"][e] is not None and p["entered"][e] <= p["steps"][i])
    at100 = next((i for i in range(n + 1) if p["x"][i] >= 100), None)
    print(f"    bundle: {size(n)} metals at the last step"
          + (f", {size(at100)} at {p['x'][at100]:.0f} Gt" if at100 is not None else "") + "; entries (step: Gt): "
          + ", ".join(f"{e} {s}: {p['x'][p['steps'].index(s)]:.1f}" for e, s in sorted(p["entered"].items(), key=lambda t: (t[1] is None, t[1], t[0])) if s is not None))
    print("    final growth: " + ", ".join(f"{e} {g:.2f}" for g, e in top) + " %/yr;  final increase: "
          + ", ".join(f"{e} {p['y'][e][-1]:.1f}%" for e in sorted(p["elements"], key=lambda e: -p["y"][e][-1])[:4]))


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("data_dir", nargs="?", type=Path, default=DEFAULT_DATA_DIR)
    ap.add_argument("--out", type=Path, default=DEFAULT_OUT_DIR)
    ap.add_argument("--pathway", default=BASIS_PATH, help="path_abb (default: fig4.py's C2 illustrative pathway)")
    args = ap.parse_args()

    export, sample, baseline_gap, cum, gaps = fig4_pipeline(args.data_dir, args.pathway)
    colors = element_colors(args.data_dir)
    panels = []
    print(f"  {args.pathway}: Reference gap {baseline_gap / 1000:.1f} Gt CO2")
    for sid, code, title in PANELS:
        p = panel(cum, export, sample, code, baseline_gap, gaps[code])
        summary(sid, title, p)
        panels.append({"id": sid, "code": code, "title": title, **p})
    used = sorted({e for p in panels for e in p["elements"]})

    data = {
        "pathway": args.pathway,
        "pathway_label": "COFFEE 1.1 EN_NPi2020_400f_lowBECCS, the AR6 C2 illustrative mitigation pathway"
                         if args.pathway == BASIS_PATH else args.pathway,
        "units": {"x": "Gt CO₂", "y": "% of baseline supply", "closure": "% of the Reference gap",
                  "growth": "% per year"},
        "x_label": "Cumulative emissions-gap closure 2025–2100 (Gt CO₂)",
        "y_label": "Cumulative increase in metal supply (% of baseline)",
        "inset_label": "Gap closure (%) by optimisation step",
        "reference_gap_gt": r(baseline_gap / 1000, 1),
        "source": "MIRACLE elasticity_evolve and co2_add_reg; logic of MIRACLE-PLOT fig4.py (main-text panels)",
        "elements": [{"id": e, "label": ELEMENT_NAMES.get(e, e), "color": colors.get(e, "#999999")} for e in used],
        "panels": panels,
    }
    write_json(args.out, "supply_expansion", data)


if __name__ == "__main__":
    main()

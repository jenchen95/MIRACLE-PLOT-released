import polars as pl
import ultraplot as uplt
import numpy as np
from matplotlib.lines import Line2D
from pathlib import Path

# Resolve paths relative to repo root (so CWD doesn't matter)
REPO_ROOT = Path(__file__).resolve().parents[2]
DATA_RAW = REPO_ROOT / "data" / "data_raw" / "miracle_oscar"
DATA_OUTPUT = REPO_ROOT / "data" / "data_output"
TESTPIC = REPO_ROOT / "testpic"
OPTIMAL_SUPPLY_DIR = TESTPIC / "optimal_supply"
OPTIMAL_SUPPLY_DIR.mkdir(parents=True, exist_ok=True)
DATA_OUTPUT.mkdir(parents=True, exist_ok=True)


pl.Config.set_tbl_width_chars(200)
pl.Config.set_tbl_cols(-1)
pl.Config.set_fmt_str_lengths(100)

NATURE_PALETTE = [
    "#332288",
    "#117733",
    "#44AA99",
    "#88CCEE",
    "#DDCC77",
    "#CC6677",
    "#AA4499",
    "#882255",
    "#6699CC",
    "#999933",
    "#AA7744",
    "#4477AA",
    "#228833",
    "#EE6677",
    "#6655AA",
    "#994455",
    "#557744",
    "#77AADD",
    "#CC8866",
    "#8866AA",
]

BAU_STRATEGIES = ["bau_bau_bau_recycl_nosub", "bau_llt_mir_recych_nosub"]
SUB_STRATEGY = ["bau_bau_bau_recycl_sub", "bau_llt_mir_recych_sub"]
ALLO_STRATEGIES = [
    "allo_bau_bau_recycl_nosub",
    "allo_llt_mir_recych_nosub",
    "allo_bau_bau_recycl_sub",
    "allo_llt_mir_recych_sub",
]
ALL_STRATEGIES = BAU_STRATEGIES + SUB_STRATEGY + ALLO_STRATEGIES

# Main-text strategy grid — keep in sync with fig3.py (2×3, six panels).
# Excluded from main text: Circular+Substitution, Global Allocation+Circular.
MAIN_TEXT_STRATEGIES = [
    "bau_bau_bau_recycl_nosub",
    "bau_llt_mir_recych_nosub",
    "bau_bau_bau_recycl_sub",
    "allo_bau_bau_recycl_nosub",
    "allo_bau_bau_recycl_sub",
    "allo_llt_mir_recych_sub",
]
MAIN_TEXT_NCOLS = 3


strategy_abbr = {
    "bau_bau_bau_recycl_nosub": "Reference",
    "bau_llt_mir_recych_nosub": "Circular",
    "bau_bau_bau_recycl_sub": "Substitution",
    "bau_llt_mir_recych_sub": "Circular+Substitution",
    "allo_bau_bau_recycl_nosub": "Global Allocation",
    "allo_llt_mir_recych_nosub": "Global Allocation+Circular",
    "allo_bau_bau_recycl_sub": "Global Allocation+Substitution",
    "allo_llt_mir_recych_sub": "All-in",
}


def main():
    """Plot all path_abb in the evolve dataset."""

    all_paths = (
        pl.read_csv(
            DATA_RAW / "illustrative_mitigation_pathways.csv"
        )
        .select("path_abb")
        .to_series()
        .to_list()
    )

    # Reference color map from COFFEE 1.1 (original 4 strategies only)
    _ref_strategies = BAU_STRATEGIES + [
        "allo_bau_bau_recycl_nosub",
        "allo_llt_mir_recych_nosub",
    ]
    ref_elements = (
        pl.scan_parquet(DATA_RAW / "elasticity_evolve.parquet")
        .filter(pl.col("path_abb") == "COFFEE 1.1|EN_NPi2020_400f_lowBECCS")
        .filter(pl.col("strategy").is_in(_ref_strategies))
        .filter(pl.col("iter_element") == 1)
        .select("element")
        .unique()
        .collect()
        .to_series()
        .sort()
        .to_list()
    )
    elem_color = {
        e: NATURE_PALETTE[i % len(NATURE_PALETTE)] for i, e in enumerate(ref_elements)
    }

    all_exports = []

    for path_abb in all_paths:
        print(f"Processing: {path_abb}")

        # ---- Load relaxation data ----
        evolve = (
            pl.scan_parquet(
                DATA_RAW / "elasticity_evolve.parquet"
            )
            .filter(pl.col("path_abb") == path_abb)
            .filter(pl.col("strategy").is_in(ALL_STRATEGIES))
            .filter(pl.col("iter_element") == 1)
            .select(pl.exclude(["run_id"]))
            .collect()
        )

        elements = evolve.select("element").unique().to_series().sort().to_list()

        # Baseline supply per element per scenario / strategy
        global_basis_supply = (
            pl.scan_parquet(
                DATA_RAW / "elasticity_evolve.parquet"
            )
            .filter(pl.col("iter_id") == 0)
            .filter(pl.col("path_abb").is_in(["COFFEE 1.1|EN_NPi2020_400f_lowBECCS"]))
            .group_by(["path_abb", "strategy", "element"])
            .agg(pl.col("supply").sum().alias("global_basis_supply"))
            .with_columns(path_abb=pl.lit(path_abb))
            .collect()
        )

        # Emission gap per scenario / strategy (5-year snapshots)
        emis_gap = (
            pl.scan_parquet(DATA_RAW / "co2_add_reg.parquet")
            .filter(pl.col("path_abb") == path_abb)
            .filter(pl.col("strategy").is_in(ALL_STRATEGIES))
            .filter(pl.col("year") % 5 == 0)
            .group_by(["path_abb", "strategy"])
            .agg(pl.sum("Emissions").alias("emis_gap"))
            .collect()
        )

        # Emission gap per scenario / strategy (annual snapshots)
        emis_gap_annual = (
            pl.scan_parquet(DATA_RAW / "co2_add_reg.parquet")
            .filter(pl.col("path_abb") == path_abb)
            .filter(pl.col("strategy").is_in(ALL_STRATEGIES))
            .group_by(["path_abb", "strategy"])
            .agg(pl.sum("Emissions").alias("emis_gap_annual"))
            .collect()
        )

        # Year-level basis supply: 2025 and 2100 endpoints for CAGR
        basis_endpoints = (
            pl.scan_parquet(
                DATA_RAW / "elasticity_evolve.parquet"
            )
            .filter(pl.col("iter_id") == 0)
            .filter(
                pl.col("path_abb").is_in(["COFFEE 1.1|EN_NPi2020_400f_lowBECCS"])
            )  # one path_abb with all supply information
            .with_columns(path_abb=pl.lit(path_abb))
            .filter(pl.col("year").is_in([2025, 2100]))
            .group_by(["path_abb", "strategy", "element", "year"])
            .agg(pl.col("supply").sum())
            .sort("year")
            .group_by(["path_abb", "strategy", "element"])
            .agg(
                [
                    pl.col("supply")
                    .filter(pl.col("year") == 2025)
                    .first()
                    .alias("supply_2025"),
                    pl.col("supply")
                    .filter(pl.col("year") == 2100)
                    .first()
                    .alias("supply_2100"),
                ]
            )
            .with_columns(
                basis_supply_rate=(pl.col("supply_2100") / pl.col("supply_2025")).pow(
                    1.0 / 75
                )
                - 1
            )
            .collect()
        )

        # Common pick grid per scenario / strategy — use max to avoid big jumps
        n_uniform = (
            evolve.group_by(["path_abb", "strategy", "year"])
            .agg(pl.max("iter_id").alias("iter_id_max"))
            .group_by(["path_abb", "strategy"])
            .agg(pl.col("iter_id_max").max().alias("n_uniform"))
        )

        # ---- Per-pick, per-element sample ----
        evolve_sample = (
            evolve.join(n_uniform, on=["path_abb", "strategy"], how="left")
            .with_columns(
                iter_id_max=pl.col("iter_id").max().over(
                    ["path_abb", "strategy", "year"]
                ),
            )
            .with_columns(
                block_start=(
                    (pl.col("iter_id") - 1)
                    * pl.col("n_uniform")
                    / pl.col("iter_id_max")
                )
                .floor()
                .cast(pl.Int64)
                + 1,
                block_end=(
                    pl.col("iter_id")
                    * pl.col("n_uniform")
                    / pl.col("iter_id_max")
                )
                .floor()
                .cast(pl.Int64),
            )
            .with_columns(
                block_size=pl.col("block_end") - pl.col("block_start") + 1,
                uniform_ids=pl.int_ranges(
                    pl.col("block_start"), pl.col("block_end") + 1
                ),
            )
            .explode("uniform_ids")
            .with_columns(
                uniform_id=pl.col("uniform_ids"),
                supply_increment=pl.col("supply_increment")
                / pl.col("block_size"),
                abate_gain=pl.col("abate_gain") / pl.col("block_size"),
            )
            .drop(
                ["iter_id_max", "block_start", "block_end", "block_size", "uniform_ids"]
            )
            .group_by(["path_abb", "strategy", "uniform_id", "element", "year"])
            .agg(
                pl.col("supply_increment").sum().alias("uniform_supply_increment"),
                pl.col("abate_gain").sum().alias("abate_gain"),
            )
        )

        evolve_sample = (
            evolve_sample.group_by(["path_abb", "strategy", "uniform_id", "element"])
            .agg(
                pl.sum("uniform_supply_increment").alias("uniform_supply_increment"),
                pl.sum("abate_gain").alias("abate_gain"),
            )
            .with_columns(
                uniform_abate_gain=pl.sum("abate_gain").over(
                    ["path_abb", "strategy", "uniform_id"]
                ),
            )
            .join(emis_gap, on=["path_abb", "strategy"], how="left")
            .join(
                global_basis_supply, on=["path_abb", "strategy", "element"], how="left"
            )
            .join(basis_endpoints, on=["path_abb", "strategy", "element"], how="left")
            .with_columns(
                ratio_increment=pl.when(
                    (
                        pl.col("uniform_supply_increment")
                        / pl.col("global_basis_supply").fill_null(0)
                    )
                    > 1e-3
                )
                .then(
                    pl.col("uniform_supply_increment")
                    / pl.col("global_basis_supply").fill_null(0)
                )
                .otherwise(0),
                ratio_abate=pl.col("uniform_abate_gain")
                / pl.col("emis_gap").fill_null(0),  # all elements
                ratio_abate_gain=pl.col("abate_gain")
                / pl.col("emis_gap").fill_null(0),  # by element
            )
            .sort("path_abb", "strategy", "uniform_id", "element")
            .with_columns(
                ratio_cumsum_increment=pl.col("uniform_supply_increment")
                .cum_sum()
                .over(["path_abb", "strategy", "element"])
                / pl.col("global_basis_supply")
                * 100,
            )
            .with_columns(
                cumsum_supply_increment=pl.col("uniform_supply_increment")
                .cum_sum()
                .over(["path_abb", "strategy", "element"]),
            )
            .with_columns(
                increment_supply_rate=(
                    (
                        (pl.col("supply_2100") + pl.col("cumsum_supply_increment"))
                        / pl.col("supply_2025")
                    ).pow(1.0 / 75)
                    - 1
                )
                * 100
            )
        )

        # ---- Cumulative % abate gain per scenario / strategy / pick ----
        abate_cum = (
            evolve_sample.select("path_abb", "strategy", "uniform_id", "ratio_abate")
            .unique()
            .sort("path_abb", "strategy", "uniform_id")
            .with_columns(
                ratio_cumsum_abate=pl.col("ratio_abate")
                .cum_sum()
                .over(["path_abb", "strategy"])
                * 100,
            )  # all elements
        )

        abate_gain_cum = (
            evolve_sample.select(
                "path_abb", "strategy", "uniform_id", "element", "ratio_abate_gain"
            )
            .unique()
            .sort("path_abb", "strategy", "uniform_id", "element")
            .with_columns(
                ratio_cumsum_abate_gain=pl.col("ratio_abate_gain")
                .cum_sum()
                .over(["path_abb", "strategy", "element"])
                * 100  # by element
            )
        )

        evolve_sample = (
            evolve_sample.join(
                emis_gap_annual,
                on=["path_abb", "strategy"],
                how="left",
            )
            .join(
                abate_cum.select(
                    "path_abb", "strategy", "uniform_id", "ratio_cumsum_abate"
                ),
                on=["path_abb", "strategy", "uniform_id"],
                how="left",
            )
            .with_columns(
                cumsum_abate=pl.col("ratio_cumsum_abate")
                / 100
                * pl.col("emis_gap_annual")
            )  # calculates cumulative abatement by aggregating across all elements within every path_abb, strategy, and uniform_id
            .join(
                abate_gain_cum.select(
                    "path_abb",
                    "strategy",
                    "uniform_id",
                    "element",
                    "ratio_cumsum_abate_gain",
                ),
                on=["path_abb", "strategy", "uniform_id", "element"],
                how="left",
            )
            .with_columns(
                cumsum_abate_gain=pl.col("ratio_cumsum_abate_gain")
                / 100
                * pl.col(
                    "emis_gap_annual"
                )  # sums abate_gain for each element within every path_abb, strategy, and uniform_id
            )
            .with_columns(
                crom=pl.col("abate_gain") / pl.col("uniform_supply_increment"),
                semi=pl.col("abate_gain")
                / 1000
                / (pl.col("ratio_increment") * 100),  # Mt / 1% global supply
                ramp=pl.col("ratio_increment") / (pl.col("abate_gain") / 1000),
            )
        )

        # Sort to guarantee forward-fill order
        evolve_sample = evolve_sample.sort(
            "path_abb", "strategy", "uniform_id", "element"
        )

        # ---- Cumulative data: wide-format pivot + forward-fill, then melt ----
        evolve_cumulative = (
            evolve_sample.with_columns(
                [
                    pl.when(pl.col("element") == el)
                    .then(pl.col("ratio_cumsum_increment"))
                    .otherwise(None)
                    .alias(el)
                    for el in elements
                ]
            )
            .with_columns(
                pl.col(elements).forward_fill().over(["path_abb", "strategy"])
            )
            .unique(subset=["path_abb", "strategy", "uniform_id"], keep="last")
            .unpivot(
                index=[
                    "path_abb",
                    "strategy",
                    "uniform_id",
                    "ratio_cumsum_abate",
                    "cumsum_abate",
                ],
                on=elements,
                variable_name="element",
                value_name="ratio_cumsum_increment",
            )
            .filter(pl.col("ratio_cumsum_increment").is_not_null())
            .sort(["path_abb", "strategy", "uniform_id"])
        )

        element_rank = (
            evolve_cumulative.group_by(["path_abb", "strategy", "element"])
            .agg(pl.min(["uniform_id"]))
            .sort(["path_abb", "strategy", "uniform_id", "element"])
            .with_columns(
                rank=pl.col(["uniform_id"])
                .rank("ordinal")
                .over(["path_abb", "strategy"], order_by=pl.col("element"))
            )
        )

        evolve_cumulative = (
            evolve_cumulative.join(
                element_rank,
                on=["path_abb", "strategy", "element", "uniform_id"],
                how="left",
            )
            .sort(["path_abb", "strategy", "element", "uniform_id"])
            .with_columns(
                rank=pl.col("rank")
                .forward_fill()
                .over(["path_abb", "strategy", "element"])
            )
            .filter(pl.col("ratio_cumsum_abate") < 100)
        )

        # ---- Split cumulative by strategy ----
        cum_data = {
            BAU_STRATEGIES[0]: evolve_cumulative.filter(
                pl.col("strategy") == BAU_STRATEGIES[0]
            ),
            BAU_STRATEGIES[1]: evolve_cumulative.filter(
                pl.col("strategy") == BAU_STRATEGIES[1]
            ),
            SUB_STRATEGY[0]: evolve_cumulative.filter(
                pl.col("strategy") == SUB_STRATEGY[0]
            ),
            SUB_STRATEGY[1]: evolve_cumulative.filter(
                pl.col("strategy") == SUB_STRATEGY[1]
            ),
            ALLO_STRATEGIES[0]: evolve_cumulative.filter(
                pl.col("strategy") == ALLO_STRATEGIES[0]
            ),
            ALLO_STRATEGIES[1]: evolve_cumulative.filter(
                pl.col("strategy") == ALLO_STRATEGIES[1]
            ),
            ALLO_STRATEGIES[2]: evolve_cumulative.filter(
                pl.col("strategy") == ALLO_STRATEGIES[2]
            ),
            ALLO_STRATEGIES[3]: evolve_cumulative.filter(
                pl.col("strategy") == ALLO_STRATEGIES[3]
            ),
        }

        # Forward-fill line-plot columns so every element appears at every step
        evolve_filled = (
            evolve_cumulative.join(
                evolve_sample.select(
                    "path_abb",
                    "strategy",
                    "uniform_id",
                    "element",
                    "increment_supply_rate",
                    "basis_supply_rate",
                    "emis_gap_annual",
                ),
                on=["path_abb", "strategy", "uniform_id", "element"],
                how="left",
            )
            .sort("path_abb", "strategy", "element", "uniform_id")
            .with_columns(
                pl.col("increment_supply_rate")
                .forward_fill()
                .over(["path_abb", "strategy", "element"]),
                pl.col("emis_gap_annual")
                .forward_fill()
                .over(["path_abb", "strategy", "element"]),
            )
        )

        # ---- Split marginal by strategy ----
        marg_data = {
            BAU_STRATEGIES[0]: evolve_filled.filter(
                pl.col("strategy") == BAU_STRATEGIES[0]
            ),
            BAU_STRATEGIES[1]: evolve_filled.filter(
                pl.col("strategy") == BAU_STRATEGIES[1]
            ),
            SUB_STRATEGY[0]: evolve_filled.filter(
                pl.col("strategy") == SUB_STRATEGY[0]
            ),
            SUB_STRATEGY[1]: evolve_filled.filter(
                pl.col("strategy") == SUB_STRATEGY[1]
            ),
            ALLO_STRATEGIES[0]: evolve_filled.filter(
                pl.col("strategy") == ALLO_STRATEGIES[0]
            ),
            ALLO_STRATEGIES[1]: evolve_filled.filter(
                pl.col("strategy") == ALLO_STRATEGIES[1]
            ),
            ALLO_STRATEGIES[2]: evolve_filled.filter(
                pl.col("strategy") == ALLO_STRATEGIES[2]
            ),
            ALLO_STRATEGIES[3]: evolve_filled.filter(
                pl.col("strategy") == ALLO_STRATEGIES[3]
            ),
        }

        # Baseline emission gap from Reference strategy for inset y-axis
        baseline_gap = (
            emis_gap_annual.filter(pl.col("strategy") == BAU_STRATEGIES[0])
            .select("emis_gap_annual")
            .item()
        )
        emis_gap_dict = {
            row["strategy"]: row["emis_gap_annual"]
            for row in emis_gap_annual.iter_rows(named=True)
        }

        # Elements present in the highlighted path (main Ref strategy)
        highlight_elements = (
            evolve_sample.filter(pl.col("strategy") == BAU_STRATEGIES[0])["element"]
            .unique()
            .sort()
            .to_list()
        )

        make_figure(
            cum_data,
            marg_data,
            path_abb,
            highlight_elements,
            elem_color,
            baseline_gap,
            emis_gap_dict,
        )
        make_line_figure(
            marg_data,
            path_abb,
            elem_color,
        )
        make_si_figure(
            cum_data,
            marg_data,
            path_abb,
            elem_color,
            baseline_gap,
            emis_gap_dict,
        )

        # Use forward-filled data so every element appears at every step
        export_df = evolve_filled.select(
            pl.col("path_abb").alias("Pathway"),
            pl.col("strategy").alias("Strategy"),
            pl.col("uniform_id").alias("Optimization step"),
            pl.col("element").alias("Element"),
            pl.col("ratio_cumsum_increment").alias(
                "Cumulative supply increment (%)"
            ),
            pl.col("ratio_cumsum_abate").alias("Cumulative abatement (%)"),
            pl.col("increment_supply_rate").alias(
                "Required annual supply growth rate (%)"
            ),
            (pl.col("cumsum_abate") / 1000).alias(
                "Cumulative gap closure (Gt CO2)"
            ),
            ((pl.col("cumsum_abate") + baseline_gap - pl.col("emis_gap_annual"))
             / baseline_gap * 100).alias("Cumulative gap closure (%)"),
            (pl.col("emis_gap_annual") / 1000).alias("Emissions gap (Gt CO2)"),
        )

        step_zero = (
            evolve_sample.select(
                "path_abb", "strategy", "element", "basis_supply_rate", "emis_gap_annual"
            )
            .unique()
            .select(
                pl.col("path_abb").alias("Pathway"),
                pl.col("strategy").alias("Strategy"),
                pl.lit(0, dtype=pl.Int64).alias("Optimization step"),
                pl.col("element").alias("Element"),
                pl.lit(0.0).alias("Cumulative supply increment (%)"),
                pl.lit(0.0).alias("Cumulative abatement (%)"),
                (pl.col("basis_supply_rate") * 100).alias(
                    "Required annual supply growth rate (%)"
                ),
                pl.lit(0.0).alias("Cumulative gap closure (Gt CO2)"),
                ((baseline_gap - pl.col("emis_gap_annual"))
                 / baseline_gap * 100).alias("Cumulative gap closure (%)"),
                (pl.col("emis_gap_annual") / 1000).alias(
                    "Emissions gap (Gt CO2)"
                ),
            )
        )

        all_exports.append(pl.concat([step_zero, export_df]))

    combined = pl.concat(all_exports).sort(['Pathway','Strategy','Optimization step','Element'])
    output_csv = DATA_OUTPUT / "companion_metals_gap_closure.csv"
    combined.write_csv(output_csv)
    print(f"Data exported to {output_csv}")


def plot_stacked(ax_stack, data, elem_color, baseline_gap, strategy_gap):
    """Stacked area fill: elements stacked by appearance order, largest first."""
    element_order = (
        data.group_by("element")
        .agg(pl.col("rank").min().alias("rank"))
        .sort("rank", "element")["element"]
        .to_list()
    )

    piv = (
        data.sort(["uniform_id", "rank", "element"])
        .pivot(
            values="ratio_cumsum_increment",
            index="cumsum_abate",
            on="element",
            maintain_order=True,
        )
        .fill_null(0)
    )
    elem_cols = [el for el in element_order if el in piv.columns]
    piv = piv.select(["cumsum_abate", *elem_cols])

    # Prepend uniform_id=0 row (all zeros) so fill starts from origin
    first_element = element_order[0]
    zero_row = {
        c: 0.0 if c in ("cumsum_abate", first_element) else None for c in piv.columns
    }
    zero_df = (
        pl.DataFrame([zero_row])
        .select(piv.columns)
        .select(pl.all().cast(pl.Float64, strict=False))
    )
    piv = pl.concat([zero_df, piv])

    # Make sure label order matches y_raw columns
    elem_cols = [el for el in element_order if el in piv.columns]
    piv = piv.select(["cumsum_abate", *elem_cols])

    x = (
        piv.select(pl.col("cumsum_abate")).to_series().to_numpy().ravel() / 1000
    )  # Mt to Gt
    y_raw = piv.select(pl.exclude("cumsum_abate")).fill_null(0).to_numpy()

    display_order = elem_cols
    display_colors = [elem_color.get(el, "#999999") for el in display_order]

    ax_stack.area(x, y_raw, stack=True, cycle=display_colors)

    # Stacked band boundaries
    y_bottom = np.zeros_like(y_raw)
    y_bottom[:, 1:] = np.cumsum(y_raw[:, :-1], axis=1)
    y_top = y_bottom + y_raw

    thickness_all = y_top - y_bottom
    max_thickness = np.max(thickness_all, axis=0)

    top_n = 4
    must_label = {"Dy", "Li"}

    top_idx = set(np.argsort(max_thickness)[-top_n:])
    label_idx = top_idx | {
        j for j, elem in enumerate(display_order) if elem in must_label
    }

    inside_idx = set()

    x_min, x_max = np.nanmin(x), np.nanmax(x)
    x_range = x_max - x_min

    for j in sorted(label_idx):
        elem = display_order[j]
        thickness = thickness_all[:, j]

        if elem == "Dy":
            valid_x = (x >= x_min + 0.20 * x_range) & (x <= x_max - 0.08 * x_range)
        else:
            valid_x = (x >= x_min + 0.45 * x_range) & (x <= x_max - 0.08 * x_range)

        candidate = np.where(valid_x & (thickness > 0))[0]
        if len(candidate) == 0:
            continue

        idx = candidate[np.argmax(thickness[candidate])]

        if thickness[idx] < 1.0:
            continue

        y_mid = 0.5 * (y_bottom[idx, j] + y_top[idx, j])
        hex_color = display_colors[j]

        r, g, b = (
            int(hex_color[1:3], 16),
            int(hex_color[3:5], 16),
            int(hex_color[5:7], 16),
        )
        luminance = 0.299 * r + 0.587 * g + 0.114 * b
        text_color = "white" if luminance < 140 else "black"

        ax_stack.text(
            x[idx],
            y_mid,
            elem,
            color=text_color,
            fontsize=8.5,
            ha="center",
            va="center",
            fontweight="bold",
        )
        inside_idx.add(j)

    # Elements not labeled inside are labeled on the right with arrows.
    # Spread labels over the full panel height so small terminal bands do not overlap.
    other_idx = [j for j in range(len(display_order)) if j not in inside_idx]

    if other_idx:
        y_span = max(np.nanmax(y_top), 1.0)
        min_gap = max(8.0, 0.07 * y_span)
        lower = 0.06 * y_span
        upper = 0.96 * y_span
        max_right_labels = max(1, int(np.floor((upper - lower) / min_gap)) + 1)

        if len(other_idx) > max_right_labels:
            terminal_thickness = y_top[-1] - y_bottom[-1]
            other_idx = sorted(
                other_idx,
                key=lambda j: terminal_thickness[j],
                reverse=True,
            )[:max_right_labels]

        other_idx.sort(key=lambda j: 0.5 * (y_bottom[-1, j] + y_top[-1, j]))
        anchor_y = [0.5 * (y_bottom[-1, j] + y_top[-1, j]) for j in other_idx]
        n_labels = len(other_idx)

        if n_labels == 1:
            y_stagger = np.array([np.clip(anchor_y[0], lower, upper)])
        else:
            label_span = min(upper - lower, min_gap * (n_labels - 1))
            center = np.clip(np.mean(anchor_y), lower + label_span / 2, upper - label_span / 2)
            y_stagger = np.linspace(
                center - label_span / 2,
                center + label_span / 2,
                n_labels,
            )

        x_right = x[-1] + 0.08 * (x[-1] - x[0])

        for k, j in enumerate(other_idx):
            elem = display_order[j]
            color = display_colors[j]
            y_band = 0.5 * (y_bottom[-1, j] + y_top[-1, j])

            ax_stack.annotate(
                elem,
                xy=(x[-1], y_band),
                xytext=(x_right, y_stagger[k]),
                color=color,
                fontsize=8.5,
                ha="left",
                va="center",
                arrowprops=dict(
                    arrowstyle="-",
                    color=color,
                    lw=0.6,
                    alpha=0.8,
                ),
            )

    # Inset: gap closure % vs pick number (baseline = Reference emissions)
    inset = ax_stack.inset_axes([0.17, 0.59, 0.31, 0.40], zoom=False)
    inset.set_facecolor("white")
    for spine in inset.spines.values():
        spine.set_edgecolor("0.6")
        spine.set_linewidth(0.4)
    abate_curve = (
        data.select("uniform_id", "cumsum_abate").unique().sort("uniform_id")
    )
    delta = baseline_gap - strategy_gap
    gap_closure = (
        abate_curve["cumsum_abate"].to_numpy() + delta
    ) / baseline_gap * 100
    inset.plot(
        abate_curve["uniform_id"].to_numpy(),
        gap_closure,
        color="0.20",
        lw=1.3,
        marker=".",
        markersize=2,
    )
    inset.format(
        xlabel="Optimization step",
        ylabel="Gap closure (%)",
        # ylim=(None, None),
        labelsize=7.5,
        ticklabelsize=7,
        tickminor=False,
        grid=False,
    )

def plot_line(ax_line, marg_data, elem_color):
    """Annual supply growth-rate lines for one strategy."""
    marg = (
        marg_data.group_by(["uniform_id", "element"])
        .agg(
            pl.mean("increment_supply_rate").alias("increment_supply_rate"),
            pl.mean("cumsum_abate").truediv(1000).alias("cumsum_abate"),
        )
        .sort(["uniform_id", "element"])
        .pivot(
            index=["cumsum_abate"],
            on=["element"],
            values="increment_supply_rate",
            maintain_order=True,
        )
    )

    basis_supply_rate = (
        marg_data.group_by(['element'])
        .agg(
            pl.mean('basis_supply_rate').mul(100).alias('basis_supply_rate')
        )
        .with_columns(cumsum_abate=pl.lit(0).cast(pl.Float64))
        .sort(['element'])
        .pivot(
            index='cumsum_abate',
            on='element',
            values='basis_supply_rate',
            maintain_order=True,
        )
        .select(pl.col(marg.columns))
    )
    marg = (
        pl.concat(
            [basis_supply_rate, marg]
        )
        .fill_null(strategy='forward')
        .with_row_index()
        .filter(
            (pl.col('index') % 10 == 0)
            | (pl.col('index') <= 4)
            | (pl.col('index') == pl.col('index').max())
        )
        .select(pl.exclude('index'))
    )

    x_vals = marg["cumsum_abate"].to_numpy()
    y_vals = marg.select(pl.exclude("cumsum_abate")).to_numpy()
    elem_cols = [c for c in marg.columns if c != "cumsum_abate"]
    display_colors = [elem_color.get(el, "#999999") for el in elem_cols]

    # Plot lines
    ax_line.line(
        x_vals, y_vals, alpha=0.65, cycle=display_colors,
    )
    # Overlay scatter markers
    ax_line.scatter(
        x_vals, y_vals, alpha=0.25, cycle=display_colors,
        # facecolor=None,
        # edgecolor='none',
        s=9,
    )


    # Collect last real data point per element for annotation
    endpoints = {}
    for j, elem in enumerate(elem_cols):
        y = y_vals[:, j]
        if y[-1] <= 0:
            continue
        color = elem_color.get(elem, display_colors[j % len(display_colors)])
        endpoints[elem] = (color, x_vals[-1], y[-1])

    # Staggered arrow annotations for all elements with element color
    all_elements = sorted(endpoints, key=lambda e: endpoints[e][2])
    if all_elements:
        y_ends = np.array([endpoints[e][2] for e in all_elements], dtype=float)
        y_span = max(np.nanmax(y_vals) - min(np.nanmin(y_vals), 0), 1.0)
        min_gap = max(8.0, 0.07 * y_span)
        lower = max(0.0, np.nanmin(y_ends) - 0.08 * y_span)
        upper = np.nanmax(y_ends) + 0.08 * y_span
        max_right_labels = max(1, int(np.floor((upper - lower) / min_gap)) + 1)

        if len(all_elements) > max_right_labels:
            all_elements = sorted(
                all_elements,
                key=lambda e: endpoints[e][2],
                reverse=True,
            )[:max_right_labels]
            all_elements.sort(key=lambda e: endpoints[e][2])
            y_ends = np.array([endpoints[e][2] for e in all_elements], dtype=float)

        y_stagger = np.clip(y_ends.copy(), lower, upper)
        for k in range(1, len(y_stagger)):
            y_stagger[k] = max(y_stagger[k], y_stagger[k - 1] + min_gap)
        if len(y_stagger) and y_stagger[-1] > upper:
            y_stagger -= y_stagger[-1] - upper
            y_stagger[0] = max(y_stagger[0], lower)
            for k in range(1, len(y_stagger)):
                y_stagger[k] = max(y_stagger[k], y_stagger[k - 1] + min_gap)

        x_right = x_vals[-1] * 1.08
        for k, elem in enumerate(all_elements):
            color, x_end, y_end = endpoints[elem]
            ax_line.annotate(
                elem,
                xy=(x_end, y_end),
                xytext=(x_right, y_stagger[k]),
                color=color,
                fontsize=10.5,
                ha="left",
                va="center",
                arrowprops=dict(arrowstyle="-", color=color, lw=0.6, alpha=0.7),
            )


def _format_axes(axs):
    """Apply shared font sizes."""
    for ax in axs:
        ax.tick_params(labelsize=8)
        ax.xaxis.label.set_size(9)
        ax.yaxis.label.set_size(9)
        ax.title.set_size(9)


def _add_centered_ylabel(fig, text, x=0.01):
    """Add a figure-level y label centered over the full figure height."""
    fig.text(
        x,
        0.5,
        text,
        rotation=90,
        va="center",
        ha="center",
        fontsize=9,
    )


def _save_pdf_and_svg(fig, pdf_path):
    fig.save(pdf_path)
    svg_path = pdf_path.with_suffix(".svg")
    fig.save(svg_path)
    print(f"Saved {pdf_path}")
    print(f"Saved {svg_path}")


def _plot_stacked_grid(
    axs,
    strategies,
    ncols,
    cum_data,
    elem_color,
    baseline_gap,
    emis_gap_dict,
):
    """Plot stacked-area strategy panels."""
    for idx, strategy in enumerate(strategies):
        row = idx // ncols
        col = idx % ncols
        ax = axs[row, col]
        plot_stacked(
            ax,
            cum_data[strategy],
            elem_color,
            baseline_gap,
            emis_gap_dict[strategy],
        )
        ax.format(title=strategy_abbr[strategy], tickminor=False, grid=False)


def _plot_line_grid(axs, strategies, ncols, marg_data, elem_color):
    """Plot annual supply growth-rate strategy panels."""
    for idx, strategy in enumerate(strategies):
        row = idx // ncols
        col = idx % ncols
        ax = axs[row, col]
        plot_line(ax, marg_data[strategy], elem_color)
        ax.format(title=strategy_abbr[strategy], tickminor=False, grid=False)


def make_figure(
    cum_data,
    marg_data,
    highlight_path,
    highlight_elements,
    elem_color,
    baseline_gap,
    emis_gap_dict,
):
    """Main-text figure: 2×3 stacked-area grid."""
    ncols = MAIN_TEXT_NCOLS
    fig, axs = uplt.subplots(
        nrows=2,
        ncols=ncols,
        sharex=False,
        sharey="limits",
        spany=False,
        figheight=7,
        abc=True,
    )

    axs.format(
        titlesize=9,
        labelsize=9,
        ticklabelsize=8,
        abcsize=10,
    )

    _plot_stacked_grid(
        axs,
        MAIN_TEXT_STRATEGIES,
        ncols,
        cum_data,
        elem_color,
        baseline_gap,
        emis_gap_dict,
    )
    _add_centered_ylabel(fig, "Cumulative supply increase (% of baseline)")
    axs[-1, :].format(xlabel="Cumulative gap closure (Gt CO2)")
    _format_axes(axs)

    safe_name = highlight_path.replace("/", "_").replace("|", "_").replace(" ", "_")
    out_path = OPTIMAL_SUPPLY_DIR / f"{safe_name}.pdf"
    _save_pdf_and_svg(fig, out_path)


def make_line_figure(
    marg_data,
    highlight_path,
    elem_color,
):
    """Separate line figure for annual supply growth rates across all strategies."""
    ncols = 4
    nrows = int(np.ceil(len(ALL_STRATEGIES) / ncols))
    fig, axs = uplt.subplots(
        nrows=nrows,
        ncols=ncols,
        sharex=False,
        sharey="limits",
        spany=False,
        figheight=6.5,
        abc=True,
    )

    for row in range(nrows):
        for col in range(1, ncols):
            axs[row, col].sharey(axs[row, 0])

    axs.format(
        titlesize=11,
        labelsize=11,
        ticklabelsize=10,
        abcsize=12,
    )

    _plot_line_grid(axs, ALL_STRATEGIES, ncols, marg_data, elem_color)
    axs[:, 0].format(ylabel="Annual supply\ngrowth rate (%)")
    axs[-1, :].format(xlabel="Cumulative gap closure (Gt CO2)")
    axs.format(
        ylim=(3, None),
        ylocator=1,
        yminorlocator=0.5,
        xtickminor=False,
        grid=False,
    )
    for ax in axs:
        ax.tick_params(labelsize=10)
        ax.xaxis.label.set_size(11)
        ax.yaxis.label.set_size(11)
        ax.title.set_size(11)

    # Metal / element legend below the panel grid
    legend_elements = sorted(elem_color)
    legend_handles = [
        Line2D([0], [0], color=elem_color[el], lw=1.8, marker="o",
               markersize=4, label=el)
        for el in legend_elements
    ]
    fig.legend(
        legend_handles,
        legend_elements,
        loc="bottom",
        ncols=len(legend_elements),
        fontsize=10,
        frame=False,
        center=True,
        handlelength=1.6,
        handletextpad=0.4,
        columnspacing=1.0,
        space=5,  # clear x-axis label ("Cumulative gap closure...")
    )

    safe_name = highlight_path.replace("/", "_").replace("|", "_").replace(" ", "_")
    out_path = OPTIMAL_SUPPLY_DIR / f"{safe_name}_line_growth_rates.pdf"
    _save_pdf_and_svg(fig, out_path)


def make_si_figure(
    cum_data,
    marg_data,
    highlight_path,
    elem_color,
    baseline_gap,
    emis_gap_dict,
):
    """Stacked-area figure for the two strategies excluded from the main-text grid."""
    si_strategies = [
        s for s in ALL_STRATEGIES if s not in MAIN_TEXT_STRATEGIES
    ]
    ncols = len(si_strategies)
    fig, axs = uplt.subplots(
        nrows=1,
        ncols=ncols,
        sharex=False,
        sharey="limits",
        spany=False,
        figheight=3.8,
        abc=True,
    )

    axs.format(
        titlesize=9,
        labelsize=9,
        ticklabelsize=8,
        abcsize=10,
    )

    for col_idx, strategy in enumerate(si_strategies):
        plot_stacked(
            axs[col_idx],
            cum_data[strategy],
            elem_color,
            baseline_gap,
            emis_gap_dict[strategy],
        )
        axs[col_idx].format(
            title=strategy_abbr[strategy], tickminor=False, grid=False
        )

    _add_centered_ylabel(fig, "Cumulative supply increase (% of baseline)")
    axs.format(xlabel="Cumulative gap closure (Gt CO2)")
    _format_axes(axs)

    safe_name = highlight_path.replace("/", "_").replace("|", "_").replace(" ", "_")
    out_path = OPTIMAL_SUPPLY_DIR / f"{safe_name}_si_strategies.pdf"
    _save_pdf_and_svg(fig, out_path)


if __name__ == "__main__":
    main()

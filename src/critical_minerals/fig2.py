import argparse
import colorsys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import polars as pl
import ultraplot as uplt
from matplotlib.patches import Patch
from matplotlib.ticker import FuncFormatter


ROOT_DIR = Path(__file__).resolve().parents[2]
INPUT_PARQUET = ROOT_DIR / "data" / "data_raw" / "miracle_oscar" / "deploy_metal_use.parquet"
OUTPUT_DIR = ROOT_DIR / "testpic"
TARGET_STRATEGIES = [
    "bau_bau_bau_recycl_nosub",
    "bau_llt_mir_recych_nosub",
    "bau_bau_bau_recycl_sub",
    "bau_llt_mir_recych_sub",
    "allo_bau_bau_recycl_nosub",
    "allo_bau_bau_recycl_sub",
    "allo_llt_mir_recych_sub",
]
STRATEGY_LABELS = {
    "bau_bau_bau_recycl_nosub": "Ref",
    "bau_llt_mir_recych_nosub": "Circular",
    "bau_bau_bau_recycl_sub": "Substitution",
    "bau_llt_mir_recych_sub": "Circular\n+Substitution",
    "allo_bau_bau_recycl_nosub": "Allocation",
    "allo_llt_mir_recych_nosub": "Allocation\n+Circular",
    "allo_bau_bau_recycl_sub": "Allocation\n+Substitution",
    "allo_llt_mir_recych_sub": "All-in",
}
REF_STRATEGY = "bau_bau_bau_recycl_nosub"
ALL_IN_STRATEGY = "allo_llt_mir_recych_sub"
METAL_TOP_K = 10
METAL_OTHER_LABEL = "Other metals"
TECH_OTHER_LABEL = "Other"
BASELINE_HATCH = "///"
GAP_HATCH = "..."
BASELINE_EDGE = "#666666"
BASELINE_FACE = "#F5F5F5"
GAP_FACE = "#E8E8E8"
METAL_OTHER_COLOR = "#4A4A4A"
TECH_OTHER_COLOR = "#4A4A4A"
DEPLOY_YLABEL = "Share of scenario baseline (%)"
METAL_YLABEL = "Share of Ref baseline (%)"
VALID_SCALES = ("global", "regional", "both")
POWER_TECH_KEYWORDS = [
    "coal", "gas", "oil", "nuclear", "hydro", "bio", "geotherm",
    "onshore", "offshore", "wind", "pv", "solar", "csp", "storage", "battery",
]
TRANSPORT_TECH_KEYWORDS = [
    "ice", "combustion", "conventional", "phev", "plug", "hybrid",
    "bev", "electric", "fcev", "fuel cell", "hydrogen",
]


def tech_sort_key(name: str, sector: str) -> tuple[int, str]:
    lname = name.lower()
    if lname in {"other", TECH_OTHER_LABEL.lower()} or name == TECH_OTHER_LABEL:
        return (9999, name)
    keywords = POWER_TECH_KEYWORDS if sector == "Power" else TRANSPORT_TECH_KEYWORDS
    for i, kw in enumerate(keywords):
        if kw in lname:
            return (i, name)
    return (5000, name)


def sort_tech_groups(groups: list[str], sector: str) -> list[str]:
    ordered = sorted(groups, key=lambda g: tech_sort_key(g, sector))
    if TECH_OTHER_LABEL in ordered:
        ordered = [g for g in ordered if g != TECH_OTHER_LABEL] + [TECH_OTHER_LABEL]
    return ordered


def sort_metal_groups(groups: list[str], other_label: str) -> list[str]:
    ordered = sorted([g for g in groups if g != other_label])
    if other_label in groups:
        ordered.append(other_label)
    return ordered


REGIONAL_PANEL_YLIM_BREAKPOINT = 1.3
REGIONAL_PANEL_YLIM_FACTOR = 1.2
REGIONAL_METAL_COLOR_SATURATION = 0.58


def clean_name(text: str) -> str:
    return "".join(ch if ch.isalnum() or ch in ("-", "_") else "_" for ch in text)


def make_color_map(names: list[str], cmap_name: str) -> dict[str, tuple[float, float, float, float]]:
    cmap = uplt.Colormap(cmap_name)
    n = max(1, len(names))
    if n == 1:
        return {names[0]: cmap(0.5)}
    return {name: cmap(i / (n - 1)) for i, name in enumerate(names)}


def make_element_color_map(names: list[str]) -> dict[str, tuple[float, float, float, float]]:
    ordered = sorted(names)
    n = max(1, len(ordered))
    color_map: dict[str, tuple[float, float, float, float]] = {}
    for i, name in enumerate(ordered):
        hue = i / n
        r, g, b = colorsys.hsv_to_rgb(hue, REGIONAL_METAL_COLOR_SATURATION, 0.88)
        color_map[name] = (r, g, b, 1.0)
    return color_map


def make_tech_color_map(names: list[str]) -> dict[str, str]:
    # Muted, colorblind-friendly base colors for key technologies.
    key_colors = {
        "bev": "#4C78A8",       # blue
        "phev": "#EECA3B",      # gold (distinct from greens)
        "pv": "#56B4E9",        # sky blue
        "csp": "#117733",       # dark green
        "onshore": "#B279A2",   # pink-purple
        "offshore": "#7A5195",  # purple
        "other": TECH_OTHER_COLOR,
    }
    fallback_cycle = [
        "#6C8EBF",
        "#D95F02",
        "#A27BB2",
        "#7570B3",
        "#1B9E77",
        "#C08A6E",
    ]

    color_map: dict[str, str] = {}
    fallback_idx = 0
    for name in names:
        lname = name.lower()
        if "offshore" in lname:
            color_map[name] = key_colors["offshore"]
        elif "onshore" in lname:
            color_map[name] = key_colors["onshore"]
        elif "phev" in lname:
            color_map[name] = key_colors["phev"]
        elif "bev" in lname:
            color_map[name] = key_colors["bev"]
        elif "csp" in lname:
            color_map[name] = key_colors["csp"]
        elif lname in {"pv", "solar pv"} or " pv" in lname:
            color_map[name] = key_colors["pv"]
        elif lname == "other" or name == TECH_OTHER_LABEL:
            color_map[name] = key_colors["other"]
        else:
            color_map[name] = fallback_cycle[fallback_idx % len(fallback_cycle)]
            fallback_idx += 1
    return color_map


def make_metal_focus_color_map(focus_elements: list[str], other_label: str) -> dict[str, str]:
    # 10-color palette for highlighted elements + neutral gray for the aggregated remainder.
    palette = [
        "#4E79A7",
        "#F28E2B",
        "#E15759",
        "#76B7B2",
        "#59A14F",
        "#EDC948",
        "#B07AA1",
        "#FF9DA7",
        "#9C755F",
        "#BAB0AC",
    ]
    color_map: dict[str, str] = {}
    for i, element in enumerate(focus_elements):
        color_map[element] = palette[i % len(palette)]
    color_map[other_label] = METAL_OTHER_COLOR
    return color_map


def get_top_metal_elements_ref_vs_allin(
    met_cat: pl.DataFrame,
    top_k: int,
    ref_strategy: str,
    all_in_strategy: str,
) -> list[str]:
    pair = (
        met_cat.filter(pl.col("strategy").is_in([ref_strategy, all_in_strategy]))
        .group_by(["sector", "element", "strategy"])
        .agg(
            [
                pl.col("baseline_metal_med").sum().alias("baseline"),
                pl.col("final_metal_med").sum().alias("final"),
            ]
        )
    )
    if pair.is_empty():
        return []

    wide = pair.pivot(on="strategy", index=["sector", "element"], values=["baseline", "final"])
    ref_baseline_col = f"baseline_{ref_strategy}"
    ref_final_col = f"final_{ref_strategy}"
    all_final_col = f"final_{all_in_strategy}"
    required_cols = {ref_baseline_col, ref_final_col, all_final_col}
    if not required_cols.issubset(set(wide.columns)):
        fallback = (
            met_cat.group_by("element")
            .agg(pl.col("final_metal_med").median().alias("score"))
            .sort("score", descending=True)
        )
        return fallback["element"].head(top_k).to_list()

    ref_totals = wide.group_by("sector").agg(pl.col(ref_baseline_col).sum().alias("ref_baseline_total"))
    scored = (
        wide.join(ref_totals, on="sector", how="left")
        .with_columns(
            (
                (pl.col(all_final_col).fill_null(0.0) - pl.col(ref_final_col).fill_null(0.0)).abs()
                / pl.col("ref_baseline_total")
            ).alias("delta_share_abs")
        )
        .group_by("element")
        .agg(pl.col("delta_share_abs").sum().alias("score"))
        .sort("score", descending=True)
    )
    return scored["element"].head(top_k).to_list()


def build_regional_metal_log_groups(
    met_cat: pl.DataFrame,
    ref_strategy: str,
    n_groups: int = 3,
) -> list[dict[str, str | list[str]]]:
    ref_mass = (
        met_cat.filter(pl.col("strategy") == ref_strategy)
        .group_by("element")
        .agg(pl.col("baseline_metal_med").sum().alias("ref_mass"))
        .filter(pl.col("ref_mass") > 0)
        .with_columns(pl.col("ref_mass").log10().alias("log_ref_mass"))
        .sort("log_ref_mass")
    )
    if ref_mass.is_empty():
        return []

    q_row = ref_mass.select(
        [
            pl.col("log_ref_mass").quantile(1 / 3).alias("q1"),
            pl.col("log_ref_mass").quantile(2 / 3).alias("q2"),
        ]
    ).to_dicts()[0]
    q1 = float(q_row["q1"])
    q2 = float(q_row["q2"])

    low = ref_mass.filter(pl.col("log_ref_mass") <= q1)["element"].to_list()
    mid = ref_mass.filter((pl.col("log_ref_mass") > q1) & (pl.col("log_ref_mass") <= q2))[
        "element"
    ].to_list()
    high = ref_mass.filter(pl.col("log_ref_mass") > q2)["element"].to_list()

    # If ties at quantile boundaries create empty bins, fall back to equal-size splits.
    if any(len(g) == 0 for g in (low, mid, high)):
        ordered = ref_mass["element"].to_list()
        chunks = [list(c) for c in np.array_split(np.array(ordered, dtype=object), n_groups)]
        low, mid, high = chunks[0], chunks[1], chunks[2]

    group_defs = [
        ("Major", "Major Metals", high),
        ("Intermediate", "Intermediate Metals", mid),
        ("Minor", "Minor Metals", low),
    ]
    groups: list[dict[str, str | list[str]]] = []
    for name, label, elements in group_defs:
        if not elements:
            continue
        groups.append(
            {
                "name": name,
                "label": label,
                "elements": sorted(elements),
            }
        )
    return groups


def get_collapsed_groups(
    panel_df: pl.DataFrame,
    group_col: str,
    baseline_col: str,
    final_col: str,
    top_n: int,
) -> list[str]:
    collapsed = collapse_top_n(
        panel_df,
        group_col=group_col,
        baseline_col=baseline_col,
        final_col=final_col,
        top_n=top_n,
    )
    return sorted(collapsed[group_col].unique().to_list())


def collapse_top_n(
    df: pl.DataFrame,
    group_col: str,
    baseline_col: str,
    final_col: str,
    top_n: int = 12,
    keep_groups: list[str] | None = None,
    other_label: str = TECH_OTHER_LABEL,
    include_other: bool = True,
) -> pl.DataFrame:
    if keep_groups is None:
        score = (
            df.group_by(group_col)
            .agg(pl.col(final_col).median().alias("score"))
            .sort("score", descending=True)
        )
        keep = score[group_col].head(top_n).to_list()
    else:
        keep = keep_groups
    prepped = (
        df.with_columns(
            pl.when(pl.col(group_col).is_in(keep))
            .then(pl.col(group_col))
            .otherwise(pl.lit(other_label))
            .alias(group_col)
        )
        if include_other
        else df.filter(pl.col(group_col).is_in(keep))
    )
    return prepped.group_by(["strategy", group_col]).agg(
        [
            pl.col(baseline_col).sum().alias(baseline_col),
            pl.col(final_col).sum().alias(final_col),
        ]
    )


def plot_panel(
    ax,
    panel_df: pl.DataFrame,
    strategy_order: list[str],
    group_col: str,
    baseline_col: str,
    final_col: str,
    main_title: str,
    urtitle: str,
    color_map: dict[str, str | tuple[float, float, float, float]],
    top_n: int,
    shared_ylim: float,
    panel_ylim: float | None = None,
    baseline_denominator_strategy: str | None = None,
    keep_groups: list[str] | None = None,
    other_label: str = TECH_OTHER_LABEL,
    include_other: bool = True,
    ylabel: str | None = None,
    xlabel: str = "",
    sector: str | None = None,
) -> list[str]:
    panel_df = collapse_top_n(
        panel_df,
        group_col=group_col,
        baseline_col=baseline_col,
        final_col=final_col,
        top_n=top_n,
        keep_groups=keep_groups,
        other_label=other_label,
        include_other=include_other,
    )
    # Baseline-anchored shares: baseline fixed at 100%, final relative to baseline total.
    if baseline_denominator_strategy is None:
        denominator = pl.col(baseline_col).sum().over("strategy")
    else:
        ref_total = (
            panel_df.filter(pl.col("strategy") == baseline_denominator_strategy)
            .select(pl.col(baseline_col).sum().alias("ref_total"))
        )
        ref_total_value = float(ref_total["ref_total"][0] or 0.0) if ref_total.height > 0 else 0.0
        if ref_total_value <= 0:
            print(
                f"Warning: missing/non-positive baseline for '{baseline_denominator_strategy}'. "
                "Falling back to strategy-specific baseline."
            )
            denominator = pl.col(baseline_col).sum().over("strategy")
        else:
            denominator = pl.lit(ref_total_value)

    panel_df = panel_df.with_columns(
        [
            (pl.col(baseline_col) / denominator).alias("baseline_share"),
            (pl.col(final_col) / denominator).alias("final_share"),
        ]
    )
    groups_all = panel_df[group_col].unique().to_list()
    if group_col == "tech" and sector is not None:
        groups = sort_tech_groups(groups_all, sector)
    elif include_other and other_label in groups_all:
        groups = sort_metal_groups(groups_all, other_label)
    else:
        groups = sorted(groups_all)
    colors = {g: color_map.get(g, METAL_OTHER_COLOR) for g in groups}

    x_positions = list(range(len(strategy_order)))
    bar_width = 0.62
    xlim = (-0.6, len(strategy_order) - 0.4)

    baseline_totals = []
    final_totals = []
    for strategy in strategy_order:
        s_df = panel_df.filter(pl.col("strategy") == strategy)
        baseline_totals.append(float(s_df["baseline_share"].sum() or 0.0))
        final_totals.append(float(s_df["final_share"].sum() or 0.0))

    # Baseline reference (hatched): scenario baseline = 100% for deployment panels.
    for x, h in zip(x_positions, baseline_totals):
        ax.bar(
            np.array([x]),
            np.array([h]),
            width=bar_width,
            color=BASELINE_FACE,
            edgecolor=BASELINE_EDGE,
            hatch=BASELINE_HATCH,
            linewidth=0.6,
            alpha=1.0,
            zorder=1,
        )

    # Gap or exceedance between baseline total and final total.
    for i, (b_total, f_total) in enumerate(zip(baseline_totals, final_totals)):
        gap_height = abs(f_total - b_total)
        if gap_height <= 1e-9:
            continue
        gap_bottom = min(f_total, b_total)
        if f_total < b_total:
            ax.bar(
                np.array([x_positions[i]]),
                np.array([gap_height]),
                width=bar_width,
                bottom=np.array([gap_bottom]),
                color=GAP_FACE,
                edgecolor=BASELINE_EDGE,
                hatch=GAP_HATCH,
                linewidth=0.55,
                zorder=1.5,
            )
        else:
            ax.bar(
                np.array([x_positions[i]]),
                np.array([gap_height]),
                width=bar_width,
                bottom=np.array([gap_bottom]),
                color="none",
                edgecolor=BASELINE_EDGE,
                linewidth=0.55,
                linestyle="--",
                zorder=1.5,
            )

    # Shared 100% reference (Ref baseline for metal panels; scenario baseline for deployment).
    ax.axhline(
        1.0,
        color=BASELINE_EDGE,
        linestyle="--",
        linewidth=0.75,
        alpha=0.75,
        zorder=0.5,
    )

    # Colored final composition overlays
    for i, strategy in enumerate(strategy_order):
        s_df = panel_df.filter(pl.col("strategy") == strategy)
        bottom = 0.0
        for g in groups:
            row = s_df.filter(pl.col(group_col) == g)
            if row.height == 0:
                continue
            h = float(row["final_share"][0])
            if h <= 0:
                continue
            ax.bar(
                np.array([x_positions[i]]),
                np.array([h]),
                width=bar_width,
                bottom=np.array([bottom]),
                color=colors[g],
                edgecolor=colors[g],
                linewidth=0.35,
                zorder=2,
            )
            bottom += h

    ax.format(
        title=main_title,
        urtitle=urtitle,
        titlesize=14,
        titlepad=6,
        xticks=x_positions,
        xticklabels=[STRATEGY_LABELS.get(s, s) for s in strategy_order],
        xticklabelsize=11,
        xrotation=60,
        xlabel=xlabel,
        xlabelsize=12,
        ylabel=ylabel,
        ylabelsize=12,
        xlim=xlim,
        yticklabelsize=11,
        ytickminor=False,
        grid=False,
        ylim=(0, panel_ylim if panel_ylim is not None else shared_ylim),
    )
    for label in ax.get_xticklabels():
        label.set_ha("right")
    ax.yaxis.set_major_formatter(FuncFormatter(lambda y, _: f"{y * 100:.0f}%"))
    return groups


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Plot structure-change 2x2 figures at global and/or regional scale."
    )
    parser.add_argument(
        "--scale",
        choices=VALID_SCALES,
        default="global",
        help="Spatial scale to render. Use 'both' to render global and regional outputs.",
    )
    parser.add_argument(
        "--regions",
        nargs="+",
        default=None,
        help="Optional list of region codes to render for regional mode (for example: R10EUROPE R10CHINA+).",
    )
    parser.add_argument(
        "--categories",
        nargs="+",
        default=None,
        help="Optional category filter (for example: C2). Omit to render all C1–C8.",
    )
    return parser.parse_args()


def aggregate_medians(
    base: pl.LazyFrame,
    extra_group_cols: list[str],
) -> tuple[pl.DataFrame, pl.DataFrame]:
    dep_level2 = ["category", *extra_group_cols, "strategy", "sector", "tech"]
    metal_level1 = ["category", *extra_group_cols, "path_abb", "strategy", "sector", "element"]
    metal_level2 = ["category", *extra_group_cols, "strategy", "sector", "element"]
    dep_dedupe_keys = [
        "category",
        "path_abb",
        "strategy",
        "region",
        "sector",
        "tech",
        "subtech",
    ]

    # Deployment rows repeat the same subtech×region totals per element; count once.
    deployment_base = base.group_by(dep_dedupe_keys).agg(
        [
            pl.col("baseline_deployment").first().alias("baseline_deployment"),
            pl.col("final_deployment").first().alias("final_deployment"),
        ]
    )

    # Deployment: dedupe element rows, then median path-level achievement (R)
    # and tech composition (w) so sector totals match model conservation.
    path_keys = ["category", *extra_group_cols, "path_abb", "strategy", "sector"]
    strat_keys = ["category", *extra_group_cols, "strategy", "sector"]

    tech_path = (
        deployment_base.group_by([*path_keys, "tech"])
        .agg(
            pl.col("baseline_deployment").sum().alias("b_tech"),
            pl.col("final_deployment").sum().alias("f_tech"),
        )
    )

    sector_path = (
        tech_path.group_by(path_keys)
        .agg(
            pl.col("b_tech").sum().alias("b_sec"),
            pl.col("f_tech").sum().alias("f_sec"),
        )
        .with_columns((pl.col("f_sec") / pl.col("b_sec")).alias("r_sec"))
    )

    tech_with_sector = tech_path.join(sector_path, on=path_keys, how="left").with_columns(
        pl.when(pl.col("f_sec") > 0)
        .then(pl.col("f_tech") / pl.col("f_sec"))
        .otherwise(0.0)
        .alias("w_tech")
    )

    r_med = sector_path.group_by(strat_keys).agg(pl.col("r_sec").median().alias("r_med"))

    baseline_med = tech_path.group_by(dep_level2).agg(
        pl.col("b_tech").median().alias("baseline_deployment_med")
    )

    w_med = tech_with_sector.group_by(dep_level2).agg(pl.col("w_tech").median().alias("w_med"))

    deployment_med = (
        baseline_med.join(w_med, on=dep_level2, how="left")
        .join(r_med, on=strat_keys, how="left")
        .with_columns(
            pl.col("w_med").sum().over(strat_keys).alias("w_sum"),
            pl.col("baseline_deployment_med").sum().over(strat_keys).alias("b_total"),
        )
        .with_columns(
            pl.when(pl.col("w_sum") > 0)
            .then(pl.col("w_med") / pl.col("w_sum"))
            .otherwise(0.0)
            .alias("w_norm"),
        )
        .with_columns(
            (pl.col("w_norm") * pl.col("r_med") * pl.col("b_total")).alias("final_deployment_med"),
        )
        .select(dep_level2 + ["baseline_deployment_med", "final_deployment_med"])
        .collect()
    )

    # Metal: sum within path_abb first, then median across path_abb.
    metal_med = (
        base.group_by(metal_level1)
        .agg(
            [
                pl.col("baseline_metal").sum().alias("baseline_metal_sum"),
                pl.col("final_metal").sum().alias("final_metal_sum"),
            ]
        )
        .group_by(metal_level2)
        .agg(
            [
                pl.col("baseline_metal_sum").median().alias("baseline_metal_med"),
                pl.col("final_metal_sum").median().alias("final_metal_med"),
            ]
        )
        .collect()
    )
    return deployment_med, metal_med


def print_scenario_counts(base: pl.LazyFrame, extra_group_cols: list[str]) -> None:
    scenario_counts = (
        base.select(["category", *extra_group_cols, "strategy", "path_abb"])
        .unique()
        .group_by(["category", *extra_group_cols, "strategy"])
        .agg(pl.col("path_abb").n_unique().alias("n_paths"))
        .collect()
        .sort(["category", *extra_group_cols, "strategy"])
    )
    print("\nScenario counts:")
    for row in scenario_counts.iter_rows(named=True):
        parts = [f"category={row['category']}"]
        for col in extra_group_cols:
            parts.append(f"{col}={row[col]}")
        parts.extend([f"strategy={row['strategy']}", f"n_paths={row['n_paths']}"])
        print(", ".join(parts))


def render_scale(
    base: pl.LazyFrame,
    scale: str,
    regions: list[str] | None,
    categories: list[str] | None = None,
) -> None:
    extra_group_cols = ["region"] if scale == "regional" else []

    base_scale = base
    if categories:
        base_scale = base_scale.filter(pl.col("category").is_in(categories))
    if scale == "regional" and regions:
        base_scale = base_scale.filter(pl.col("region").is_in(regions))

    deployment_med, metal_med = aggregate_medians(base_scale, extra_group_cols=extra_group_cols)
    if deployment_med.is_empty() or metal_med.is_empty():
        print(f"No data found for scale={scale}. Skipping.")
        return

    print_scenario_counts(base_scale, extra_group_cols=extra_group_cols)

    available_strategies = deployment_med["strategy"].unique().to_list()
    strategy_order = [s for s in TARGET_STRATEGIES if s in available_strategies]
    print(f"\nUsing strategies for scale={scale}: {strategy_order}")
    if not strategy_order:
        print(f"No matching strategies for scale={scale}. Skipping.")
        return

    strata_cols = ["category", *extra_group_cols]
    strata = deployment_med.select(strata_cols).unique().sort(strata_cols)
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    regional_element_colors: dict[str, tuple[float, float, float, float]] | None = None
    if scale == "regional":
        regional_elements = sorted(metal_med["element"].unique().to_list())
        regional_element_colors = make_element_color_map(regional_elements)

    uplt.rc["grid"] = False
    print(f"\nRendering structure-change figures for scale={scale}...")
    for stratum in strata.iter_rows(named=True):
        category = stratum["category"]
        region = stratum.get("region")

        dep_cat = deployment_med.filter(pl.col("category") == category)
        met_cat = metal_med.filter(pl.col("category") == category)
        if region is not None:
            dep_cat = dep_cat.filter(pl.col("region") == region)
            met_cat = met_cat.filter(pl.col("region") == region)

        dep_power_df = dep_cat.filter(pl.col("sector") == "Power").select(
            ["strategy", "tech", "baseline_deployment_med", "final_deployment_med"]
        )
        dep_transport_df = dep_cat.filter(pl.col("sector") == "Transport").select(
            ["strategy", "tech", "baseline_deployment_med", "final_deployment_med"]
        )
        met_power_df = met_cat.filter(pl.col("sector") == "Power").select(
            ["strategy", "element", "baseline_metal_med", "final_metal_med"]
        )
        met_transport_df = met_cat.filter(pl.col("sector") == "Transport").select(
            ["strategy", "element", "baseline_metal_med", "final_metal_med"]
        )
        if (
            dep_power_df.is_empty()
            or dep_transport_df.is_empty()
            or met_power_df.is_empty()
            or met_transport_df.is_empty()
        ):
            print(
                f"Skipping stratum with missing sector data: "
                f"category={category}"
                + (f", region={region}" if region is not None else "")
            )
            continue

        tech_groups_power = get_collapsed_groups(
            dep_power_df, "tech", "baseline_deployment_med", "final_deployment_med", top_n=10
        )
        tech_groups_transport = get_collapsed_groups(
            dep_transport_df, "tech", "baseline_deployment_med", "final_deployment_med", top_n=10
        )
        tech_groups = sort_tech_groups(
            sorted(set(tech_groups_power + tech_groups_transport)),
            "Power",
        )
        all_elements = sorted(met_cat["element"].unique().to_list())
        if scale == "regional":
            met_power_cat = met_cat.filter(pl.col("sector") == "Power")
            met_transport_cat = met_cat.filter(pl.col("sector") == "Transport")
            metal_group_specs_power = build_regional_metal_log_groups(
                met_cat=met_power_cat,
                ref_strategy=REF_STRATEGY,
                n_groups=3,
            )
            metal_group_specs_transport = build_regional_metal_log_groups(
                met_cat=met_transport_cat,
                ref_strategy=REF_STRATEGY,
                n_groups=3,
            )
            if not metal_group_specs_power or not metal_group_specs_transport:
                print(
                    f"Skipping stratum with no regional metal groups: category={category}"
                    + (f", region={region}" if region is not None else "")
                )
                continue
            group_order = [
                ("Major", "Major Metals"),
                ("Intermediate", "Intermediate Metals"),
                ("Minor", "Minor Metals"),
            ]
            power_group_map = {
                str(spec["name"]): [str(e) for e in spec["elements"]]
                for spec in metal_group_specs_power
            }
            transport_group_map = {
                str(spec["name"]): [str(e) for e in spec["elements"]]
                for spec in metal_group_specs_transport
            }
            power_all_elements = sorted(met_power_cat["element"].unique().to_list())
            transport_all_elements = sorted(met_transport_cat["element"].unique().to_list())
            regional_row_specs: list[dict[str, str | list[str]]] = []
            for name, label in group_order:
                power_elements = power_group_map.get(name, power_all_elements)
                transport_elements = transport_group_map.get(name, transport_all_elements)
                regional_row_specs.append(
                    {
                        "name": name,
                        "label": label,
                        "power_elements": power_elements,
                        "transport_elements": transport_elements,
                    }
                )
            metal_color_order: list[str] = []
            for row in regional_row_specs:
                for e in row["power_elements"] + row["transport_elements"]:
                    e_str = str(e)
                    if e_str not in metal_color_order:
                        metal_color_order.append(e_str)
            if regional_element_colors is None:
                metal_colors = make_element_color_map(metal_color_order)
            else:
                metal_colors = regional_element_colors
            print(
                f"category={category}"
                + (f", region={region}" if region is not None else "")
                + ", regional metal groups by sector baseline mass (log quantiles): "
                + "; ".join(
                    (
                        f"{str(row['name'])} "
                        f"(P={len(row['power_elements'])}, T={len(row['transport_elements'])})"
                    )
                    for row in regional_row_specs
                )
            )
        else:
            metal_focus_elements = get_top_metal_elements_ref_vs_allin(
                met_cat=met_cat,
                top_k=METAL_TOP_K,
                ref_strategy=REF_STRATEGY,
                all_in_strategy=ALL_IN_STRATEGY,
            )
            has_other_metals = any(e not in set(metal_focus_elements) for e in all_elements)
            metal_groups = sort_metal_groups(
                [e for e in metal_focus_elements if e in all_elements],
                METAL_OTHER_LABEL,
            )
            if has_other_metals:
                metal_groups.append(METAL_OTHER_LABEL)
            metal_keep_groups = metal_focus_elements
            metal_top_n = METAL_TOP_K
            metal_other_label = METAL_OTHER_LABEL
            metal_colors = make_metal_focus_color_map(metal_focus_elements, METAL_OTHER_LABEL)
            print(
                f"category={category}"
                + (f", region={region}" if region is not None else "")
                + f", metal focus elements (Ref vs All-in top {METAL_TOP_K}): "
                f"{metal_focus_elements}"
            )

        tech_colors = make_tech_color_map(tech_groups)

        def panel_top(
            df: pl.DataFrame,
            gcol: str,
            bcol: str,
            fcol: str,
            top_n: int,
            baseline_denominator_strategy: str | None = None,
            keep_groups: list[str] | None = None,
            other_label: str = TECH_OTHER_LABEL,
            include_other: bool = True,
        ) -> float:
            cdf = collapse_top_n(
                df,
                group_col=gcol,
                baseline_col=bcol,
                final_col=fcol,
                top_n=top_n,
                keep_groups=keep_groups,
                other_label=other_label,
                include_other=include_other,
            )
            if baseline_denominator_strategy is None:
                denominator = pl.col(bcol).sum().over("strategy")
            else:
                ref_total = (
                    cdf.filter(pl.col("strategy") == baseline_denominator_strategy)
                    .select(pl.col(bcol).sum().alias("ref_total"))
                )
                ref_total_value = float(ref_total["ref_total"][0] or 0.0) if ref_total.height > 0 else 0.0
                if ref_total_value <= 0:
                    denominator = pl.col(bcol).sum().over("strategy")
                else:
                    denominator = pl.lit(ref_total_value)
            cdf = cdf.with_columns((pl.col(fcol) / denominator).alias("final_share"))
            totals = cdf.group_by("strategy").agg(pl.col("final_share").sum().alias("total"))
            if totals.is_empty():
                return 1.0
            return max(1.0, float(totals["total"].max() or 1.0))

        if scale == "regional":
            tech_power_top = panel_top(
                dep_power_df, "tech", "baseline_deployment_med", "final_deployment_med", 10
            )
            tech_transport_top = panel_top(
                dep_transport_df, "tech", "baseline_deployment_med", "final_deployment_med", 10
            )
            metal_panel_tops: dict[str, tuple[float, float]] = {}
            all_panel_tops = [tech_power_top, tech_transport_top]
            for row in regional_row_specs:
                spec_name = str(row["name"])
                power_elements = [str(e) for e in row["power_elements"]]
                transport_elements = [str(e) for e in row["transport_elements"]]
                power_top_n_spec = max(1, len(power_elements))
                transport_top_n_spec = max(1, len(transport_elements))
                power_top = panel_top(
                    met_power_df,
                    "element",
                    "baseline_metal_med",
                    "final_metal_med",
                    power_top_n_spec,
                    keep_groups=power_elements,
                    include_other=False,
                )
                transport_top = panel_top(
                    met_transport_df,
                    "element",
                    "baseline_metal_med",
                    "final_metal_med",
                    transport_top_n_spec,
                    keep_groups=transport_elements,
                    include_other=False,
                )
                metal_panel_tops[spec_name] = (power_top, transport_top)
                all_panel_tops.extend([power_top, transport_top])

            # Regional default: all panels use 130% unless an individual panel exceeds it.
            shared_ylim = REGIONAL_PANEL_YLIM_BREAKPOINT
            regional_has_outlier = any(v > REGIONAL_PANEL_YLIM_BREAKPOINT for v in all_panel_tops)

            def panel_ylim_from_top(top_value: float) -> float:
                return (
                    REGIONAL_PANEL_YLIM_FACTOR * top_value
                    if top_value > REGIONAL_PANEL_YLIM_BREAKPOINT
                    else shared_ylim
                )

            tech_power_ylim = panel_ylim_from_top(tech_power_top)
            tech_transport_ylim = panel_ylim_from_top(tech_transport_top)
            deploy_row_ylim = max(tech_power_ylim, tech_transport_ylim)
            tech_power_ylim = deploy_row_ylim
            tech_transport_ylim = deploy_row_ylim
            regional_max_top = max(all_panel_tops) if all_panel_tops else shared_ylim
            regional_max_ylim = panel_ylim_from_top(regional_max_top)
            regional_wspace = 3.2
            if regional_has_outlier:
                # Dynamic spacing for center gap: long percentage tick labels on right-column
                # y-axes need extra room to avoid colliding with left-column panels.
                max_pct_value = int(np.ceil(regional_max_ylim * 100))
                max_pct_digits = len(str(abs(max_pct_value)))
                regional_wspace = 3.6 + 0.25 * max(0, max_pct_digits - 3)
                regional_wspace = min(4.6, regional_wspace)
            regional_hspace = 3.7
        else:
            tech_power_top = panel_top(
                dep_power_df, "tech", "baseline_deployment_med", "final_deployment_med", 10
            )
            tech_transport_top = panel_top(
                dep_transport_df, "tech", "baseline_deployment_med", "final_deployment_med", 10
            )
            metal_power_top = panel_top(
                met_power_df,
                "element",
                "baseline_metal_med",
                "final_metal_med",
                metal_top_n,
                baseline_denominator_strategy=REF_STRATEGY,
                keep_groups=metal_keep_groups,
                other_label=metal_other_label,
            )
            metal_transport_top = panel_top(
                met_transport_df,
                "element",
                "baseline_metal_med",
                "final_metal_med",
                metal_top_n,
                baseline_denominator_strategy=REF_STRATEGY,
                keep_groups=metal_keep_groups,
                other_label=metal_other_label,
            )
            shared_ylim = 1.12 * max(
                tech_power_top,
                tech_transport_top,
                metal_power_top,
                metal_transport_top,
            )
            tech_power_ylim = shared_ylim
            tech_transport_ylim = shared_ylim
            regional_has_outlier = False

        if scale == "regional":
            fig = uplt.figure(
                refwidth=3.2,
                refheight=2.0,
                facecolor=(0, 0, 0, 0),
            )
            axes = fig.subplots(nrows=4, ncols=2, wspace=regional_wspace, hspace=regional_hspace)
        else:
            fig = uplt.figure(
                refwidth=3.2,
                refheight=2.4,
                facecolor=(0, 0, 0, 0),
            )
            axes = fig.subplots(nrows=2, ncols=2, wspace=2.6, hspace=3.4)
        if scale == "regional":
            region_label = region.replace("R10", "") if region is not None else ""
            fig.format(
                suptitle=f"{region_label} | Technology Deployment and Metal Use Shares",
                suptitle_kw={"fontsize": 14},
                suptitlepad=10,
            )

        plot_panel(
            axes[0, 0],
            dep_power_df,
            strategy_order=strategy_order,
            group_col="tech",
            baseline_col="baseline_deployment_med",
            final_col="final_deployment_med",
            main_title="Technology deployment",
            urtitle="Power",
            color_map=tech_colors,
            top_n=10,
            shared_ylim=shared_ylim,
            panel_ylim=tech_power_ylim,
            ylabel=DEPLOY_YLABEL,
            sector="Power",
        )
        plot_panel(
            axes[0, 1],
            dep_transport_df,
            strategy_order=strategy_order,
            group_col="tech",
            baseline_col="baseline_deployment_med",
            final_col="final_deployment_med",
            main_title="Technology deployment",
            urtitle="Transport",
            color_map=tech_colors,
            top_n=10,
            shared_ylim=shared_ylim,
            panel_ylim=tech_transport_ylim,
            sector="Transport",
        )

        tech_handles = [
            Patch(facecolor=tech_colors.get(g, TECH_OTHER_COLOR), edgecolor="none", label=g)
            for g in tech_groups
        ]
        ref_handles = [
            Patch(
                facecolor=GAP_FACE,
                edgecolor=BASELINE_EDGE,
                hatch=GAP_HATCH,
                label="Gap to baseline",
            ),
        ]
        tech_legend = axes[0, 1].legend(
            handles=tech_handles + ref_handles,
            fontsize=12,
            frameon=False,
            ncol=1,
            loc="center left",
            bbox_to_anchor=(1.02, 0.5),
            title="Technology",
            title_fontsize=14,
        )
        tech_legend._legend_box.align = "left"
        tech_legend.get_title().set_ha("left")

        if scale == "regional":
            for row_idx, row in enumerate(regional_row_specs, start=1):
                spec_name = str(row["name"])
                spec_label = str(row["label"])
                power_elements = [str(e) for e in row["power_elements"]]
                transport_elements = [str(e) for e in row["transport_elements"]]
                power_top_n_spec = max(1, len(power_elements))
                transport_top_n_spec = max(1, len(transport_elements))
                spec_power_top, spec_transport_top = metal_panel_tops.get(spec_name, (shared_ylim, shared_ylim))
                spec_row_ylim = max(
                    panel_ylim_from_top(spec_power_top),
                    panel_ylim_from_top(spec_transport_top),
                )

                plot_panel(
                    axes[row_idx, 0],
                    met_power_df,
                    strategy_order=strategy_order,
                    group_col="element",
                    baseline_col="baseline_metal_med",
                    final_col="final_metal_med",
                    main_title=f"Metal Use | {spec_label}",
                    urtitle="Power",
                    color_map=metal_colors,
                    top_n=power_top_n_spec,
                    shared_ylim=shared_ylim,
                    panel_ylim=spec_row_ylim,
                    keep_groups=power_elements,
                    include_other=False,
                    baseline_denominator_strategy=REF_STRATEGY,
                    ylabel=METAL_YLABEL,
                )
                plot_panel(
                    axes[row_idx, 1],
                    met_transport_df,
                    strategy_order=strategy_order,
                    group_col="element",
                    baseline_col="baseline_metal_med",
                    final_col="final_metal_med",
                    main_title=f"Metal Use | {spec_label}",
                    urtitle="Transport",
                    color_map=metal_colors,
                    top_n=transport_top_n_spec,
                    shared_ylim=shared_ylim,
                    panel_ylim=spec_row_ylim,
                    keep_groups=transport_elements,
                    include_other=False,
                    baseline_denominator_strategy=REF_STRATEGY,
                )

                row_elements = sort_metal_groups(sorted(set(power_elements + transport_elements)), METAL_OTHER_LABEL)
                metal_handles = [Patch(facecolor=metal_colors[e], edgecolor="none", label=e) for e in row_elements]
                metal_legend = axes[row_idx, 1].legend(
                    handles=metal_handles,
                    fontsize=14,
                    frameon=False,
                    ncol=1,
                    loc="center left",
                    bbox_to_anchor=(1.02, 0.5),
                    title=f"Elements ({spec_name})",
                    title_fontsize=14,
                )
                metal_legend._legend_box.align = "left"
                metal_legend.get_title().set_ha("left")
        else:
            plot_panel(
                axes[1, 0],
                met_power_df,
                strategy_order=strategy_order,
                group_col="element",
                baseline_col="baseline_metal_med",
                final_col="final_metal_med",
                main_title="Metal Use",
                urtitle="Power",
                color_map=metal_colors,
                top_n=metal_top_n,
                shared_ylim=shared_ylim,
                baseline_denominator_strategy=REF_STRATEGY,
                keep_groups=metal_keep_groups,
                other_label=metal_other_label,
                ylabel=METAL_YLABEL,
            )
            plot_panel(
                axes[1, 1],
                met_transport_df,
                strategy_order=strategy_order,
                group_col="element",
                baseline_col="baseline_metal_med",
                final_col="final_metal_med",
                main_title="Metal Use",
                urtitle="Transport",
                color_map=metal_colors,
                top_n=metal_top_n,
                shared_ylim=shared_ylim,
                baseline_denominator_strategy=REF_STRATEGY,
                keep_groups=metal_keep_groups,
                other_label=metal_other_label,
            )

            metal_handles = [Patch(facecolor=metal_colors[g], edgecolor="none", label=g) for g in metal_groups]
            metal_legend = axes[1, 1].legend(
                handles=metal_handles,
                fontsize=14,
                frameon=False,
                ncol=1,
                loc="center left",
                bbox_to_anchor=(1.02, 0.5),
                title="Element",
                title_fontsize=14,
            )
            metal_legend._legend_box.align = "left"
            metal_legend.get_title().set_ha("left")

        axes.format(abc=True, abcloc="ul", abcsize=14, grid=False, ytickminor=False)
        # Y-axis ticks and labels only on the first column; enforce shared y-limits per row.
        nrows = len(axes[:, 0])
        for row_idx in range(nrows):
            ax_left = axes[row_idx, 0]
            ax_right = axes[row_idx, 1]
            row_ylim = ax_left.get_ylim()
            ax_right.set_ylim(row_ylim)
            ax_left.tick_params(axis="y", labelleft=True, labelright=False)
            ax_left.yaxis.set_ticks_position("left")
            ax_left.yaxis.set_label_position("left")
            ax_right.tick_params(axis="y", labelleft=False, labelright=False)
            ax_right.set_ylabel("")
        if region is None:
            out_path = OUTPUT_DIR / f"structure_change_2x2_{clean_name(category)}.svg"
        else:
            out_path = (
                OUTPUT_DIR
                / f"structure_change_2x2_regional_{clean_name(category)}_{clean_name(region)}.svg"
            )
        fig.save(out_path)
        plt.close("all")
        print(f"Saved: {out_path}")


def main() -> None:
    args = parse_args()
    print("Loading deploy_metal_use with lazy scan...")
    base = (
        pl.scan_parquet(INPUT_PARQUET.as_posix())
        .filter(pl.col("strategy").is_in(TARGET_STRATEGIES))
        .filter(pl.col("category").is_not_null())
        .with_columns(
            pl.concat_str(["sector", "tech", "subtech"], separator="_").alias("sector_tech_subtech")
        )
    )

    scales = ["global", "regional"] if args.scale == "both" else [args.scale]
    for scale in scales:
        render_scale(base, scale=scale, regions=args.regions, categories=args.categories)

    print("\nDone.")


if __name__ == "__main__":
    main()

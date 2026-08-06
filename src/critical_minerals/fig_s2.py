import numpy as np
import polars as pl
import ultraplot as uplt
from pathlib import Path


ROOT_DIR = Path(__file__).resolve().parents[2]
SECO_RATE_PATH = ROOT_DIR / "data" / "data_raw" / "miracle_oscar" / "seco_rate.parquet"
ELASTICITY_PATH = ROOT_DIR / "data" / "data_raw" / "miracle_oscar" / "elasticity.parquet"
CATEGORY_PATH = ROOT_DIR / "data" / "data_raw" / "miracle_oscar" / "category.csv"
OUTPUT_PATH = ROOT_DIR / "testpic" / "seco_rate_heatmap_demo.svg"

CATEGORY = "C2"
HEATMAP_COLUMNS = ["bau_bau_recycl", "llt_mir_recych"]
HEATMAP_TITLES = {
    "bau_bau_recycl": "Ref",
    "llt_mir_recych": "Circular",
}
SEMI_ELASTICITY_STRATEGY_MAP = {
    "bau_bau_recycl": "bau_bau_bau_recycl_nosub",
    "llt_mir_recych": "bau_llt_mir_recych_nosub",
}


def build_heatmap_matrix(
    df: pl.DataFrame,
    value_col: str,
    elements: list[str],
    regions: list[str],
) -> np.ndarray:
    """Create matrix with rows=elements and cols=regions."""
    matrix = np.zeros((len(elements), len(regions)), dtype=float)
    lookup = {(row["element"], row["region"]): row[value_col] for row in df.iter_rows(named=True)}

    for i, element in enumerate(elements):
        for j, region in enumerate(regions):
            matrix[i, j] = lookup.get((element, region), 0.0)

    return matrix


def build_positive_symbol_matrix(
    semi_df: pl.DataFrame,
    strategy: str,
    elements: list[str],
    regions: list[str],
) -> np.ndarray:
    """Return boolean matrix where semi_elasticity > 0 for each cell."""
    matrix = np.zeros((len(elements), len(regions)), dtype=bool)
    filtered = semi_df.filter(pl.col("strategy") == strategy)
    lookup = {
        (row["element"], row["region"]): row["semi_elasticity"] > 0
        for row in filtered.iter_rows(named=True)
    }
    for i, element in enumerate(elements):
        for j, region in enumerate(regions):
            matrix[i, j] = lookup.get((element, region), False)
    return matrix


def plot_heatmap_panel(
    ax,
    matrix: np.ndarray,
    symbol_matrix: np.ndarray,
    title: str,
    elements: list[str],
    regions: list[str],
    show_ylabels: bool,
    ylabel: str,
    vmin: float,
    vmax: float,
):
    image_handle = ax.heatmap(
        matrix,
        aspect="auto",
        cmap="Greens",
        vmin=vmin,
        vmax=vmax,
        labels=False,
    )

    ax.set_title(title, fontsize=10, pad=6)
    ax.set_xticks(range(len(regions)))
    ax.set_yticks(range(len(elements)))
    ax.set_xticklabels([r.replace("R10", "") for r in regions], rotation=45, ha="right", fontsize=8)
    if show_ylabels:
        ax.set_yticklabels(elements, fontsize=8)
        ax.set_ylabel(ylabel, fontsize=9)
    else:
        ax.set_yticklabels([])
        ax.set_ylabel("")
        ax.tick_params(axis="y", which="both", left=False)
    ax.set_xlabel("")
    ax.set_box_aspect(1)

    # Cell borders for readability
    ax.set_xticks(np.arange(len(regions)) - 0.5, minor=True)
    ax.set_yticks(np.arange(len(elements)) - 0.5, minor=True)
    ax.grid(which="minor", color="white", linestyle="-", linewidth=0.5)
    ax.tick_params(which="minor", size=0)

    # Overlay symbol where semi-elasticity > 0
    for i in range(len(elements)):
        for j in range(len(regions)):
            if symbol_matrix[i, j]:
                ax.scatter(
                    j,
                    i,
                    marker="*",
                    s=28,
                    c="black",
                    linewidths=0.0,
                    zorder=5,
                )

    return image_handle


def main() -> None:
    print("Loading seco rate data...")
    seco_rate = pl.scan_parquet(SECO_RATE_PATH.as_posix())
    category = (
        pl.scan_csv(CATEGORY_PATH.as_posix())
        .with_columns(path_abb=pl.concat_str(pl.col("model"), pl.col("scenario"), separator="|"))
        .select(["path_abb", "category"])
    )

    print(f"Filtering category == {CATEGORY} and taking median across path_abb...")
    aggregated = (
        seco_rate
        .filter(pl.col("category") == CATEGORY)
        .group_by(["element", "region"])
        .agg(
            pl.median("bau_bau_recycl").alias("bau_bau_recycl"),
            pl.median("llt_mir_recych").alias("llt_mir_recych"),
        )
        .collect()
    )

    if aggregated.is_empty():
        raise ValueError(f"No rows found after filtering category == '{CATEGORY}'.")

    print("Loading semi-elasticity and building >0 mask...")
    semi_elasticity = (
        pl.scan_parquet(ELASTICITY_PATH.as_posix())
        .filter(pl.col("strategy").is_in(list(SEMI_ELASTICITY_STRATEGY_MAP.values())))
        .join(category, on=["path_abb"], how="left")
        .filter(pl.col("category") == CATEGORY)
        .filter(pl.col("year") >= 2026)
        .group_by(["strategy", "element", "region", "year"])
        .agg(pl.median("semi_elasticity").alias("semi_elasticity_median_by_year"))
        .group_by(["strategy", "element", "region"])
        .agg(pl.mean("semi_elasticity_median_by_year").alias("semi_elasticity"))
        .collect()
    )

    elements = sorted(aggregated["element"].unique().to_list(), reverse=True)
    regions = sorted(aggregated["region"].unique().to_list())
    print(f"Elements ({len(elements)}): {elements}")
    print(f"Regions ({len(regions)}): {regions}")

    # Build seco_rate matrices
    matrices = {
        col: build_heatmap_matrix(aggregated, col, elements, regions) for col in HEATMAP_COLUMNS
    }

    # Build positive semi-elasticity symbol matrices (strategy-aligned to each subplot)
    symbol_matrices = {
        metric: build_positive_symbol_matrix(
            semi_df=semi_elasticity,
            strategy=SEMI_ELASTICITY_STRATEGY_MAP[metric],
            elements=elements,
            regions=regions,
        )
        for metric in HEATMAP_COLUMNS
    }

    # Shared fixed color scale across both subplots
    vmin = 0.0
    vmax = 1.0
    print(f"Shared value range: [{vmin:.4f}, {vmax:.4f}]")

    uplt.rc["grid"] = False
    uplt.rc["font.family"] = "Arial"
    uplt.rc["abc.size"] = 9

    fig = uplt.figure(
        refwidth=3.4,
        refheight=3.4,
        facecolor=(0, 0, 0, 0),
        share=False,
        wspace=1.0,
    )
    axes = fig.subplots([[1, 2]])

    image_handle = None
    for idx, metric in enumerate(HEATMAP_COLUMNS):
        image_handle = plot_heatmap_panel(
            ax=axes[idx],
            matrix=matrices[metric],
            symbol_matrix=symbol_matrices[metric],
            title=HEATMAP_TITLES[metric],
            elements=elements,
            regions=regions,
            show_ylabels=(idx == 0),
            ylabel="Element",
            vmin=vmin,
            vmax=vmax,
        )

    fig.colorbar(
        image_handle,
        loc="r",
        label="Secondary use rate",
        labelsize=8,
        ticklabelsize=8,
        length=0.7,
    )
    fig.format(tickminor=False)
    fig.save(OUTPUT_PATH.as_posix())
    print(f"Saved heatmap demo to: {OUTPUT_PATH}")


if __name__ == "__main__":
    main()

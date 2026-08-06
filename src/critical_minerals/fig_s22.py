"""5×5 grid: one panel per metal, regional baseline absolute supply (2025–2100).

Source: elasticity_evolve.parquet, iter_id=0 (baseline supply). Paths → median.
Y-axis: linear scale, independent limits per metal; ×10^n in panel upper-left.
"""
from pathlib import Path

import numpy as np
import polars as pl
import ultraplot as uplt
from matplotlib.lines import Line2D
from matplotlib.ticker import MaxNLocator

REPO_ROOT = Path(__file__).resolve().parents[2]
DATA_RAW = REPO_ROOT / "data" / "data_raw" / "miracle_oscar"
OUT_DIR = REPO_ROOT / "testpic" / "regional_supply"
OUT_PATH = OUT_DIR / "regional_supply_growth_5x5.svg"

STRATEGY = "bau_bau_bau_recycl_nosub"

NROWS, NCOLS = 5, 5

REGIONS = [
    "R10AFRICA",
    "R10CHINA+",
    "R10EUROPE",
    "R10INDIA+",
    "R10LATIN_AM",
    "R10MIDDLE_EAST",
    "R10NORTH_AM",
    "R10PAC_OECD",
    "R10REF_ECON",
    "R10REST_ASIA",
]

REG_LABELS = {
    "R10AFRICA": "Africa",
    "R10CHINA+": "China+",
    "R10EUROPE": "Europe",
    "R10INDIA+": "India+",
    "R10LATIN_AM": "Latin Am",
    "R10MIDDLE_EAST": "Middle East",
    "R10NORTH_AM": "North Am",
    "R10PAC_OECD": "Pac OECD",
    "R10REF_ECON": "Ref Econ",
    "R10REST_ASIA": "Rest Asia",
}

REG_COLORS = {
    "R10AFRICA": "#e6194b",
    "R10CHINA+": "#f58231",
    "R10EUROPE": "#3cb44b",
    "R10INDIA+": "#4363d8",
    "R10LATIN_AM": "#42d4f4",
    "R10MIDDLE_EAST": "#f032e6",
    "R10NORTH_AM": "#a9a9ff",
    "R10PAC_OECD": "#9A6324",
    "R10REF_ECON": "#800000",
    "R10REST_ASIA": "#469990",
}


def load_regional_supply() -> pl.DataFrame:
    print("Loading baseline regional supply...")
    return (
        pl.scan_parquet(DATA_RAW / "elasticity_evolve.parquet")
        .filter(pl.col("iter_id") == 0)
        .filter(pl.col("strategy") == STRATEGY)
        .group_by(["path_abb", "element", "region", "year"])
        .agg(pl.col("supply").first())
        .group_by(["element", "region", "year"])
        .agg(pl.col("supply").median().alias("supply"))
        .sort(["element", "region", "year"])
        .collect()
    )


def _scale_exponent(ymax: float) -> int:
    if ymax <= 0 or not np.isfinite(ymax):
        return 0
    return int(np.floor(np.log10(ymax)))


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    uplt.rc["grid"] = False

    supply = load_regional_supply()
    elements = sorted(supply["element"].unique().to_list())
    if len(elements) != NROWS * NCOLS:
        raise ValueError(
            f"Expected {NROWS * NCOLS} elements for {NROWS}×{NCOLS}, got {len(elements)}: {elements}"
        )
    print(f"Elements ({len(elements)}): {elements}")

    fig = uplt.figure(
        refwidth=1.65,
        facecolor=(0, 0, 0, 0),
        sharex=1,
        sharey=0,
        spanx=True,
        spany=False,
    )
    axes = fig.subplots(
        nrows=NROWS,
        ncols=NCOLS,
        hspace=0.7,
        wspace=2.0,  # 子图横向间距：越大越疏
    )

    for idx, elem in enumerate(elements):
        row, col = divmod(idx, NCOLS)
        ax = axes[row, col]
        elem_df = supply.filter(pl.col("element") == elem)

        series = []
        for region in REGIONS:
            reg = elem_df.filter(pl.col("region") == region).sort("year")
            if reg.is_empty():
                continue
            years = reg["year"].to_numpy()
            vals = reg["supply"].to_numpy()
            if not np.any(np.isfinite(vals) & (vals > 0)):
                continue
            series.append((region, years, vals))

        ymax_raw = max((float(np.nanmax(v)) for _, _, v in series), default=0.0)
        exp = _scale_exponent(ymax_raw)
        scale = 10.0 ** exp

        for region, years, vals in series:
            ax.plot(
                years,
                vals / scale,
                color=REG_COLORS[region],
                lw=1.1,
                alpha=0.9,
            )

        if ymax_raw > 0:
            ax.set_ylim(0, (ymax_raw / scale) * 1.08)

        ax.format(
            urtitle=elem,
            titleweight="bold",
            titlesize=9,
            xlim=(2025, 2100),
            xticks=[2030, 2060, 2090],
            tickminor=False,
            grid=False,
            yticklabelsize=10,   # y轴数字字号
            xticklabelsize=10,   # x轴年份字号（axes.format 里也会设）
        )
        ax.yaxis.set_major_locator(MaxNLocator(nbins=3, prune="both", integer=False))
        ax.tick_params(axis="y", pad=1.5, length=2)
        for label in ax.get_yticklabels():
            label.set_horizontalalignment("right")

        # Exponent badge in upper-left (y ticks show mantissa only)
        if exp != 0:
            ax.text(
                0.03,
                0.97,
                rf"$\times 10^{{{exp}}}$",
                transform=ax.transAxes,
                ha="left",
                va="top",
                fontsize=7,
                color="0.25",
            )

    axes.format(
        xlabel="Year",
        ylabel="",
        xlabelsize=10,
        xticklabelsize=9,  # 底行 x 轴年份字号
    )

    for row in range(NROWS):
        for col in range(NCOLS):
            ax = axes[row, col]
            ax.set_ylabel("")
            if row < NROWS - 1:
                ax.set_xlabel("")
                ax.tick_params(axis="x", labelbottom=False)
                for label in ax.get_xticklabels():
                    label.set_visible(False)
            else:
                for label in ax.get_xticklabels():
                    label.set_rotation(45)
                    label.set_ha("right")
                    label.set_fontsize(9)  # 底行年份最终字号（会覆盖上面）
            ax.yaxis.set_major_locator(MaxNLocator(nbins=3, prune="both"))

    # One shared y label for the whole figure (vertically centered on the left)
    # 第一个数：越大越靠近子图（往右）；越小越靠纸张左边缘
    fig.text(
        0.015,
        0.55,
        "Regional supply (kt)",
        rotation=90,
        va="center",
        ha="center",
        fontsize=10,
    )

    legend_handles = [
        Line2D([0], [0], color=REG_COLORS[r], lw=1.8, label=REG_LABELS[r])
        for r in REGIONS
    ]
    fig.legend(
        legend_handles,
        [h.get_label() for h in legend_handles],
        loc="bottom",
        ncols=5,
        fontsize=8,
        frame=False,
        center=True,
        handlelength=1.6,
        handletextpad=0.4,
        columnspacing=1.1,
        space=4.5,  # legend 与图的间距：越大越往下
    )

    fig.save(OUT_PATH)
    print(f"Saved {OUT_PATH}")


if __name__ == "__main__":
    main()

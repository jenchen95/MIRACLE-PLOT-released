"""Slide 6: temperature.json, global warming by strategy, from the logic of fig1.py.

    uv run python src/overshoot2026/export_temperature.py [DATA_DIR] [--out DECK_DATA_DIR]

As fig1.py:
- history: dtg_weighted_hist_2000mc (measure "mean") up to the first scenario year;
- Ideal: dtg_weighted_scen_2000mc; strategies: dtg_weighted_cf_2000mc_category*_scen_all;
- category C2, measure "mean"; the line is the median over the C2 pathways of each year;
- everything relative to the 1850-1900 mean of the history.
Band (not drawn by fig1.py's time series): the median over the same pathways of the
OSCAR-ensemble p33 and p66 of each pathway (measures "p33"/"p66" of the same files).
`stats_2100` repeats fig1.py's right panel: percentiles over pathways in 2100 and the
median of the paired (strategy - Ideal) difference.
"""

import numpy as np
import polars as pl

from _shared import CATEGORY, CODE_TO_ID, STRATEGIES, parse_args, r, write_json

MEASURES = ["mean", "p33", "p66"]


def main() -> None:
    args = parse_args(__doc__.splitlines()[0])
    d = args.data_dir

    hist = pl.read_parquet(d / "dtg_weighted_hist_2000mc.parquet").sort("year")
    hist_mean = hist.filter(pl.col("measure") == "mean")
    base_temp = hist_mean.filter(pl.col("year").is_between(1850, 1900))["value"].mean()

    scen = (
        pl.scan_parquet(d / "dtg_weighted_scen_2000mc.parquet")
        .with_columns(pl.col("scen").str.splitn("|", 2).struct.rename_fields(["model", "scenario"]))
        .unnest("scen")
        .with_columns(strategy=pl.lit("ideal"))
        .select("model", "scenario", "category", "strategy", "measure", "year", "value")
    )
    cf = (
        pl.scan_parquet(d / "dtg_weighted_cf_2000mc_category*_scen_all.parquet")
        .with_columns(pl.col("scen").str.splitn("|", 3).struct.rename_fields(["model", "scenario", "strategy"]))
        .unnest("scen")
        .filter(pl.col("strategy").is_in(list(STRATEGIES.values())))
        .select("model", "scenario", "category", "strategy", "measure", "year", "value")
    )
    stats = (
        pl.concat([scen, cf])
        .filter(pl.col("category") == CATEGORY, pl.col("measure").is_in(MEASURES))
        .with_columns(value=pl.col("value") - base_temp)
        .collect()
    )
    first_year = stats["year"].min()

    # history up to the first scenario year (fig1.py: hist_year <= future_year.min())
    h = hist_mean.filter(pl.col("year").is_between(2000, first_year))
    history = {"years": h["year"].to_list(), "median": [r(v, 3) for v in (h["value"] - base_temp)]}

    med = (
        stats.group_by("strategy", "measure", "year")
        .agg(pl.col("value").median(), pl.len().alias("n"))
        .sort("year")
    )
    paths, n_paths = [], {}
    for code in ["ideal", *STRATEGIES.values()]:
        sid = CODE_TO_ID.get(code, code)
        m = med.filter(pl.col("strategy") == code)
        if m.is_empty():
            print(f"no data for {code}; skipped")
            continue
        wide = m.pivot(on="measure", index="year", values="value").sort("year")
        n_paths[sid] = int(m.filter(pl.col("measure") == "mean")["n"].max())
        paths.append({
            "id": sid,
            "years": wide["year"].to_list(),
            "median": [r(v) for v in wide["mean"]],
            "lo": [r(v) for v in wide["p33"]],
            "hi": [r(v) for v in wide["p66"]],
        })

    # fig1.py right panel: 2100 percentiles over pathways, and median paired difference to Ideal
    y2100 = stats.filter(pl.col("year") == 2100, pl.col("measure") == "mean")
    ideal = y2100.filter(pl.col("strategy") == "ideal").select("model", "scenario", pl.col("value").alias("ideal"))
    stats_2100 = {}
    for code in ["ideal", *STRATEGIES.values()]:
        sid = CODE_TO_ID.get(code, code)
        v = y2100.filter(pl.col("strategy") == code)
        if v.is_empty():
            continue
        p05, p25, p50, p75, p95 = np.percentile(v["value"].to_numpy(), [5, 25, 50, 75, 95])
        entry = {"p05": r(p05), "p25": r(p25), "median": r(p50), "p75": r(p75), "p95": r(p95)}
        if code != "ideal":
            diff = v.join(ideal, on=["model", "scenario"], how="inner").select(pl.col("value") - pl.col("ideal"))
            entry["median_diff_to_ideal"] = r(diff.to_series().median())
        stats_2100[sid] = entry

    data = {
        "unit": "°C above 1850–1900",
        "band": "33rd–66th percentile of the climate ensemble (median over pathways)",
        "thresholds": [1.5, 2.0],
        "source": f"MIRACLE-OSCAR, category {CATEGORY}, median over pathways (n={n_paths.get('ideal')}); "
                  "logic of MIRACLE-PLOT fig1.py",
        "category": CATEGORY,
        "n_pathways": n_paths,
        "history": history,
        "paths": paths,
        "stats_2100": stats_2100,
    }
    write_json(args.out, "temperature", data)
    for sid, s in stats_2100.items():
        print(f"  2100 {sid:13s} median {s['median']:.2f} °C  IQR {s['p25']:.2f}-{s['p75']:.2f}"
              + (f"  vs Ideal +{s['median_diff_to_ideal']:.3f}" if "median_diff_to_ideal" in s else ""))


if __name__ == "__main__":
    main()

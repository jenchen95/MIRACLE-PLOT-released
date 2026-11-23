"""Slide 6: temperature.json, global warming by strategy, from the logic of fig1.py.

    uv run python src/overshoot2026/export_temperature.py [DATA_DIR] [--out DECK_DATA_DIR]

As fig1.py:
- history: dtg_weighted_hist_2000mc (measure "mean") up to the first scenario year;
- Ideal: dtg_weighted_scen_2000mc; strategies: dtg_weighted_cf_2000mc_category*_scen_all,
  every strategy of _shared.STRATEGIES, among them Circular + Global Allocation
  (allo_llt_mir_recych_nosub: free trade x circular x no substitution);
- category C2, measure "mean"; the line is the median over the C2 pathways of each year;
- everything relative to the 1850-1900 mean of the history.
Band (not drawn by fig1.py's time series): the median over the same pathways of the
OSCAR-ensemble p33 and p66 of each pathway (measures "p33"/"p66" of the same files).
`stats_2100` repeats fig1.py's right panel: percentiles over pathways in 2100 and the
median of the paired (strategy - Ideal) difference.

`delta_to_ideal` holds the numbers the slide writes next to the legend, per strategy:
- dT: 2100 warming above Ideal, as tab_warming_add_global_stats.py (diff_median): the
  median over the C2 pathways of each pathway's (strategy - Ideal) difference. The
  difference of the two medians (the gap between the drawn lines) is kept as dT_of_medians.
- dCO2: additional CO2 2025-2100 in Gt, from co2_add_reg.parquet as
  tab_emissions_add_global_stats.py: per pathway the sum over regions and years (Mt -> Gt),
  then the median over the C2 pathways (category.csv). That script sums every year of the
  file (2021-2100); the slide's label says 2025-2100, so dCO2 sums 2025-2100 and
  dCO2_2021_2100 keeps the table's number.
"""

import numpy as np
import polars as pl

from _shared import CATEGORY, CODE_TO_ID, STRATEGIES, parse_args, r, write_json

MEASURES = ["mean", "p33", "p66"]
CO2_YEARS = (2025, 2100)


def additional_co2(data_dir, pathways: set) -> dict:
    """Median over `pathways` of each pathway's additional CO2 (Gt), by model strategy code."""
    category = (
        pl.read_csv(data_dir / "category.csv")
        .with_columns(path_abb=pl.concat_str("model", "scenario", separator="|"))
        .filter(pl.col("category") == CATEGORY)
    )
    c2 = set(category["path_abb"].to_list())
    if c2 != pathways:
        print(f"  note: category.csv has {len(c2)} {CATEGORY} pathways, the temperature files {len(pathways)}; "
              f"using the {len(c2 & pathways)} in both")
    co2 = (
        pl.scan_parquet(data_dir / "co2_add_reg.parquet")
        .filter(pl.col("strategy").is_in(list(STRATEGIES.values())), pl.col("path_abb").is_in(sorted(c2 & pathways)))
        .group_by("path_abb", "strategy")
        .agg(
            (pl.col("Emissions").filter(pl.col("year").is_between(*CO2_YEARS)).sum() / 1000).alias("window"),   # Mt -> Gt
            (pl.col("Emissions").sum() / 1000).alias("all_years"),
            pl.col("year").min().alias("y0"), pl.col("year").max().alias("y1"),
        )
        .collect()
    )
    out = {}
    for code in STRATEGIES.values():
        v = co2.filter(pl.col("strategy") == code)
        if v.is_empty():
            continue
        out[code] = {"window": v["window"].median(), "all_years": v["all_years"].median(), "n": v.height,
                     "years": [int(v["y0"].min()), int(v["y1"].max())]}
    return out


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

    # the gap to Ideal the slide writes next to each strategy: 2100 warming and cumulative CO2
    pathways = {f"{m}|{s}" for m, s in ideal.select("model", "scenario").iter_rows()}
    co2 = additional_co2(d, pathways)
    file_years = next(iter(co2.values()))["years"] if co2 else list(CO2_YEARS)
    deltas = {}
    for code in STRATEGIES.values():
        sid = CODE_TO_ID[code]
        if sid not in stats_2100 or code not in co2:
            continue
        deltas[sid] = {
            "dT": stats_2100[sid]["median_diff_to_ideal"],
            "dT_of_medians": r(stats_2100[sid]["median"] - stats_2100["ideal"]["median"]),
            "dCO2": r(co2[code]["window"], 1),
            f"dCO2_{file_years[0]}_{file_years[1]}": r(co2[code]["all_years"], 1),
            "n_pathways": co2[code]["n"],
        }
    delta_to_ideal = {
        "temperature": {"year": 2100, "unit": "°C",
                        "statistic": f"median over the {CATEGORY} pathways of each pathway's 2100 warming minus its Ideal "
                                     "(MIRACLE-PLOT tab_warming_add_global_stats.py, diff_median)"},
        "co2": {"years": list(CO2_YEARS), "unit": "Gt CO₂",
                "statistic": f"median over the {CATEGORY} pathways of each pathway's additional CO₂ (co2_add_reg), summed over "
                             f"the ten regions and {CO2_YEARS[0]}–{CO2_YEARS[1]} (MIRACLE-PLOT tab_emissions_add_global_stats.py, "
                             f"which sums {file_years[0]}–{file_years[1]})"},
        "values": deltas,
    }

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
        "delta_to_ideal": delta_to_ideal,
    }
    write_json(args.out, "temperature", data)
    for sid, s in stats_2100.items():
        print(f"  2100 {sid:13s} median {s['median']:.2f} °C  IQR {s['p25']:.2f}-{s['p75']:.2f}"
              + (f"  vs Ideal +{s['median_diff_to_ideal']:.3f}" if "median_diff_to_ideal" in s else ""))
    for sid, v in deltas.items():
        print(f"  delta {sid:13s} dT +{v['dT']:.3f} °C (of medians +{v['dT_of_medians']:.3f})  "
              f"dCO2 {CO2_YEARS[0]}-{CO2_YEARS[1]} +{v['dCO2']:.1f} Gt  ({file_years[0]}-{file_years[1]}: "
              f"+{v[f'dCO2_{file_years[0]}_{file_years[1]}']:.1f} Gt, n={v['n_pathways']})")


if __name__ == "__main__":
    main()

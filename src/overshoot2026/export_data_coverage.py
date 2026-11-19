"""Slide 4: data_coverage.json, what the MIRACLE-Stats dataset holds (sources, commodities, stages, coverage).

    uv run python src/overshoot2026/export_data_coverage.py [MINERALS_DUCKDB] [--out DECK_DATA_DIR]

Reads MIRACLE-Stats clean/minerals.duckdb. The database is never opened in place: it is
copied to a temporary folder and the copy is opened read-only with the duckdb CLI.

Only the open-licence sources are shown (seeds.ref_source_licences = 'open'); the
licensed and private tables of the database stay off the slide. Per commodity and stage:
- mine and refined/smelter production: production_mine, production_refined (2000 on);
- trade: CEPII BACI bilateral flows (v_trade_by_stage) and BGS imports/exports (trade_bgs);
- reserves: USGS MCS reserves (reserves);
the countries with a value, the years and the sources. `reliability` is the cross-source
check of the earlier slide 4 (table validation: check_type 'cross_source', level
'country', independent_check; a pair is flagged above a 10% gap and kept).
"""

import argparse
import csv
import io
import shutil
import statistics
import subprocess
import tempfile
from pathlib import Path

from _shared import DEFAULT_OUT_DIR, r, write_json

DEFAULT_DB = Path("/Users/huangc/Code/MIRACLE-Stats/clean/minerals.duckdb")

STAGES = [
    ("mine", "Mine production"),
    ("refined", "Refined / smelter production"),
    ("trade", "Trade"),
    ("reserves", "Reserves"),
]
SOURCES = [  # (source id in the database, label, short label)
    ("bgs_wms", "BGS World Mineral Statistics", "BGS WMS"),
    ("usgs_mcs", "USGS Mineral Commodity Summaries", "USGS MCS"),
    ("usgs_myb", "USGS Minerals Yearbook", "USGS Minerals Yearbook"),
    ("wmd", "World Mining Data", "World Mining Data"),
    ("baci", "CEPII BACI (HS6 trade)", "CEPII BACI"),
]
GROUPS = [  # display order of the commodities
    ("Battery", ["lithium", "cobalt", "nickel", "manganese", "graphite", "graphite_synthetic"]),
    ("Magnets, electronics, alloys", ["rare_earths", "gallium", "germanium", "tungsten", "antimony"]),
    ("Base metals", ["copper", "aluminium", "zinc", "lead", "tin"]),
    ("Reserves only", ["molybdenum", "niobium", "titanium", "vanadium"]),
]

CELLS = """
with x as (
  select commodity_id, 'mine' as stage, source, iso3, year from production_mine where value_t is not null
  union all select commodity_id, 'refined', source, iso3, year from production_refined where value_t is not null
  union all select commodity_id, 'reserves', source, iso3, reference_year from reserves where value_t is not null
  union all select commodity_id, 'trade', 'baci', exporter_iso3, year from v_trade_by_stage
  union all select commodity_id, 'trade', 'baci', importer_iso3, year from v_trade_by_stage
  union all select commodity_id, 'trade', 'bgs_wms', iso3, year from trade_bgs where value_original is not null
)
select commodity_id, stage, coalesce(source, 'all') as source, countries, y0, y1 from (
select commodity_id, stage, source, count(distinct iso3) as countries, min(year) as y0, max(year) as y1
from x where iso3 is not null group by grouping sets ((commodity_id, stage, source), (commodity_id, stage))
) order by all
"""
COUNTRIES = """
select
  (select count(distinct iso3) from (select iso3 from production_mine where value_t is not null
     union select iso3 from production_refined where value_t is not null)) as production_countries,
  (select count(distinct iso3) from (select exporter_iso3 as iso3 from trade union select importer_iso3 from trade)) as trade_economies,
  (select count(*) from trade) as trade_flows,
  (select min(year) from trade) as trade_y0, (select max(year) from trade) as trade_y1,
  (select count(*) from trade_bgs) as trade_bgs_rows,
  (select min(year) from production_history) as history_y0,
  (select count(*) from production_mine) + (select count(*) from production_refined) as production_rows
"""
LICENCES = "select source, licence from seeds.ref_source_licences"
NAMES = "select commodity_id, name from commodities"
RELIABILITY = """
select abs_rel_diff, flagged from validation
where check_type = 'cross_source' and level = 'country' and independent_check
"""


def run(db: Path, sql: str) -> list[dict]:
    out = subprocess.run(["duckdb", "-readonly", "-no-agent", "-csv", str(db), "-c", sql],
                         check=True, capture_output=True, text=True).stdout
    return list(csv.DictReader(io.StringIO(out)))


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("db", nargs="?", type=Path, default=DEFAULT_DB, help=f"minerals.duckdb (default: {DEFAULT_DB})")
    p.add_argument("--out", type=Path, default=DEFAULT_OUT_DIR)
    args = p.parse_args()
    if not args.db.is_file():
        p.error(f"no such database: {args.db}")

    with tempfile.TemporaryDirectory() as tmp:
        copy = Path(tmp) / args.db.name
        shutil.copy2(args.db, copy)
        cells = run(copy, CELLS)
        totals = run(copy, COUNTRIES)[0]
        licence = {row["source"]: row["licence"] for row in run(copy, LICENCES)}
        names = {row["commodity_id"]: row["name"] for row in run(copy, NAMES)}
        checks = run(copy, RELIABILITY)

    shown = [s for s, _, _ in SOURCES if licence.get(s) == "open"]
    union = {(c["commodity_id"], c["stage"]): int(c["countries"]) for c in cells if c["source"] == "all"}
    cells = [c for c in cells if c["source"] in shown]
    if len({c["source"] for c in cells}) != len(shown):
        raise SystemExit("a source shown has no rows")
    ids = [c for _, g in GROUPS for c in g]
    missing = sorted({c["commodity_id"] for c in cells} - set(ids))
    if missing:
        raise SystemExit(f"commodities without a group: {missing}")

    commodities = []
    for group, members in GROUPS:
        for cid in members:
            row = {"id": cid, "label": names.get(cid, cid), "group": group, "cells": {}}
            for st, _ in STAGES:
                cc = [c for c in cells if c["commodity_id"] == cid and c["stage"] == st]
                if not cc:
                    continue
                row["cells"][st] = {
                    "countries": union[(cid, st)],
                    "sources": [s for s in shown if any(c["source"] == s for c in cc)],
                    "years": [min(int(c["y0"]) for c in cc), max(int(c["y1"]) for c in cc)],
                    "countries_by_source": {c["source"]: int(c["countries"]) for c in cc},
                }
            if row["cells"]:
                commodities.append(row)

    # sources x stages: years each source covers in each stage (any commodity)
    sources = []
    for sid, label, short in SOURCES:
        if sid not in shown:
            continue
        st = {}
        for stage, _ in STAGES:
            cc = [c for c in cells if c["source"] == sid and c["stage"] == stage]
            if cc:
                st[stage] = [min(int(c["y0"]) for c in cc), max(int(c["y1"]) for c in cc)]
        sources.append({"id": sid, "label": label, "short": short, "stages": st,
                        "commodities": len({c["commodity_id"] for c in cells if c["source"] == sid})})

    diffs = [float(c["abs_rel_diff"]) for c in checks]
    flagged = sum(c["flagged"] == "true" for c in checks)
    prod_years = [v for c in commodities for st in ("mine", "refined") if st in c["cells"] for v in c["cells"][st]["years"]]
    data = {
        "source": "MIRACLE-Stats clean/minerals.duckdb (open-licence sources only)",
        "stages": [{"id": s, "label": l} for s, l in STAGES],
        "sources": sources,
        "commodities": commodities,
        "totals": {
            "commodities": len(commodities),
            "production_countries": int(totals["production_countries"]),
            "production_years": [min(prod_years), max(prod_years)],
            "production_history_from": int(totals["history_y0"]),
            "trade_flows": int(totals["trade_flows"]),
            "trade_years": [int(totals["trade_y0"]), int(totals["trade_y1"])],
            "trade_economies": int(totals["trade_economies"]),
            "trade_bgs_rows": int(totals["trade_bgs_rows"]),
        },
        "reliability": {
            "pairs": len(checks),
            "median_gap_pct": r(statistics.median(diffs) * 100, 2),
            "share_within_10pct": r(1 - flagged / len(checks), 3),
            "flagged": flagged,
            "rule": "pairs more than 10% apart are flagged and kept, never deleted",
        },
    }
    write_json(args.out, "data_coverage", data)
    t = data["totals"]
    print(f"  {t['commodities']} commodities, {t['production_countries']} producing countries "
          f"{t['production_years'][0]}-{t['production_years'][1]} (history from {t['production_history_from']}), "
          f"{t['trade_flows']:,} BACI flows {t['trade_years'][0]}-{t['trade_years'][1]}, {t['trade_economies']} economies")
    for s in sources:
        print(f"  {s['short']:10s} {s['commodities']:2d} commodities  {s['stages']}")
    rl = data["reliability"]
    print(f"  reliability: {rl['pairs']:,} pairs, median gap {rl['median_gap_pct']}%, "
          f"{rl['share_within_10pct'] * 100:.0f}% within 10%, {rl['flagged']:,} flagged")


if __name__ == "__main__":
    main()

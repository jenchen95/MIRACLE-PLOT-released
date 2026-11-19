#!/usr/bin/env bash
# Write the chart JSONs of the overshoot-2026 deck (slides 4 and 6-8) and rebuild its bundle.
#
#   src/overshoot2026/run_all.sh [DATA_DIR] [DECK_DATA_DIR] [STATS_DB]
#
# DATA_DIR: the MIRACLE-OSCAR outputs (read-only), default below.
# DECK_DATA_DIR: miracle-overshoot-assets/data of the deck, default below.
# STATS_DB: MIRACLE-Stats clean/minerals.duckdb (read from a temporary copy), default below.
set -euo pipefail

DATA_DIR="${1:-/Users/huangc/Code/MIRACLE-OSCAR-released/data/data_raw/miracle_oscar}"
OUT_DIR="${2:-/Users/huangc/Code/miracle-overshoot-talk/miracle-overshoot-assets/data}"
STATS_DB="${3:-/Users/huangc/Code/MIRACLE-Stats/clean/minerals.duckdb}"
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$HERE/../.."   # repo root, so uv uses the repo's environment

echo "== data_coverage"
uv run python src/overshoot2026/export_data_coverage.py "$STATS_DB" --out "$OUT_DIR"
for chart in temperature deployment_gap regional supply_expansion; do
    echo "== $chart"
    uv run python "src/overshoot2026/export_${chart}.py" "$DATA_DIR" --out "$OUT_DIR"
done

echo "== bundle"
python3 "$OUT_DIR/build_bundle.py"

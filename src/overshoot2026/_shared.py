"""Shared settings of the overshoot-2026 deck exports (slides 6-8).

Each export_<chart>.py reads the MIRACLE-OSCAR outputs (read-only) and writes one
JSON into the deck's data folder; the schemas are in that folder's README.md.
"""

import argparse
import json
import math
from pathlib import Path

DEFAULT_DATA_DIR = Path("/Users/huangc/Code/MIRACLE-OSCAR-released/data/data_raw/miracle_oscar")
DEFAULT_OUT_DIR = Path("/Users/huangc/Code/miracle-overshoot-talk/miracle-overshoot-assets/data")

# Deck strategy id -> model strategy code (fig1.py / fig2.py / fig3.py / fig4.py)
STRATEGIES = {
    "bau": "bau_bau_bau_recycl_nosub",
    "circular": "bau_llt_mir_recych_nosub",
    "substitution": "bau_bau_bau_recycl_sub",
    "allocation": "allo_bau_bau_recycl_nosub",
    "allin": "allo_llt_mir_recych_sub",
}
CODE_TO_ID = {code: sid for sid, code in STRATEGIES.items()}

# Category shown on slides 6 and 7 (fig1.py target_category)
CATEGORY = "C2"


def parse_args(description: str) -> argparse.Namespace:
    p = argparse.ArgumentParser(description=description)
    p.add_argument("data_dir", nargs="?", type=Path, default=DEFAULT_DATA_DIR,
                   help=f"folder with the MIRACLE-OSCAR outputs (default: {DEFAULT_DATA_DIR})")
    p.add_argument("--out", type=Path, default=DEFAULT_OUT_DIR,
                   help=f"deck data folder to write the JSON into (default: {DEFAULT_OUT_DIR})")
    args = p.parse_args()
    if not args.data_dir.is_dir():
        p.error(f"no such data folder: {args.data_dir}")
    if not args.out.is_dir():
        p.error(f"no such output folder: {args.out}")
    return args


def r(v, nd=3):
    """Round for JSON; NaN/None -> None."""
    if v is None or (isinstance(v, float) and math.isnan(v)):
        return None
    return round(float(v), nd)


def write_json(out_dir: Path, name: str, data: dict) -> Path:
    f = out_dir / f"{name}.json"
    f.write_text(json.dumps(data, indent=1, ensure_ascii=False) + "\n", encoding="utf-8")
    print(f"wrote {f}")
    return f

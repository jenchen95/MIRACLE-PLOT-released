# Critical minerals — figure code

Plotting scripts for the manuscript figures (main text and SI).

## Setup

```bash
uv sync
```

Requires Python 3.11 (see `pyproject.toml`). Run scripts with `uv run`, not bare `python`.

## Figures

| Figure | Script | Notes |
|--------|--------|-------|
| Fig. 1 | `src/critical_minerals/fig1.py` | |
| Fig. 2 | `src/critical_minerals/fig2.py` | Same script as Figs. S6–S15 |
| Fig. 3 | `src/critical_minerals/fig3.py` | Same script as Figs. S20–S21 |
| Fig. 4 | `src/critical_minerals/fig4.py` | |
| Fig. S1 | `src/critical_minerals/fig_s1.py` | |
| Fig. S2 | `src/critical_minerals/fig_s2.py` | |
| Figs. S3–S4 | `src/critical_minerals/fig_s3_s4.py` | |
| Fig. S5 | `src/critical_minerals/fig_s5.py` | |
| Figs. S6–S15 | `src/critical_minerals/fig2.py` | Same as Fig. 2 (category / regional variants) |
| Fig. S16 | `src/critical_minerals/fig_s16.py` | |
| Fig. S17 | `src/critical_minerals/fig_s17.py` | |
| Fig. S18 | `src/critical_minerals/fig_s18.py` | |
| Fig. S19 | — | No dedicated plotting script |
| Figs. S20–S21 | `src/critical_minerals/fig3.py` | Same as Fig. 3 (SI panels / roses) |
| Fig. S22 | `src/critical_minerals/fig_s22.py` | |

Example:

```bash
uv run src/critical_minerals/fig1.py
uv run src/critical_minerals/fig2.py --scale both
```

Outputs are written under `testpic/` (gitignored).

## Supporting scripts

- `src/critical_minerals/tab_*.py` — SI table / summary exports to `data/data_output/`
- `src/critical_minerals/funcs.py` — shared helpers
- `src/critical_minerals/temp_diff_by_reg.py` — builds intermediate temperature parquet inputs

## Data

Raw model outputs live under `data/data_raw/` (not included in this repository). Place the corresponding parquet/CSV files locally before running the figure scripts.

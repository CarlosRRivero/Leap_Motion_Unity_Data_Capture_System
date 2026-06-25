# Analysis pipeline

Python scripts that reproduce every figure and machine-learning result reported in the manuscript. The scripts read the dataset from `../Users/` (downloaded separately from Zenodo — see the top-level [`README.md`](../README.md)) and write their outputs to `../Plots/` (created on first run).

## Prerequisites

- Python ≥ 3.9
- Dependencies from the top-level `requirements.txt` (`pip install -r ../requirements.txt`)
- The dataset extracted into `../Users/` (download from <https://doi.org/10.5281/zenodo.19892935>)

### Data inputs

The scripts read from the project root (the parent of `code/`). Besides the bulk
recordings in `Users/`, the following companion files — provided with the dataset —
must sit at the project root:

- `Users/` — per-participant recordings (`dataCompilation*.csv`, `*_Rock_Status.csv`, `fuel*.csv`).
- `SUS_Stats_Total.csv`, `User_Stats_Total.csv` — aggregate tables used by `plot_sus.py` and `plot_demographics.py`.
- `users.json`, `fuel_locations.json`, and the per-participant `exercise*.json` trajectory files — participant age/sex metadata and the plane-trajectory data used by `plot_showcase.py` (the trajectory panels, Figures 13–14).

If a companion file is absent the corresponding script skips that figure (it prints a
warning rather than failing), so the remaining figures still reproduce.

## Quick start

```bash
# from the repository root, after placing the Zenodo dataset in ./Users/
cd code
python plot_groups_sturges.py   # Figure 7   (groups_gender_distribution.png)
python plot_sus.py              # Figure 8   (sus_response_distribution.png, sus_scores.png)
python plot_showcase.py         # Figures 9–14 (speed, grip and trajectory showcases)
python create_ml_plots.py       # Figures 15–16 (LOOCV training takes a few minutes)
```

These four scripts have a `__main__` entry point and regenerate every manuscript
figure (7–16) into `../Plots/`. The remaining `plot_*.py` modules (`plot_speed.py`,
`plot_grip.py`, `plot_position.py`, `plot_orientation.py`, `plot_grip_freq.py`) expose a
`run()` function that is invoked by the `generate_plots_v2.py` batch driver rather than
run directly.

All scripts expect to be launched from this directory (`code/`). They derive every other path from `__file__`, so no environment variables are required.

## Publication-quality figures

The manuscript figures are produced at an enlarged font size and 300 dpi so the
embedded text stays legible at the journal's column width. `regenerate_hq_figures.py`
runs the standard plotting scripts unchanged but, via lightweight matplotlib
monkeypatches, scales every font by a configurable factor, forces 300 dpi, and saves
with a tight bounding box (data-table fonts are capped so the cells do not overflow):

```bash
cd code
python regenerate_hq_figures.py 1.4 plot_groups_sturges.py plot_sus.py \
       plot_showcase.py create_ml_plots.py
```

The first argument is the font scale factor (default `1.4`). This reproduces
Figures 7–16 with the publication-grade typography used in the manuscript.

## Script reference

| Script                          | Reproduces |
|---------------------------------|------------|
| `analyze_exercises.py`          | Quality filtering and descriptive statistics (Table 1 supporting numbers). |
| `plot_showcase.py`              | Representative-participant showcases — speed, grip and trajectory (**Figures 9–14**). Has `__main__`. |
| `plot_groups_sturges.py`        | Cohort gender/age distribution (**Figure 7**, `groups_gender_distribution.png`) plus alternative bin stratifications. Has `__main__`. |
| `plot_sus.py`                   | SUS response distribution and overall score (**Figure 8**). Has `__main__`. |
| `create_ml_plots.py`            | LOOCV training of SVM, RF, and Grouped RF; confusion matrices and Gini importance (**Figures 15–16**). Has `__main__`. |
| `plot_speed.py`                 | Per-exercise-group hand-speed box plots. `run()` helper, invoked by `generate_plots_v2.py`. |
| `plot_grip.py`                  | Per-exercise-group grip box plots. `run()` helper, invoked by `generate_plots_v2.py`. |
| `plot_grip_freq.py`             | Grip-strength frequency-domain summaries. `run()` helper. |
| `plot_position.py`              | Palm-position statistics (per-exercise). `run()` helper. |
| `plot_orientation.py`           | Palm-normal orientation statistics. `run()` helper. |
| `plot_demographics.py`          | Standalone demographic summaries (age/gender box & pie). Note: the manuscript's Figure 7 is produced by `plot_groups_sturges.py`. |
| `regenerate_hq_figures.py`      | Re-runs the four figure scripts with enlarged fonts and 300 dpi (publication-quality Figures 7–16). |
| `create_pilot_study_plots.py`   | Pilot study plots (IWINAC 2022). |
| `generate_user_plots.py`        | Per-participant kinematic dashboards (one PNG per user, per exercise). |
| `generate_plots_v2.py`          | Batch driver for the v2 plot set used on the MotionInsightHub web platform. |
| `export_user_data_json.py`      | Convert per-participant CSVs into JSON for the web visualisation platform. |
| `export_v1_user_data.py`        | Same as above for the v1 (pilot) participants. |
| `ensure_hands_consistency.py`   | Sanity check on bilateral column presence and frame alignment. |
| `translate_user_stats.py`       | Helper for cross-language stats tables in the manuscript. |
| `compile_and_fix.py`            | LaTeX manuscript build helper (not required to reproduce the figures). |
| `plot_utils.py`                 | Shared CSV-loading and plotting primitives — imported by the other scripts. |

## Output

Generated artefacts live in `../Plots/<topic>/` (created on demand). To reproduce the manuscript figures in the same naming scheme used by the LaTeX source, run the scripts above in order — file names match those referenced by `\includegraphics` in the paper. 

All the graphics can be checked in https://motioninsighthub.web.app

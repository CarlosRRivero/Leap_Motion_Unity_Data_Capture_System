# Analysis pipeline

Python scripts that reproduce every figure and machine-learning result reported in the manuscript. The scripts read the dataset from `../Users/` (the unzipped contents of `Users.zip` in the repository root) and write their outputs to `../Plots/` (created on first run).

## Prerequisites

- Python ≥ 3.9
- Dependencies from the top-level `requirements.txt` (`pip install -r ../requirements.txt`)

## Quick start

```bash
# from the repository root
unzip Users.zip
cd code
python plot_demographics.py   # Figure 7
python plot_sus.py            # Figure 8
python plot_speed.py          # Figures 9–10
python plot_grip.py           # Figures 11–12
python plot_showcase.py       # Figures 13–14
python create_ml_plots.py     # Figures 15–16 (LOOCV training takes a few minutes)
```

All scripts expect to be launched from this directory (`code/`). They derive every other path from `__file__`, so no environment variables are required.

## Script reference

| Script                          | Reproduces |
|---------------------------------|------------|
| `analyze_exercises.py`          | Quality filtering and descriptive statistics (Table 1 supporting numbers). |
| `plot_speed.py`                 | Bilateral hand-speed traces (Figures 9–10). |
| `plot_grip.py`                  | Grip-strength time series and summary panels (Figures 11–12). |
| `plot_grip_freq.py`             | Grip-strength frequency-domain summaries. |
| `plot_position.py`              | Palm-position statistics (per-exercise). |
| `plot_orientation.py`           | Palm-normal orientation statistics. |
| `plot_showcase.py`              | Per-participant trajectory showcase (Figures 13–14). |
| `plot_demographics.py`          | Demographic distribution by age, gender, and cohort (Figure 7). |
| `plot_groups_sturges.py`        | Alternative stratifications (Sturges, Scott, Freedman–Diaconis). |
| `plot_sus.py`                   | SUS response distribution and overall score (Figure 8). |
| `create_ml_plots.py`            | LOOCV training of SVM, RF, and Grouped RF; confusion matrices and Gini importance (Figures 15–16). |
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

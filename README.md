# Leap Motion Unity Data Capture System

**Companion code and dataset for the manuscript:**

> *A Technical Framework for Detecting Dyskinesia in Parkinson's Disease Using Motion Tracking and Serious Games*
>
> Carlos Rodrigo-Rivero, Nikola Hristov-Kalamov, Raúl Fernández-Ruiz, Carlos Garre del Olmo, Agustín Álvarez-Marquina, Paulo Peixoto, Daniel Palacios-Alonso (corresponding author).
>
> Submitted to *PeerJ Computer Science* (2026).

Project website: **https://motioninsighthub.web.app/**
Persistent data DOI: **https://doi.org/10.5281/zenodo.19892935**

---

## Description

This repository contains the anonymized motion-capture dataset and the analysis code used to evaluate the proposed framework for the objective, contactless assessment of dyskinesia and motor dysfunction in Parkinson's Disease (PD). The system integrates a consumer-grade Leap Motion Controller (LMC v1.0) with three custom-designed Unity serious games (exergames) targeting resting tremor, bradykinesia, and visuomotor coordination under fatigue. Bilateral hand kinematics are recorded at frame-level granularity and processed into 62 kinematic, temporal, and spectral features that feed supervised classifiers (Support Vector Machines, Random Forests, and Grouped Random Forests) evaluated under Leave-One-Out cross-validation.

The data and code allow independent researchers to:

- reproduce the figures and machine-learning results reported in the paper,
- re-use the kinematic recordings of 46 participants (20 PD + 26 normative) for further analyses, and
- adapt the analysis pipeline (or the underlying Unity exergames) to other neuromotor research questions.

---

## Repository contents

```
.
├── README.md                # This file
├── LICENSE                  # CC-BY-4.0 license
├── CITATION.cff             # Machine-readable citation metadata
├── requirements.txt         # Python dependencies for the analysis pipeline
└── code/                    # Python analysis and visualization pipeline
    ├── analyze_exercises.py
    ├── create_ml_plots.py
    ├── plot_speed.py
    ├── plot_grip.py
    ├── plot_position.py
    ├── plot_orientation.py
    ├── plot_showcase.py
    ├── plot_sus.py
    ├── plot_demographics.py
    ├── plot_utils.py
    └── ...                  # See code/README.md for the full list
```

> **Unity exergames.** The Unity project that records the data (Unity 2022.3.10f1, Leap Motion SDK Orion 4.1) is large and is not stored in this repository. A compiled release of the data-acquisition application, together with build instructions, is available from the corresponding author on request and at the project website.

---

## Dataset information

### Source

Bilateral upper-limb kinematic data captured with the **Leap Motion Controller v1.0** (Ultraleap, formerly Leap Motion Inc.) at up to 120 Hz, with an effective application-side rate of ~44 fps. Data were collected at the *Asociación de Parkinson de Alcorcón/Leganés* (APARKAM) and at *Residencia Pacífico Los Nogales* (Madrid).

### Participants (n = 46)

| Group                | n  | Mean age (years) | Gender (F / M) |
|----------------------|----|------------------|----------------|
| Normative — study    | 24 | 75.6             | 65% / 35%      |
| Normative — reference| 2  | 27, 28           | 50% / 50%      |
| Non-normative (PD)   | 20 | 71.0             | 30% / 70%      |
| **Total**            | **46** | —            | —              |

Two younger normative reference users were excluded from the kinematic and machine-learning analyses, yielding `n = 44` for classification.

Disease severity of the non-normative group (Hoehn & Yahr scale):

`S_PD = {1.0: 6, 1.5: 3, 2.0: 8, 3.0: 4}`

### File layout (zenodo dataset)

Zenodo Dataset must have the following file structure:

```
Users/
├── fuelSpaceLocation.csv          # Bonus-item positions, space scene  (game asset)
├── fuelSkyLocation.csv            # Bonus-item positions, sky scene    (game asset)
├── fuelWaterLocation.csv          # Bonus-item positions, ocean scene  (game asset)
└── User_<group>_<timestamp>_ID_<id>/
    ├── dataCompilation1.csv       # Baseline resting tremor (both hands, 10 s)
    ├── dataCompilation2.csv       # Bilateral grip open/close (10 s)
    ├── dataCompilation3.csv       # Vertical obstacle avoidance — scene 1 (30 s)
    ├── dataCompilation4.csv       # Vertical obstacle avoidance — scene 2 (30 s)
    ├── dataCompilation5.csv       # Vertical obstacle avoidance — scene 3 (30 s)
    ├── dataCompilation{N}_Rock_Status.csv   # Per-frame obstacle status for exercises 3–5
    └── *.meta                     # Unity asset metadata (safe to ignore)
```

`<group>` is either `normative` or `non_normative`. Each `dataCompilationN.csv` is a frame-level table containing palm positions, palm-normal orientation, hand velocities, fingertip coordinates, grab strength, and device-side quality flags (`Is_lighting_bad`, `Is_smudged`, `Is_low_resource`) plus two timestamps (application clock and LMC hardware microsecond clock). The `.meta` files are auto-generated by the Unity asset pipeline that produced the dataset and may be ignored by analysis code. See Table 1 of the manuscript for the complete column dictionary (26 used variables, 62 derived features).

### Privacy and ethics

- The protocol was approved by the **Ethics Committee of Universidad Politécnica de Madrid (UPM)**, ID `CDHYLCDVESDNPMGSDWYDSPTYDDUHPPDLS(-RMO-DATOS-20250226)`.
- All participants signed a written informed-consent form before data acquisition.
- No personally identifiable information (PII), audio, video, or image of any participant is included in the dataset.
- Participant identifiers are non-reversible pseudonyms.

---

## Code information

The `code/` folder contains the Python pipeline that ingests the raw CSV recordings in Zenodo (must be downloaded locally in order to use them) and produces the kinematic plots, demographic figures, usability summaries, and machine-learning results reported in the manuscript.

| Script                          | Purpose                                                      |
|---------------------------------|--------------------------------------------------------------|
| `analyze_exercises.py`          | Per-exercise descriptive statistics and quality filtering.   |
| `plot_speed.py`                 | Hand-speed traces (Figures 9–10).                            |
| `plot_grip.py`, `plot_grip_freq.py` | Grip strength and grip-frequency plots (Figures 11–12). |
| `plot_position.py`              | Palm-position summaries.                                     |
| `plot_orientation.py`           | Palm-normal orientation summaries.                           |
| `plot_showcase.py`              | Per-participant showcase trajectories (Figures 13–14).       |
| `plot_demographics.py`, `plot_groups_sturges.py` | Demographic distributions (Figure 7).       |
| `plot_sus.py`                   | System Usability Scale results (Figure 8).                   |
| `create_ml_plots.py`            | LOOCV training of SVM / Random Forest / Grouped RF, confusion matrices, and feature importance (Figures 15–16). |
| `create_pilot_study_plots.py`   | Pilot-study plots (IWINAC 2022).                             |
| `generate_user_plots.py`, `generate_plots_v2.py` | Batch driver scripts.                       |
| `export_user_data_json.py`, `export_v1_user_data.py` | JSON exporters for the web visualisation platform. |
| `ensure_hands_consistency.py`   | Consistency checks across bilateral CSV columns.             |
| `compile_and_fix.py`            | Manuscript build helper (LaTeX).                             |
| `translate_user_stats.py`       | Stats translation utility.                                   |
| `plot_utils.py`                 | Shared plotting and CSV-loading utilities.                   |

---

## Requirements

- **Python** ≥ 3.9 (tested on 3.9 and 3.11)
- The Python libraries listed in [`requirements.txt`](requirements.txt):
  - `numpy`, `pandas`, `scipy`
  - `matplotlib`, `pillow`
  - `scikit-learn`
  - `openpyxl`

For data acquisition (only if rebuilding the exergames):

- **Unity** 2022.3.10f1 (Universal Render Pipeline)
- **Leap Motion SDK** Orion 4.1 with Unity Modules 4.5.1
- **Leap Motion Controller v1.0** hardware
- **Windows 10/11** (the acquisition application was tested on Windows 11 Home build 22631)

---

## Usage instructions

### 1. Get the data and code

```bash
git clone https://github.com/CarlosRRivero/Leap_Motion_Unity_Data_Capture_System.git
cd Leap_Motion_Unity_Data_Capture_System
# extracts Users from Zenodo and copy them such as Users/<participant_id>/exercise_*.csv
```

### 2. Install the Python dependencies

```bash
python -m venv .venv
# Linux / macOS
source .venv/bin/activate
# Windows (PowerShell)
.venv\Scripts\Activate.ps1

pip install -r requirements.txt
```

### 3. Reproduce the figures and machine-learning results

```bash
cd code

# Demographics (Figure 7)
python plot_demographics.py

# System Usability Scale (Figure 8)
python plot_sus.py

# Speed plots (Figures 9–10)
python plot_speed.py

# Grip plots (Figures 11–12)
python plot_grip.py

# Trajectory showcase (Figures 13–14)
python plot_showcase.py

# Machine learning — confusion matrices and feature importance (Figures 15–16)
python create_ml_plots.py
```

Each script reads from `../Users/` (the zenodo dataset) and writes its outputs to a `Plots/` folder created alongside the script. Adjust paths in `plot_utils.py` if your dataset is located elsewhere.

---

## Methodology

A succinct summary of the analysis pipeline (full details in the manuscript):

1. **Pre-processing.** Duplicate frames are discarded; missing-hand intervals are preserved with the corresponding columns zeroed (no interpolation); frames flagged by the LMC for poor lighting, smudged optics, or low-resource tracking are excluded.
2. **Feature extraction.** Per-exercise descriptors are derived from raw frame-level kinematics: palm-position statistics, hand-speed magnitudes and components, palm-normal orientation, fingertip positions, and grab strength (plus their means and standard deviations).
3. **Statistical normalisation.** Within every Leave-One-Out cross-validation fold, features are z-score-standardised using *training-fold* statistics only, then applied to the held-out sample. This prevents any form of data leakage.
4. **Classification.** Three models are compared per exercise group: SVM (RBF kernel, C = 1.0), Random Forest (200 trees, default Gini), and Grouped Random Forest (six semantic feature groups). LOOCV is used because it makes maximal use of the limited cohort (`n = 44`).
5. **Interpretability.** Gini-based feature importance is reported per exercise group; confusion matrices are reported per (model, exercise group) pair.

---

## Citation

If you use this dataset or code, please cite:

```bibtex
@article{rodrigorivero2026dyskinesia,
  author  = {Rodrigo-Rivero, Carlos and Hristov-Kalamov, Nikola and
             Fern{\'a}ndez-Ruiz, Ra{\'u}l and Garre del Olmo, Carlos and
             {\'A}lvarez-Marquina, Agust{\'i}n and Peixoto, Paulo and
             Palacios-Alonso, Daniel},
  title   = {A Technical Framework for Detecting Dyskinesia in {P}arkinson's
             Disease Using Motion Tracking and Serious Games},
  journal = {PeerJ Computer Science},
  year    = {2026},
  note    = {Submitted}
}
```

A machine-readable citation file is also provided in [`CITATION.cff`](CITATION.cff).

---

## Funding

This work was supported by the **State Research Agency of Spain** (*Agencia Estatal de Investigación — AEI*) under grant **CarHaVoz: PID2023-152984OB-I00**.

---

## License

This dataset and code are released under the **Creative Commons Attribution 4.0 International (CC-BY-4.0)** license, consistent with PeerJ's open-access policy. See [`LICENSE`](LICENSE) for the full text.

You are free to share and adapt the material for any purpose, including commercially, provided that appropriate attribution is given.

---

## Contributing

Issues and pull requests are welcome. For substantive questions about the dataset, the protocol, or the analysis pipeline, please contact the corresponding author:

**Daniel Palacios-Alonso** — daniel.palacios@urjc.es
*Escuela Técnica Superior de Ingeniería Informática, Universidad Rey Juan Carlos, Móstoles, Madrid, Spain.*

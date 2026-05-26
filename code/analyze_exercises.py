"""
analyze_exercises.py – main analysis script.

Loads all user data from Paper/Users, then calls the individual plot scripts
for each exercise to compare normative vs non-normative users.
Also generates global SUS and demographics plots.

Output structure:
  Paper/Plots/
    sus/
      sus_response_distribution.png
      sus_scores.png
    demographics/
      demographics_age.png
      demographics_gender.png
      demographics_overview.png
      demographics_stages.png
    exercise_1/
      exercise1_speed.png
      exercise1_orientation.png
      exercise1_position_mean.png
      exercise1_position_std.png
      exercise1_grip_fps.png
    exercise_2/ ...
    ...
    exercise_5/ ...

Usage:
  python analyze_exercises.py
"""

import os
import sys
import warnings
warnings.filterwarnings("ignore")

import pandas as pd

# Make sure the Scripts directory is on the path so sibling modules are found
SCRIPTS_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, SCRIPTS_DIR)

import plot_speed
import plot_orientation
import plot_position
import plot_grip
import plot_sus
import plot_demographics

USERS_DIR = os.path.normpath(os.path.join(SCRIPTS_DIR, "..", "Users"))
PLOTS_DIR = os.path.normpath(os.path.join(SCRIPTS_DIR, "..", "Plots"))


# ── Data loading ──────────────────────────────────────────────────────────────

def load_all_data(users_dir):
    """
    Walk every user folder in `users_dir` and load dataCompilation1–5.csv.

    Returns:
        dict  {exercise_num (1-5): {"normative": [(name, df), ...],
                                    "non_normative": [(name, df), ...]}}
    """
    data = {i: {"normative": [], "non_normative": []} for i in range(1, 6)}

    if not os.path.isdir(users_dir):
        print(f"[ERROR] Users directory not found: {users_dir}")
        return data

    for user_name in sorted(os.listdir(users_dir)):
        user_dir = os.path.join(users_dir, user_name)
        if not os.path.isdir(user_dir):
            continue

        if "non_normative" in user_name:
            group = "non_normative"
        elif "normative" in user_name:
            group = "normative"
        else:
            continue  # skip unexpected folders

        for i in range(1, 6):
            path = os.path.join(user_dir, f"dataCompilation{i}.csv")
            if not os.path.exists(path):
                continue
            try:
                df = pd.read_csv(path, sep=";", low_memory=False)
                data[i][group].append((user_name, df))
            except Exception as exc:
                print(f"  [ERROR] {user_name}/dataCompilation{i}.csv: {exc}")

    return data


def print_summary(data):
    print("\n+---------------------------------------------+")
    print("|              Data Summary                   |")
    print("+--------------+--------------+---------------+")
    print("|  Exercise    |  Normative   | Non-normative |")
    print("+--------------+--------------+---------------+")
    for ex in range(1, 6):
        n  = len(data[ex]["normative"])
        nn = len(data[ex]["non_normative"])
        print(f"|  Exercise {ex}  |     {n:>3}      |      {nn:>3}      |")
    print("+--------------+--------------+---------------+")


# ── Per-exercise plotting ────────────────────────────────────────────────────

PLOT_STEPS = [
    ("Speed",           plot_speed,       "speed"),
    ("Orientation",     plot_orientation, "orientation"),
    ("Position",        plot_position,    "position"),
    ("Grip & FPS",      plot_grip,        "grip_fps"),
]


def process_exercise(ex_data, ex_num, ex_out_dir):
    for label, module, _ in PLOT_STEPS:
        try:
            module.run(ex_data, ex_num, ex_out_dir)
            print(f"    [OK] {label}")
        except Exception as exc:
            print(f"    [ERR] {label}: {exc}")


# ── Entry point ───────────────────────────────────────────────────────────────

def main():
    print(f"Users directory : {USERS_DIR}")
    print(f"Output directory: {PLOTS_DIR}")

    data = load_all_data(USERS_DIR)
    print_summary(data)

    os.makedirs(PLOTS_DIR, exist_ok=True)
    print("\n=== Generating plots ===")

    # ── Global: SUS & Demographics ────────────────────────────────────────────
    print("\n  Global analyses")
    sus_out  = os.path.join(PLOTS_DIR, "sus")
    demo_out = os.path.join(PLOTS_DIR, "demographics")
    os.makedirs(sus_out,  exist_ok=True)
    os.makedirs(demo_out, exist_ok=True)
    try:
        plot_sus.run(sus_out)
        print("    [OK] SUS Statistics")
    except Exception as exc:
        print(f"    [ERR] SUS Statistics: {exc}")
    try:
        plot_demographics.run(demo_out)
        print("    [OK] Demographics")
    except Exception as exc:
        print(f"    [ERR] Demographics: {exc}")

    # ── Per-exercise ──────────────────────────────────────────────────────────
    for ex in range(1, 6):
        ex_data = data[ex]
        n_total = len(ex_data["normative"]) + len(ex_data["non_normative"])

        if n_total == 0:
            print(f"\n  Exercise {ex}: no data found - skipping")
            continue

        n_norm = len(ex_data["normative"])
        n_non  = len(ex_data["non_normative"])
        print(f"\n  Exercise {ex}  ({n_norm} normative, {n_non} non-normative)")

        ex_out = os.path.join(PLOTS_DIR, f"exercise_{ex}")
        os.makedirs(ex_out, exist_ok=True)

        process_exercise(ex_data, ex, ex_out)

    print(f"\nDone.  All plots saved under: {PLOTS_DIR}")


if __name__ == "__main__":
    main()

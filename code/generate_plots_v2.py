"""
generate_plots_v2.py

Regenerates all V2 analysis plots, excluding the two youngest normative
participants:
  - file_id "0"  (age 28, session 2025-08-04)
  - file_id "1"  (age 27, session 2025-08-04)

Steps:
  1. Load exercise data from Paper/Users, skipping the excluded folders.
  2. Run speed / orientation / position / grip plots for each exercise.
  3. Run SUS plot.
  4. Run demographics with the two excluded users removed from the CSV.
  5. Copy all results to web/src/assets/plots/.

Usage:
  python generate_plots_v2.py
"""

import os
import sys
import shutil
import warnings
warnings.filterwarnings("ignore")

import pandas as pd

SCRIPTS_DIR = os.path.dirname(os.path.abspath(__file__))
PAPER_DIR   = os.path.normpath(os.path.join(SCRIPTS_DIR, ".."))
sys.path.insert(0, SCRIPTS_DIR)

import plot_speed
import plot_orientation
import plot_position
import plot_grip
import plot_sus
import plot_demographics

USERS_DIR = os.path.join(PAPER_DIR, "Users")
PLOTS_DIR = os.path.join(PAPER_DIR, "Plots")
WEB_PLOTS = os.path.normpath(
    os.path.join(PAPER_DIR, "MotionInsightHub", "web", "src", "assets", "plots")
)

# ── Exclusion list ─────────────────────────────────────────────────────────────
# Normative file_ids to exclude: the two youngest participants (ages 27 and 28).
# These are from the 2025 session (start_date 2025-08-04).
# The 2026 batch reuses numeric file_ids 1-12, so we only exclude simple
# single-digit IDs that are NOT part of the 2026 folder naming scheme.
EXCLUDE_IDS = {"0", "1"}


def is_excluded(folder_name: str) -> bool:
    """Return True for normative folders belonging to excluded users."""
    if "non_normative" in folder_name or "_ID_" not in folder_name:
        return False
    fid = folder_name.split("_ID_")[-1]
    # 2026-batch folder IDs look like "20262261", "202622615" — never exclude those
    return fid in EXCLUDE_IDS and not fid.startswith("2026")


# ── Data loading ───────────────────────────────────────────────────────────────

def load_all_data():
    """Load exercise CSVs, skipping excluded users."""
    data = {i: {"normative": [], "non_normative": []} for i in range(1, 6)}

    if not os.path.isdir(USERS_DIR):
        print(f"[ERROR] Users directory not found: {USERS_DIR}")
        return data

    for folder in sorted(os.listdir(USERS_DIR)):
        user_dir = os.path.join(USERS_DIR, folder)
        if not os.path.isdir(user_dir):
            continue

        if is_excluded(folder):
            print(f"  [SKIP] {folder}  (excluded — too young)")
            continue

        if "non_normative" in folder:
            group = "non_normative"
        elif "normative" in folder:
            group = "normative"
        else:
            continue

        for ex in range(1, 6):
            path = os.path.join(user_dir, f"dataCompilation{ex}.csv")
            if not os.path.exists(path):
                continue
            try:
                df = pd.read_csv(path, sep=";", low_memory=False)
                data[ex][group].append((folder, df))
            except Exception as e:
                print(f"  [ERROR] {folder}/dataCompilation{ex}.csv: {e}")

    return data


# ── Demographics with exclusion ────────────────────────────────────────────────

def run_demographics_excl(out_dir: str):
    """
    Run the demographics plot but strip the two excluded users from the
    data before classifying, so the stats and charts reflect the final V2
    cohort.
    """
    csv_path = os.path.join(PAPER_DIR, "User_Stats_Total.csv")
    if not os.path.exists(csv_path):
        print("  [WARN] User_Stats_Total.csv not found — skipping demographics")
        return

    orig_classify = plot_demographics.classify

    def classify_patched(df):
        # Remove rows where File_ID is in EXCLUDE_IDS AND the session is from
        # the 2025 batch (Start_Date does NOT begin with "2026").
        excl_mask = (
            df["File_ID"].astype(str).isin(EXCLUDE_IDS)
            & ~df["Start_Date"].astype(str).str.startswith("2026")
        )
        return orig_classify(df[~excl_mask].reset_index(drop=True))

    plot_demographics.classify = classify_patched
    try:
        plot_demographics.run(out_dir)
    finally:
        plot_demographics.classify = orig_classify


# ── Copy to web assets ─────────────────────────────────────────────────────────

COPY_CATS = ["demographics", "sus"] + [f"exercise_{i}" for i in range(1, 6)]


def copy_to_web():
    if not os.path.isdir(WEB_PLOTS):
        print(f"  [WARN] Web plots directory not found: {WEB_PLOTS}")
        return
    for cat in COPY_CATS:
        src = os.path.join(PLOTS_DIR, cat)
        dst = os.path.join(WEB_PLOTS, cat)
        if not os.path.isdir(src):
            continue
        if os.path.exists(dst):
            shutil.rmtree(dst)
        shutil.copytree(src, dst)
        print(f"  [OK] {cat}")


# ── Main ───────────────────────────────────────────────────────────────────────

PLOT_STEPS = [
    ("Speed",       plot_speed),
    ("Orientation", plot_orientation),
    ("Position",    plot_position),
    ("Grip",        plot_grip),
]


def main():
    print("=" * 60)
    print("  V2 Plot Generation")
    print(f"  Excluding normative file_ids: {EXCLUDE_IDS}")
    print("=" * 60)

    data = load_all_data()
    print()
    for ex in range(1, 6):
        n  = len(data[ex]["normative"])
        nn = len(data[ex]["non_normative"])
        print(f"  Exercise {ex}: {n:2d} normative | {nn:2d} non-normative")

    os.makedirs(PLOTS_DIR, exist_ok=True)

    # ── Global analyses ────────────────────────────────────────────────────────
    print("\n-- Global analyses --")
    sus_out  = os.path.join(PLOTS_DIR, "sus")
    demo_out = os.path.join(PLOTS_DIR, "demographics")
    os.makedirs(sus_out,  exist_ok=True)
    os.makedirs(demo_out, exist_ok=True)

    try:
        plot_sus.run(sus_out)
        print("  [OK] SUS")
    except Exception as e:
        print(f"  [ERR] SUS: {e}")

    try:
        run_demographics_excl(demo_out)
        print("  [OK] Demographics")
    except Exception as e:
        print(f"  [ERR] Demographics: {e}")

    # ── Exercise plots ─────────────────────────────────────────────────────────
    print("\n-- Exercise plots --")
    for ex in range(1, 6):
        ex_data = data[ex]
        if not ex_data["normative"] and not ex_data["non_normative"]:
            print(f"\n  Exercise {ex}: no data — skipped")
            continue
        ex_out = os.path.join(PLOTS_DIR, f"exercise_{ex}")
        os.makedirs(ex_out, exist_ok=True)
        n  = len(ex_data["normative"])
        nn = len(ex_data["non_normative"])
        print(f"\n  Exercise {ex}  ({n} normative, {nn} non-normative)")
        for label, module in PLOT_STEPS:
            try:
                module.run(ex_data, ex, ex_out)
                print(f"    [OK] {label}")
            except Exception as e:
                print(f"    [ERR] {label}: {e}")

    # ── Copy to web assets ─────────────────────────────────────────────────────
    print("\n-- Copying to web assets --")
    copy_to_web()

    print(f"\nDone.  Plots saved to: {PLOTS_DIR}")
    print(f"       Web assets:     {WEB_PLOTS}")


if __name__ == "__main__":
    main()

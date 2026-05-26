"""
generate_user_plots.py
Generates per-user per-exercise time-series plots for all variables.
Output: Paper/Plots/users/{file_id}/exercise{n}_{variable}.png
Also copies to web/src/assets/plots/users/{file_id}/
"""

import os, sys, shutil
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# ── Paths ─────────────────────────────────────────────────────────────────────
SCRIPT_DIR  = os.path.dirname(os.path.abspath(__file__))
PAPER_DIR   = os.path.dirname(SCRIPT_DIR)
USERS_DIR   = os.path.join(PAPER_DIR, "Users")
PLOTS_DIR   = os.path.join(PAPER_DIR, "Plots", "users")
WEB_ASSETS  = os.path.join(PAPER_DIR, "MotionInsightHub", "web", "src", "assets", "plots", "users")

MAX_POINTS  = 600   # downsample to keep plots readable

# ── Style ─────────────────────────────────────────────────────────────────────
LEFT_COLOR  = "#42A5F5"   # blue
RIGHT_COLOR = "#EF5350"   # red-orange
SINGLE_COLOR = "#AB47BC"  # purple (single-hand exercises)

plt.rcParams.update({
    "figure.facecolor":  "#0d1340",
    "axes.facecolor":    "#111830",
    "axes.edgecolor":    "#334477",
    "axes.labelcolor":   "#aac4ff",
    "axes.titlecolor":   "white",
    "xtick.color":       "#667aaa",
    "ytick.color":       "#667aaa",
    "grid.color":        "#1e2d5a",
    "grid.linestyle":    "--",
    "grid.alpha":        0.5,
    "text.color":        "white",
    "legend.facecolor":  "#111830",
    "legend.edgecolor":  "#334477",
    "legend.labelcolor": "white",
    "figure.titlesize":  14,
    "axes.titlesize":    11,
    "axes.labelsize":    9,
    "font.size":         9,
})


def read_csv(path):
    """Read exercise CSV, normalise decimal comma→dot, return DataFrame."""
    try:
        df = pd.read_csv(path, sep=";", encoding="utf-8-sig", low_memory=False)
        for col in df.columns:
            if df[col].dtype == object:
                df[col] = df[col].str.replace(",", ".").pipe(pd.to_numeric, errors="coerce")
        return df
    except Exception as e:
        print(f"  [WARN] Could not read {path}: {e}")
        return pd.DataFrame()


def downsample(series, max_pts=MAX_POINTS):
    """Return a uniformly downsampled copy of a 1-D array/Series."""
    arr = np.asarray(series, dtype=float)
    if len(arr) <= max_pts:
        return arr
    idx = np.linspace(0, len(arr) - 1, max_pts, dtype=int)
    return arr[idx]


def dominant_hand(df):
    left  = pd.to_numeric(df.get("Left_hand_speed",  pd.Series(dtype=float)), errors="coerce").abs().sum()
    right = pd.to_numeric(df.get("Right_hand_speed", pd.Series(dtype=float)), errors="coerce").abs().sum()
    return "Left" if left >= right else "Right"


def is_bimanual(df):
    left  = pd.to_numeric(df.get("Left_hand_speed",  pd.Series(dtype=float)), errors="coerce").abs().sum()
    right = pd.to_numeric(df.get("Right_hand_speed", pd.Series(dtype=float)), errors="coerce").abs().sum()
    return left > 0 and right > 0


def plot_col(ax, y, label, color, alpha=1.0):
    x = np.arange(len(y))
    ax.plot(x, y, color=color, linewidth=0.9, alpha=alpha, label=label)
    ax.grid(True)
    ax.set_xlabel("Frame", fontsize=8)


# ── Per-variable plot generators ──────────────────────────────────────────────

def plot_speed(df, out_path, exercise_num):
    bimanual = is_bimanual(df)
    if bimanual:
        panels = [
            ("Left_hand_speed",   "Left Speed",    "mm/s"),
            ("Left_hand_speed_x", "Left Speed X",  "mm/s"),
            ("Left_hand_speed_y", "Left Speed Y",  "mm/s"),
            ("Left_hand_speed_z", "Left Speed Z",  "mm/s"),
            ("Right_hand_speed",   "Right Speed",   "mm/s"),
            ("Right_hand_speed_x", "Right Speed X", "mm/s"),
            ("Right_hand_speed_y", "Right Speed Y", "mm/s"),
            ("Right_hand_speed_z", "Right Speed Z", "mm/s"),
        ]
        colors = [LEFT_COLOR]*4 + [RIGHT_COLOR]*4
    else:
        hand = dominant_hand(df)
        col_pre = hand + "_"
        panels = [
            (col_pre + "hand_speed",   f"{hand} Speed",   "mm/s"),
            (col_pre + "hand_speed_x", f"{hand} Speed X", "mm/s"),
            (col_pre + "hand_speed_y", f"{hand} Speed Y", "mm/s"),
            (col_pre + "hand_speed_z", f"{hand} Speed Z", "mm/s"),
        ]
        colors = [SINGLE_COLOR]*4

    fig, axes = plt.subplots(2, 4, figsize=(16, 6)) if bimanual else plt.subplots(1, 4, figsize=(16, 3.5))
    axes = np.array(axes).flatten()

    for i, (col, label, unit) in enumerate(panels):
        if col in df.columns:
            y = downsample(pd.to_numeric(df[col], errors="coerce").fillna(0))
            plot_col(axes[i], y, label, colors[i])
        axes[i].set_title(label)
        axes[i].set_ylabel(unit)

    for j in range(len(panels), len(axes)):
        axes[j].set_visible(False)

    fig.suptitle(f"Exercise {exercise_num} – Hand Speed", y=1.01, color="white")
    fig.tight_layout()
    fig.savefig(out_path, dpi=100, bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.close(fig)


def plot_orientation(df, out_path, exercise_num):
    bimanual = is_bimanual(df)
    if bimanual:
        panels = [
            ("Left_hand_normal_x", "Left Normal X", ""),
            ("Left_hand_normal_y", "Left Normal Y", ""),
            ("Left_hand_normal_z", "Left Normal Z", ""),
            ("Right_hand_normal_x", "Right Normal X", ""),
            ("Right_hand_normal_y", "Right Normal Y", ""),
            ("Right_hand_normal_z", "Right Normal Z", ""),
        ]
        colors = [LEFT_COLOR]*3 + [RIGHT_COLOR]*3
        ncols = 3
    else:
        hand = dominant_hand(df)
        panels = [
            (f"{hand}_hand_normal_x", f"{hand} Normal X", ""),
            (f"{hand}_hand_normal_y", f"{hand} Normal Y", ""),
            (f"{hand}_hand_normal_z", f"{hand} Normal Z", ""),
        ]
        colors = [SINGLE_COLOR]*3
        ncols = 3

    fig, axes = plt.subplots(2 if bimanual else 1, ncols, figsize=(12, 6 if bimanual else 3.5))
    axes = np.array(axes).flatten()

    for i, (col, label, unit) in enumerate(panels):
        if col in df.columns:
            y = downsample(pd.to_numeric(df[col], errors="coerce").fillna(0))
            plot_col(axes[i], y, label, colors[i])
        axes[i].set_title(label)
        axes[i].set_ylabel("Normal")

    for j in range(len(panels), len(axes)):
        axes[j].set_visible(False)

    fig.suptitle(f"Exercise {exercise_num} – Palm Orientation (Normal Vector)", y=1.01, color="white")
    fig.tight_layout()
    fig.savefig(out_path, dpi=100, bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.close(fig)


def plot_position(df, out_path, exercise_num):
    bimanual = is_bimanual(df)
    if bimanual:
        panels = [
            ("Left_hand_palm_position_x",  "Left Palm X",  "mm"),
            ("Left_hand_palm_position_y",  "Left Palm Y",  "mm"),
            ("Left_hand_palm_position_z",  "Left Palm Z",  "mm"),
            ("Right_hand_palm_position_x", "Right Palm X", "mm"),
            ("Right_hand_palm_position_y", "Right Palm Y", "mm"),
            ("Right_hand_palm_position_z", "Right Palm Z", "mm"),
        ]
        colors = [LEFT_COLOR]*3 + [RIGHT_COLOR]*3
        ncols = 3
    else:
        hand = dominant_hand(df)
        panels = [
            (f"{hand}_hand_palm_position_x", f"{hand} Palm X", "mm"),
            (f"{hand}_hand_palm_position_y", f"{hand} Palm Y", "mm"),
            (f"{hand}_hand_palm_position_z", f"{hand} Palm Z", "mm"),
        ]
        colors = [SINGLE_COLOR]*3
        ncols = 3

    fig, axes = plt.subplots(2 if bimanual else 1, ncols, figsize=(12, 6 if bimanual else 3.5))
    axes = np.array(axes).flatten()

    for i, (col, label, unit) in enumerate(panels):
        if col in df.columns:
            y = downsample(pd.to_numeric(df[col], errors="coerce").fillna(0))
            plot_col(axes[i], y, label, colors[i])
        axes[i].set_title(label)
        axes[i].set_ylabel(unit)

    for j in range(len(panels), len(axes)):
        axes[j].set_visible(False)

    fig.suptitle(f"Exercise {exercise_num} – Palm Position", y=1.01, color="white")
    fig.tight_layout()
    fig.savefig(out_path, dpi=100, bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.close(fig)


def plot_grip(df, out_path, exercise_num):
    bimanual = is_bimanual(df)
    if bimanual:
        panels = [
            ("Left_hand_grab_strength",  "Left Grip",  ""),
            ("Right_hand_grab_strength", "Right Grip", ""),
        ]
        colors = [LEFT_COLOR, RIGHT_COLOR]
    else:
        hand = dominant_hand(df)
        panels = [(f"{hand}_hand_grab_strength", f"{hand} Grip", "")]
        colors = [SINGLE_COLOR]

    fig, axes = plt.subplots(1, len(panels), figsize=(8, 3.5))
    axes = np.array([axes]).flatten() if len(panels) == 1 else np.array(axes).flatten()

    for i, (col, label, unit) in enumerate(panels):
        if col in df.columns:
            y = downsample(pd.to_numeric(df[col], errors="coerce").fillna(0))
            plot_col(axes[i], y, label, colors[i])
        axes[i].set_title(label)
        axes[i].set_ylabel("Grip Strength (0-1)")
        axes[i].set_ylim(-0.05, 1.05)

    fig.suptitle(f"Exercise {exercise_num} – Grip Strength", y=1.01, color="white")
    fig.tight_layout()
    fig.savefig(out_path, dpi=100, bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.close(fig)


# ── Main ──────────────────────────────────────────────────────────────────────

def process_user(folder_name):
    folder_path = os.path.join(USERS_DIR, folder_name)
    file_id = folder_name.split("_ID_")[-1]
    out_dir_plots = os.path.join(PLOTS_DIR, file_id)
    out_dir_web   = os.path.join(WEB_ASSETS, file_id)
    os.makedirs(out_dir_plots, exist_ok=True)
    os.makedirs(out_dir_web,   exist_ok=True)

    exercises_done = []
    for n in range(1, 6):
        csv_path = os.path.join(folder_path, f"dataCompilation{n}.csv")
        if not os.path.exists(csv_path):
            continue
        df = read_csv(csv_path)
        if df.empty:
            continue

        exercises_done.append(n)
        prefix = f"exercise{n}"

        # Speed
        plot_speed(df, os.path.join(out_dir_plots, f"{prefix}_speed.png"), n)
        # Orientation
        plot_orientation(df, os.path.join(out_dir_plots, f"{prefix}_orientation.png"), n)
        # Position
        plot_position(df, os.path.join(out_dir_plots, f"{prefix}_position.png"), n)
        # Grip
        plot_grip(df, os.path.join(out_dir_plots, f"{prefix}_grip.png"), n)

        print(f"    Exercise {n}: speed, orientation, position, grip")

    # Copy all PNGs to web assets
    for f in os.listdir(out_dir_plots):
        if f.endswith(".png"):
            shutil.copy2(os.path.join(out_dir_plots, f), os.path.join(out_dir_web, f))

    return file_id, exercises_done


def main():
    folders = sorted([
        f for f in os.listdir(USERS_DIR)
        if os.path.isdir(os.path.join(USERS_DIR, f)) and "_ID_" in f
    ])
    total = len(folders)
    print(f"Generating plots for {total} users...\n")

    user_exercise_map = {}
    for i, folder in enumerate(folders, 1):
        file_id = folder.split("_ID_")[-1]
        print(f"[{i}/{total}] {file_id}")
        fid, exercises = process_user(folder)
        user_exercise_map[fid] = exercises

    # Write exercise availability map to JSON for Angular
    import json
    map_path = os.path.join(
        PAPER_DIR, "MotionInsightHub", "web", "src", "assets", "data", "user_exercises.json"
    )
    os.makedirs(os.path.dirname(map_path), exist_ok=True)
    with open(map_path, "w", encoding="utf-8") as f:
        json.dump(user_exercise_map, f, indent=2)
    print(f"\n[OK] user_exercises.json written -> {map_path}")
    print(f"[OK] All plots saved to {PLOTS_DIR}")
    print(f"[OK] All plots copied to {WEB_ASSETS}")
    print(f"\nDone. {total} users processed.")


if __name__ == "__main__":
    main()

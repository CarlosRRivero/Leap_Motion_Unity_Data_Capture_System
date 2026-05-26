"""
plot_showcase.py

15 visually-significant comparison plots (3 per exercise × 5 exercises).
Each plot pairs a normative user vs a non-normative user of matching sex and
age band, selected automatically to maximise visual contrast.

Age bands  (matching 3-division scheme):
  45–60 yr  |  61–75 yr  |  75+ yr

  Exercise 1  — Active-hand speed time-series  (shared y-axis)
  Exercise 2  — Grip: time-series · FFT spectrum · L/R covariance scatter
  Exercise 3  — Plane trajectory with rocks, fuels & time zones
  Exercise 4  — Plane trajectory with rocks, fuels & time zones
  Exercise 5  — Plane trajectory with rocks, fuels & time zones

Plus a showcase_overview.png card preview.

Outputs:
  Paper/Plots/showcase/
  web/src/assets/plots/showcase/

Usage:
  python plot_showcase.py
"""

import os
import sys
import json
import shutil
import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection
from scipy.ndimage import uniform_filter1d

SCRIPTS_DIR = os.path.dirname(os.path.abspath(__file__))
PAPER_DIR   = os.path.normpath(os.path.join(SCRIPTS_DIR, ".."))
sys.path.insert(0, SCRIPTS_DIR)

import plot_utils as pu

USERS_DIR  = os.path.join(PAPER_DIR, "Users")
USERS_JSON = os.path.normpath(
    os.path.join(PAPER_DIR, "MotionInsightHub", "web",
                 "src", "assets", "data", "users.json")
)
PLOTS_DIR = os.path.join(PAPER_DIR, "Plots", "showcase")
WEB_PLOTS = os.path.normpath(
    os.path.join(PAPER_DIR, "MotionInsightHub", "web",
                 "src", "assets", "plots", "showcase")
)

EXCLUDE_IDS = {"0", "1"}
NORM_C      = "#2196F3"
NN_C        = "#E64A19"
FPS         = 44.0

USERS_DATA_DIR = os.path.normpath(
    os.path.join(PAPER_DIR, "MotionInsightHub", "web", "src", "assets", "data", "users")
)
FUEL_JSON = os.path.normpath(
    os.path.join(PAPER_DIR, "MotionInsightHub", "web", "src", "assets", "data", "fuel_locations.json")
)

# Age bands matching the 3-division scheme
AGE_EDGES  = [45, 61, 76, 110]
AGE_LABELS = ["45\u201360", "61\u201375", "75+"]
AGE_KEYS   = ["45_60", "61_75", "75plus"]


# ═══════════════════════════════════════════════════════════════════════════════
#  Demographic helpers (self-contained — no dependency on other plot scripts)
# ═══════════════════════════════════════════════════════════════════════════════

def _folder_ts(name):
    for prefix in ("User_non_normative_", "User_normative_"):
        if name.startswith(prefix):
            return name[len(prefix):].split("_ID_")[0]
    return name


def is_excluded(folder):
    if "non_normative" in folder or "_ID_" not in folder:
        return False
    fid = folder.split("_ID_")[-1]
    return fid in EXCLUDE_IDS and not fid.startswith("2026")


def assign_bin(age):
    age = float(age)
    for i in range(len(AGE_EDGES) - 1):
        if AGE_EDGES[i] <= age < AGE_EDGES[i + 1]:
            return i
    return len(AGE_EDGES) - 2


def build_age_sex_maps():
    if not os.path.exists(USERS_JSON):
        return {}, {}
    with open(USERS_JSON, encoding="utf-8") as f:
        j = json.load(f)

    n25, n26, nn = {}, [], {}
    for u in j.get("normative", []):
        sd = str(u["start_date"])
        if sd.startswith("2026"):
            n26.append((sd, int(u["age"]), u.get("sex", "")))
        else:
            n25[u["file_id"]] = (int(u["age"]), u.get("sex", ""))
    for u in j.get("non_normative", []):
        nn[u["file_id"]] = (int(u["age"]), u.get("sex", ""))
    n26.sort(key=lambda x: x[0])

    all_folders = [f for f in sorted(os.listdir(USERS_DIR))
                   if os.path.isdir(os.path.join(USERS_DIR, f))]
    n26_folders = sorted(
        [f for f in all_folders
         if "non_normative" not in f and "normative" in f and "_ID_" in f
         and f.split("_ID_")[-1].startswith("2026")],
        key=_folder_ts
    )

    age_map, sex_map = {}, {}
    for i, folder in enumerate(n26_folders):
        if i < len(n26):
            age_map[folder] = n26[i][1]
            sex_map[folder] = n26[i][2]

    for folder in all_folders:
        if folder in age_map or "_ID_" not in folder:
            continue
        fid = folder.split("_ID_")[-1]
        if "non_normative" in folder:
            if fid in nn:
                age_map[folder] = nn[fid][0]
                sex_map[folder] = nn[fid][1]
            else:
                for kfid, (age, sex) in nn.items():
                    if len(fid) == len(kfid) and sum(a != b for a, b in zip(fid, kfid)) <= 1:
                        age_map[folder] = age
                        sex_map[folder] = sex
                        break
        elif "normative" in folder and fid in n25:
            age_map[folder] = n25[fid][0]
            sex_map[folder] = n25[fid][1]

    return age_map, sex_map


# ═══════════════════════════════════════════════════════════════════════════════
#  Data loading & small utilities
# ═══════════════════════════════════════════════════════════════════════════════

def load_df(folder, ex_num):
    path = os.path.join(USERS_DIR, folder, f"dataCompilation{ex_num}.csv")
    if not os.path.exists(path):
        return None
    try:
        return pd.read_csv(path, sep=";", low_memory=False)
    except Exception:
        return None


def num(df, col):
    return pd.to_numeric(df.get(col, pd.Series(dtype=float)), errors="coerce")


def _dominant(df):
    try:
        return pu.dominant_hand(df)
    except Exception:
        l = num(df, "Left_hand_speed").abs().sum()
        r = num(df, "Right_hand_speed").abs().sum()
        return "Left" if l >= r else "Right"


def _resample(arr, n=300):
    if len(arr) < 2:
        return np.zeros(n)
    return np.interp(np.linspace(0, 1, n), np.linspace(0, 1, len(arr)), arr)


# ═══════════════════════════════════════════════════════════════════════════════
#  User pool by age band and exercise
# ═══════════════════════════════════════════════════════════════════════════════

def build_pool(age_map, sex_map, ex_num):
    """
    {bin: {"normative": [(folder, age, sex)], "non_normative": [...]}}
    """
    pool = {i: {"normative": [], "non_normative": []}
            for i in range(len(AGE_LABELS))}

    for folder in sorted(os.listdir(USERS_DIR)):
        if not os.path.isdir(os.path.join(USERS_DIR, folder)):
            continue
        if is_excluded(folder):
            continue
        age = age_map.get(folder)
        if age is None:
            continue
        b = assign_bin(age)
        if not os.path.exists(os.path.join(USERS_DIR, folder,
                                           f"dataCompilation{ex_num}.csv")):
            continue
        sex = sex_map.get(folder, "")
        if "non_normative" in folder:
            pool[b]["non_normative"].append((folder, age, sex))
        elif "normative" in folder:
            pool[b]["normative"].append((folder, age, sex))

    return pool


# ═══════════════════════════════════════════════════════════════════════════════
#  Scoring functions  (higher = more visually different)
# ═══════════════════════════════════════════════════════════════════════════════

def score_speed(n_df, p_df):
    """Score using both hands combined for maximum contrast detection."""
    total = 0.0
    for col in ("Left_hand_speed", "Right_hand_speed"):
        n_s = num(n_df, col).dropna().values
        p_s = num(p_df, col).dropna().values
        if len(n_s) < 30 or len(p_s) < 30:
            continue
        n_r = _resample(n_s)
        p_r = _resample(p_s)
        total += float(np.mean(np.abs(n_r - p_r)) + abs(np.std(n_r) - np.std(p_r)))
    return total


def score_grip(n_df, p_df):
    """Score grip contrast using both hands. Penalises near-constant signals."""
    total = 0.0
    for col in ("Left_hand_grab_strength", "Right_hand_grab_strength"):
        n_g = num(n_df, col).dropna().values
        p_g = num(p_df, col).dropna().values
        if len(n_g) < 30 or len(p_g) < 30:
            continue
        # Reject pairs where either user has a flat/constant grip
        if np.std(n_g) < 0.05 or np.std(p_g) < 0.05:
            return 0.0
        # Reward: large std difference and mean difference
        total += abs(np.std(n_g) - np.std(p_g)) * 4 + abs(np.mean(n_g) - np.mean(p_g))
    return total


def _file_id(folder):
    """Extract file_id from folder name (part after last _ID_)."""
    if "_ID_" in folder:
        return folder.split("_ID_")[-1]
    return None


def load_rocks(folder, ex_num):
    fid  = _file_id(folder)
    if not fid:
        return []
    path = os.path.join(USERS_DATA_DIR, fid, f"exercise{ex_num}.json")
    if not os.path.exists(path):
        return []
    try:
        with open(path, encoding="utf-8") as f:
            return json.load(f).get("rocks", [])
    except Exception:
        return []


def load_fuels(ex_num):
    if not os.path.exists(FUEL_JSON):
        return []
    try:
        with open(FUEL_JSON, encoding="utf-8") as f:
            return json.load(f).get(str(ex_num), [])
    except Exception:
        return []


def score_traj(n_df, p_df):
    """Score using plane trajectory from exercise JSON (exercises 3/4/5).
    n_df and p_df are (folder, csv_df) tuples when called from find_best_pair,
    so we accept both tuple and DataFrame forms."""
    # This scorer is called as score_fn(n_df, p_df) where the args are DataFrames.
    # We stash folder names on the DataFrame via a workaround: the caller passes
    # plain DataFrames, so fall back to a simple CSV-based heuristic.
    def path_len_csv(df):
        # Plane_position_x has only 1 row in CSV — use hand position as proxy
        hand = _dominant(df)
        x = num(df, f"{hand}_hand_palm_position_x").dropna().values
        z = num(df, f"{hand}_hand_palm_position_z").dropna().values
        mn = min(len(x), len(z))
        if mn < 10:
            return 0.0, 0.0
        length = float(np.sum(np.sqrt(np.diff(x[:mn])**2 + np.diff(z[:mn])**2)))
        spread = float(np.std(z[:mn]))
        return length, spread

    n_len, n_sp = path_len_csv(n_df)
    p_len, p_sp = path_len_csv(p_df)
    if n_len == 0 and p_len == 0:
        return 0.0
    rel = abs(n_len - p_len) / max(n_len, p_len, 1.0)
    spr = abs(n_sp  - p_sp) * 0.005
    return rel + spr


_SCORE_FN = {1: score_speed, 2: score_grip, 3: score_traj,
             4: score_traj,  5: score_traj}


# ═══════════════════════════════════════════════════════════════════════════════
#  Pair selection
# ═══════════════════════════════════════════════════════════════════════════════

def find_best_pair(norm_list, nn_list, score_fn, ex_num, force_sex=None):
    """
    Returns (norm_folder, nn_folder) with maximum contrast score.
    If force_sex is given (e.g. "Female"), only pairs of that sex are considered.
    Otherwise same-sex pairs are preferred; if none exist any-sex is used.
    """
    best_score = -1.0
    best_pair  = (None, None)

    if force_sex:
        passes = [(force_sex,)]   # only one pass, sex-forced
    else:
        passes = [(True,), (False,)]  # prefer same-sex, then any

    for (pass_arg,) in passes:
        for n_folder, _, n_sex in norm_list:
            if force_sex and n_sex != force_sex:
                continue
            n_df = load_df(n_folder, ex_num)
            if n_df is None or len(n_df) < 30:
                continue
            for p_folder, _, p_sex in nn_list:
                if force_sex and p_sex != force_sex:
                    continue
                if not force_sex and pass_arg is True and n_sex and p_sex and n_sex != p_sex:
                    continue
                p_df = load_df(p_folder, ex_num)
                if p_df is None or len(p_df) < 30:
                    continue
                try:
                    s = score_fn(n_df, p_df)
                except Exception:
                    s = 0.0
                if s > best_score:
                    best_score = s
                    best_pair  = (n_folder, p_folder)
        if best_pair[0] is not None:
            break

    return best_pair


# ═══════════════════════════════════════════════════════════════════════════════
#  Shared style helper
# ═══════════════════════════════════════════════════════════════════════════════

def _style(ax, grid_both=False):
    axis = "both" if grid_both else "y"
    ax.grid(axis=axis, linestyle="--", alpha=0.32)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)


# ═══════════════════════════════════════════════════════════════════════════════
#  Exercise 1 — Speed time-series
# ═══════════════════════════════════════════════════════════════════════════════

def plot_speed(n_folder, p_folder, age_label, age_key, info, out_dir):
    n_df = load_df(n_folder, 1)
    p_df = load_df(p_folder, 1)

    def smooth_col(df, col):
        arr = num(df, col).dropna().values
        return uniform_filter1d(arr, size=max(1, len(arr) // 60))

    n_L = smooth_col(n_df, "Left_hand_speed")
    n_R = smooth_col(n_df, "Right_hand_speed")
    p_L = smooth_col(p_df, "Left_hand_speed")
    p_R = smooth_col(p_df, "Right_hand_speed")

    # Shared y-axis across all four series
    all_vals = np.concatenate([n_L, n_R, p_L, p_R])
    y_max = float(np.percentile(all_vals[all_vals > 0], 98)) * 1.10 if (all_vals > 0).any() else 1.0

    # Colour pairs: main (Left) and lighter (Right)
    NORM_L_C = "#1565C0"   # dark blue  — normative left
    NORM_R_C = "#64B5F6"   # light blue — normative right
    NN_L_C   = "#BF360C"   # dark orange — non-norm left
    NN_R_C   = "#FFAB91"   # light orange — non-norm right

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5.0), sharey=True)

    for ax, L_arr, R_arr, Lc, Rc, folder_key, title_prefix in [
        (ax1, n_L, n_R, NORM_L_C, NORM_R_C,
         n_folder, f"Normative  ({info[n_folder]})"),
        (ax2, p_L, p_R, NN_L_C,   NN_R_C,
         p_folder, f"Non-normative / PD  ({info[p_folder]})"),
    ]:
        xi_L = np.linspace(0, 100, len(L_arr))
        xi_R = np.linspace(0, 100, len(R_arr))

        ax.fill_between(xi_L, 0, L_arr, color=Lc, alpha=0.12)
        ax.fill_between(xi_R, 0, R_arr, color=Rc, alpha=0.12)
        ax.plot(xi_L, L_arr, color=Lc, linewidth=1.4, alpha=0.92, label="Left hand")
        ax.plot(xi_R, R_arr, color=Rc, linewidth=1.4, alpha=0.92,
                linestyle="--", label="Right hand")

        ax.set_title(title_prefix, fontweight="bold", fontsize=10)
        ax.set_xlabel("Normalised time (%)", fontsize=9)
        ax.set_ylim(0, y_max)
        ax.legend(fontsize=8, loc="upper right")
        _style(ax)

    ax1.set_ylabel("Hand speed  (mm/s)", fontsize=9)

    fig.suptitle(
        f"Exercise 1 \u2014 Both Hands Speed  [{age_label} yr]",
        fontsize=12, fontweight="bold"
    )
    fig.tight_layout()
    fname = f"exercise1_showcase_speed_{age_key}.png"
    fig.savefig(os.path.join(out_dir, fname), dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  [OK] {fname}")
    return fname


# ═══════════════════════════════════════════════════════════════════════════════
#  Exercise 2 — Grip: time-series · FFT · L/R covariance
# ═══════════════════════════════════════════════════════════════════════════════

def _grip_fft(arr, fps=FPS, max_hz=8.0):
    n = len(arr)
    if n < 8:
        return np.array([0.0]), np.array([0.0])
    w   = np.hanning(n)
    yf  = np.fft.rfft((arr - arr.mean()) * w)
    xf  = np.fft.rfftfreq(n, 1.0 / fps)
    psd = np.abs(yf) ** 2
    mask = xf <= max_hz
    return xf[mask], psd[mask]


def _fft_both_hands(ax, df_n, df_p, fps=FPS, max_hz=10.0):
    """Plot FFT spectra for both hands of both users on one axis for direct comparison."""
    NORM_L_C = "#1565C0"; NORM_R_C = "#64B5F6"
    NN_L_C   = "#BF360C"; NN_R_C   = "#FFAB91"

    pairs = [
        ("Left_hand_grab_strength",  df_n, NORM_L_C, "Norm — Left",  "-"),
        ("Right_hand_grab_strength", df_n, NORM_R_C, "Norm — Right", "--"),
        ("Left_hand_grab_strength",  df_p, NN_L_C,   "NN — Left",   "-"),
        ("Right_hand_grab_strength", df_p, NN_R_C,   "NN — Right",  "--"),
    ]
    for col, df, c, lbl, ls in pairs:
        arr = num(df, col).dropna().values
        if len(arr) < 8:
            continue
        xf, psd = _grip_fft(arr, fps=fps, max_hz=max_hz)
        if psd.max() == 0:
            continue
        psd_norm = psd / psd.max()
        ax.fill_between(xf, 0, psd_norm, color=c, alpha=0.13)
        ax.plot(xf, psd_norm, color=c, linewidth=1.4, alpha=0.92,
                linestyle=ls, label=lbl)
        peak = xf[np.argmax(psd)]
        ax.axvline(peak, color=c, linestyle=":", linewidth=1.0, alpha=0.60)
        ax.text(peak + 0.08, psd_norm.max() * 0.90 if psd_norm.max() > 0 else 0.90,
                f"{peak:.1f} Hz", fontsize=7.5, color=c, fontweight="bold")


def plot_grip(n_folder, p_folder, age_label, age_key, info, out_dir):
    """
    Redesigned 2x2 layout:
      Row 0: Grip time-series side-by-side  (normative | non-normative),
             each panel shows Left (solid) + Right (dashed) hands.
      Row 1: Overlaid FFT spectra for all 4 signals | summary stats panel.
    """
    n_df = load_df(n_folder, 2)
    p_df = load_df(p_folder, 2)

    NORM_L_C = "#1565C0"; NORM_R_C = "#64B5F6"
    NN_L_C   = "#BF360C"; NN_R_C   = "#FFAB91"

    fig, axes = plt.subplots(2, 2, figsize=(14, 8.0))
    ax_ts_n, ax_ts_p = axes[0]
    ax_fft,  ax_stat  = axes[1]

    # ── Row 0: Time-series panels ─────────────────────────────────────────────
    for ax, df, folder, lc, rc, label in [
        (ax_ts_n, n_df, n_folder, NORM_L_C, NORM_R_C,
         f"Normative  ({info[n_folder]})"),
        (ax_ts_p, p_df, p_folder, NN_L_C,   NN_R_C,
         f"Non-normative / PD  ({info[p_folder]})"),
    ]:
        grip_l = num(df, "Left_hand_grab_strength").dropna().values
        grip_r = num(df, "Right_hand_grab_strength").dropna().values
        xi_l = np.linspace(0, 100, len(grip_l))
        xi_r = np.linspace(0, 100, len(grip_r))

        ax.fill_between(xi_l, 0, grip_l, color=lc, alpha=0.14)
        ax.fill_between(xi_r, 0, grip_r, color=rc, alpha=0.14)
        ax.plot(xi_l, grip_l, color=lc, linewidth=1.1, alpha=0.90, label="Left hand")
        ax.plot(xi_r, grip_r, color=rc, linewidth=1.1, alpha=0.90,
                linestyle="--", label="Right hand")
        ax.set_ylim(-0.05, 1.12)
        ax.set_title(f"{label}\nGrip strength over time",
                     fontweight="bold", fontsize=9.5)
        ax.set_xlabel("Normalised time (%)", fontsize=9)
        ax.set_ylabel("Grip  (0-1)", fontsize=9)
        ax.legend(fontsize=8, loc="upper right")
        _style(ax)

    # ── Row 1 left: Overlaid FFT comparison ──────────────────────────────────
    _fft_both_hands(ax_fft, n_df, p_df)
    ax_fft.set_title("Frequency spectrum  (normalised power, all signals)",
                     fontweight="bold", fontsize=9.5)
    ax_fft.set_xlabel("Frequency  (Hz)", fontsize=9)
    ax_fft.set_ylabel("Normalised power  (0-1)", fontsize=9)
    ax_fft.legend(fontsize=7.5, ncol=2, loc="upper right")
    ax_fft.set_ylim(0, 1.10)
    _style(ax_fft, grid_both=True)

    # ── Row 1 right: Summary stats table ─────────────────────────────────────
    ax_stat.axis("off")
    rows_data = []
    for label_short, df, lc, rc in [
        ("Normative", n_df, NORM_L_C, NORM_R_C),
        ("Non-norm",  p_df, NN_L_C,   NN_R_C),
    ]:
        for hand_label, col, c in [
            ("Left",  "Left_hand_grab_strength",  lc),
            ("Right", "Right_hand_grab_strength", rc),
        ]:
            arr = num(df, col).dropna().values
            if len(arr) < 8:
                rows_data.append([f"{label_short}\n{hand_label}", "—", "—", "—", "—"])
                continue
            xf, psd = _grip_fft(arr)
            peak_hz = xf[np.argmax(psd)] if psd.max() > 0 else 0.0
            rows_data.append([
                f"{label_short}\n{hand_label}",
                f"{np.mean(arr):.3f}",
                f"{np.std(arr):.3f}",
                f"{arr.min():.2f} – {arr.max():.2f}",
                f"{peak_hz:.2f} Hz",
            ])

    col_labels = ["Group\nHand", "Mean", "Std Dev", "Range", "Peak freq"]
    table = ax_stat.table(
        cellText=rows_data,
        colLabels=col_labels,
        cellLoc="center",
        loc="center",
        bbox=[0.0, 0.05, 1.0, 0.90],
    )
    table.auto_set_font_size(False)
    table.set_fontsize(8.5)
    # Colour header row
    for j in range(len(col_labels)):
        table[(0, j)].set_facecolor("#ECEFF1")
        table[(0, j)].set_text_props(fontweight="bold")
    # Colour data rows by group
    colours = [NORM_L_C, NORM_R_C, NN_L_C, NN_R_C]
    for i in range(len(rows_data)):
        for j in range(len(col_labels)):
            cell = table[(i + 1, j)]
            cell.set_facecolor(colours[i] + "22")  # hex alpha ~13 %
    ax_stat.set_title("Summary statistics", fontweight="bold", fontsize=9.5)

    fig.suptitle(
        f"Exercise 2 \u2014 Grip Analysis  [{age_label} yr]",
        fontsize=13, fontweight="bold"
    )
    fig.tight_layout()
    fname = f"exercise2_showcase_grip_{age_key}.png"
    fig.savefig(os.path.join(out_dir, fname), dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  [OK] {fname}")
    return fname


# ═══════════════════════════════════════════════════════════════════════════════
#  Exercises 3 / 4 / 5 — Plane trajectory with rocks, fuels & time zones
# ═══════════════════════════════════════════════════════════════════════════════

def _zone_bands(px, duration=30.0):
    """Return (x0, x10, x20, x_end) plane-x positions at 0 / 10 / 20 s and end.
    Uses proportional indexing so it works correctly on sampled (downsampled) data."""
    if len(px) == 0:
        return 0, 0, 0, 0
    n    = len(px)
    x0   = px[0]
    x10  = px[min(int(n * 10 / duration), n - 1)]
    x20  = px[min(int(n * 20 / duration), n - 1)]
    xend = px[-1]
    return x0, x10, x20, xend


def _draw_plane_panel(ax, px, py, folder, label, color, rocks, fuels, x0, x10, x20, xend):
    """Draw one plane-trajectory panel (matching the explorer style)."""

    # ── Time-zone background bands ───────────────────────────────────────────
    # Fixed zones (0-10 s and 20-30 s): same green; Random zone (10-20 s): yellow
    ax.axvspan(min(x0, x10),   max(x0, x10),   alpha=0.18, color="#50C878",
               label="0-10 s (fixed)",   zorder=0)
    ax.axvspan(min(x10, x20),  max(x10, x20),  alpha=0.18, color="#FFC832",
               label="10-20 s (random)", zorder=0)
    ax.axvspan(min(x20, xend), max(x20, xend), alpha=0.18, color="#50C878",
               label="20-30 s (fixed)",  zorder=0)

    # ── Plane path coloured by time ──────────────────────────────────────────
    t    = np.linspace(0, 1, len(px))
    pts  = np.array([px, py]).T.reshape(-1, 1, 2)
    segs = np.concatenate([pts[:-1], pts[1:]], axis=1)
    lc   = LineCollection(segs, cmap="plasma", linewidth=2.0, alpha=0.90, zorder=3)
    lc.set_array(t[:-1])
    ax.add_collection(lc)

    ax.scatter(px[0],  py[0],  color="limegreen", s=100, zorder=6,
               marker="o", edgecolors="k", linewidths=0.7, label="Start")
    ax.scatter(px[-1], py[-1], color="crimson",   s=130, zorder=6,
               marker="*", edgecolors="k", linewidths=0.7, label="End")

    # ── Rocks (obstacles) ────────────────────────────────────────────────────
    if rocks:
        rx = [r["x"] for r in rocks]
        ry = [r["y"] for r in rocks]
        ax.scatter(rx, ry, color="rgba(239,83,80,0.85)" if False else "#EF5350",
                   s=55, marker="^", zorder=5, alpha=0.85,
                   edgecolors="#B71C1C", linewidths=0.5, label="Obstacles")

    # ── Fuels ────────────────────────────────────────────────────────────────
    if fuels:
        fx = [f["x"] for f in fuels]
        fy = [f["y"] for f in fuels]
        ax.scatter(fx, fy, color="#FFD54F", s=110, marker="*", zorder=5,
                   edgecolors="#F57F17", linewidths=0.5, label="Fuels")

    ax.set_xlabel("Position X — forward  (m)", fontsize=9)
    ax.set_ylabel("Position Y — height  (m)", fontsize=9)
    ax.set_title(label, fontweight="bold", fontsize=10)
    ax.legend(fontsize=7.5, loc="upper left", ncol=2)
    _style(ax, grid_both=True)


def load_exercise_json(folder, ex_num):
    """Load the pre-processed exercise JSON used by the explorer."""
    fid  = _file_id(folder)
    if not fid:
        return None
    path = os.path.join(USERS_DATA_DIR, fid, f"exercise{ex_num}.json")
    if not os.path.exists(path):
        return None
    try:
        with open(path, encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return None


def plot_trajectory(n_folder, p_folder, ex_num, age_label, age_key, info, out_dir):
    n_ex = load_exercise_json(n_folder, ex_num)
    p_ex = load_exercise_json(p_folder, ex_num)

    def plane_xy(ex):
        if ex is None:
            return np.array([]), np.array([])
        cols = ex.get("columns", {})
        px = np.array(cols.get("Plane_position_x", []), dtype=float)
        py = np.array(cols.get("Plane_position_y", []), dtype=float)
        mn = min(len(px), len(py))
        return px[:mn], py[:mn]

    n_px, n_py = plane_xy(n_ex)
    p_px, p_py = plane_xy(p_ex)

    if len(n_px) < 10 or len(p_px) < 10:
        print(f"  [SKIP] Exercise {ex_num} {age_label}: no plane data")
        return None

    # Rocks and fuels
    n_rocks = n_ex.get("rocks", []) if n_ex else []
    p_rocks = p_ex.get("rocks", []) if p_ex else []
    fuels   = load_fuels(ex_num)

    # Shared axis limits
    all_x = np.concatenate([n_px, p_px, [r["x"] for r in n_rocks + p_rocks + fuels]] or [n_px])
    all_y = np.concatenate([n_py, p_py, [r["y"] for r in n_rocks + p_rocks + fuels]] or [n_py])
    pad_x = max((all_x.max() - all_x.min()) * 0.06, 5)
    pad_y = max((all_y.max() - all_y.min()) * 0.15, 2)
    xlim  = (all_x.min() - pad_x, all_x.max() + pad_x)
    ylim  = (all_y.min() - pad_y, all_y.max() + pad_y)

    # Zone boundaries (use normative user as reference for x positions)
    x0, x10, x20, xend = _zone_bands(n_px)

    # Reserve a narrow column on the right for the colorbar
    fig = plt.figure(figsize=(15, 5.5))
    ax1 = fig.add_axes([0.04, 0.12, 0.43, 0.78])
    ax2 = fig.add_axes([0.52, 0.12, 0.43, 0.78])
    cax = fig.add_axes([0.96, 0.20, 0.015, 0.60])

    for ax, px, py, folder, rocks in [
        (ax1, n_px, n_py, n_folder, n_rocks),
        (ax2, p_px, p_py, p_folder, p_rocks),
    ]:
        lbl = (f"Normative  ({info[folder]})"
               if folder == n_folder
               else f"Non-normative / PD  ({info[folder]})")
        _draw_plane_panel(ax, px, py, folder, lbl,
                          NORM_C if folder == n_folder else NN_C,
                          rocks, fuels, x0, x10, x20, xend)
        ax.set_xlim(xlim)
        ax.set_ylim(ylim)

    # Shared time colorbar in dedicated axes — never overlaps the plots
    sm = plt.cm.ScalarMappable(cmap="plasma", norm=plt.Normalize(0, 100))
    sm.set_array([])
    cbar = fig.colorbar(sm, cax=cax)
    cbar.set_label("Normalised time  (%)", fontsize=9)
    cbar.ax.tick_params(labelsize=8)

    fig.suptitle(
        f"Exercise {ex_num} \u2014 Plane Trajectory  [{age_label} yr]",
        fontsize=12, fontweight="bold", y=0.98
    )
    fname = f"exercise{ex_num}_showcase_trajectory_{age_key}.png"
    fig.savefig(os.path.join(out_dir, fname), dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  [OK] {fname}")
    return fname


# ═══════════════════════════════════════════════════════════════════════════════
#  3×5 overview mosaic
# ═══════════════════════════════════════════════════════════════════════════════

def make_overview(fnames, out_dir):
    """5 rows (exercises) × 3 columns (age bands) thumbnail mosaic."""
    try:
        from PIL import Image
        TW, TH = 460, 230
        rows, cols = 5, 3
        mosaic = Image.new("RGB", (cols * TW, rows * TH), (248, 248, 248))
        for idx, fname in enumerate(fnames):
            if not fname:
                continue
            path = os.path.join(out_dir, fname)
            if not os.path.exists(path):
                continue
            img = Image.open(path)
            img.thumbnail((TW, TH), Image.LANCZOS)
            r, c = divmod(idx, cols)
            ox = c * TW + (TW - img.width)  // 2
            oy = r * TH + (TH - img.height) // 2
            mosaic.paste(img, (ox, oy))
        mosaic.save(os.path.join(out_dir, "showcase_overview.png"))
        print("  [OK] showcase_overview.png  (mosaic)")
    except ImportError:
        # Pillow not installed — fall back to copying the first valid plot
        for fname in fnames:
            if fname and os.path.exists(os.path.join(out_dir, fname)):
                shutil.copy2(
                    os.path.join(out_dir, fname),
                    os.path.join(out_dir, "showcase_overview.png")
                )
                print("  [OK] showcase_overview.png  (fallback copy)")
                break


# ═══════════════════════════════════════════════════════════════════════════════
#  Main
# ═══════════════════════════════════════════════════════════════════════════════

def main():
    print("=" * 60)
    print("  Showcase — Normative vs Non-normative Comparison Plots")
    print("=" * 60)

    age_map, sex_map = build_age_sex_maps()
    info = {f: f"age {age_map[f]}, {sex_map.get(f, '?')}" for f in age_map}

    os.makedirs(PLOTS_DIR, exist_ok=True)

    plot_fns = {
        1: lambda nf, pf, al, ak: plot_speed(nf, pf, al, ak, info, PLOTS_DIR),
        2: lambda nf, pf, al, ak: plot_grip(nf, pf, al, ak, info, PLOTS_DIR),
        3: lambda nf, pf, al, ak: plot_trajectory(nf, pf, 3, al, ak, info, PLOTS_DIR),
        4: lambda nf, pf, al, ak: plot_trajectory(nf, pf, 4, al, ak, info, PLOTS_DIR),
        5: lambda nf, pf, al, ak: plot_trajectory(nf, pf, 5, al, ak, info, PLOTS_DIR),
    }

    # fnames[ex_idx * 3 + band_idx]  → used for mosaic layout
    all_fnames = []

    for ex in range(1, 6):
        print(f"\n{'='*50}")
        print(f"  Exercise {ex}")
        print(f"{'='*50}")
        pool = build_pool(age_map, sex_map, ex)
        score_fn = _SCORE_FN[ex]

        for b, (age_label, age_key) in enumerate(zip(AGE_LABELS, AGE_KEYS)):
            norm_l = pool[b]["normative"]
            nn_l   = pool[b]["non_normative"]
            if not norm_l or not nn_l:
                print(f"  [SKIP] {age_label} yr: "
                      f"norm={len(norm_l)}, nn={len(nn_l)}")
                all_fnames.append(None)
                continue

            n_f, p_f = find_best_pair(norm_l, nn_l, score_fn, ex)
            if n_f is None:
                print(f"  [SKIP] {age_label} yr: no valid pair found")
                all_fnames.append(None)
                continue

            print(f"  {age_label} yr -> norm: {info.get(n_f,'?')} "
                  f"| nn: {info.get(p_f,'?')}")
            fname = plot_fns[ex](n_f, p_f, age_label, age_key)
            all_fnames.append(fname)

    # ── Female Exercise 2 grip plots ─────────────────────────────────────────
    print(f"\n{'='*50}")
    print("  Exercise 2 — Female showcase (forced female pairs)")
    print(f"{'='*50}")
    pool2 = build_pool(age_map, sex_map, 2)
    for b, (age_label, age_key) in enumerate(zip(AGE_LABELS, AGE_KEYS)):
        norm_l = pool2[b]["normative"]
        nn_l   = pool2[b]["non_normative"]
        n_f, p_f = find_best_pair(norm_l, nn_l, score_grip, 2, force_sex="Female")
        if n_f is None:
            print(f"  [SKIP] {age_label} yr: no valid female pair found")
            continue
        print(f"  {age_label} yr -> norm: {info.get(n_f,'?')} | nn: {info.get(p_f,'?')}")
        plot_grip(n_f, p_f, age_label, f"female_{age_key}", info, PLOTS_DIR)

    # Overview mosaic
    print(f"\n{'='*50}")
    print("  Overview mosaic")
    print(f"{'='*50}")
    make_overview(all_fnames, PLOTS_DIR)

    # Copy to web assets
    print("\n-- Copying to web assets --")
    if os.path.exists(WEB_PLOTS):
        shutil.rmtree(WEB_PLOTS)
    shutil.copytree(PLOTS_DIR, WEB_PLOTS)
    print(f"  [OK] showcase  ->  {WEB_PLOTS}")

    print(f"\nDone.  {sum(1 for f in all_fnames if f)} / 15 plots generated.")
    print(f"       Saved to: {PLOTS_DIR}")


if __name__ == "__main__":
    main()

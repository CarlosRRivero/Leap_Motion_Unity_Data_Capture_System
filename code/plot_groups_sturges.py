"""
plot_groups_sturges.py

Age-group analysis for V2 using MANUAL unified age bands.
Youngsters (file_ids 0 and 1, ages 27/28) are excluded from all analyses.

Two grouping schemes are produced:

  3-division (folder: groups)
    Group 1 — 45–60 yrs:  normative & PD present
    Group 2 — 61–75 yrs:  normative & PD present
    Group 3 — 75+   yrs:  includes elderly normative cohort (Feb 2026)

  4-division (folder: groups_4)
    Group 1 — 45–55 yrs:  normative & PD present
    Group 2 — 55–65 yrs:  normative & PD present
    Group 3 — 65–75 yrs:  normative & PD present
    Group 4 — 75+   yrs:  normative & PD present

For each exercise (1-5) and each scheme, four figures are produced:
  exercise{n}_groups_speed.png
  exercise{n}_groups_orientation.png
  exercise{n}_groups_position.png
  exercise{n}_groups_grip.png

Plus a demographics overview:
  groups_age_distribution.png   (also saved as groups_overview.png for preview)

Outputs:
  Paper/Plots/groups/          Paper/Plots/groups_4/
  web/src/assets/plots/groups/ web/src/assets/plots/groups_4/

Usage:
  python plot_groups_sturges.py
"""

import os
import sys
import json
import math
import shutil
import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

SCRIPTS_DIR = os.path.dirname(os.path.abspath(__file__))
PAPER_DIR   = os.path.normpath(os.path.join(SCRIPTS_DIR, ".."))
sys.path.insert(0, SCRIPTS_DIR)

import plot_utils as pu

USERS_DIR  = os.path.join(PAPER_DIR, "Users")
# Prefer the repo-bundled metadata so the script runs from the public repository
# together with the Zenodo Users/ folder (no local web project required).
USERS_JSON = next(
    (p for p in (
        os.path.join(PAPER_DIR, "metadata", "users.json"),
        os.path.normpath(os.path.join(PAPER_DIR, "MotionInsightHub", "web", "src", "assets", "data", "users.json")),
    ) if os.path.exists(p)),
    os.path.join(PAPER_DIR, "metadata", "users.json"),
)
PLOTS_BASE = os.path.join(PAPER_DIR, "Plots")
WEB_BASE   = os.path.normpath(
    os.path.join(PAPER_DIR, "MotionInsightHub", "web", "src", "assets", "plots")
)

EXCLUDE_IDS = {"0", "1"}   # two youngest normative users (ages 27, 28)
NORM_C = "#2196F3"
NN_C   = "#E64A19"

# ── Age group configurations ───────────────────────────────────────────────────
# 3-division: wide bands so every group has both normative and PD participants.
# Boundaries: [45,61) → 45–60 yr | [61,76) → 61–75 yr | [76,110) → 75+ yr
EDGES_3  = [45, 61, 76, 110]
LABELS_3 = ["45\u201360", "61\u201375", "75+"]

# 4-division: decade-aligned bands (youngsters already excluded via EXCLUDE_IDS).
# Boundaries: [45,55)→45–55 | [55,65)→55–65 | [65,75)→65–75 | [75,110)→75+
EDGES_4  = [45, 55, 65, 75, 110]
LABELS_4 = ["45\u201355", "55\u201365", "65\u201375", "75+"]

# Configurations: (subfolder_name, edges, labels, scheme_title)
# The two manual schemes are always included; Sturges/Scott/FD are computed
# dynamically in main() from the real age data and appended at runtime.
AGE_SCHEMES = [
    ("groups",   EDGES_3,  LABELS_3, "3 Age Divisions"),
    ("groups_4", EDGES_4,  LABELS_4, "4 Age Divisions"),
]


# ═══════════════════════════════════════════════════════════════════════════════
#  Statistical binning helpers  (Sturges, Scott, Freedman-Diaconis)
# ═══════════════════════════════════════════════════════════════════════════════

def compute_binning_schemes(ages: list) -> list:
    """
    Given a list of participant ages (youngsters already excluded),
    compute three statistically-motivated age-group schemes:
      * Sturges   : k = ceil(1 + log2(n))
      * Scott     : h = 3.5 * sigma * n^(-1/3)
      * Freedman-Diaconis: h = 2 * IQR * n^(-1/3)

    Returns a list of (subfolder, edges, labels, title) tuples ready to
    append to AGE_SCHEMES.
    """
    arr = np.array([a for a in ages if a is not None and a >= 45], dtype=float)
    n   = len(arr)
    if n < 4:
        print("  [WARN] Too few participants to compute statistical bins — skipping.")
        return []

    age_min = 45          # fixed lower bound (youngsters excluded)
    age_max = int(np.ceil(arr.max()))
    rng     = age_max - age_min

    # ── Sturges ───────────────────────────────────────────────────────────────
    k_sturges = max(2, int(np.ceil(1.0 + np.log2(n))))
    h_sturges = rng / k_sturges

    # ── Scott ─────────────────────────────────────────────────────────────────
    h_scott   = 3.5 * float(arr.std(ddof=1)) * (n ** (-1.0 / 3.0))
    k_scott   = max(2, int(np.ceil(rng / h_scott)))
    h_scott   = rng / k_scott   # recompute from rounded k for clean edges

    # ── Freedman-Diaconis ─────────────────────────────────────────────────────
    iqr       = float(np.percentile(arr, 75) - np.percentile(arr, 25))
    if iqr < 1.0:                           # guard against degenerate data
        iqr = float(arr.std(ddof=1))
    h_fd      = 2.0 * iqr * (n ** (-1.0 / 3.0))
    k_fd      = max(2, int(np.ceil(rng / h_fd)))
    h_fd      = rng / k_fd

    def _make(k, h, name):
        """Build integer edges and Unicode-dash labels from (k, h)."""
        edges = [int(round(age_min + i * h)) for i in range(k)]
        edges.append(110)        # open-ended last bin
        labels = []
        for i in range(len(edges) - 2):
            labels.append(f"{edges[i]}\u2013{edges[i + 1] - 1}")
        labels.append(f"{edges[-2]}+")
        return edges, labels

    edges_s,  labels_s  = _make(k_sturges, h_sturges, "sturges")
    edges_sc, labels_sc = _make(k_scott,   h_scott,   "scott")
    edges_fd, labels_fd = _make(k_fd,      h_fd,      "fd")

    print(f"  Sturges  : k={k_sturges}, h={h_sturges:.1f} yr  -> {labels_s}")
    print(f"  Scott    : k={k_scott},   h={h_scott:.1f} yr  -> {labels_sc}")
    print(f"  Freed-D  : k={k_fd},      h={h_fd:.1f} yr  -> {labels_fd}")

    return [
        ("groups_sturges", edges_s,  labels_s,  f"Sturges ({k_sturges} divisions)"),
        ("groups_scott",   edges_sc, labels_sc, f"Scott ({k_scott} divisions)"),
        ("groups_fd",      edges_fd, labels_fd, f"Freedman-Diaconis ({k_fd} divisions)"),
    ]


# ═══════════════════════════════════════════════════════════════════════════════
#  Folder helpers
# ═══════════════════════════════════════════════════════════════════════════════

def _folder_ts(name: str) -> str:
    for prefix in ("User_non_normative_", "User_normative_"):
        if name.startswith(prefix):
            return name[len(prefix):].split("_ID_")[0]
    return name


def is_excluded(folder: str) -> bool:
    if "non_normative" in folder or "_ID_" not in folder:
        return False
    fid = folder.split("_ID_")[-1]
    return fid in EXCLUDE_IDS and not fid.startswith("2026")


# ═══════════════════════════════════════════════════════════════════════════════
#  Age map
# ═══════════════════════════════════════════════════════════════════════════════

def build_age_map() -> dict:
    """Returns {folder_name: age_int} for all user folders."""
    if not os.path.exists(USERS_JSON):
        print(f"  [WARN] users.json not found: {USERS_JSON}")
        return {}
    with open(USERS_JSON, encoding="utf-8") as f:
        j = json.load(f)

    norm_2025_map  = {}
    norm_2026_list = []
    non_norm_map   = {}

    for u in j.get("normative", []):
        fid = u["file_id"]
        age = int(u["age"])
        sd  = str(u["start_date"])
        if sd.startswith("2026"):
            norm_2026_list.append((sd, age))
        else:
            norm_2025_map[fid] = age

    for u in j.get("non_normative", []):
        non_norm_map[u["file_id"]] = int(u["age"])

    norm_2026_list.sort(key=lambda x: x[0])

    if not os.path.isdir(USERS_DIR):
        return {}
    all_folders = [
        f for f in sorted(os.listdir(USERS_DIR))
        if os.path.isdir(os.path.join(USERS_DIR, f))
    ]

    # 2026 normative folders (IDs like "20262261")
    norm_2026_folders = sorted(
        [f for f in all_folders
         if "non_normative" not in f and "normative" in f
         and "_ID_" in f and f.split("_ID_")[-1].startswith("2026")],
        key=_folder_ts
    )

    age_map: dict = {}

    # Positional mapping for 2026 normative
    for i, folder in enumerate(norm_2026_folders):
        if i < len(norm_2026_list):
            age_map[folder] = norm_2026_list[i][1]

    # Exact / near-match for all others
    for folder in all_folders:
        if folder in age_map or "_ID_" not in folder:
            continue
        fid = folder.split("_ID_")[-1]

        if "non_normative" in folder:
            if fid in non_norm_map:
                age_map[folder] = non_norm_map[fid]
            else:
                for kfid, age in non_norm_map.items():
                    if len(fid) == len(kfid) and sum(a != b for a, b in zip(fid, kfid)) <= 1:
                        age_map[folder] = age
                        break
        elif "normative" in folder and fid in norm_2025_map:
            age_map[folder] = norm_2025_map[fid]

    return age_map


def build_sex_map() -> dict:
    """Returns {folder_name: sex_str} for all user folders."""
    if not os.path.exists(USERS_JSON):
        return {}
    with open(USERS_JSON, encoding="utf-8") as f:
        j = json.load(f)

    norm_2025_map  = {}
    norm_2026_list = []
    non_norm_map   = {}

    for u in j.get("normative", []):
        fid = u["file_id"]
        sex = u.get("sex", "")
        sd  = str(u["start_date"])
        if sd.startswith("2026"):
            norm_2026_list.append((sd, sex))
        else:
            norm_2025_map[fid] = sex

    for u in j.get("non_normative", []):
        non_norm_map[u["file_id"]] = u.get("sex", "")

    norm_2026_list.sort(key=lambda x: x[0])

    if not os.path.isdir(USERS_DIR):
        return {}
    all_folders = [
        f for f in sorted(os.listdir(USERS_DIR))
        if os.path.isdir(os.path.join(USERS_DIR, f))
    ]

    norm_2026_folders = sorted(
        [f for f in all_folders
         if "non_normative" not in f and "normative" in f
         and "_ID_" in f and f.split("_ID_")[-1].startswith("2026")],
        key=_folder_ts
    )

    sex_map: dict = {}

    for i, folder in enumerate(norm_2026_folders):
        if i < len(norm_2026_list):
            sex_map[folder] = norm_2026_list[i][1]

    for folder in all_folders:
        if folder in sex_map or "_ID_" not in folder:
            continue
        fid = folder.split("_ID_")[-1]

        if "non_normative" in folder:
            if fid in non_norm_map:
                sex_map[folder] = non_norm_map[fid]
            else:
                for kfid, sex in non_norm_map.items():
                    if len(fid) == len(kfid) and sum(a != b for a, b in zip(fid, kfid)) <= 1:
                        sex_map[folder] = sex
                        break
        elif "normative" in folder and fid in norm_2025_map:
            sex_map[folder] = norm_2025_map[fid]

    return sex_map


# ═══════════════════════════════════════════════════════════════════════════════
#  Age bin helpers
# ═══════════════════════════════════════════════════════════════════════════════

def assign_bin(age: float, edges: list) -> int:
    for i in range(len(edges) - 1):
        if edges[i] <= age < edges[i + 1]:
            return i
    return len(edges) - 2


# ═══════════════════════════════════════════════════════════════════════════════
#  Data loading
# ═══════════════════════════════════════════════════════════════════════════════

def load_all_data(age_map: dict) -> dict:
    """
    Load exercise data for all users (excluding the two youngest normative).
    Each entry is a (folder, df, age) triple.
    """
    data = {i: {"normative": [], "non_normative": []} for i in range(1, 6)}

    if not os.path.isdir(USERS_DIR):
        print(f"[ERROR] Users directory not found: {USERS_DIR}")
        return data

    for folder in sorted(os.listdir(USERS_DIR)):
        user_dir = os.path.join(USERS_DIR, folder)
        if not os.path.isdir(user_dir) or is_excluded(folder):
            continue

        if "non_normative" in folder:
            group = "non_normative"
        elif "normative" in folder:
            group = "normative"
        else:
            continue

        age = age_map.get(folder)

        for ex in range(1, 6):
            path = os.path.join(user_dir, f"dataCompilation{ex}.csv")
            if not os.path.exists(path):
                continue
            try:
                df = pd.read_csv(path, sep=";", low_memory=False)
                data[ex][group].append((folder, df, age))
            except Exception as e:
                print(f"  [ERROR] {folder}/dataCompilation{ex}.csv: {e}")

    return data


# ═══════════════════════════════════════════════════════════════════════════════
#  Plotting — side-by-side box plot (normative vs non-normative per age bin)
# ═══════════════════════════════════════════════════════════════════════════════

def _side_by_side_box(ax, norm_by_bin: list, nn_by_bin: list,
                      bin_labels: list, title: str, ylabel: str):
    """
    For each age bin, draw normative (blue) and non-normative (orange) boxes
    side by side. Bins with only 1 observation show as a single dot.
    """
    n = len(bin_labels)
    sep = 3.0          # units per bin group
    pw  = 0.80         # box width

    pos_n = [i * sep + 0.75 for i in range(n)]   # normative positions
    pos_p = [i * sep + 2.25 for i in range(n)]   # non-normative positions
    ticks = [i * sep + 1.50 for i in range(n)]   # tick centres

    def draw(positions, data_by_bin, color):
        clean = [np.array(pd.Series(d).dropna(), dtype=float) for d in data_by_bin]
        non_empty = [(p, d) for p, d in zip(positions, clean) if len(d) > 0]
        if not non_empty:
            return
        pps, dds = zip(*non_empty)

        bp = ax.boxplot(
            list(dds), positions=list(pps), widths=pw,
            patch_artist=True,
            medianprops=dict(color="black", linewidth=2),
            whiskerprops=dict(linewidth=1.2),
            capprops=dict(linewidth=1.2),
            flierprops=dict(marker=""),
            manage_ticks=False,
        )
        for box in bp["boxes"]:
            box.set_facecolor(color + "44")
            box.set_edgecolor(color)

        rng = np.random.default_rng(0)
        for p, d in zip(pps, dds):
            jitter = rng.uniform(-0.22, 0.22, len(d))
            ax.scatter(p + jitter, d, color=color, s=24, alpha=0.88,
                       zorder=3, edgecolors="white", linewidths=0.4)

    draw(pos_n, norm_by_bin, NORM_C)
    draw(pos_p, nn_by_bin,   NN_C)

    ax.set_xticks(ticks)
    ax.set_xticklabels(bin_labels, fontsize=6.5, rotation=25, ha="right")
    ax.set_ylabel(ylabel, fontsize=8)
    ax.set_title(title, fontsize=9, fontweight="bold", pad=4)
    ax.grid(axis="y", linestyle="--", alpha=0.35)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)


def _make_grid(n_panels: int, n_cols: int = 4):
    nc = min(n_cols, n_panels)
    nr = math.ceil(n_panels / nc)
    fig, axes = plt.subplots(nr, nc,
                             figsize=(nc * 4.0, nr * 3.6),
                             squeeze=False)
    return fig, axes.flatten()


# ═══════════════════════════════════════════════════════════════════════════════
#  Age distribution overview
# ═══════════════════════════════════════════════════════════════════════════════

def plot_age_distribution(age_map, edges, labels, out_dir):
    norm_ages = [
        age for folder, age in age_map.items()
        if "normative" in folder and "non_normative" not in folder
        and not is_excluded(folder) and age is not None
    ]
    nn_ages = [
        age for folder, age in age_map.items()
        if "non_normative" in folder and age is not None
    ]

    def counts(ages):
        c = [0] * len(labels)
        for a in ages:
            c[assign_bin(a, edges)] += 1
        return c

    nc = counts(norm_ages)
    pc = counts(nn_ages)

    x   = np.arange(len(labels))
    w   = 0.38
    fig, ax = plt.subplots(figsize=(12, 5))

    bn = ax.bar(x - w / 2, nc, width=w, color=NORM_C,   alpha=0.85,
                label=f"Normative (n={len(norm_ages)})",    edgecolor="white", linewidth=0.7)
    bp = ax.bar(x + w / 2, pc, width=w, color=NN_C,     alpha=0.85,
                label=f"Non-normative (n={len(nn_ages)})", edgecolor="white", linewidth=0.7)

    for bar, v in list(zip(bn, nc)) + list(zip(bp, pc)):
        if v > 0:
            ax.text(bar.get_x() + bar.get_width() / 2, v + 0.06, str(v),
                    ha="center", va="bottom", fontsize=9, fontweight="bold",
                    color=bar.get_facecolor())

    n_total = len(norm_ages) + len(nn_ages)
    ax.set_title(
        f"Age Group Distribution — Manual Boundaries  (n = {n_total})",
        fontsize=12, fontweight="bold"
    )
    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=9)
    ax.set_xlabel("Age group (years)", fontsize=10)
    ax.set_ylabel("Participants", fontsize=10)
    ax.legend(fontsize=10)
    ax.grid(axis="y", linestyle="--", alpha=0.35)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)


    fig.tight_layout()
    path = os.path.join(out_dir, "groups_age_distribution.png")
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    shutil.copy2(path, os.path.join(out_dir, "groups_overview.png"))
    print("  [OK] groups_age_distribution.png")


# ═══════════════════════════════════════════════════════════════════════════════
#  Gender distribution by age group
# ═══════════════════════════════════════════════════════════════════════════════

def plot_gender_by_group(age_map, sex_map, edges, labels, out_dir):
    """Stacked bar chart: M/F counts for normative and non-normative per age group."""
    n_groups = len(labels)

    # Collect counts
    cts = [{"norm": {"Male": 0, "Female": 0}, "nn": {"Male": 0, "Female": 0}}
           for _ in range(n_groups)]

    for folder, age in age_map.items():
        if is_excluded(folder) or age is None:
            continue
        sex = sex_map.get(folder, "")
        if sex not in ("Male", "Female"):
            continue
        b = assign_bin(float(age), edges)
        if "non_normative" in folder:
            cts[b]["nn"][sex] += 1
        elif "normative" in folder:
            cts[b]["norm"][sex] += 1

    norm_m = [cts[i]["norm"]["Male"]   for i in range(n_groups)]
    norm_f = [cts[i]["norm"]["Female"] for i in range(n_groups)]
    nn_m   = [cts[i]["nn"]["Male"]     for i in range(n_groups)]
    nn_f   = [cts[i]["nn"]["Female"]   for i in range(n_groups)]

    NORM_M_C = "#1565C0"   # dark blue
    NORM_F_C = "#90CAF9"   # light blue
    NN_M_C   = "#BF360C"   # dark orange
    NN_F_C   = "#FFAB91"   # light orange

    x = np.arange(n_groups)
    w = 0.35
    fig, ax = plt.subplots(figsize=(10, 6))

    # Stacked bars: normative (left of group), non-normative (right)
    ax.bar(x - w / 2, norm_m, width=w, color=NORM_M_C, alpha=0.92,
           label="Normative — Male",   edgecolor="white", linewidth=0.6)
    ax.bar(x - w / 2, norm_f, width=w, bottom=norm_m, color=NORM_F_C, alpha=0.92,
           label="Normative — Female", edgecolor="white", linewidth=0.6)
    ax.bar(x + w / 2, nn_m,   width=w, color=NN_M_C,   alpha=0.92,
           label="Non-Normative — Male",   edgecolor="white", linewidth=0.6)
    ax.bar(x + w / 2, nn_f,   width=w, bottom=nn_m,   color=NN_F_C, alpha=0.92,
           label="Non-Normative — Female", edgecolor="white", linewidth=0.6)

    # Annotations inside bars and totals on top
    for i in range(n_groups):
        n_tot = norm_m[i] + norm_f[i]
        p_tot = nn_m[i]   + nn_f[i]

        # Total on top
        if n_tot > 0:
            ax.text(i - w / 2, n_tot + 0.12, f"n={n_tot}",
                    ha="center", va="bottom", fontsize=9, fontweight="bold",
                    color=NORM_M_C)
        if p_tot > 0:
            ax.text(i + w / 2, p_tot + 0.12, f"n={p_tot}",
                    ha="center", va="bottom", fontsize=9, fontweight="bold",
                    color=NN_M_C)

        # M/F counts inside segments (only if segment tall enough)
        if norm_m[i] >= 1:
            ax.text(i - w / 2, norm_m[i] / 2, f"M:{norm_m[i]}",
                    ha="center", va="center", fontsize=8,
                    color="white", fontweight="bold")
        if norm_f[i] >= 1:
            ax.text(i - w / 2, norm_m[i] + norm_f[i] / 2, f"F:{norm_f[i]}",
                    ha="center", va="center", fontsize=8,
                    color="#0d3780", fontweight="bold")
        if nn_m[i] >= 1:
            ax.text(i + w / 2, nn_m[i] / 2, f"M:{nn_m[i]}",
                    ha="center", va="center", fontsize=8,
                    color="white", fontweight="bold")
        if nn_f[i] >= 1:
            ax.text(i + w / 2, nn_m[i] + nn_f[i] / 2, f"F:{nn_f[i]}",
                    ha="center", va="center", fontsize=8,
                    color="#7f1f03", fontweight="bold")

    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=10)
    ax.set_xlabel("Age Group (years)", fontsize=11)
    ax.set_ylabel("Number of Participants", fontsize=11)
    ax.set_title("Gender Distribution by Age Group and Cohort", fontsize=12, fontweight="bold")
    # Headroom so the legend clears the tallest bars (avoids overlap at large fonts).
    _ymax = max([norm_m[i] + norm_f[i] for i in range(n_groups)] +
                [nn_m[i] + nn_f[i] for i in range(n_groups)] + [1])
    ax.set_ylim(0, _ymax * 1.35)
    ax.legend(fontsize=9, loc="upper left")
    ax.grid(axis="y", linestyle="--", alpha=0.35)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    fig.tight_layout()
    path = os.path.join(out_dir, "groups_gender_distribution.png")
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print("  [OK] groups_gender_distribution.png")


# ═══════════════════════════════════════════════════════════════════════════════
#  Per-exercise figures
# ═══════════════════════════════════════════════════════════════════════════════

def _col_mean(df, col):
    return pd.to_numeric(df.get(col, pd.Series(dtype=float)), errors="coerce").mean()


def _active_mean(df, suf):
    hand = pu.dominant_hand(df)
    return _col_mean(df, f"{hand}_{suf}")


def _make_bins(users, col_fn, edges):
    """Group per-user metric values by age bin."""
    n = len(edges) - 1
    bins = [[] for _ in range(n)]
    for folder, df, age in users:
        if age is None:
            continue
        val = col_fn(df)
        if pd.isna(val):
            continue
        bins[assign_bin(float(age), edges)].append(float(val))
    return bins


def plot_exercise_groups(ex_data, ex_num, edges, labels, out_dir):
    norm_list = ex_data.get("normative",     [])
    nn_list   = ex_data.get("non_normative", [])

    # pu.is_bimanual expects 2-tuples
    ex_data_2 = {
        "normative":     [(f, df) for f, df, _ in norm_list],
        "non_normative": [(f, df) for f, df, _ in nn_list],
    }
    bimanual  = pu.is_bimanual(ex_data_2)
    hand_note = "Both Hands" if bimanual else "Active Hand"

    legend_handles = [
        mpatches.Patch(color=NORM_C, alpha=0.85, label="Normative"),
        mpatches.Patch(color=NN_C,   alpha=0.85, label="Non-normative"),
    ]

    def save(fig, name, suptitle):
        fig.legend(handles=legend_handles, loc="upper right",
                   fontsize=9, title="Group", title_fontsize=8)
        fig.suptitle(suptitle, fontsize=11, fontweight="bold", y=1.02)
        fig.tight_layout()
        path = os.path.join(out_dir, name)
        fig.savefig(path, dpi=150, bbox_inches="tight")
        plt.close(fig)
        print(f"  [OK] {name}")

    # ── Speed ─────────────────────────────────────────────────────────────────
    if bimanual:
        speed_items = [
            ("L Speed",   lambda df: _col_mean(df, "Left_hand_speed"),   "mm/s"),
            ("L Speed X", lambda df: _col_mean(df, "Left_hand_speed_x"), "mm/s"),
            ("L Speed Y", lambda df: _col_mean(df, "Left_hand_speed_y"), "mm/s"),
            ("L Speed Z", lambda df: _col_mean(df, "Left_hand_speed_z"), "mm/s"),
            ("R Speed",   lambda df: _col_mean(df, "Right_hand_speed"),   "mm/s"),
            ("R Speed X", lambda df: _col_mean(df, "Right_hand_speed_x"), "mm/s"),
            ("R Speed Y", lambda df: _col_mean(df, "Right_hand_speed_y"), "mm/s"),
            ("R Speed Z", lambda df: _col_mean(df, "Right_hand_speed_z"), "mm/s"),
        ]
    else:
        speed_items = [
            ("Speed",   lambda df: _active_mean(df, "hand_speed"),   "mm/s"),
            ("Speed X", lambda df: _active_mean(df, "hand_speed_x"), "mm/s"),
            ("Speed Y", lambda df: _active_mean(df, "hand_speed_y"), "mm/s"),
            ("Speed Z", lambda df: _active_mean(df, "hand_speed_z"), "mm/s"),
        ]
    fig, axes = _make_grid(len(speed_items), 4)
    for i, (lbl, fn, unit) in enumerate(speed_items):
        _side_by_side_box(axes[i],
                          _make_bins(norm_list, fn, edges),
                          _make_bins(nn_list,   fn, edges),
                          labels, lbl, unit)
    for j in range(len(speed_items), len(axes)):
        axes[j].set_visible(False)
    save(fig, f"exercise{ex_num}_groups_speed.png",
         f"Exercise {ex_num} — Hand Speed by Age Group  [{hand_note}]")

    # ── Orientation ───────────────────────────────────────────────────────────
    if bimanual:
        orient_items = [
            ("L Normal X", lambda df: _col_mean(df, "Left_hand_normal_x"),  "unit"),
            ("L Normal Y", lambda df: _col_mean(df, "Left_hand_normal_y"),  "unit"),
            ("L Normal Z", lambda df: _col_mean(df, "Left_hand_normal_z"),  "unit"),
            ("R Normal X", lambda df: _col_mean(df, "Right_hand_normal_x"), "unit"),
            ("R Normal Y", lambda df: _col_mean(df, "Right_hand_normal_y"), "unit"),
            ("R Normal Z", lambda df: _col_mean(df, "Right_hand_normal_z"), "unit"),
        ]
    else:
        orient_items = [
            ("Normal X", lambda df: _active_mean(df, "hand_normal_x"), "unit"),
            ("Normal Y", lambda df: _active_mean(df, "hand_normal_y"), "unit"),
            ("Normal Z", lambda df: _active_mean(df, "hand_normal_z"), "unit"),
        ]
    fig, axes = _make_grid(len(orient_items), 3)
    for i, (lbl, fn, unit) in enumerate(orient_items):
        _side_by_side_box(axes[i],
                          _make_bins(norm_list, fn, edges),
                          _make_bins(nn_list,   fn, edges),
                          labels, lbl, unit)
    for j in range(len(orient_items), len(axes)):
        axes[j].set_visible(False)
    save(fig, f"exercise{ex_num}_groups_orientation.png",
         f"Exercise {ex_num} — Palm Orientation by Age Group  [{hand_note}]")

    # ── Position ──────────────────────────────────────────────────────────────
    if bimanual:
        pos_items = [
            ("L Palm X", lambda df: _col_mean(df, "Left_hand_palm_position_x"),  "mm"),
            ("L Palm Y", lambda df: _col_mean(df, "Left_hand_palm_position_y"),  "mm"),
            ("L Palm Z", lambda df: _col_mean(df, "Left_hand_palm_position_z"),  "mm"),
            ("R Palm X", lambda df: _col_mean(df, "Right_hand_palm_position_x"), "mm"),
            ("R Palm Y", lambda df: _col_mean(df, "Right_hand_palm_position_y"), "mm"),
            ("R Palm Z", lambda df: _col_mean(df, "Right_hand_palm_position_z"), "mm"),
        ]
    else:
        pos_items = [
            ("Palm X", lambda df: _active_mean(df, "hand_palm_position_x"), "mm"),
            ("Palm Y", lambda df: _active_mean(df, "hand_palm_position_y"), "mm"),
            ("Palm Z", lambda df: _active_mean(df, "hand_palm_position_z"), "mm"),
        ]
    fig, axes = _make_grid(len(pos_items), 3)
    for i, (lbl, fn, unit) in enumerate(pos_items):
        _side_by_side_box(axes[i],
                          _make_bins(norm_list, fn, edges),
                          _make_bins(nn_list,   fn, edges),
                          labels, lbl, unit)
    for j in range(len(pos_items), len(axes)):
        axes[j].set_visible(False)
    save(fig, f"exercise{ex_num}_groups_position.png",
         f"Exercise {ex_num} — Palm Position by Age Group  [{hand_note}]")

    # ── Grip ──────────────────────────────────────────────────────────────────
    if bimanual:
        grip_items = [
            ("L Grip", lambda df: _col_mean(df, "Left_hand_grab_strength"),  "0-1"),
            ("R Grip", lambda df: _col_mean(df, "Right_hand_grab_strength"), "0-1"),
        ]
    else:
        grip_items = [
            ("Grip", lambda df: _active_mean(df, "hand_grab_strength"), "0-1"),
        ]
    fig, axes = _make_grid(len(grip_items), 2)
    for i, (lbl, fn, unit) in enumerate(grip_items):
        _side_by_side_box(axes[i],
                          _make_bins(norm_list, fn, edges),
                          _make_bins(nn_list,   fn, edges),
                          labels, lbl, unit)
    for j in range(len(grip_items), len(axes)):
        axes[j].set_visible(False)
    save(fig, f"exercise{ex_num}_groups_grip.png",
         f"Exercise {ex_num} — Grip Strength by Age Group  [{hand_note}]")


# ═══════════════════════════════════════════════════════════════════════════════
#  Main
# ═══════════════════════════════════════════════════════════════════════════════

def main():
    print("=" * 60)
    print("  Age-Group Analysis  —  Manual Boundaries  (V2)")
    print(f"  Excluding normative file_ids: {EXCLUDE_IDS}")
    print("=" * 60)

    # ── Age / sex maps ─────────────────────────────────────────────────────────
    print("\n-- Building age & sex maps --")
    age_map = build_age_map()
    sex_map = build_sex_map()
    print(f"  Age map: {len(age_map)} folders  |  Sex map: {len(sex_map)} folders")

    combined_ages = sorted([
        age for folder, age in age_map.items()
        if not is_excluded(folder) and age is not None
    ])
    n_norm = sum(1 for f in age_map if "normative" in f and "non_normative" not in f and not is_excluded(f))
    n_nn   = sum(1 for f in age_map if "non_normative" in f)

    print(f"\n  Combined participants: {len(combined_ages)}")
    print(f"  Normative: {n_norm}  |  Non-normative: {n_nn}")
    print(f"  Age range: {int(min(combined_ages))}\u2013{int(max(combined_ages))}")

    # ── Load exercise data (shared across schemes) ─────────────────────────────
    print("\n-- Loading exercise data --")
    data = load_all_data(age_map)
    for ex in range(1, 6):
        n  = len(data[ex]["normative"])
        nn = len(data[ex]["non_normative"])
        print(f"  Exercise {ex}: {n:2d} normative | {nn:2d} non-normative")

    # ── Compute statistical schemes from real age data ────────────────────────
    print("\n-- Computing statistical binning schemes (Sturges, Scott, FD) --")
    dynamic_schemes = compute_binning_schemes(combined_ages)
    all_schemes = AGE_SCHEMES + dynamic_schemes

    # ── Run all age-division schemes ───────────────────────────────────────────
    for subfolder, edges, labels, scheme_title in all_schemes:
        plots_dir = os.path.join(PLOTS_BASE, subfolder)
        web_plots = os.path.join(WEB_BASE, subfolder)

        print(f"\n{'=' * 60}")
        print(f"  Scheme: {scheme_title}  ({len(labels)} groups: {labels})")
        print(f"  Output: {plots_dir}")
        print(f"{'=' * 60}")

        os.makedirs(plots_dir, exist_ok=True)

        # Age distribution
        print("\n-- Age distribution --")
        plot_age_distribution(age_map, edges, labels, plots_dir)

        # Gender by group
        print("\n-- Gender distribution by group --")
        plot_gender_by_group(age_map, sex_map, edges, labels, plots_dir)

        # Per-exercise plots
        print("\n-- Exercise group plots --")
        for ex in range(1, 6):
            ex_data = data[ex]
            if not ex_data["normative"] and not ex_data["non_normative"]:
                print(f"\n  Exercise {ex}: no data — skipped")
                continue
            print(f"\n  Exercise {ex}:")
            try:
                plot_exercise_groups(ex_data, ex, edges, labels, plots_dir)
            except Exception as e:
                import traceback
                print(f"    [ERR] {e}")
                traceback.print_exc()

        # Copy to web assets
        print("\n-- Copying to web assets --")
        if os.path.exists(web_plots):
            shutil.rmtree(web_plots)
        shutil.copytree(plots_dir, web_plots)
        print(f"  [OK] {subfolder}  ->  {web_plots}")

    subfolders = [s for s, _, _, _ in all_schemes]
    print(f"\nDone.  Plots saved to: {PLOTS_BASE}/[{', '.join(subfolders)}]")
    print(f"       Web assets:     {WEB_BASE}/[{', '.join(subfolders)}]")


if __name__ == "__main__":
    main()

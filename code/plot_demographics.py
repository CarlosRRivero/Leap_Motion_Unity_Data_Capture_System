"""
plot_demographics.py – Age and gender comparison between groups.

Groups:
  Normative     – Parkinson_Phase starts with "Stage 0.0"  (any Stage 0 variant)
  Non-normative – all other stages (Parkinson patients)

Produces:
  demographics_age.png      – box + strip plot of age by group
  demographics_gender.png   – gender distribution by group (pie charts)
  demographics_overview.png – combined one-page summary

Output: Paper/Plots/demographics/
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

try:
    from scipy import stats as scipy_stats
    HAS_SCIPY = True
except ImportError:
    HAS_SCIPY = False

SCRIPTS_DIR = os.path.dirname(os.path.abspath(__file__))
PAPER_DIR   = os.path.normpath(os.path.join(SCRIPTS_DIR, ".."))
CSV_PATH    = os.path.join(PAPER_DIR, "User_Stats_Total.csv")
USERS_DIR   = os.path.join(PAPER_DIR, "Users")
OUT_DIR     = os.path.join(PAPER_DIR, "Plots", "demographics")

NORM_COLOR     = "#2196F3"
NON_NORM_COLOR = "#E64A19"
GENDER_COLORS  = ["#42a5f5", "#ef9a9a", "#ab47bc"]  # M, F, other


def _folder_counts(users_dir):
    """
    Returns (norm_ids, nn_ids, n_2026_norm) where:
      norm_ids     – set of normative folder File_ID strings
      nn_ids       – set of non-normative folder File_ID strings
      n_2026_norm  – count of 2026 normative folders (IDs starting with "2026")
    """
    norm_ids, nn_ids = set(), set()
    if not os.path.isdir(users_dir):
        return norm_ids, nn_ids, 0
    for name in os.listdir(users_dir):
        if "_ID_" not in name or not os.path.isdir(os.path.join(users_dir, name)):
            continue
        fid = name.split("_ID_")[-1]
        if "non_normative" in name:
            nn_ids.add(fid)
        else:
            norm_ids.add(fid)
    n_2026 = sum(1 for f in norm_ids if str(f).startswith("2026"))
    return norm_ids, nn_ids, n_2026


def _near_match(fid_str, folder_ids):
    """True if fid_str matches any folder ID exactly or with <=1 digit difference."""
    if fid_str in folder_ids:
        return True
    for fid2 in folder_ids:
        if len(fid_str) == len(fid2) and sum(a != b for a, b in zip(fid_str, fid2)) <= 1:
            return True
    return False


def classify(df):
    """
    Returns (norm_df, non_norm_df) restricted to users with exercise data.

    Non-normative: near-match against folder IDs (handles 1-digit typos).
    Normative:
      - Pre-2026 batches: exact File_ID match against folder IDs.
      - 2026 batch (Start_Date year=2026): keep first N rows by Start_Date,
        where N = number of 2026 normative folders.
    """
    norm_folder_ids, nn_folder_ids, n_2026_norm = _folder_counts(USERS_DIR)
    is_norm = df["Parkinson_Phase"].astype(str).str.startswith("Stage 0.0")

    # ── Non-normative ──────────────────────────────────────────────────────────
    nn_df = df[~is_norm].copy()
    if nn_folder_ids and "File_ID" in nn_df.columns:
        nn_df = nn_df[nn_df["File_ID"].astype(str).apply(
            lambda x: _near_match(x, nn_folder_ids)
        )].copy()

    # ── Normative ──────────────────────────────────────────────────────────────
    norm_df = df[is_norm].copy()
    if norm_folder_ids and "File_ID" in norm_df.columns:
        # Identify 2026 batch by Start_Date year
        is_2026 = norm_df["Start_Date"].astype(str).str.startswith("2026")
        pre2026 = norm_df[~is_2026]
        batch26 = norm_df[is_2026]

        # Pre-2026: keep only rows whose File_ID matches a folder
        pre2026_keep = pre2026[
            pre2026["File_ID"].astype(str).isin(norm_folder_ids)
        ]
        # 2026 batch: keep first N by Start_Date (positional match to folders)
        batch26_keep = (batch26.sort_values("Start_Date")
                               .head(n_2026_norm))

        norm_df = pd.concat([pre2026_keep, batch26_keep], ignore_index=True)

    return norm_df, nn_df


def _box_strip(ax, data_list, labels, colors, ylabel, title, stat_text=None):
    rng = np.random.default_rng(42)
    bp  = ax.boxplot(data_list, patch_artist=True, widths=0.45,
                     medianprops=dict(color="black", linewidth=2),
                     flierprops=dict(marker="o", markersize=4, alpha=0.5))
    for patch, col in zip(bp["boxes"], colors):
        patch.set_facecolor(col)
        patch.set_alpha(0.6)
    for i, (data, col) in enumerate(zip(data_list, colors)):
        x = rng.uniform(i + 0.72, i + 1.28, size=len(data))
        ax.scatter(x, data, color=col, alpha=0.7, s=45,
                   zorder=3, edgecolors="white", linewidth=0.5)
        if len(data):
            ax.text(i + 1, max(data) + 1.5,
                    f"μ = {np.mean(data):.1f}\nn = {len(data)}",
                    ha="center", fontsize=9, color=col, fontweight="bold")
    ax.set_xticks(range(1, len(labels) + 1))
    ax.set_xticklabels(labels, fontsize=10)
    ax.set_ylabel(ylabel, fontsize=10)
    ax.set_title(title, fontsize=11, fontweight="bold")
    ax.grid(axis="y", linestyle="--", alpha=0.35)
    if stat_text:
        ax.text(0.98, 0.02, stat_text, transform=ax.transAxes,
                ha="right", va="bottom", fontsize=8, color="dimgray",
                bbox=dict(boxstyle="round,pad=0.3", facecolor="lightyellow",
                          edgecolor="gray", alpha=0.8))


def _pie(ax, series, title, color):
    counts = series.value_counts()
    wedge_cols = GENDER_COLORS[:len(counts)]
    if len(counts) == 0:
        ax.axis("off")
        return
    wedges, texts, autotexts = ax.pie(
        counts.values, labels=counts.index,
        autopct="%1.0f%%", colors=wedge_cols,
        startangle=90,
        wedgeprops=dict(edgecolor="white", linewidth=1.8),
    )
    for at in autotexts:
        at.set_fontsize(11)
        at.set_fontweight("bold")
    ax.set_title(title, fontsize=11, fontweight="bold", color=color)


def run(out_dir=None):
    if out_dir is None:
        out_dir = OUT_DIR
    os.makedirs(out_dir, exist_ok=True)

    if not os.path.exists(CSV_PATH):
        print(f"  [WARN] User stats CSV not found: {CSV_PATH}")
        return

    df       = pd.read_csv(CSV_PATH, sep=";", encoding="utf-8-sig")
    norm, nn = classify(df)

    age_norm = norm["Age"].dropna().values.astype(float)
    age_nn   = nn["Age"].dropna().values.astype(float)

    GROUP_LABELS = ["Normative", "Non-Normative"]
    COLORS       = [NORM_COLOR, NON_NORM_COLOR]

    # ── Figure 1: Age comparison ──────────────────────────────────────────────
    fig, ax = plt.subplots(figsize=(8, 6))
    _box_strip(ax, [age_norm, age_nn], GROUP_LABELS, COLORS,
               "Age (years)", "Age by Group", stat_text=None)

    handles = [mpatches.Patch(color=c, alpha=0.7, label=l)
               for c, l in zip(COLORS, GROUP_LABELS)]
    ax.legend(handles=handles, fontsize=9)
    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, "demographics_age.png"), dpi=150, bbox_inches="tight")
    plt.close(fig)

    # ── Figure 2: Gender distribution ─────────────────────────────────────────
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 5))
    _pie(ax1, norm["Sex"], f"Normative  (n={len(norm)})", NORM_COLOR)
    _pie(ax2, nn["Sex"],   f"Non-Normative  (n={len(nn)})", NON_NORM_COLOR)
    fig.suptitle("Gender Distribution by Group", fontsize=13, fontweight="bold")
    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, "demographics_gender.png"), dpi=150, bbox_inches="tight")
    plt.close(fig)

    # ── Figure 3: Combined overview ────────────────────────────────────────────
    fig = plt.figure(figsize=(16, 6))
    gs  = fig.add_gridspec(1, 3, width_ratios=[1.6, 1, 1], wspace=0.3)
    ax_age = fig.add_subplot(gs[0])
    ax_g1  = fig.add_subplot(gs[1])
    ax_g2  = fig.add_subplot(gs[2])

    _box_strip(ax_age, [age_norm, age_nn], GROUP_LABELS, COLORS,
               "Age (years)", "Age by Group")
    handles = [mpatches.Patch(color=c, alpha=0.7, label=l)
               for c, l in zip(COLORS, GROUP_LABELS)]
    ax_age.legend(handles=handles, fontsize=9)

    _pie(ax_g1, norm["Sex"], f"Normative\n(n={len(norm)})", NORM_COLOR)
    _pie(ax_g2, nn["Sex"],   f"Non-Normative\n(n={len(nn)})", NON_NORM_COLOR)

    fig.suptitle("Demographics: Normative vs Non-Normative",
                 fontsize=14, fontweight="bold", y=1.01)
    fig.savefig(os.path.join(out_dir, "demographics_overview.png"),
                dpi=150, bbox_inches="tight")
    plt.close(fig)

    # ── Figure 4: Parkinson stage distribution (PD patients only, no Stage 0) ──
    fig, ax = plt.subplots(figsize=(10, 5))
    stage_labels = {
        "Stage 1.0":  "1.0",
        "Stage 1.5":  "1.5",
        "Stage 2.0":  "2.0",
        "Stage 3.0":  "3.0",
    }
    # Exclude Stage 0 (normative/healthy users don't have Parkinson's)
    pd_only = df[~df["Parkinson_Phase"].astype(str).str.startswith("Stage 0")]
    phase_short = pd_only["Parkinson_Phase"].astype(str).apply(
        lambda x: next((v for k, v in stage_labels.items() if x.startswith(k[:12])), x[:12])
    )
    counts = phase_short.value_counts().sort_index()
    ax.bar(counts.index, counts.values, color=NON_NORM_COLOR, edgecolor="white",
           linewidth=0.8, alpha=0.85)
    for i, v in enumerate(counts.values):
        ax.text(i, v + 0.1, str(v), ha="center", va="bottom", fontsize=9, fontweight="bold")

    ax.set_xlabel("Parkinson Stage", fontsize=11)
    ax.set_ylabel("Number of participants", fontsize=11)
    ax.set_title("Participant Distribution by Parkinson Stage (PD Patients)",
                 fontsize=12, fontweight="bold")
    ax.grid(axis="y", linestyle="--", alpha=0.35)
    ax.tick_params(axis="x", rotation=20)

    nonnp = mpatches.Patch(color=NON_NORM_COLOR, alpha=0.85, label="PD Patients")
    ax.legend(handles=[nonnp], fontsize=9)

    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, "demographics_stages.png"), dpi=150, bbox_inches="tight")
    plt.close(fig)


if __name__ == "__main__":
    run()

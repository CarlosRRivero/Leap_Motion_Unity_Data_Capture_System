"""
plot_utils.py – shared utilities for exercise analysis plots.
Provides data loading helpers, per-user aggregation, and box-plot primitives.
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

# ── Colour palette ────────────────────────────────────────────────────────────
NORM_COLOR     = "#2196F3"   # blue
NON_NORM_COLOR = "#E64A19"   # deep orange
NORM_LABEL     = "Normative"
NON_NORM_LABEL = "Non-normative"

# ── Hand detection ────────────────────────────────────────────────────────────

def dominant_hand(df):
    """Return 'Left' or 'Right': whichever hand has the greater total |speed|."""
    left  = pd.to_numeric(df.get("Left_hand_speed",  pd.Series(dtype=float)), errors="coerce").abs().sum()
    right = pd.to_numeric(df.get("Right_hand_speed", pd.Series(dtype=float)), errors="coerce").abs().sum()
    return "Left" if left >= right else "Right"


def is_bimanual(exercise_data):
    """
    Return True when the majority of users have BOTH hands active in this exercise.
    A hand is considered active if its total |speed| > 0.
    """
    users = exercise_data.get("normative", []) + exercise_data.get("non_normative", [])
    if not users:
        return False
    bimanual = sum(
        1 for _, df in users
        if pd.to_numeric(df.get("Left_hand_speed",  pd.Series(dtype=float)), errors="coerce").abs().sum() > 0
        and pd.to_numeric(df.get("Right_hand_speed", pd.Series(dtype=float)), errors="coerce").abs().sum() > 0
    )
    return bimanual > len(users) / 2


# ── Per-user aggregation ──────────────────────────────────────────────────────

def user_means(user_list, columns):
    """
    Compute the time-series mean of each column, per user.
    Returns a DataFrame with one row per user.
    """
    rows = []
    for user_name, df in user_list:
        row = {"user": user_name}
        for col in columns:
            row[col] = pd.to_numeric(df[col], errors="coerce").mean() if col in df.columns else np.nan
        rows.append(row)
    return pd.DataFrame(rows) if rows else pd.DataFrame(columns=["user"] + list(columns))


def user_stds(user_list, columns):
    """Standard-deviation variant of user_means (captures movement variability)."""
    rows = []
    for user_name, df in user_list:
        row = {"user": user_name}
        for col in columns:
            row[col] = pd.to_numeric(df[col], errors="coerce").std() if col in df.columns else np.nan
        rows.append(row)
    return pd.DataFrame(rows) if rows else pd.DataFrame(columns=["user"] + list(columns))


def active_hand_means(user_list, suffixes):
    """
    Single-hand exercises: detect each user's dominant hand, then compute
    the per-user mean for <hand>_<suffix> columns.
    `suffixes` are column name parts after the 'Left_' / 'Right_' prefix,
    e.g. ['hand_speed', 'hand_normal_x'].
    """
    rows = []
    for user_name, df in user_list:
        hand = dominant_hand(df)
        row = {"user": user_name}
        for suf in suffixes:
            col = f"{hand}_{suf}"
            row[suf] = pd.to_numeric(df[col], errors="coerce").mean() if col in df.columns else np.nan
        rows.append(row)
    return pd.DataFrame(rows) if rows else pd.DataFrame(columns=["user"] + list(suffixes))


def active_hand_stds(user_list, suffixes):
    """Standard-deviation variant of active_hand_means."""
    rows = []
    for user_name, df in user_list:
        hand = dominant_hand(df)
        row = {"user": user_name}
        for suf in suffixes:
            col = f"{hand}_{suf}"
            row[suf] = pd.to_numeric(df[col], errors="coerce").std() if col in df.columns else np.nan
        rows.append(row)
    return pd.DataFrame(rows) if rows else pd.DataFrame(columns=["user"] + list(suffixes))


# ── Plotting primitives ───────────────────────────────────────────────────────

def box_compare(ax, norm_vals, non_norm_vals, title, ylabel="Mean value", show_points=True):
    """
    Draw a side-by-side box plot on `ax` comparing normative vs non-normative.
    If a group has no data the corresponding box is omitted gracefully.
    """
    norm_clean = np.array(pd.Series(norm_vals).dropna(), dtype=float)
    non_clean  = np.array(pd.Series(non_norm_vals).dropna(), dtype=float)

    # Both empty → show placeholder text
    if len(norm_clean) == 0 and len(non_clean) == 0:
        ax.text(0.5, 0.5, "No data", ha="center", va="center",
                transform=ax.transAxes, fontsize=9, color="grey")
        ax.set_title(title, fontsize=9, fontweight="bold", pad=4)
        ax.axis("off")
        return

    groups  = []
    pos     = []
    colors  = []
    labels  = []
    for vals, p, color, lbl in [
        (norm_clean, 1, NORM_COLOR,     NORM_LABEL),
        (non_clean,  2, NON_NORM_COLOR, NON_NORM_LABEL),
    ]:
        if len(vals) > 0:
            groups.append(vals)
            pos.append(p)
            colors.append(color)
            labels.append(lbl)

    bp = ax.boxplot(
        groups,
        positions=pos,
        widths=0.5,
        patch_artist=True,
        medianprops=dict(color="black", linewidth=2),
        whiskerprops=dict(linewidth=1.5),
        capprops=dict(linewidth=1.5),
        flierprops=dict(marker=""),
    )
    for box, color in zip(bp["boxes"], colors):
        box.set_facecolor(color + "55")
        box.set_edgecolor(color)

    if show_points:
        rng = np.random.default_rng(0)
        for p, vals, color in zip(pos, groups, colors):
            jitter = rng.uniform(-0.12, 0.12, len(vals))
            ax.scatter(p + jitter, vals, color=color, s=22, alpha=0.85,
                       zorder=3, edgecolors="white", linewidths=0.4)

    ax.set_xticks(pos)
    ax.set_xticklabels(labels, fontsize=8)
    ax.set_ylabel(ylabel, fontsize=8)
    ax.set_title(title, fontsize=9, fontweight="bold", pad=4)
    ax.grid(axis="y", linestyle="--", alpha=0.4)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)


def legend_handles():
    return [
        mpatches.Patch(color=NORM_COLOR,     alpha=0.8, label=NORM_LABEL),
        mpatches.Patch(color=NON_NORM_COLOR, alpha=0.8, label=NON_NORM_LABEL),
    ]


def save_fig(fig, path, suptitle=None):
    if suptitle:
        fig.suptitle(suptitle, fontsize=12, fontweight="bold", y=1.02)
    fig.tight_layout()
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return path


def make_axes_grid(n_panels, n_cols=4, panel_size=(3.5, 3.5)):
    """Create a figure with enough subplots for n_panels, up to n_cols wide."""
    import math
    nc = min(n_cols, n_panels)
    nr = math.ceil(n_panels / nc)
    fig, axes = plt.subplots(nr, nc, figsize=(nc * panel_size[0], nr * panel_size[1]),
                              squeeze=False)
    return fig, axes.flatten()

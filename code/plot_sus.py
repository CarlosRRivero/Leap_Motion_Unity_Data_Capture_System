"""
plot_sus.py – SUS (System Usability Scale) statistics visualisation.

Reads SUS_Stats_Total.csv and produces:
  1. sus_response_distribution.png  – stacked 100% bar chart per question
  2. sus_scores.png                 – per-question mean response + SUS gauge

Output: Paper/Plots/sus/
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

SCRIPTS_DIR = os.path.dirname(os.path.abspath(__file__))
PAPER_DIR   = os.path.normpath(os.path.join(SCRIPTS_DIR, ".."))
CSV_PATH    = os.path.join(PAPER_DIR, "SUS_Stats_Total.csv")
OUT_DIR     = os.path.join(PAPER_DIR, "Plots", "sus")

SCALE_COLS = [
    "1 - Strongly disagree",
    "2 - Disagree",
    "3 - Neutral",
    "4 - Agree",
    "5 - Strongly agree",
]
SCALE_COLORS = ["#d32f2f", "#ef9a9a", "#fff176", "#a5d6a7", "#388e3c"]
SHORT_LABELS  = [f"Q{i}" for i in range(1, 11)]

# Odd questions are positive (1,3,5,7,9); even are negative (2,4,6,8,10)
ODD_COLOR  = "#388e3c"
EVEN_COLOR = "#d32f2f"


def _weighted_mean(row):
    total = sum(row[c] for c in SCALE_COLS)
    if total == 0:
        return 3.0
    return sum((i + 1) * row[SCALE_COLS[i]] for i in range(5)) / total


def compute_sus_score(df):
    """
    Standard SUS calculation from aggregated counts.
    Odd questions: score contribution = mean_response - 1
    Even questions: score contribution = 5 - mean_response
    Final = sum * 2.5
    """
    sus = 0.0
    for q_idx, (_, row) in enumerate(df.iterrows()):
        wm = _weighted_mean(row)
        q_num = q_idx + 1
        sus += (wm - 1) if q_num % 2 == 1 else (5 - wm)
    return sus * 2.5


def _grade(score):
    if score >= 90:  return "A+", "Excellent"
    if score >= 80.3: return "A", "Excellent"
    if score >= 68:  return "B",  "Good"
    if score >= 51:  return "C",  "Okay"
    if score >= 25:  return "D",  "Poor"
    return "F", "Awful"


def run(out_dir=None):
    if out_dir is None:
        out_dir = OUT_DIR
    os.makedirs(out_dir, exist_ok=True)

    if not os.path.exists(CSV_PATH):
        print(f"  [WARN] SUS CSV not found: {CSV_PATH}")
        return

    df = pd.read_csv(CSV_PATH)
    totals = df[SCALE_COLS].sum(axis=1)

    # ── Figure 1: 100% stacked bar – response distribution ────────────────────
    fig, ax = plt.subplots(figsize=(14, 6))
    bottoms = np.zeros(len(df))

    for i, (col, color) in enumerate(zip(SCALE_COLS, SCALE_COLORS)):
        pcts = (df[col] / totals * 100).fillna(0).values
        bars = ax.bar(SHORT_LABELS, pcts, bottom=bottoms,
                      color=color, label=col,
                      edgecolor="white", linewidth=0.8)
        for bar, pct in zip(bars, pcts):
            if pct > 6:
                ax.text(bar.get_x() + bar.get_width() / 2,
                        bar.get_y() + bar.get_height() / 2,
                        f"{pct:.0f}%", ha="center", va="center",
                        fontsize=8, color="black", fontweight="bold")
        bottoms += pcts

    ax.set_xlabel("Question", fontsize=11)
    ax.set_ylabel("Percentage of responses (%)", fontsize=11)
    ax.set_title("SUS – Response Distribution per Question",
                 fontsize=13, fontweight="bold", pad=12)
    ax.legend(loc="upper right", fontsize=9, framealpha=0.9)
    ax.set_ylim(0, 102)
    ax.grid(axis="y", linestyle="--", alpha=0.3)
    ax.tick_params(axis="both", labelsize=10)

    # Annotate total respondents
    n_total = int(totals.max())
    ax.text(0.01, 0.98, f"n = {n_total} respondents per question",
            transform=ax.transAxes, fontsize=9, va="top", color="gray")

    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, "sus_response_distribution.png"), dpi=150, bbox_inches="tight")
    plt.close(fig)

    # ── Figure 2: mean response bars + SUS score gauge ────────────────────────
    fig, (ax_bars, ax_gauge) = plt.subplots(1, 2, figsize=(15, 6),
                                            gridspec_kw={"width_ratios": [1.4, 1]})

    # Left: per-question weighted mean
    means  = [_weighted_mean(row) for _, row in df.iterrows()]
    colors = [ODD_COLOR if (i % 2 == 0) else EVEN_COLOR for i in range(10)]
    bars   = ax_bars.bar(SHORT_LABELS, means, color=colors,
                         edgecolor="white", linewidth=0.8, alpha=0.85)

    for bar, m in zip(bars, means):
        ax_bars.text(bar.get_x() + bar.get_width() / 2,
                     bar.get_height() + 0.04,
                     f"{m:.2f}", ha="center", va="bottom", fontsize=8)

    ax_bars.axhline(3, color="gray", linestyle="--", linewidth=1.2, label="Neutral (3.0)")
    ax_bars.set_ylim(1, 5.4)
    ax_bars.set_xlabel("Question", fontsize=11)
    ax_bars.set_ylabel("Weighted mean response (1–5)", fontsize=11)
    ax_bars.set_title("Mean Response per Question", fontsize=11, fontweight="bold")
    ax_bars.tick_params(axis="both", labelsize=10)
    ax_bars.grid(axis="y", linestyle="--", alpha=0.3)

    odd_patch  = mpatches.Patch(color=ODD_COLOR,  alpha=0.85, label="Positive phrasing (odd)")
    even_patch = mpatches.Patch(color=EVEN_COLOR, alpha=0.85, label="Negative phrasing (even)")
    neutral_l  = plt.Line2D([0], [0], color="gray", linestyle="--", label="Neutral (3.0)")
    ax_bars.legend(handles=[odd_patch, even_patch, neutral_l], fontsize=9, loc="upper right")

    # Right: SUS gauge
    sus_score = compute_sus_score(df)
    grade, adj = _grade(sus_score)

    grade_bands = [
        (0,   25,   "#d32f2f", "F\n0–25"),
        (25,  51,   "#ef6c00", "D\n26–51"),
        (51,  68,   "#fbc02d", "C\n52–68"),
        (68,  80.3, "#7cb342", "B\n69–80"),
        (80.3,90,   "#388e3c", "A\n81–90"),
        (90,  100,  "#1b5e20", "A+\n91–100"),
    ]

    ax_gauge.set_xlim(0, 100)
    ax_gauge.set_ylim(-0.2, 2.0)
    ax_gauge.axis("off")

    for lo, hi, col, lbl in grade_bands:
        ax_gauge.barh(0.5, hi - lo, left=lo, height=0.55,
                      color=col, alpha=0.85, edgecolor="white", linewidth=1.5)
        ax_gauge.text((lo + hi) / 2, 0.5, lbl,
                      ha="center", va="center", fontsize=8,
                      color="white", fontweight="bold")

    # Needle
    ax_gauge.axvline(sus_score, ymin=0.15, ymax=0.85,
                     color="black", linewidth=3.5, zorder=5)
    ax_gauge.plot(sus_score, 1.0, marker="v", markersize=16,
                  color="black", zorder=6)
    ax_gauge.text(sus_score, 1.35,
                  f"SUS Score\n{sus_score:.1f}",
                  ha="center", va="center", fontsize=14,
                  fontweight="bold", color="black")
    ax_gauge.text(sus_score, 1.75,
                  f"{grade}  –  {adj}",
                  ha="center", va="center", fontsize=11,
                  fontweight="bold",
                  color=grade_bands[[b[3].startswith(grade) for b in grade_bands].index(True)][2])

    ax_gauge.set_title("Overall SUS Score", fontsize=11, fontweight="bold")

    fig.suptitle("SUS – System Usability Scale Summary", fontsize=14, fontweight="bold", y=1.01)
    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, "sus_scores.png"), dpi=150, bbox_inches="tight")
    plt.close(fig)

    print(f"    SUS score: {sus_score:.1f}  ->  {grade} ({adj})")


if __name__ == "__main__":
    run()

"""
create_pilot_study_plots.py  –  IWINAC 2022 / TFM predecessor-project visualisations.

Source:
  ../../Previous_Paper/First_Project/all_normative_users_output.xlsx

Output (one sub-folder per graphics category):
  ../MotionInsightHub/web/src/assets/plots/pilot_demographics/
  ../MotionInsightHub/web/src/assets/plots/pilot_overview/
  ../MotionInsightHub/web/src/assets/plots/pilot_exercise_1/
  ...

Context:
  Pilot / feasibility study (9 normative participants, no PD patients).
  Exercises: 1, 3, 4, 5, 6, 8, 9.
  This study preceded and motivated the larger current project (n = 46).
  Reference: Rodrigo-Rivero et al., IWINAC 2022.
"""

import os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.colors import Normalize
from matplotlib.cm import ScalarMappable

# ── Paths ─────────────────────────────────────────────────────────────────────
SCRIPTS_DIR = os.path.dirname(os.path.abspath(__file__))
PREV_DIR    = os.path.normpath(os.path.join(
    SCRIPTS_DIR, "..", "..", "Previous_Paper", "First_Project"))
XLSX_PATH   = os.path.join(PREV_DIR, "all_normative_users_output.xlsx")
PLOTS_BASE  = os.path.normpath(os.path.join(
    SCRIPTS_DIR, "..", "MotionInsightHub", "web", "src", "assets", "plots"))

def out_dir(category: str) -> str:
    d = os.path.join(PLOTS_BASE, category)
    os.makedirs(d, exist_ok=True)
    return d

# ── Participant metadata ───────────────────────────────────────────────────────
# Gender in source: H = Hombre (Male), M = Mujer (Female)
PARTICIPANT_INFO = {
    "MCH":  {"age": 24, "gender": "F"},
    "CRR":  {"age": 25, "gender": "M"},
    "DPA":  {"age": 39, "gender": "M"},
    "ENV":  {"age": 39, "gender": "F"},
    "MFA":  {"age": 46, "gender": "F"},
    "AJRA": {"age": 54, "gender": "F"},
    "JCR":  {"age": 54, "gender": "M"},
    "LAS":  {"age": 72, "gender": "F"},
    "BPA":  {"age": 72, "gender": "M"},
}
NICK_ORDER = ["MCH", "CRR", "DPA", "ENV", "MFA", "AJRA", "JCR", "LAS", "BPA"]
EXERCISES  = [1, 3, 4, 5, 6, 8, 9]

# Age-based colouring (viridis: young=purple, old=yellow)
AGE_CMAP = plt.cm.viridis
AGE_MIN, AGE_MAX = 24, 72
_age_norm = Normalize(vmin=AGE_MIN, vmax=AGE_MAX)

def age_color(age):
    return AGE_CMAP(_age_norm(age))

def nick_color(nick):
    return age_color(PARTICIPANT_INFO[nick]["age"])

GENDER_COLORS = {"M": "#42a5f5", "F": "#ef9a9a"}
SPEED_SCALE   = 1e-3   # raw units → ×10³ (display as thousands)

# ── Style helpers ─────────────────────────────────────────────────────────────

def style_ax(ax):
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.grid(axis="y", linestyle="--", alpha=0.35)

def save(fig, category, name, tight=True):
    path = os.path.join(out_dir(category), name)
    if tight:
        fig.tight_layout()
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  saved: {category}/{name}")
    return path

# ── Load data ─────────────────────────────────────────────────────────────────

def load_data():
    print("Loading XLSX …")
    df = pd.read_excel(XLSX_PATH, engine="openpyxl")
    # Normalise column names (strip spaces)
    df.columns = df.columns.str.strip()
    # Speed columns are numeric – coerce just in case
    for col in ["Left_hand_speed", "Right_hand_speed",
                "Left_hand_palm_position_x", "Left_hand_palm_position_y",
                "Left_hand_palm_position_z",
                "Right_hand_palm_position_x", "Right_hand_palm_position_y",
                "Right_hand_palm_position_z",
                "Left_hand_grab_strength", "Right_hand_grab_strength"]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")
    print(f"  {len(df):,} rows, {df['Nickname'].nunique()} participants, "
          f"{df['Exercise'].nunique()} exercises")
    return df

# ── 1.  Demographics overview ─────────────────────────────────────────────────

def plot_demographics(df):
    fig = plt.figure(figsize=(14, 5))
    fig.suptitle("Pilot Study — Participant Demographics (n = 9)",
                 fontsize=13, fontweight="bold")

    gs = fig.add_gridspec(1, 3, wspace=0.35)

    # ── Panel A: age bar ──────────────────────────────────────────────────────
    ax_age = fig.add_subplot(gs[0])
    ages   = [PARTICIPANT_INFO[n]["age"] for n in NICK_ORDER]
    colors = [age_color(a) for a in ages]
    bars   = ax_age.barh(NICK_ORDER, ages, color=colors, edgecolor="white", linewidth=0.5)
    ax_age.set_xlabel("Age (years)", fontsize=9)
    ax_age.set_title("A  Age by participant", fontsize=9, fontweight="bold")
    ax_age.set_xlim(0, 80)
    for bar, age in zip(bars, ages):
        ax_age.text(bar.get_width() + 1, bar.get_y() + bar.get_height() / 2,
                    str(age), va="center", fontsize=8)
    sm = ScalarMappable(cmap=AGE_CMAP, norm=_age_norm)
    sm.set_array([])
    fig.colorbar(sm, ax=ax_age, label="Age", shrink=0.7, pad=0.02)
    ax_age.spines["top"].set_visible(False)
    ax_age.spines["right"].set_visible(False)

    # ── Panel B: gender pie ───────────────────────────────────────────────────
    ax_pie = fig.add_subplot(gs[1])
    genders  = [PARTICIPANT_INFO[n]["gender"] for n in NICK_ORDER]
    n_male   = genders.count("M")
    n_female = genders.count("F")
    ax_pie.pie([n_male, n_female],
               labels=[f"Male\n(n={n_male})", f"Female\n(n={n_female})"],
               colors=[GENDER_COLORS["M"], GENDER_COLORS["F"]],
               autopct="%1.0f%%", startangle=90,
               textprops={"fontsize": 9},
               wedgeprops={"edgecolor": "white", "linewidth": 1.5})
    ax_pie.set_title("B  Gender distribution", fontsize=9, fontweight="bold")

    # ── Panel C: participant table ────────────────────────────────────────────
    ax_tbl = fig.add_subplot(gs[2])
    ax_tbl.axis("off")
    rows   = [[n,
               PARTICIPANT_INFO[n]["age"],
               "Male" if PARTICIPANT_INFO[n]["gender"] == "M" else "Female"]
              for n in NICK_ORDER]
    col_labels = ["Nickname", "Age", "Gender"]
    tbl = ax_tbl.table(cellText=rows, colLabels=col_labels,
                       loc="center", cellLoc="center")
    tbl.auto_set_font_size(False)
    tbl.set_fontsize(8)
    tbl.scale(1.0, 1.4)
    for j, _ in enumerate(col_labels):
        tbl[0, j].set_facecolor("#7c9bff")
        tbl[0, j].set_text_props(color="white", fontweight="bold")
    for i, nick in enumerate(NICK_ORDER):
        color = age_color(PARTICIPANT_INFO[nick]["age"])
        for j in range(3):
            tbl[i + 1, j].set_facecolor((*color[:3], 0.20))
    ax_tbl.set_title("C  Participant overview", fontsize=9, fontweight="bold")

    save(fig, "pilot_demographics", "pilot_demographics.png")


# ── 2.  Speed heatmap (all exercises × all participants) ──────────────────────

def plot_speed_heatmap(df):
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    fig.suptitle("Pilot Study — Mean Hand Speed  (×10³ units) per Exercise",
                 fontsize=12, fontweight="bold")

    for ax, hand, col in [
        (axes[0], "Left",  "Left_hand_speed"),
        (axes[1], "Right", "Right_hand_speed"),
    ]:
        matrix = np.full((len(NICK_ORDER), len(EXERCISES)), np.nan)
        for j, ex in enumerate(EXERCISES):
            sub = df[df["Exercise"] == ex]
            for i, nick in enumerate(NICK_ORDER):
                vals = sub.loc[sub["Nickname"] == nick, col].dropna()
                if len(vals) > 0:
                    matrix[i, j] = vals.mean() * SPEED_SCALE

        im = ax.imshow(matrix, aspect="auto", cmap="YlOrRd",
                       vmin=np.nanmin(matrix), vmax=np.nanmax(matrix))
        ax.set_xticks(range(len(EXERCISES)))
        ax.set_xticklabels([f"Ex{e}" for e in EXERCISES], fontsize=8)
        ax.set_yticks(range(len(NICK_ORDER)))
        ax.set_yticklabels(
            [f"{n}  ({PARTICIPANT_INFO[n]['age']})" for n in NICK_ORDER],
            fontsize=8)
        ax.set_title(f"{hand} hand", fontsize=10, fontweight="bold")
        for i in range(len(NICK_ORDER)):
            for j in range(len(EXERCISES)):
                val = matrix[i, j]
                if not np.isnan(val):
                    ax.text(j, i, f"{val:.0f}", ha="center", va="center",
                            fontsize=7, color="black")
        plt.colorbar(im, ax=ax, label="Mean speed (×10³)")

    save(fig, "pilot_overview", "pilot_speed_heatmap.png")


# ── 3.  Grab strength heatmap ─────────────────────────────────────────────────

def plot_grab_heatmap(df):
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    fig.suptitle("Pilot Study — Mean Grab Strength (0–1) per Exercise",
                 fontsize=12, fontweight="bold")

    for ax, hand, col in [
        (axes[0], "Left",  "Left_hand_grab_strength"),
        (axes[1], "Right", "Right_hand_grab_strength"),
    ]:
        matrix = np.full((len(NICK_ORDER), len(EXERCISES)), np.nan)
        for j, ex in enumerate(EXERCISES):
            sub = df[df["Exercise"] == ex]
            for i, nick in enumerate(NICK_ORDER):
                vals = sub.loc[sub["Nickname"] == nick, col].dropna()
                if len(vals) > 0:
                    matrix[i, j] = vals.mean()

        im = ax.imshow(matrix, aspect="auto", cmap="Blues",
                       vmin=0, vmax=1)
        ax.set_xticks(range(len(EXERCISES)))
        ax.set_xticklabels([f"Ex{e}" for e in EXERCISES], fontsize=8)
        ax.set_yticks(range(len(NICK_ORDER)))
        ax.set_yticklabels(
            [f"{n}  ({PARTICIPANT_INFO[n]['age']})" for n in NICK_ORDER],
            fontsize=8)
        ax.set_title(f"{hand} hand", fontsize=10, fontweight="bold")
        for i in range(len(NICK_ORDER)):
            for j in range(len(EXERCISES)):
                val = matrix[i, j]
                if not np.isnan(val):
                    ax.text(j, i, f"{val:.2f}", ha="center", va="center",
                            fontsize=7, color="black")
        plt.colorbar(im, ax=ax, label="Mean grab strength")

    save(fig, "pilot_overview", "pilot_grab_heatmap.png")


# ── 4.  Per-exercise speed (bar, per participant) ─────────────────────────────

def plot_exercise_speed(df, exercise):
    sub = df[df["Exercise"] == exercise]
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5), sharey=False)
    fig.suptitle(f"Exercise {exercise} — Hand Speed per Participant",
                 fontsize=12, fontweight="bold")

    for ax, col, hand in [
        (axes[0], "Left_hand_speed",  "Left hand"),
        (axes[1], "Right_hand_speed", "Right hand"),
    ]:
        means = []
        stds  = []
        for nick in NICK_ORDER:
            vals = sub.loc[sub["Nickname"] == nick, col].dropna() * SPEED_SCALE
            means.append(vals.mean() if len(vals) else np.nan)
            stds.append( vals.std()  if len(vals) else 0.0)

        colors = [nick_color(n) for n in NICK_ORDER]
        x      = np.arange(len(NICK_ORDER))
        ax.bar(x, means, color=colors, edgecolor="white",
               linewidth=0.5, yerr=stds, capsize=4,
               error_kw={"elinewidth": 1.2, "alpha": 0.7})
        ax.set_xticks(x)
        ax.set_xticklabels(
            [f"{n}\n({PARTICIPANT_INFO[n]['age']})" for n in NICK_ORDER],
            fontsize=7, rotation=0)
        ax.set_ylabel("Mean speed (×10³)", fontsize=8)
        ax.set_title(hand, fontsize=10, fontweight="bold")
        style_ax(ax)

    # Age colorbar
    sm = ScalarMappable(cmap=AGE_CMAP, norm=_age_norm)
    sm.set_array([])
    fig.colorbar(sm, ax=axes, label="Age", shrink=0.7, pad=0.02)
    save(fig, f"pilot_exercise_{exercise}", f"pilot_exercise{exercise}_speed.png")


# ── 5.  Per-exercise grab strength ────────────────────────────────────────────

def plot_exercise_grab(df, exercise):
    sub = df[df["Exercise"] == exercise]
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5), sharey=True)
    fig.suptitle(f"Exercise {exercise} — Grab Strength per Participant",
                 fontsize=12, fontweight="bold")

    for ax, col, hand in [
        (axes[0], "Left_hand_grab_strength",  "Left hand"),
        (axes[1], "Right_hand_grab_strength", "Right hand"),
    ]:
        means, stds = [], []
        for nick in NICK_ORDER:
            vals = sub.loc[sub["Nickname"] == nick, col].dropna()
            means.append(vals.mean() if len(vals) else np.nan)
            stds.append( vals.std()  if len(vals) else 0.0)

        colors = [nick_color(n) for n in NICK_ORDER]
        x      = np.arange(len(NICK_ORDER))
        ax.bar(x, means, color=colors, edgecolor="white",
               linewidth=0.5, yerr=stds, capsize=4,
               error_kw={"elinewidth": 1.2, "alpha": 0.7})
        ax.set_xticks(x)
        ax.set_xticklabels(
            [f"{n}\n({PARTICIPANT_INFO[n]['age']})" for n in NICK_ORDER],
            fontsize=7)
        ax.set_ylabel("Mean grab strength (0–1)", fontsize=8)
        ax.set_ylim(0, 1.1)
        ax.set_title(hand, fontsize=10, fontweight="bold")
        style_ax(ax)

    sm = ScalarMappable(cmap=AGE_CMAP, norm=_age_norm)
    sm.set_array([])
    fig.colorbar(sm, ax=axes, label="Age", shrink=0.7, pad=0.02)
    save(fig, f"pilot_exercise_{exercise}", f"pilot_exercise{exercise}_grab.png")


# ── 6.  Per-exercise hand trajectory (X–Z top view) ──────────────────────────

def plot_exercise_position(df, exercise):
    sub = df[df["Exercise"] == exercise]
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    fig.suptitle(f"Exercise {exercise} — Hand Trajectory (top view: X–Z plane)",
                 fontsize=12, fontweight="bold")

    for ax, px, pz, hand in [
        (axes[0], "Left_hand_palm_position_x",  "Left_hand_palm_position_z",  "Left hand"),
        (axes[1], "Right_hand_palm_position_x", "Right_hand_palm_position_z", "Right hand"),
    ]:
        legend_patches = []
        for nick in NICK_ORDER:
            usr = sub[sub["Nickname"] == nick]
            x   = pd.to_numeric(usr[px], errors="coerce").dropna()
            z   = pd.to_numeric(usr[pz], errors="coerce").dropna()
            min_len = min(len(x), len(z))
            if min_len == 0:
                continue
            x, z = x.iloc[:min_len].values, z.iloc[:min_len].values
            col  = nick_color(nick)
            # Subsample to ≤ 2000 points for performance
            step = max(1, min_len // 2000)
            ax.scatter(x[::step], z[::step], s=2, alpha=0.4, color=col)
            # Start marker
            ax.scatter(x[0], z[0], s=30, color=col, marker="o",
                       edgecolors="black", linewidths=0.5, zorder=5)
            legend_patches.append(
                mpatches.Patch(color=col,
                               label=f"{nick} ({PARTICIPANT_INFO[nick]['age']})"))

        ax.set_xlabel("X position", fontsize=8)
        ax.set_ylabel("Z position", fontsize=8)
        ax.set_title(hand, fontsize=10, fontweight="bold")
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.legend(handles=legend_patches, fontsize=7,
                  loc="upper right", framealpha=0.6,
                  markerscale=1.5, ncol=2)

    save(fig, f"pilot_exercise_{exercise}", f"pilot_exercise{exercise}_position.png")


# ── 7.  Overview card image ───────────────────────────────────────────────────

def plot_overview(df):
    """
    2×2 summary card:
      TL: participant table  TR: speed heatmap (left hand)
      BL: age scatter        BR: grab heatmap (left hand)
    """
    fig = plt.figure(figsize=(14, 9))
    fig.suptitle(
        "Predecessor Study (IWINAC 2022) — Overview\n"
        "Acquisition of Relevant Hand-Wrist Features Using Leap Motion Controller",
        fontsize=11, fontweight="bold", y=0.99)

    gs = fig.add_gridspec(2, 2, hspace=0.38, wspace=0.32)

    # ── TL: participant table ─────────────────────────────────────────────────
    ax_tbl = fig.add_subplot(gs[0, 0])
    ax_tbl.axis("off")
    rows = [[n, PARTICIPANT_INFO[n]["age"],
             "M" if PARTICIPANT_INFO[n]["gender"] == "M" else "F"]
            for n in NICK_ORDER]
    tbl = ax_tbl.table(cellText=rows,
                       colLabels=["Nickname", "Age", "Gender"],
                       loc="center", cellLoc="center")
    tbl.auto_set_font_size(False)
    tbl.set_fontsize(8)
    tbl.scale(1.1, 1.35)
    for j in range(3):
        tbl[0, j].set_facecolor("#7c9bff")
        tbl[0, j].set_text_props(color="white", fontweight="bold")
    for i, nick in enumerate(NICK_ORDER):
        c = age_color(PARTICIPANT_INFO[nick]["age"])
        for j in range(3):
            tbl[i + 1, j].set_facecolor((*c[:3], 0.18))
    ax_tbl.set_title("Participants (n = 9, all normative)",
                     fontsize=8, fontweight="bold", pad=6)

    # ── TR: speed heatmap – left hand ─────────────────────────────────────────
    ax_spd = fig.add_subplot(gs[0, 1])
    matrix = np.full((len(NICK_ORDER), len(EXERCISES)), np.nan)
    for j, ex in enumerate(EXERCISES):
        sub = df[df["Exercise"] == ex]
        for i, nick in enumerate(NICK_ORDER):
            vals = sub.loc[sub["Nickname"] == nick,
                           "Left_hand_speed"].dropna() * SPEED_SCALE
            if len(vals):
                matrix[i, j] = vals.mean()
    im = ax_spd.imshow(matrix, aspect="auto", cmap="YlOrRd",
                       vmin=np.nanmin(matrix), vmax=np.nanmax(matrix))
    ax_spd.set_xticks(range(len(EXERCISES)))
    ax_spd.set_xticklabels([f"Ex{e}" for e in EXERCISES], fontsize=7)
    ax_spd.set_yticks(range(len(NICK_ORDER)))
    ax_spd.set_yticklabels(
        [f"{n} ({PARTICIPANT_INFO[n]['age']})" for n in NICK_ORDER], fontsize=7)
    for i in range(len(NICK_ORDER)):
        for j in range(len(EXERCISES)):
            v = matrix[i, j]
            if not np.isnan(v):
                ax_spd.text(j, i, f"{v:.0f}", ha="center", va="center",
                            fontsize=6.5)
    plt.colorbar(im, ax=ax_spd, label="×10³")
    ax_spd.set_title("Left-hand mean speed (×10³) per exercise",
                     fontsize=8, fontweight="bold")

    # ── BL: age distribution bar ──────────────────────────────────────────────
    ax_age = fig.add_subplot(gs[1, 0])
    ages   = [PARTICIPANT_INFO[n]["age"] for n in NICK_ORDER]
    gcols  = [GENDER_COLORS[PARTICIPANT_INFO[n]["gender"]] for n in NICK_ORDER]
    bars   = ax_age.bar(NICK_ORDER, ages, color=gcols,
                        edgecolor="white", linewidth=0.5)
    ax_age.set_ylabel("Age (years)", fontsize=8)
    ax_age.set_xticklabels(NICK_ORDER, fontsize=7)
    ax_age.set_ylim(0, 80)
    style_ax(ax_age)
    for bar, age in zip(bars, ages):
        ax_age.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 1,
                    str(age), ha="center", fontsize=7)
    m_patch = mpatches.Patch(color=GENDER_COLORS["M"], label="Male")
    f_patch = mpatches.Patch(color=GENDER_COLORS["F"], label="Female")
    ax_age.legend(handles=[m_patch, f_patch], fontsize=7, framealpha=0.7)
    ax_age.set_title("Age by participant", fontsize=8, fontweight="bold")

    # ── BR: grab heatmap – both hands mean ────────────────────────────────────
    ax_grb = fig.add_subplot(gs[1, 1])
    gmatrix = np.full((len(NICK_ORDER), len(EXERCISES)), np.nan)
    for j, ex in enumerate(EXERCISES):
        sub = df[df["Exercise"] == ex]
        for i, nick in enumerate(NICK_ORDER):
            l = sub.loc[sub["Nickname"] == nick,
                        "Left_hand_grab_strength"].dropna()
            r = sub.loc[sub["Nickname"] == nick,
                        "Right_hand_grab_strength"].dropna()
            vals = pd.concat([l, r])
            if len(vals):
                gmatrix[i, j] = vals.mean()
    im2 = ax_grb.imshow(gmatrix, aspect="auto", cmap="Blues", vmin=0, vmax=1)
    ax_grb.set_xticks(range(len(EXERCISES)))
    ax_grb.set_xticklabels([f"Ex{e}" for e in EXERCISES], fontsize=7)
    ax_grb.set_yticks(range(len(NICK_ORDER)))
    ax_grb.set_yticklabels(
        [f"{n} ({PARTICIPANT_INFO[n]['age']})" for n in NICK_ORDER], fontsize=7)
    for i in range(len(NICK_ORDER)):
        for j in range(len(EXERCISES)):
            v = gmatrix[i, j]
            if not np.isnan(v):
                ax_grb.text(j, i, f"{v:.2f}", ha="center", va="center",
                            fontsize=6.5)
    plt.colorbar(im2, ax=ax_grb, label="0–1")
    ax_grb.set_title("Mean grab strength (both hands) per exercise",
                     fontsize=8, fontweight="bold")

    save(fig, "pilot_overview", "pilot_overview.png", tight=False)


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    df = load_data()

    print("Generating overview …")
    plot_overview(df)

    print("Generating demographics …")
    plot_demographics(df)

    print("Generating speed heatmap …")
    plot_speed_heatmap(df)

    print("Generating grab heatmap …")
    plot_grab_heatmap(df)

    print("Generating per-exercise plots …")
    for ex in EXERCISES:
        print(f"  Exercise {ex}")
        plot_exercise_speed(df, ex)
        plot_exercise_grab(df, ex)
        plot_exercise_position(df, ex)

    print(f"\nAll plots saved under:\n  {PLOTS_BASE}")


if __name__ == "__main__":
    main()

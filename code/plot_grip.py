"""
plot_grip.py – grab strength and system quality (FPS) analysis.

Grab strength (0–1): how closed the hand is on average.
Leap Motion FPS: system performance – should be similar across groups
  but useful to spot recording-quality differences.

Bimanual exercises (1, 2): Left and Right plotted separately.
Single-hand exercises (3, 4, 5): active hand detected per user.
"""

import os
import plot_utils as pu

GRIP_SUFFIXES = [
    ("hand_grab_strength", "Grab Strength", "0–1"),
]

BIMANUAL_GRIP_COLS = (
    [("Left_"  + s, f"Left {lbl}",  unit) for s, lbl, unit in GRIP_SUFFIXES]
    + [("Right_" + s, f"Right {lbl}", unit) for s, lbl, unit in GRIP_SUFFIXES]
)

FPS_COL = "Leap_Motion_Controller_fps"


def run(exercise_data, exercise_num, out_dir):
    norm     = exercise_data.get("normative",     [])
    non_norm = exercise_data.get("non_normative", [])
    bimanual = pu.is_bimanual(exercise_data)
    hand_note = "Both Hands" if bimanual else "Active Hand"

    # ── Gather grip stats ────────────────────────────────────────────────────
    if bimanual:
        col_pairs = [(col, lbl, unit) for col, lbl, unit in BIMANUAL_GRIP_COLS]
        cols = [c for c, _, _ in col_pairs]
        norm_grip     = pu.user_means(norm,     cols)
        non_norm_grip = pu.user_means(non_norm, cols)
    else:
        col_pairs = [(s, lbl, unit) for s, lbl, unit in GRIP_SUFFIXES]
        suffixes = [s for s, _, _ in col_pairs]
        norm_grip     = pu.active_hand_means(norm,     suffixes)
        non_norm_grip = pu.active_hand_means(non_norm, suffixes)

    # ── Build panel list: grip columns only ─────────────────────────────────
    n_panels = len(col_pairs)
    fig, axes = pu.make_axes_grid(n_panels, n_cols=min(n_panels, 4))

    for i, (col, label, unit) in enumerate(col_pairs):
        n_v  = norm_grip[col]     if col in norm_grip.columns     else []
        nn_v = non_norm_grip[col] if col in non_norm_grip.columns else []
        pu.box_compare(axes[i], n_v, nn_v, label, ylabel=unit)

    for j in range(n_panels, len(axes)):
        axes[j].set_visible(False)

    fig.legend(handles=pu.legend_handles(), loc="upper right", fontsize=9)
    pu.save_fig(
        fig,
        os.path.join(out_dir, f"exercise{exercise_num}_grip_fps.png"),
        suptitle=f"Exercise {exercise_num} – Grip Strength  [{hand_note}]",
    )

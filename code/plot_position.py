"""
plot_position.py – palm position analysis (mean position + movement variability).

Two figures are produced per exercise:
  1. Mean position  – where the hand is on average (spatial centre).
  2. Position std   – how much the hand moves around (movement range).

Bimanual exercises (1, 2): Left and Right plotted separately.
Single-hand exercises (3, 4, 5): active hand detected per user.
"""

import os
import plot_utils as pu

POSITION_SUFFIXES = [
    ("hand_palm_position_x", "Palm X", "mm"),
    ("hand_palm_position_y", "Palm Y", "mm"),
    ("hand_palm_position_z", "Palm Z", "mm"),
]

BIMANUAL_COLS = (
    [("Left_"  + s, f"Left {lbl}",  unit) for s, lbl, unit in POSITION_SUFFIXES]
    + [("Right_" + s, f"Right {lbl}", unit) for s, lbl, unit in POSITION_SUFFIXES]
)


def _plot_stat(exercise_num, col_pairs, norm_stats, non_norm_stats,
               stat_label, filename, out_dir):
    fig, axes = pu.make_axes_grid(len(col_pairs), n_cols=3)

    for i, (col, label, unit) in enumerate(col_pairs):
        n_v  = norm_stats[col]     if col in norm_stats.columns     else []
        nn_v = non_norm_stats[col] if col in non_norm_stats.columns else []
        pu.box_compare(axes[i], n_v, nn_v, label, ylabel=unit)

    for j in range(len(col_pairs), len(axes)):
        axes[j].set_visible(False)

    fig.legend(handles=pu.legend_handles(), loc="upper right", fontsize=9)
    pu.save_fig(fig, os.path.join(out_dir, filename),
                suptitle=stat_label)


def run(exercise_data, exercise_num, out_dir):
    norm     = exercise_data.get("normative",     [])
    non_norm = exercise_data.get("non_normative", [])
    bimanual = pu.is_bimanual(exercise_data)
    hand_note = "Both Hands" if bimanual else "Active Hand"

    if bimanual:
        col_pairs = [(col, lbl, unit) for col, lbl, unit in BIMANUAL_COLS]
        cols = [c for c, _, _ in col_pairs]
        norm_means     = pu.user_means(norm,     cols)
        non_norm_means = pu.user_means(non_norm, cols)
        norm_stds      = pu.user_stds(norm,      cols)
        non_norm_stds  = pu.user_stds(non_norm,  cols)
    else:
        col_pairs = [(s, lbl, unit) for s, lbl, unit in POSITION_SUFFIXES]
        suffixes = [s for s, _, _ in col_pairs]
        norm_means     = pu.active_hand_means(norm,     suffixes)
        non_norm_means = pu.active_hand_means(non_norm, suffixes)
        norm_stds      = pu.active_hand_stds(norm,      suffixes)
        non_norm_stds  = pu.active_hand_stds(non_norm,  suffixes)

    # Figure 1: mean position
    _plot_stat(
        exercise_num, col_pairs,
        norm_means, non_norm_means,
        stat_label=f"Exercise {exercise_num} – Palm Mean Position  [{hand_note}]",
        filename=f"exercise{exercise_num}_position_mean.png",
        out_dir=out_dir,
    )

    # Figure 2: position std (movement variability / range)
    _plot_stat(
        exercise_num, col_pairs,
        norm_stds, non_norm_stds,
        stat_label=f"Exercise {exercise_num} – Palm Position Std (Movement Range)  [{hand_note}]",
        filename=f"exercise{exercise_num}_position_std.png",
        out_dir=out_dir,
    )

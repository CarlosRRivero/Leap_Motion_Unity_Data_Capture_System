"""
plot_orientation.py – hand palm-normal vector analysis.

Palm normal vectors (x, y, z) represent the orientation / tilt of each hand.
Bimanual exercises (1, 2): Left and Right plotted separately.
Single-hand exercises (3, 4, 5): active hand detected per user.
"""

import os
import plot_utils as pu

NORMAL_SUFFIXES = [
    ("hand_normal_x", "Normal X", "unit"),
    ("hand_normal_y", "Normal Y", "unit"),
    ("hand_normal_z", "Normal Z", "unit"),
]

BIMANUAL_COLS = (
    [("Left_"  + s, f"Left {lbl}",  unit) for s, lbl, unit in NORMAL_SUFFIXES]
    + [("Right_" + s, f"Right {lbl}", unit) for s, lbl, unit in NORMAL_SUFFIXES]
)


def run(exercise_data, exercise_num, out_dir):
    norm     = exercise_data.get("normative",     [])
    non_norm = exercise_data.get("non_normative", [])
    bimanual = pu.is_bimanual(exercise_data)

    if bimanual:
        col_pairs = [(col, lbl, unit) for col, lbl, unit in BIMANUAL_COLS]
        cols = [c for c, _, _ in col_pairs]
        norm_stats     = pu.user_means(norm,     cols)
        non_norm_stats = pu.user_means(non_norm, cols)
    else:
        col_pairs = [(s, lbl, unit) for s, lbl, unit in NORMAL_SUFFIXES]
        suffixes = [s for s, _, _ in col_pairs]
        norm_stats     = pu.active_hand_means(norm,     suffixes)
        non_norm_stats = pu.active_hand_means(non_norm, suffixes)

    fig, axes = pu.make_axes_grid(len(col_pairs), n_cols=3)

    for i, (col, label, unit) in enumerate(col_pairs):
        n_v  = norm_stats[col]     if col in norm_stats.columns     else []
        nn_v = non_norm_stats[col] if col in non_norm_stats.columns else []
        pu.box_compare(axes[i], n_v, nn_v, label, ylabel="mean component")

    for j in range(len(col_pairs), len(axes)):
        axes[j].set_visible(False)

    fig.legend(handles=pu.legend_handles(), loc="upper right", fontsize=9)
    hand_note = "Both Hands" if bimanual else "Active Hand"
    pu.save_fig(
        fig,
        os.path.join(out_dir, f"exercise{exercise_num}_orientation.png"),
        suptitle=f"Exercise {exercise_num} – Hand Orientation / Normal  [{hand_note}]",
    )

"""
regenerate_hq_figures.py — reproducible high-quality figure regeneration for the
manuscript "A Technical Framework for Detecting Dyskinesia in Parkinson's Disease
Using Motion Tracking and Serious Games" (PeerJ Computer Science).

It runs the standard plotting scripts unchanged but, via lightweight matplotlib
monkeypatches, (1) scales up every font so the embedded text remains legible at the
journal's column width, (2) forces 300 dpi, and (3) saves with a tight bounding box
so no label is clipped. The web-asset mirroring (MotionInsightHub) is neutralised so
the web project is never modified.

Usage:
    python regenerate_hq_figures.py [SCALE] <script1.py> <script2.py> ...

If the first argument parses as a float it is used as the font SCALE factor
(default 1.7). Example:
    python regenerate_hq_figures.py 1.7 plot_groups_sturges.py plot_sus.py \
        plot_showcase.py create_ml_plots.py
"""
import sys
import os
import runpy
import shutil
import traceback

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.figure
import matplotlib.axes
import matplotlib.table
from matplotlib.text import Text

# ── Parse optional leading SCALE argument ──────────────────────────────────────
args = sys.argv[1:]
SCALE = 1.4
if args:
    try:
        SCALE = float(args[0])
        args = args[1:]
    except ValueError:
        pass

# 1. Larger rc defaults (affects any text without an explicit size).
plt.rcParams.update({
    "font.size":        round(10 * SCALE),
    "axes.titlesize":   round(11 * SCALE),
    "axes.labelsize":   round(10 * SCALE),
    "xtick.labelsize":  round(9 * SCALE),
    "ytick.labelsize":  round(9 * SCALE),
    "legend.fontsize":  round(8 * SCALE),
    "figure.titlesize": round(12 * SCALE),
    "savefig.dpi":      300,
    "figure.dpi":       300,
})

# 2. Force every savefig to 300 dpi + tight bbox (scripts hard-code dpi=150).
_orig_savefig = matplotlib.figure.Figure.savefig
def _savefig(self, *a, **k):
    k["dpi"] = 300
    k.setdefault("bbox_inches", "tight")
    k.setdefault("pad_inches", 0.15)
    return _orig_savefig(self, *a, **k)
matplotlib.figure.Figure.savefig = _savefig

# 3. Scale every explicit Text font size (labels, titles, annotations, colorbar
#    labels and suptitles all route through Text.set_fontsize).
_orig_set_fontsize = Text.set_fontsize
def _set_fontsize(self, fontsize):
    if isinstance(fontsize, (int, float)):
        fontsize = float(fontsize) * SCALE
    return _orig_set_fontsize(self, fontsize)
Text.set_fontsize = _set_fontsize

# 4. Scale legend fontsize kwarg (legends use FontProperties, not set_fontsize).
_orig_legend = matplotlib.axes.Axes.legend
def _legend(self, *a, **k):
    fs = k.get("fontsize")
    if isinstance(fs, (int, float)):
        k["fontsize"] = fs * SCALE
    return _orig_legend(self, *a, **k)
matplotlib.axes.Axes.legend = _legend

_orig_fig_legend = matplotlib.figure.Figure.legend
def _fig_legend(self, *a, **k):
    fs = k.get("fontsize")
    if isinstance(fs, (int, float)):
        k["fontsize"] = fs * SCALE
    return _orig_fig_legend(self, *a, **k)
matplotlib.figure.Figure.legend = _fig_legend

# 5. Scale tick_params labelsize.
_orig_tick_params = matplotlib.axes.Axes.tick_params
def _tick_params(self, *a, **k):
    if isinstance(k.get("labelsize"), (int, float)):
        k["labelsize"] = k["labelsize"] * SCALE
    return _orig_tick_params(self, *a, **k)
matplotlib.axes.Axes.tick_params = _tick_params

# 6. Data tables: scale by a CAPPED factor (so the text grows but still fits inside
#    the cells). A full SCALE (e.g. 2.0x) overflows the narrow summary-table cells,
#    so cap at 1.5x. We bypass the global Text scaling inside the table to avoid
#    double-scaling.
TABLE_SCALE = min(SCALE, 1.5)
_orig_table_set_fontsize = matplotlib.table.Table.set_fontsize
def _table_set_fontsize(self, size):
    saved = Text.set_fontsize
    Text.set_fontsize = _orig_set_fontsize  # cells keep the (capped) size, no double scale
    try:
        return _orig_table_set_fontsize(self, size * TABLE_SCALE)
    finally:
        Text.set_fontsize = saved
matplotlib.table.Table.set_fontsize = _table_set_fontsize

# 7. Neutralise web-asset mirroring so MotionInsightHub is not touched.
shutil.copytree = lambda *a, **k: None
shutil.rmtree = lambda *a, **k: None

print(f"[regenerate_hq_figures] SCALE={SCALE}, scripts={args}", flush=True)
for script in args:
    script = os.path.abspath(script)
    print(f"\n===== RUNNING {os.path.basename(script)} =====", flush=True)
    try:
        runpy.run_path(script, run_name="__main__")
        print(f"===== DONE {os.path.basename(script)} =====", flush=True)
    except SystemExit:
        print(f"===== DONE (SystemExit) {os.path.basename(script)} =====", flush=True)
    except Exception:
        print(f"===== ERROR in {os.path.basename(script)} =====", flush=True)
        traceback.print_exc()

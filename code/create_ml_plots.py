"""
create_ml_plots.py

Classification of normative vs non-normative participants evaluated
SEPARATELY per exercise group:

  Exercise 1          — bimanual (Left + Right hand)
  Exercise 2          — bimanual (Left + Right hand)
  Exercise 3 (3+4+5)  — single dominant hand

Models
------
  SVM        — RBF kernel, standardised features
  RF         — Random Forest (200 trees)
  Grouped RF — one RF per sensor-type group, probabilities averaged

Note: the 2 young reference users (file_ids 0 & 1, ages 27-28) are
EXCLUDED from all ML evaluation (n=44 participants).

Evaluation strategies
---------------------
  LOOCV         — Leave-One-Out (existing baseline)
  3-Fold × 10   — RepeatedStratifiedKFold(n_splits=3,  n_repeats=10)
  5-Fold × 10   — RepeatedStratifiedKFold(n_splits=5,  n_repeats=10)
  10-Fold × 5   — RepeatedStratifiedKFold(n_splits=10, n_repeats=5)

Outputs
-------
  Paper/Plots/ml_classification/
    ml_confusion_matrices.png   — 3x3 grid (rows=exercise, cols=model)
    ml_metrics_table.png        — F1 / Accuracy / Precision / Recall (LOOCV)
    ml_feature_importance.png   — top-5 Gini features per exercise
    ml_cv_comparison.png        — F1 comparison across all CV strategies
    ml_cv_full_table.png        — full metric table for all CV strategies
  web/src/assets/plots/ml_classification/  (mirror)
  web/src/assets/cv_results.json           (structured JSON for web display)

Usage
-----
  python create_ml_plots.py
"""

import os
import sys
import shutil
import json
import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

SCRIPTS_DIR = os.path.dirname(os.path.abspath(__file__))
PAPER_DIR   = os.path.normpath(os.path.join(SCRIPTS_DIR, ".."))
sys.path.insert(0, SCRIPTS_DIR)

import plot_utils as pu

USERS_DIR  = os.path.join(PAPER_DIR, "Users")
PLOTS_DIR  = os.path.join(PAPER_DIR, "Plots", "ml_classification")
WEB_PLOTS  = os.path.normpath(
    os.path.join(PAPER_DIR, "MotionInsightHub", "web", "src",
                 "assets", "plots", "ml_classification")
)
WEB_ASSETS = os.path.normpath(
    os.path.join(PAPER_DIR, "MotionInsightHub", "web", "src", "assets")
)

NORM_C  = "#2196F3"   # blue
NONNC   = "#E64A19"   # deep orange
GRFC    = "#388E3C"   # green

# File-IDs of the 2 young healthy reference participants (ages 27–28)
YOUNGSTER_IDS = {"0", "1"}

# CV strategy definitions
CV_KEYS        = ["loocv", "skf3", "skf5", "skf10"]
CV_LABELS      = ["LOOCV", "Leave-3-Out\n(3-Fold ×10)", "Leave-5-Out\n(5-Fold ×10)", "K-Fold Strat.\n(10-Fold ×5)"]
CV_LABELS_FLAT = ["LOOCV", "Leave-3-Out (3-Fold ×10)", "Leave-5-Out (5-Fold ×10)", "K-Fold Strat. (10-Fold ×5)"]
CV_COLORS      = ["#1565C0", "#2E7D32", "#E65100", "#6A1B9A"]

try:
    from sklearn.svm import SVC
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.model_selection import LeaveOneOut, RepeatedStratifiedKFold
    from sklearn.preprocessing import StandardScaler
    from sklearn.impute import SimpleImputer
    from sklearn.metrics import (
        confusion_matrix, accuracy_score, f1_score,
        precision_score, recall_score,
    )
except ImportError:
    print("[ERROR] scikit-learn is required:  pip install scikit-learn")
    sys.exit(1)

# Exercise groups definition
EX_GROUPS = [
    {"label": "Exercise 1",   "exs": [1]},
    {"label": "Exercise 2",   "exs": [2]},
    {"label": "Exercise 3",   "exs": [3, 4, 5]},
]

CLASS_NAMES = ["Normative", "Non-norm."]

# ===============================================================================
#  Feature engineering constants
# ===============================================================================

BIMAN_SCALAR = [
    "Left_hand_speed",  "Right_hand_speed",
    "Left_hand_grab_strength", "Right_hand_grab_strength",
]
BIMAN_TRIPLETS = [
    ("Left_hand_speed_x",        "Left_hand_speed_y",        "Left_hand_speed_z"),
    ("Right_hand_speed_x",       "Right_hand_speed_y",       "Right_hand_speed_z"),
    ("Left_hand_normal_x",       "Left_hand_normal_y",       "Left_hand_normal_z"),
    ("Right_hand_normal_x",      "Right_hand_normal_y",      "Right_hand_normal_z"),
    ("Left_hand_palm_position_x","Left_hand_palm_position_y","Left_hand_palm_position_z"),
    ("Right_hand_palm_position_x","Right_hand_palm_position_y","Right_hand_palm_position_z"),
]

SINGLE_SCALAR = [
    "{H}_hand_speed",
    "{H}_hand_grab_strength",
]
SINGLE_TRIPLETS = [
    ("{H}_hand_speed_x",         "{H}_hand_speed_y",         "{H}_hand_speed_z"),
    ("{H}_hand_normal_x",        "{H}_hand_normal_y",        "{H}_hand_normal_z"),
    ("{H}_hand_palm_position_x", "{H}_hand_palm_position_y", "{H}_hand_palm_position_z"),
]


def _shorten(name):
    return (name
            .replace("Left_hand_",    "L_")
            .replace("Right_hand_",   "R_")
            .replace("palm_position_","pos_")
            .replace("grab_strength", "grip")
            .replace("_speed",        "_spd")
            .replace("_normal_",      "_nrm_")
            .replace("hand_",         ""))


def _col_stats(df, col):
    s = pd.to_numeric(df.get(col, pd.Series(dtype=float)), errors="coerce")
    return float(s.mean()), float(s.std())


def _mag_stats(df, cx, cy, cz):
    x = pd.to_numeric(df.get(cx, pd.Series(dtype=float)), errors="coerce")
    y = pd.to_numeric(df.get(cy, pd.Series(dtype=float)), errors="coerce")
    z = pd.to_numeric(df.get(cz, pd.Series(dtype=float)), errors="coerce")
    mag = np.sqrt(x**2 + y**2 + z**2)
    return float(mag.mean()), float(mag.std())


# ===============================================================================
#  Data loading
# ===============================================================================

def load_data():
    per_user = {}
    per_ex   = {i: {"normative": [], "non_normative": []} for i in range(1, 6)}

    if not os.path.isdir(USERS_DIR):
        print(f"[ERROR] Users dir not found: {USERS_DIR}")
        return per_user, per_ex

    for folder in sorted(os.listdir(USERS_DIR)):
        user_dir = os.path.join(USERS_DIR, folder)
        if not os.path.isdir(user_dir):
            continue
        if "non_normative" in folder:
            label = 1; group = "non_normative"
        elif "normative" in folder:
            label = 0; group = "normative"
        else:
            continue

        file_id    = folder.split("_ID_")[-1] if "_ID_" in folder else ""
        is_youngster = file_id in YOUNGSTER_IDS

        per_user[folder] = {"label": label, "exs": {}, "youngster": is_youngster}
        for ex in range(1, 6):
            path = os.path.join(user_dir, f"dataCompilation{ex}.csv")
            if not os.path.exists(path):
                continue
            try:
                df = pd.read_csv(path, sep=";", low_memory=False)
                per_user[folder]["exs"][ex] = df
                per_ex[ex][group].append((folder, df))
            except Exception as e:
                print(f"  [WARN] {folder}/dataCompilation{ex}.csv: {e}")

    return per_user, per_ex


# ===============================================================================
#  Feature matrix
# ===============================================================================

def build_feature_matrix(per_user, ex_nums, bimanual_per_ex):
    rows, labels, names = [], [], []

    for folder, info in per_user.items():
        if info.get("youngster", False):
            continue
        available = [ex for ex in ex_nums if ex in info["exs"]]
        if not available:
            continue

        row = {}
        for ex in ex_nums:
            pfx = f"ex{ex}_"
            if ex not in info["exs"]:
                if bimanual_per_ex[ex]:
                    for col in BIMAN_SCALAR:
                        k = pfx + _shorten(col)
                        row[k + "_mean"] = np.nan; row[k + "_std"] = np.nan
                    for trip in BIMAN_TRIPLETS:
                        k = pfx + _shorten(trip[0]).replace("_x","") + "_mag"
                        row[k + "_mean"] = np.nan; row[k + "_std"] = np.nan
                else:
                    for tmpl in SINGLE_SCALAR:
                        col = tmpl.replace("{H}", "Left")
                        k   = pfx + _shorten(col)
                        row[k + "_mean"] = np.nan; row[k + "_std"] = np.nan
                    for trip_t in SINGLE_TRIPLETS:
                        col = trip_t[0].replace("{H}", "Left")
                        k   = pfx + _shorten(col).replace("_x","") + "_mag"
                        row[k + "_mean"] = np.nan; row[k + "_std"] = np.nan
                continue

            df = info["exs"][ex]
            if bimanual_per_ex[ex]:
                for col in BIMAN_SCALAR:
                    k = pfx + _shorten(col)
                    m, s = _col_stats(df, col)
                    row[k + "_mean"] = m; row[k + "_std"] = s
                for trip in BIMAN_TRIPLETS:
                    k = pfx + _shorten(trip[0]).replace("_x","") + "_mag"
                    m, s = _mag_stats(df, *trip)
                    row[k + "_mean"] = m; row[k + "_std"] = s
            else:
                hand = pu.dominant_hand(df)
                for tmpl in SINGLE_SCALAR:
                    col = tmpl.replace("{H}", hand)
                    k   = pfx + _shorten(col)
                    m, s = _col_stats(df, col)
                    row[k + "_mean"] = m; row[k + "_std"] = s
                for trip_t in SINGLE_TRIPLETS:
                    trip = tuple(t.replace("{H}", hand) for t in trip_t)
                    k    = pfx + _shorten(trip[0]).replace("_x","") + "_mag"
                    m, s = _mag_stats(df, *trip)
                    row[k + "_mean"] = m; row[k + "_std"] = s

        rows.append(row)
        labels.append(info["label"])
        names.append(folder)

    feat_df       = pd.DataFrame(rows)
    feature_names = list(feat_df.columns)
    return feat_df.values.astype(float), np.array(labels), feature_names, names


# ===============================================================================
#  Feature groups
# ===============================================================================

def build_feature_groups(feature_names):
    groups = {
        "speed":       [],
        "position":    [],
        "orientation": [],
        "grip":        [],
        "left_hand":   [],
        "right_hand":  [],
    }
    for i, name in enumerate(feature_names):
        low = name.lower()
        if "spd" in low or "speed" in low:   groups["speed"].append(i)
        if "pos" in low:                      groups["position"].append(i)
        if "nrm" in low or "normal" in low:   groups["orientation"].append(i)
        if "grip" in low:                     groups["grip"].append(i)
        after_ex = name.split("_", 1)[1] if "_" in name else name
        if after_ex.startswith("L_"):         groups["left_hand"].append(i)
        elif after_ex.startswith("R_"):       groups["right_hand"].append(i)

    return {k: v for k, v in groups.items() if v}


# ===============================================================================
#  Generic CV infrastructure
# ===============================================================================

def _impute_split(X_tr, X_te):
    imp = SimpleImputer(strategy="mean")
    return imp.fit_transform(X_tr), imp.transform(X_te)


def _run_one_fold(model_type, X_tr_raw, X_te_raw, y_tr, feature_groups):
    """Train one model fold and return predictions for the test set."""
    X_tr, X_te = _impute_split(X_tr_raw, X_te_raw)

    if model_type == "svm":
        sc   = StandardScaler()
        X_tr = sc.fit_transform(X_tr)
        X_te = sc.transform(X_te)
        clf  = SVC(kernel="rbf", C=1.0, gamma="scale", random_state=42)
        clf.fit(X_tr, y_tr)
        return clf.predict(X_te)

    if model_type == "rf":
        clf = RandomForestClassifier(n_estimators=200, random_state=42, n_jobs=-1)
        clf.fit(X_tr, y_tr)
        return clf.predict(X_te)

    if model_type == "grouped_rf":
        group_probs = []
        for _, idx in feature_groups.items():
            if not idx:
                continue
            clf = RandomForestClassifier(n_estimators=150, random_state=42, n_jobs=-1)
            clf.fit(X_tr[:, idx], y_tr)
            group_probs.append(clf.predict_proba(X_te[:, idx])[:, 1])
        mean_prob = np.mean(group_probs, axis=0)
        return (mean_prob >= 0.5).astype(int)

    raise ValueError(f"Unknown model_type: {model_type}")


def run_loocv_full(X_raw, y, model_type, feature_groups=None):
    """
    LOOCV: collects all predictions into a single array.
    Returns y_pred (needed for confusion matrices and global metrics).
    """
    y_pred = np.zeros(len(y), dtype=int)
    for tr, te in LeaveOneOut().split(X_raw):
        y_pred[te] = _run_one_fold(
            model_type, X_raw[tr], X_raw[te], y[tr], feature_groups or {}
        )
    return y_pred


def run_repeated_skf(X_raw, y, n_splits, n_repeats, model_type,
                     feature_groups=None, seed=42):
    """
    RepeatedStratifiedKFold: returns per-metric {mean, std} dict.
    Metrics are computed per fold (each fold has n/k test samples),
    then averaged across all n_splits × n_repeats folds.
    """
    rskf = RepeatedStratifiedKFold(
        n_splits=n_splits, n_repeats=n_repeats, random_state=seed
    )
    fold_metrics = []
    for tr, te in rskf.split(X_raw, y):
        if len(np.unique(y[tr])) < 2:
            continue
        y_pred = _run_one_fold(
            model_type, X_raw[tr], X_raw[te], y[tr], feature_groups or {}
        )
        fold_metrics.append(metrics(y[te], y_pred))

    return {
        k: {
            "mean": float(np.mean([m[k] for m in fold_metrics])),
            "std":  float(np.std( [m[k] for m in fold_metrics])),
        }
        for k in ["accuracy", "precision", "recall", "f1"]
    }


# ===============================================================================
#  Metrics helper
# ===============================================================================

def metrics(y_true, y_pred):
    return {
        "accuracy":  round(float(accuracy_score(y_true, y_pred)), 3),
        "precision": round(float(precision_score(y_true, y_pred, zero_division=0)), 3),
        "recall":    round(float(recall_score(y_true, y_pred, zero_division=0)), 3),
        "f1":        round(float(f1_score(y_true, y_pred, zero_division=0)), 3),
    }


def loocv_metrics_as_cv_dict(m):
    """Wrap LOOCV point estimates in the same {mean, std} structure (std=0)."""
    return {k: {"mean": m[k], "std": 0.0} for k in m}


# ===============================================================================
#  Figure 1 — Confusion Matrices (LOOCV)
# ===============================================================================

def _draw_cm(ax, cm, title, bg_color):
    im = ax.imshow(cm, interpolation="nearest", cmap="Blues", vmin=0)
    ax.set_title(title, fontsize=10, fontweight="bold", pad=6)
    ax.set_xlabel("Predicted", fontsize=12)
    ax.set_ylabel("True", fontsize=12)
    ax.set_xticks([0, 1]); ax.set_xticklabels(CLASS_NAMES, fontsize=11)
    ax.set_yticks([0, 1]); ax.set_yticklabels(CLASS_NAMES, fontsize=11,
                                                rotation=90, va="center")
    thresh = cm.max() / 2.0
    for i in range(2):
        for j in range(2):
            ax.text(j, i, str(cm[i, j]), ha="center", va="center",
                    fontsize=16, fontweight="bold",
                    color="white" if cm[i, j] > thresh else "#111111")
    ax.set_facecolor(bg_color)
    plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)


ROW_COLORS = ["#E3F2FD", "#FFF3E0", "#E8F5E9"]


def create_confusion_figure(results, out_dir):
    n_rows = len(results)
    n_cols = 3
    fig, axes = plt.subplots(n_rows, n_cols,
                             figsize=(14, 4.8 * n_rows),
                             squeeze=False)

    col_titles  = ["SVM", "Random Forest", "Grouped RF"]
    model_keys  = ["y_svm", "y_rf", "y_grf"]
    model_colors = [NORM_C + "18", NONNC + "18", GRFC + "18"]

    for r, res in enumerate(results):
        for c, (mkey, mc) in enumerate(zip(model_keys, model_colors)):
            ax = axes[r][c]
            cm = confusion_matrix(res["y"], res[mkey])
            _draw_cm(ax, cm, f"{col_titles[c]}\n({res['label']})", mc)

    fig.suptitle(
        "Confusion Matrices — LOOCV per Exercise Group\n"
        "(Normative = 0, Non-normative = 1)",
        fontsize=13, fontweight="bold", y=1.01
    )
    fig.tight_layout()
    path = os.path.join(out_dir, "ml_confusion_matrices.png")
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print("  [OK] ml_confusion_matrices.png")


# ===============================================================================
#  Figure 2 — LOOCV Metrics Table
# ===============================================================================

def create_metrics_table_figure(results, out_dir):
    fig, ax = plt.subplots(figsize=(12, 5.5))
    ax.axis("off")

    headers = ["Exercise", "Model", "Accuracy", "Precision", "Recall", "F1"]
    rows    = []
    model_names = ["SVM", "Random Forest", "Grouped RF"]
    model_keys  = ["y_svm", "y_rf", "y_grf"]

    model_tints = [
        ["#E3F2FD", "#FFF3E0", "#E8F5E9"],
        ["#BBDEFB", "#FFE0B2", "#A5D6A7"],
        ["#90CAF9", "#FFCC80", "#66BB6A"],
    ]
    row_colors = []

    for r_idx, res in enumerate(results):
        for m_idx, (mname, mkey) in enumerate(zip(model_names, model_keys)):
            m = metrics(res["y"], res[mkey])
            ex_label = res["label"] if m_idx == 0 else ""
            rows.append([
                ex_label, mname,
                f"{m['accuracy']:.3f}", f"{m['precision']:.3f}",
                f"{m['recall']:.3f}",   f"{m['f1']:.3f}",
            ])
            row_colors.append([model_tints[m_idx][r_idx]] * 6)

    tbl = ax.table(cellText=rows, colLabels=headers, loc="center",
                   cellLoc="center", cellColours=row_colors)
    tbl.auto_set_font_size(False)
    tbl.set_fontsize(11)
    tbl.scale(1.1, 2.5)
    for j in range(len(headers)):
        cell = tbl[0, j]
        cell.set_facecolor("#1a3a6b")
        cell.set_text_props(color="white", fontweight="bold")

    fig.suptitle("LOOCV Classification Metrics per Exercise Group",
                 fontsize=13, fontweight="bold", y=1.04)
    fig.tight_layout()
    path = os.path.join(out_dir, "ml_metrics_table.png")
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print("  [OK] ml_metrics_table.png")


# ===============================================================================
#  Figure 3 — Feature Importance
# ===============================================================================

def _readable_label(name):
    parts = name.split("_")
    ex   = parts[0].replace("ex", "Ex") if parts[0].startswith("ex") else parts[0]
    rest = "_".join(parts[1:])
    stat = ""
    if rest.endswith("_mean"):
        stat = " (mean)"; rest = rest[:-5]
    elif rest.endswith("_std"):
        stat = " (std)";  rest = rest[:-4]
    label_map = {
        "L_spd":       "L Speed",    "R_spd":       "R Speed",
        "L_grip":      "L Grip",     "R_grip":      "R Grip",
        "L_pos_x_mag": "L Pos mag",  "R_pos_x_mag": "R Pos mag",
        "L_spd_x_mag": "L Speed mag","R_spd_x_mag": "R Speed mag",
        "L_nrm_x_mag": "L Orient mag","R_nrm_x_mag":"R Orient mag",
        "spd":         "Speed",      "grip":        "Grip",
        "pos_x_mag":   "Pos mag",    "spd_x_mag":   "Speed mag",
        "nrm_x_mag":   "Orient mag",
    }
    friendly = label_map.get(rest, rest.replace("_", " "))
    return f"{ex} · {friendly}{stat}"


def create_feature_importance_figure(fi_data, out_dir, top_n=5):
    n_groups  = len(fi_data)
    ex_colors = [NORM_C, NONNC, GRFC]
    fig, axes = plt.subplots(1, n_groups, figsize=(6 * n_groups, 5.5))
    if n_groups == 1:
        axes = [axes]

    for ax, data, color in zip(axes, fi_data, ex_colors):
        imp   = np.array(data["importances"])
        names = data["feature_names"]
        idx   = np.argsort(imp)[::-1][:top_n]
        top_imp   = imp[idx][::-1]
        top_names = [_readable_label(names[i]) for i in idx][::-1]

        bars = ax.barh(range(top_n), top_imp, color=color, alpha=0.82,
                       edgecolor="white", linewidth=0.6)
        for bar, val in zip(bars, top_imp):
            ax.text(val + 0.001, bar.get_y() + bar.get_height() / 2,
                    f"{val:.3f}", va="center", ha="left", fontsize=9,
                    fontweight="bold", color="#333333")

        ax.set_yticks(range(top_n))
        ax.set_yticklabels(top_names, fontsize=10)
        ax.set_xlabel("Mean decrease in impurity (Gini)", fontsize=11)
        ax.set_title(f"{data['label']}\nTop {top_n} Features",
                     fontsize=11, fontweight="bold")
        ax.set_xlim(0, max(top_imp) * 1.30)
        # Limit the number of x ticks so the values never overlap (narrow ranges).
        ax.locator_params(axis="x", nbins=4)
        ax.grid(axis="x", linestyle="--", alpha=0.35)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)

    fig.suptitle(
        "Random Forest — Top 5 Most Important Features per Exercise Group\n"
        "(trained on full dataset, Gini impurity)",
        fontsize=13, fontweight="bold", y=1.02
    )
    fig.tight_layout()
    path = os.path.join(out_dir, "ml_feature_importance.png")
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print("  [OK] ml_feature_importance.png")


# ===============================================================================
#  Figure 4 — CV Strategy Comparison (F1, all strategies, all models)
# ===============================================================================

def create_cv_comparison_figure(cv_results, out_dir):
    """
    9 subplots: 3 exercises × 3 models.
    Each subplot: grouped bar for each CV strategy showing F1 ± std.
    """
    n_ex  = len(EX_GROUPS)
    model_labels = ["SVM", "Random Forest", "Grouped RF"]
    model_keys   = ["svm", "rf", "grouped_rf"]

    fig, axes = plt.subplots(n_ex, 3, figsize=(15, 4.5 * n_ex), squeeze=False)

    for r, grp in enumerate(EX_GROUPS):
        ex_label = grp["label"]
        for c, (mkey, mlabel) in enumerate(zip(model_keys, model_labels)):
            ax = axes[r][c]

            means = [cv_results[ex_label][mkey][cvk]["f1"]["mean"] for cvk in CV_KEYS]
            stds  = [cv_results[ex_label][mkey][cvk]["f1"]["std"]  for cvk in CV_KEYS]

            x    = np.arange(len(CV_KEYS))
            bars = ax.bar(x, means, color=CV_COLORS, alpha=0.85,
                          edgecolor="white", linewidth=0.8)
            ax.errorbar(x, means, yerr=stds, fmt="none",
                        color="#333333", capsize=5, linewidth=1.5)

            for bar, m in zip(bars, means):
                ax.text(bar.get_x() + bar.get_width() / 2,
                        min(m + 0.02, 1.0),
                        f"{m:.3f}", ha="center", va="bottom",
                        fontsize=8, fontweight="bold", color="#111111")

            ax.set_xticks(x)
            ax.set_xticklabels(CV_LABELS, fontsize=8, rotation=15, ha="right")
            ax.set_ylim(0, 1.15)
            ax.set_ylabel("F1 Score" if c == 0 else "", fontsize=9)
            ax.set_title(f"{ex_label} — {mlabel}", fontsize=10, fontweight="bold")
            ax.axhline(0.5, color="#aaa", linestyle=":", linewidth=0.8)
            ax.grid(axis="y", linestyle="--", alpha=0.3)
            ax.spines["top"].set_visible(False)
            ax.spines["right"].set_visible(False)

    # Legend
    from matplotlib.patches import Patch
    legend_elements = [Patch(facecolor=c, label=l)
                       for c, l in zip(CV_COLORS, CV_LABELS_FLAT)]
    fig.legend(handles=legend_elements, loc="lower center", ncol=4,
               fontsize=10, frameon=True, bbox_to_anchor=(0.5, -0.02))

    fig.suptitle(
        "Cross-Validation Strategy Comparison — F1 Score per (Exercise × Model)\n"
        "Error bars = ±1 SD across folds (repeated strategies)",
        fontsize=13, fontweight="bold", y=1.01
    )
    fig.tight_layout(rect=[0, 0.04, 1, 1])
    path = os.path.join(out_dir, "ml_cv_comparison.png")
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print("  [OK] ml_cv_comparison.png")


# ===============================================================================
#  Figure 5 — Full CV Metrics Table (mean ± std for all strategies)
# ===============================================================================

def create_cv_full_table_figure(cv_results, out_dir):
    """
    One large table: rows = (exercise, model, CV strategy), cols = 4 metrics.
    """
    model_names = ["SVM", "Random Forest", "Grouped RF"]
    model_keys  = ["svm", "rf", "grouped_rf"]
    headers = ["Exercise", "Model", "CV Strategy",
               "Accuracy", "Precision", "Recall", "F1"]

    rows       = []
    row_colors = []
    ex_tints   = ["#E3F2FD", "#FFF3E0", "#E8F5E9"]
    cv_alphas  = ["FF", "CC", "99", "66"]  # hex opacity per CV strategy

    for r_idx, grp in enumerate(EX_GROUPS):
        ex_label = grp["label"]
        base     = ex_tints[r_idx]

        for m_idx, (mkey, mname) in enumerate(zip(model_keys, model_names)):
            for cv_idx, (cvk, cvlabel) in enumerate(zip(CV_KEYS, CV_LABELS_FLAT)):
                d = cv_results[ex_label][mkey][cvk]

                def fmt(k):
                    m, s = d[k]["mean"], d[k]["std"]
                    return f"{m:.3f} ± {s:.3f}" if s > 0 else f"{m:.3f}"

                rows.append([
                    ex_label if (m_idx == 0 and cv_idx == 0) else "",
                    mname    if cv_idx == 0 else "",
                    cvlabel,
                    fmt("accuracy"), fmt("precision"), fmt("recall"), fmt("f1"),
                ])
                row_colors.append([base] * 7)

    n_rows = len(rows)
    fig_h  = max(6, n_rows * 0.42 + 1.5)
    fig, ax = plt.subplots(figsize=(16, fig_h))
    ax.axis("off")

    tbl = ax.table(cellText=rows, colLabels=headers, loc="center",
                   cellLoc="center", cellColours=row_colors)
    tbl.auto_set_font_size(False)
    tbl.set_fontsize(8.5)
    tbl.scale(1.0, 1.6)

    for j in range(len(headers)):
        cell = tbl[0, j]
        cell.set_facecolor("#1a3a6b")
        cell.set_text_props(color="white", fontweight="bold")

    fig.suptitle(
        "Classification Metrics — All CV Strategies (mean ± SD across folds)\n"
        "LOOCV reports single-estimate; repeated strategies report mean ± SD",
        fontsize=12, fontweight="bold", y=1.02
    )
    fig.tight_layout()
    path = os.path.join(out_dir, "ml_cv_full_table.png")
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print("  [OK] ml_cv_full_table.png")


# ===============================================================================
#  JSON export for web
# ===============================================================================

def export_cv_json(cv_results, assets_dir):
    """
    Serialise cv_results to cv_results.json in web assets.
    Structure: {ex_label: {model_key: {cv_key: {metric: {mean, std}}}}}
    """
    os.makedirs(assets_dir, exist_ok=True)
    path = os.path.join(assets_dir, "cv_results.json")
    with open(path, "w", encoding="utf-8") as f:
        json.dump(cv_results, f, indent=2)
    print(f"  [OK] cv_results.json -> {path}")


# ===============================================================================
#  Main
# ===============================================================================

def main():
    print("=" * 60)
    print("  ML Classification  —  per-exercise-group evaluation")
    print("  CV strategies: LOOCV | 3-Fold×10 | 5-Fold×10 | 10-Fold×5")
    print("=" * 60)

    print("\n-- Loading data --")
    per_user, per_ex = load_data()
    n_norm  = sum(1 for v in per_user.values() if v["label"] == 0 and not v.get("youngster"))
    n_nn    = sum(1 for v in per_user.values() if v["label"] == 1)
    n_young = sum(1 for v in per_user.values() if v.get("youngster"))
    print(f"  Total users loaded: {len(per_user)}  "
          f"({n_norm} normative, {n_nn} non-normative, {n_young} young ref excluded)")

    bimanual_per_ex = {ex: pu.is_bimanual(per_ex[ex]) for ex in range(1, 6)}

    results    = []   # LOOCV y_pred arrays (for confusion matrices)
    fi_data    = []   # feature importance data
    cv_results = {}   # all CV strategy metrics

    for grp in EX_GROUPS:
        label   = grp["label"]
        ex_nums = grp["exs"]
        print(f"\n{'='*60}")
        print(f"  {label}  (exercises {ex_nums})")
        print(f"{'='*60}")

        X, y, feat_names, user_names = build_feature_matrix(
            per_user, ex_nums, bimanual_per_ex
        )
        n0 = int(np.sum(y == 0)); n1 = int(np.sum(y == 1))
        print(f"  Users with data: {len(y)}  ({n0} norm, {n1} non-norm)")
        print(f"  Features: {X.shape[1]}")

        feat_groups   = build_feature_groups(feat_names)
        cv_results[label] = {}

        y_svm = y_rf = y_grf = None

        for model_type, model_label, fgs in [
            ("svm",        "SVM",        None),
            ("rf",         "RF",         None),
            ("grouped_rf", "Grouped RF", feat_groups),
        ]:
            print(f"\n  [{model_label}]")

            print("    LOOCV...", end=" ", flush=True)
            y_pred_loo = run_loocv_full(X, y, model_type, fgs)
            m_loo      = metrics(y, y_pred_loo)
            print(f"Acc={m_loo['accuracy']}  F1={m_loo['f1']}")

            print("    3-Fold × 10...", end=" ", flush=True)
            m_skf3 = run_repeated_skf(X, y, 3, 10, model_type, fgs)
            print(f"F1={m_skf3['f1']['mean']:.3f} ± {m_skf3['f1']['std']:.3f}")

            print("    5-Fold × 10...", end=" ", flush=True)
            m_skf5 = run_repeated_skf(X, y, 5, 10, model_type, fgs)
            print(f"F1={m_skf5['f1']['mean']:.3f} ± {m_skf5['f1']['std']:.3f}")

            print("    10-Fold × 5...", end=" ", flush=True)
            m_skf10 = run_repeated_skf(X, y, 10, 5, model_type, fgs)
            print(f"F1={m_skf10['f1']['mean']:.3f} ± {m_skf10['f1']['std']:.3f}")

            cv_results[label][model_type] = {
                "loocv": loocv_metrics_as_cv_dict(m_loo),
                "skf3":  m_skf3,
                "skf5":  m_skf5,
                "skf10": m_skf10,
            }

            if model_type == "svm":        y_svm = y_pred_loo
            elif model_type == "rf":       y_rf  = y_pred_loo
            elif model_type == "grouped_rf": y_grf = y_pred_loo

        results.append({
            "label": label, "y": y,
            "y_svm": y_svm, "y_rf": y_rf, "y_grf": y_grf,
        })

        # Feature importance on full dataset
        imp_full = SimpleImputer(strategy="mean")
        X_full   = imp_full.fit_transform(X)
        rf_full  = RandomForestClassifier(n_estimators=500, random_state=42, n_jobs=-1)
        rf_full.fit(X_full, y)
        fi_data.append({
            "label":         label,
            "feature_names": feat_names,
            "importances":   rf_full.feature_importances_,
        })

    # ── Generate figures ──────────────────────────────────────────────────────
    print("\n-- Generating figures --")
    os.makedirs(PLOTS_DIR, exist_ok=True)

    create_confusion_figure(results, PLOTS_DIR)
    create_metrics_table_figure(results, PLOTS_DIR)
    create_feature_importance_figure(fi_data, PLOTS_DIR)
    create_cv_comparison_figure(cv_results, PLOTS_DIR)
    create_cv_full_table_figure(cv_results, PLOTS_DIR)

    # ── Export JSON ───────────────────────────────────────────────────────────
    print("\n-- Exporting JSON --")
    export_cv_json(cv_results, WEB_ASSETS)

    # ── Copy to web assets ────────────────────────────────────────────────────
    print("\n-- Copying to web assets --")
    if os.path.exists(WEB_PLOTS):
        shutil.rmtree(WEB_PLOTS)
    shutil.copytree(PLOTS_DIR, WEB_PLOTS)
    print(f"  [OK] -> {WEB_PLOTS}")

    # ── Final summary ─────────────────────────────────────────────────────────
    print("\n" + "=" * 60)
    print("  FINAL METRICS SUMMARY  (LOOCV)")
    print("=" * 60)
    print(f"{'Exercise':<14} {'Model':<14} {'Acc':>6} {'Prec':>7} {'Rec':>7} {'F1':>7}")
    print("-" * 58)
    for res in results:
        for mname, mkey in [("SVM","y_svm"),("RF","y_rf"),("Grouped RF","y_grf")]:
            m = metrics(res["y"], res[mkey])
            print(f"{res['label']:<14} {mname:<14} "
                  f"{m['accuracy']:>6.3f} {m['precision']:>7.3f} "
                  f"{m['recall']:>7.3f} {m['f1']:>7.3f}")
        print()

    print(f"\nDone.  Plots -> {PLOTS_DIR}")


if __name__ == "__main__":
    main()

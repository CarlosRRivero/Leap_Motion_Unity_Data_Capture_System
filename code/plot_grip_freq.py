"""
plot_grip_freq.py – grip frequency and rhythm-consistency analysis.

Pipeline per user:
  1. Low-pass  filter (4th-order Butterworth, fc = 3 Hz)
       → removes high-frequency sensor jitter, keeps rhythmic grip motion.
  2. High-pass filter (4th-order Butterworth, fc = 0.2 Hz)
       → removes slow DC offset / constant-grip drift.
  3. Dominant grip frequency (Hz) via FFT peak in 0.1–4 Hz range.
  4. Grip-cycle CV = std(inter-peak intervals) / mean(inter-peak intervals)
       → measures rhythm consistency; high CV = inconsistent open/close cycles.

Two figures per exercise hand:
  exercise{n}_grip_frequency[_left|_right].png
  exercise{n}_grip_cv[_left|_right].png

Bimanual exercises (1, 2): Left and Right plotted separately.
Single-hand exercises (3, 4, 5): dominant hand detected per user.
"""

import os
import warnings
import numpy as np
import pandas as pd
from scipy import signal
import plot_utils as pu

warnings.filterwarnings("ignore")

# ── Filter parameters ─────────────────────────────────────────────────────────
LP_CUTOFF    = 3.0   # Hz – low-pass  (keep rhythmic grip, remove jitter)
HP_CUTOFF    = 0.2   # Hz – high-pass (remove slow drift / DC offset)
FILTER_ORDER = 4
DEFAULT_FPS  = 44.0  # fallback if FPS column is missing/invalid


# ── Filter helpers ────────────────────────────────────────────────────────────

def _sos_lp(fs):
    nyq = fs / 2.0
    return signal.butter(FILTER_ORDER, LP_CUTOFF / nyq, btype="low",  output="sos")


def _sos_hp(fs):
    nyq = fs / 2.0
    return signal.butter(FILTER_ORDER, HP_CUTOFF / nyq, btype="high", output="sos")


def _apply(x, sos):
    """Zero-phase forward-backward filter; returns copy unchanged if too short."""
    if len(x) < 3 * FILTER_ORDER:
        return x.copy()
    return signal.sosfiltfilt(sos, x)


# ── Per-signal metric extraction ──────────────────────────────────────────────

def _grip_metrics(series, fps):
    """
    Compute (dominant_freq_hz, inter_peak_cv) from a raw grip time series.

    dominant_freq_hz : peak FFT frequency of the LP-filtered signal in 0.1–4 Hz.
    inter_peak_cv    : std/mean of inter-peak intervals on the bandpass signal.
                       A high value means irregular open/close rhythm.

    Returns (nan, nan) on failure or when signal is too flat to analyse.
    """
    x = pd.to_numeric(series, errors="coerce").dropna().values
    if len(x) < 30 or x.std() < 1e-4:
        return np.nan, np.nan

    sos_lp = _sos_lp(fps)
    sos_hp = _sos_hp(fps)

    # ── Low-pass filtered signal ──────────────────────────────────────────────
    x_lp = _apply(x, sos_lp)

    # ── Dominant frequency (FFT on LP signal) ─────────────────────────────────
    freqs = np.fft.rfftfreq(len(x_lp), d=1.0 / fps)
    power = np.abs(np.fft.rfft(x_lp)) ** 2
    mask  = (freqs >= 0.1) & (freqs <= 4.0)
    dom_freq = freqs[mask][np.argmax(power[mask])] if mask.any() else np.nan

    # ── Bandpass signal = LP then HP (removes drift) ──────────────────────────
    x_bp = _apply(x_lp, sos_hp)

    # ── Peak detection for CV ─────────────────────────────────────────────────
    min_dist = max(3, int(fps * 0.3))            # minimum 0.3 s between peaks
    peaks, _ = signal.find_peaks(
        x_bp, distance=min_dist, prominence=0.03
    )
    if len(peaks) < 3:
        inter_cv = np.nan
    else:
        intervals = np.diff(peaks) / fps          # frames → seconds
        inter_cv  = (intervals.std() / intervals.mean()
                     if intervals.mean() > 0 else np.nan)

    return dom_freq, inter_cv


# ── Per-user aggregation ──────────────────────────────────────────────────────

def _extract_for_col(user_list, grip_col):
    """Return (freq_list, cv_list) for all users using a fixed column name."""
    freqs, cvs = [], []
    for _, df in user_list:
        fps = pd.to_numeric(
            df.get("Leap_Motion_Controller_fps", pd.Series(dtype=float)),
            errors="coerce"
        ).median()
        fps = fps if (np.isfinite(fps) and fps > 5) else DEFAULT_FPS
        gc  = df[grip_col] if grip_col in df.columns else pd.Series(dtype=float)
        f, c = _grip_metrics(gc, fps)
        freqs.append(f)
        cvs.append(c)
    return freqs, cvs


def _extract_active_hand(user_list):
    """Return (freq_list, cv_list) using each user's dominant hand."""
    freqs, cvs = [], []
    for _, df in user_list:
        fps = pd.to_numeric(
            df.get("Leap_Motion_Controller_fps", pd.Series(dtype=float)),
            errors="coerce"
        ).median()
        fps = fps if (np.isfinite(fps) and fps > 5) else DEFAULT_FPS
        hand = pu.dominant_hand(df)
        col  = f"{hand}_hand_grab_strength"
        gc   = df[col] if col in df.columns else pd.Series(dtype=float)
        f, c = _grip_metrics(gc, fps)
        freqs.append(f)
        cvs.append(c)
    return freqs, cvs


# ── Plotting ──────────────────────────────────────────────────────────────────

def _panel_fig(norm_vals, nn_vals, title, ylabel, filename, out_dir):
    fig, axes = pu.make_axes_grid(1, n_cols=1, panel_size=(4.5, 4.0))
    pu.box_compare(axes[0], norm_vals, nn_vals, title=title, ylabel=ylabel)
    fig.legend(handles=pu.legend_handles(), loc="upper right", fontsize=9)
    pu.save_fig(fig, os.path.join(out_dir, filename), suptitle=title)


# ── Public entry point ────────────────────────────────────────────────────────

def run(exercise_data, exercise_num, out_dir):
    norm     = exercise_data.get("normative",     [])
    non_norm = exercise_data.get("non_normative", [])
    bimanual = pu.is_bimanual(exercise_data)

    if bimanual:
        for hand in ("Left", "Right"):
            col    = f"{hand}_hand_grab_strength"
            suffix = f"_{hand.lower()}"
            label  = f"{hand} Hand"

            n_f,  n_c  = _extract_for_col(norm,     col)
            nn_f, nn_c = _extract_for_col(non_norm, col)

            _panel_fig(
                n_f, nn_f,
                title    = f"Exercise {exercise_num} – Grip Dominant Frequency  [{label}]",
                ylabel   = "Frequency (Hz)",
                filename = f"exercise{exercise_num}_grip_frequency{suffix}.png",
                out_dir  = out_dir,
            )
            _panel_fig(
                n_c, nn_c,
                title    = (f"Exercise {exercise_num} – Grip Rhythm Consistency  [{label}]\n"
                            "CV of inter-peak intervals  (higher = more irregular)"),
                ylabel   = "CV  (std / mean of intervals)",
                filename = f"exercise{exercise_num}_grip_cv{suffix}.png",
                out_dir  = out_dir,
            )
    else:
        n_f,  n_c  = _extract_active_hand(norm)
        nn_f, nn_c = _extract_active_hand(non_norm)

        _panel_fig(
            n_f, nn_f,
            title    = f"Exercise {exercise_num} – Grip Dominant Frequency  [Active Hand]",
            ylabel   = "Frequency (Hz)",
            filename = f"exercise{exercise_num}_grip_frequency.png",
            out_dir  = out_dir,
        )
        _panel_fig(
            n_c, nn_c,
            title    = (f"Exercise {exercise_num} – Grip Rhythm Consistency  [Active Hand]\n"
                        "CV of inter-peak intervals  (higher = more irregular)"),
            ylabel   = "CV  (std / mean of intervals)",
            filename = f"exercise{exercise_num}_grip_cv.png",
            out_dir  = out_dir,
        )

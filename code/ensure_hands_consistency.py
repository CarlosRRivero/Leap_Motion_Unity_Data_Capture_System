import os
import shutil
import pandas as pd

# Script to ensure consistent hand usage across exercise data files.
# Reads from Paper/Raw_Users (never modified) and writes processed files to
# Paper/Users, mirroring the same directory/file structure.
#
#  1. Checks that dataCompilation1.csv and dataCompilation3.csv contain data
#     for both hands; logs a warning if only one hand is found.
#
#  2. For exercises 3, 4 and 5: determines the dominant hand (the one with
#     greater total activity across all three files) and writes those CSVs to
#     the output folder so that only the dominant hand carries non-zero values.
#       - Rows where only the minority hand is active: values are moved
#         (copied) to the dominant-hand columns, minority columns zeroed.
#       - Rows where both hands are active: dominant-hand data is kept as-is,
#         minority-hand columns are zeroed (no addition).
#       - Rows where only the dominant hand is active: no change.
#
# All other files (dataCompilation0, 1, 2 …) are copied to the output folder
# unchanged.
#
# Usage: run the script from anywhere; it will locate Raw_Users and Users
# relative to its own location.


def hand_activity(df):
    """Return (left_total, right_total) as the sum of absolute speed values."""
    left_col = df.get("Left_hand_speed")
    right_col = df.get("Right_hand_speed")
    tot_left = float(left_col.abs().sum()) if left_col is not None else 0.0
    tot_right = float(right_col.abs().sum()) if right_col is not None else 0.0
    return tot_left, tot_right


def hand_active_mask(df, side):
    """Return boolean Series: True where the given hand has non-zero speed."""
    col = f"{side}_hand_speed"
    if col not in df.columns:
        return pd.Series([False] * len(df), index=df.index)
    return df[col].fillna(0).abs() > 0


def uses_both_hands(df):
    """Return True if both hands show activity anywhere in the file."""
    left_active = hand_active_mask(df, "Left").any()
    right_active = hand_active_mask(df, "Right").any()
    return left_active and right_active


def unify_to_hand(df, main_hand):
    """
    Ensure only `main_hand` carries data in every row.

    For rows where the minority hand is active but the main hand is not:
      the minority values are copied to the corresponding main-hand columns.
    For rows where both hands are active:
      the main-hand values are kept and the minority columns are zeroed.
    For rows where only the main hand is active: no change.

    The minority-hand columns are zeroed in all rows afterward.
    """
    other = "Right" if main_hand == "Left" else "Left"

    other_active = hand_active_mask(df, other)
    main_active = hand_active_mask(df, main_hand)

    # Rows where we need to copy other -> main (other active, main inactive)
    copy_mask = other_active & ~main_active

    for col in list(df.columns):
        if col.startswith(f"{other}_hand"):
            counterpart = col.replace(f"{other}_hand", f"{main_hand}_hand", 1)
            if counterpart in df.columns and copy_mask.any():
                # Ensure target column is float so float values can be assigned
                df[counterpart] = df[counterpart].astype(float)
                df.loc[copy_mask, counterpart] = df.loc[copy_mask, col].values
            # Zero out minority-hand column for every row
            df[col] = 0.0

    return df, int(copy_mask.sum()), int((other_active & main_active).sum())


def process_user(src_dir, out_dir):
    user_name = os.path.basename(src_dir)
    print(f"\nProcessing: {user_name}")

    os.makedirs(out_dir, exist_ok=True)

    # Copy every file in the source directory to the output directory first
    for fname in os.listdir(src_dir):
        src_file = os.path.join(src_dir, fname)
        if os.path.isfile(src_file):
            shutil.copy2(src_file, os.path.join(out_dir, fname))

    # Build paths pointing at the OUTPUT directory (files already copied above)
    out_paths = {i: os.path.join(out_dir, f"dataCompilation{i}.csv") for i in range(1, 6)}

    # --- 1. Check that files 1 and 3 report both hands ---
    for idx in (1, 3):
        path = out_paths[idx]
        if not os.path.exists(path):
            print(f"  [WARNING] dataCompilation{idx}.csv not found")
            continue
        try:
            df = pd.read_csv(path, sep=";")
        except Exception as exc:
            print(f"  [ERROR] could not read dataCompilation{idx}.csv: {exc}")
            continue
        if uses_both_hands(df):
            print(f"  [OK] dataCompilation{idx}.csv uses both hands")
        else:
            left_act, right_act = hand_activity(df)
            only = "Left" if left_act > 0 else "Right"
            print(
                f"  [WARNING] dataCompilation{idx}.csv appears to use only the "
                f"{only} hand (left_activity={left_act:.1f}, right_activity={right_act:.1f})"
            )

    # --- 2. Unify hand usage for exercises 3, 4, 5 ---
    available = []
    for idx in (3, 4, 5):
        path = out_paths[idx]
        if not os.path.exists(path):
            continue
        try:
            df = pd.read_csv(path, sep=";")
            available.append((idx, path, df))
        except Exception as exc:
            print(f"  [ERROR] could not read dataCompilation{idx}.csv: {exc}")

    if not available:
        print("  No dataCompilation3/4/5 files found – nothing to unify")
        return

    # Decide dominant hand by total activity across all available files
    total_left = sum(hand_activity(df)[0] for _, _, df in available)
    total_right = sum(hand_activity(df)[1] for _, _, df in available)
    main_hand = "Left" if total_left >= total_right else "Right"
    other_hand = "Right" if main_hand == "Left" else "Left"
    print(
        f"  Dominant hand for exercises 3/4/5: {main_hand} "
        f"(left_total={total_left:.1f}, right_total={total_right:.1f})"
    )

    for idx, path, df in available:
        other_active_before = hand_active_mask(df, other_hand).sum()

        if other_active_before == 0:
            print(f"  [OK] dataCompilation{idx}.csv already uses only the {main_hand} hand")
            continue

        unified, copied, both_active = unify_to_hand(df, main_hand)

        try:
            unified.to_csv(path, sep=";", index=False)
            print(
                f"  [FIXED] dataCompilation{idx}.csv – "
                f"copied {copied} rows ({other_hand}→{main_hand}), "
                f"zeroed {both_active} rows where both hands were active"
            )
        except Exception as exc:
            print(f"  [ERROR] could not write dataCompilation{idx}.csv: {exc}")


def main():
    base = os.path.dirname(os.path.abspath(__file__))
    raw_users = os.path.normpath(os.path.join(base, "..", "Raw_Users"))
    users_out = os.path.normpath(os.path.join(base, "..", "Users"))

    if not os.path.isdir(raw_users):
        print(f"Raw_Users directory not found at: {raw_users}")
        return

    os.makedirs(users_out, exist_ok=True)

    user_dirs = sorted(
        os.path.join(raw_users, name)
        for name in os.listdir(raw_users)
        if os.path.isdir(os.path.join(raw_users, name))
    )

    print(f"Found {len(user_dirs)} user directories in {raw_users}")
    print(f"Output directory: {users_out}")

    for src_dir in user_dirs:
        user_name = os.path.basename(src_dir)
        out_dir = os.path.join(users_out, user_name)
        process_user(src_dir, out_dir)

    print("\nDone.")


if __name__ == "__main__":
    main()

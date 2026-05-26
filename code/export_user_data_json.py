"""
export_user_data_json.py
Converts user exercise CSVs to JSON asset files for Firebase-hosted Explorer.
Outputs to: Paper/MotionInsightHub/web/src/assets/data/users/{fileId}/exercise{n}.json
Only includes the columns required for the 4 visualisation groups (speed, orientation,
position, grip) plus plane position for exercises 3-5.
"""

import os, json, math, re
from pathlib import Path

# ── Paths ──────────────────────────────────────────────────────────────────
SCRIPT_DIR   = Path(__file__).resolve().parent
USERS_BASE   = SCRIPT_DIR.parent / "Users"
ASSETS_OUT   = SCRIPT_DIR.parent / "MotionInsightHub" / "web" / "src" / "assets" / "data" / "users"
MAX_PTS      = 600

# Columns to keep (all that the Angular component uses)
KEEP_COLS = {
    "Left_hand_speed", "Right_hand_speed",
    "Left_hand_speed_x", "Left_hand_speed_y", "Left_hand_speed_z",
    "Right_hand_speed_x", "Right_hand_speed_y", "Right_hand_speed_z",
    "Left_hand_normal_x", "Left_hand_normal_y", "Left_hand_normal_z",
    "Right_hand_normal_x", "Right_hand_normal_y", "Right_hand_normal_z",
    "Left_hand_palm_position_x", "Left_hand_palm_position_y", "Left_hand_palm_position_z",
    "Right_hand_palm_position_x", "Right_hand_palm_position_y", "Right_hand_palm_position_z",
    "Left_hand_grab_strength", "Right_hand_grab_strength",
    # Plane position for exercises 3-5
    "Plane_position_x", "Plane_position_y",
}

# Exercises 3-5 are plane exercises; rock status file index = exercise_num - 3
PLANE_EXERCISES = {3, 4, 5}
ROCK_STATUS_IDX = {3: 0, 4: 1, 5: 2}


def parse_csv(csv_path: Path):
    text = csv_path.read_text(encoding="utf-8-sig")
    lines = [l for l in text.splitlines() if l.strip()]
    if len(lines) < 2:
        return None
    headers = [h.strip() for h in lines[0].split(";")]
    rows = []
    for line in lines[1:]:
        vals = line.split(";")
        row = []
        for v in vals:
            try:
                row.append(float(v.replace(",", ".")))
            except ValueError:
                row.append(0.0)
        rows.append(row)
    return headers, rows


def read_rock_status(csv_path: Path):
    """Return list of {x, y} dicts from a Rock_Status CSV."""
    rocks = []
    try:
        text = csv_path.read_text(encoding="utf-8-sig")
        lines = [l for l in text.splitlines() if l.strip()]
        if len(lines) < 2:
            return rocks
        # first line = header; columns: Rock_position_x ; y ; z
        for line in lines[1:]:
            parts = line.split(";")
            if len(parts) < 2:
                continue
            try:
                rx = round(float(parts[0].strip().replace(",", ".")), 3)
                ry = round(float(parts[1].strip().replace(",", ".")), 3)
                rocks.append({"x": rx, "y": ry})
            except ValueError:
                continue
    except Exception:
        pass
    return rocks


def downsample(rows, max_pts):
    n = len(rows)
    if n <= max_pts:
        return rows
    step = n / max_pts
    return [rows[int(i * step)] for i in range(max_pts)]


def process_csv(file_id: str, exercise: int, csv_path: Path, out_dir: Path):
    result = parse_csv(csv_path)
    if result is None:
        print(f"  [SKIP] {csv_path.name} – empty")
        return

    headers, rows = result
    total_rows = len(rows)
    sampled = downsample(rows, MAX_PTS)

    # Build column arrays for only the needed columns
    columns = {}
    for i, h in enumerate(headers):
        if h in KEEP_COLS:
            columns[h] = [round(row[i], 4) if i < len(row) else 0.0 for row in sampled]

    # Detect bimanual
    left_sum  = sum(abs(v) for v in columns.get("Left_hand_speed", []))
    right_sum = sum(abs(v) for v in columns.get("Right_hand_speed", []))
    bimanual  = left_sum > 0 and right_sum > 0
    dominant  = "Left" if left_sum >= right_sum else "Right"

    # Plane exercise: read rock positions
    plane_exercise = exercise in PLANE_EXERCISES
    rocks = []
    if plane_exercise:
        rock_idx  = ROCK_STATUS_IDX[exercise]
        rock_path = csv_path.parent / f"dataCompilation{rock_idx}_Rock_Status.csv"
        if rock_path.exists():
            rocks = read_rock_status(rock_path)

    payload = {
        "fileId":        file_id,
        "exercise":      exercise,
        "totalRows":     total_rows,
        "sampledRows":   len(sampled),
        "bimanual":      bimanual,
        "dominantHand":  dominant,
        "planeExercise": plane_exercise,
        "rocks":         rocks,
        "columns":       columns
    }

    out_path = out_dir / f"exercise{exercise}.json"
    out_path.write_text(json.dumps(payload, separators=(",", ":")), encoding="utf-8")
    kb = out_path.stat().st_size / 1024
    rock_note = f", {len(rocks)} rocks" if plane_exercise else ""
    print(f"  exercise{exercise}.json  ({total_rows} rows -> {len(sampled)} pts{rock_note}, {kb:.0f} KB)")


def main():
    if not USERS_BASE.exists():
        print(f"ERROR: Users folder not found: {USERS_BASE}")
        return

    ASSETS_OUT.mkdir(parents=True, exist_ok=True)

    folders = sorted(USERS_BASE.iterdir())
    id_pattern = re.compile(r"_ID_(\d+)$")

    total_files = 0
    for folder in folders:
        if not folder.is_dir():
            continue
        m = id_pattern.search(folder.name)
        if not m:
            continue
        file_id = m.group(1)
        out_dir = ASSETS_OUT / file_id
        out_dir.mkdir(exist_ok=True)

        print(f"\nUser {file_id}  ({folder.name})")
        for n in range(1, 6):
            csv_path = folder / f"dataCompilation{n}.csv"
            if csv_path.exists():
                process_csv(file_id, n, csv_path, out_dir)
            else:
                # Remove stale JSON if exercise no longer exists
                stale = out_dir / f"exercise{n}.json"
                if stale.exists():
                    stale.unlink()
        total_files += 1

    print(f"\nDone - processed {total_files} users -> {ASSETS_OUT}")


if __name__ == "__main__":
    main()

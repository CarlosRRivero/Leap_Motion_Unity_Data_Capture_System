"""
export_v1_user_data.py — IWINAC 2022 pilot study.

Reads all_normative_users_output.xlsx and writes per-user per-exercise JSON files
to assets/data/users_v1/{NICK}/exercise{N}.json  (ExerciseData format).
"""

import os, json
import pandas as pd

SCRIPTS_DIR = os.path.dirname(os.path.abspath(__file__))
PREV_DIR    = os.path.normpath(os.path.join(SCRIPTS_DIR, '..', '..', 'Previous_Paper', 'First_Project'))
XLSX_PATH   = os.path.join(PREV_DIR, 'all_normative_users_output.xlsx')
OUT_BASE    = os.path.normpath(os.path.join(
    SCRIPTS_DIR, '..', 'MotionInsightHub', 'web', 'src', 'assets', 'data', 'users_v1'))

NICK_ORDER = ['MCH', 'CRR', 'DPA', 'ENV', 'MFA', 'AJRA', 'JCR', 'LAS', 'BPA']
EXERCISES  = [1, 3, 4, 5, 6, 8, 9]
MAX_ROWS   = 1000

EXPORT_COLS = [
    'Left_hand_speed',  'Right_hand_speed',
    'Left_hand_palm_position_x', 'Left_hand_palm_position_y', 'Left_hand_palm_position_z',
    'Right_hand_palm_position_x', 'Right_hand_palm_position_y', 'Right_hand_palm_position_z',
    'Left_hand_grab_strength',  'Right_hand_grab_strength',
]


def main():
    print('Loading XLSX...')
    df = pd.read_excel(XLSX_PATH, engine='openpyxl')
    df.columns = df.columns.str.strip()
    for col in EXPORT_COLS:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='coerce').fillna(0.0)

    os.makedirs(OUT_BASE, exist_ok=True)

    for nick in NICK_ORDER:
        nick_dir = os.path.join(OUT_BASE, nick)
        os.makedirs(nick_dir, exist_ok=True)

        for ex in EXERCISES:
            sub = df[(df['Nickname'] == nick) & (df['Exercise'] == ex)].copy()
            total = len(sub)
            if total == 0:
                print(f'  WARNING: no data for {nick} ex{ex}')
                continue

            if total > MAX_ROWS:
                step = total / MAX_ROWS
                idx  = [int(i * step) for i in range(MAX_ROWS)]
                sub  = sub.iloc[idx]
            sampled = len(sub)

            cols_dict: dict = {}
            for col in EXPORT_COLS:
                if col in sub.columns:
                    cols_dict[col] = [round(float(v), 6) for v in sub[col].values]
                else:
                    cols_dict[col] = [0.0] * sampled

            record = {
                'fileId':      nick,
                'exercise':    ex,
                'totalRows':   total,
                'sampledRows': sampled,
                'bimanual':    True,
                'dominantHand': 'Right',
                'planeExercise': False,
                'rocks':       [],
                'columns':     cols_dict,
            }

            out_path = os.path.join(nick_dir, f'exercise{ex}.json')
            with open(out_path, 'w', encoding='utf-8') as f:
                json.dump(record, f, separators=(',', ':'))
            print(f'  {nick}/exercise{ex}.json  ({total} -> {sampled} rows)')

    print(f'\nDone. Files written to:\n  {OUT_BASE}')


if __name__ == '__main__':
    main()

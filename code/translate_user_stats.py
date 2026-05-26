"""
translate_user_stats.py – Translates Spanish text in User_Stats_Total.csv to English.
Also fixes Parkinson_Phase for normative users ('No' -> 'Stage 0.0 - No Parkinson's').
"""

import pandas as pd
import os

PAPER_DIR = os.path.normpath(os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
CSV_PATH = os.path.join(PAPER_DIR, "User_Stats_Total.csv")

df = pd.read_csv(CSV_PATH, sep=";", encoding="utf-8-sig")

# ── Sex ───────────────────────────────────────────────────────────────────────
df["Sex"] = df["Sex"].map({"Mujer": "Female", "Hombre": "Male"}).fillna(df["Sex"])

# ── Parkinson_Phase ───────────────────────────────────────────────────────────
phase_map = {
    "No": "Stage 0.0 - No Parkinson's",
    "Etapa 0.0 - Sin afectaci\xf3n":
        "Stage 0.0 - No affectation",
    "Etapa 1.0 \x96 Afectaci\xf3n unilateral.\xa0S\xedntomas son leves y afectan solo un lado del cuerpo. Puede haber temblores o rigidez en una extremidad, pero la persona mantiene una postura corporal correcta y una expresi\xf3n facial normal.":
        "Stage 1.0 - Unilateral affectation. Symptoms are mild and affect only one side of the body. There may be tremors or rigidity in a limb, but the person maintains correct body posture and a normal facial expression.",
    "Etapa 1.5 \x96 Afectaci\xf3n unilateral y axial. Los s\xedntomas siguen siendo unilaterales pero tambi\xe9n afectan el tronco, lo que puede incluir rigidez en el cuello o la espalda.":
        "Stage 1.5 - Unilateral and axial affectation. Symptoms remain unilateral but also affect the trunk, which may include stiffness in the neck or back.",
    "Etapa 2.0 \x96 Afectaci\xf3n bilateral sin alteraci\xf3n del equilibrio. La enfermedad ya afecta ambos lados del cuerpo, aunque el equilibrio a\xfan no est\xe1 comprometido. En esta fase, pueden observarse alteraciones en la expresi\xf3n facial, disminuci\xf3n del parpadeo y cierta torpeza en las actividades cotidianas.":
        "Stage 2.0 - Bilateral affectation without balance impairment. The disease affects both sides of the body, although balance is not yet compromised. Alterations in facial expression, decreased blinking, and some clumsiness in daily activities may be observed.",
    "Etapa 3.0 \x96 Afectaci\xf3n bilateral con inestabilidad postural leve. El/La paciente sigue siendo f\xedsicamente independiente, pero comienza a presentar inestabilidad postural. Aparecen dificultades para caminar, como pasos m\xe1s cortos o dificultad para girar, y pueden surgir s\xedntomas de disfunci\xf3n auton\xf3mica, fatiga y dolores. Tambi\xe9n es com\xfan que se presenten los llamados fen\xf3menos \x93on/off\x94, donde los efectos de la medicaci\xf3n fluct\xfaan.":
        "Stage 3.0 - Bilateral affectation with mild postural instability. The patient remains physically independent but begins to show postural instability. Walking difficulties appear (shorter steps, difficulty turning), and symptoms of autonomic dysfunction, fatigue, and pain may arise. On/off phenomena, where medication effects fluctuate, are also common.",
}
df["Parkinson_Phase"] = df["Parkinson_Phase"].map(phase_map).fillna(df["Parkinson_Phase"])

# ── Pharmacology_and_Dosage ───────────────────────────────────────────────────
pharma_map = {
    "Sinemed (4 al dia), Ungenis (1 al dia), Sinemed retard (1 al dia) ":
        "Sinemet (4 per day), Ungenis (1 per day), Sinemet retard (1 per day)",
    "Sinemed (8 al dia), Sadago (1 al dia), Oprimea (1 al dia)":
        "Sinemet (8 per day), Xadago (1 per day), Oprymea (1 per day)",
    "Escalebo (cada 4 horas), Mirapesin (1 al dia), Xadago (1 al dia)":
        "Stalevo (every 4 hours), Mirapexin (1 per day), Xadago (1 per day)",
    "Sinemet plus (3 al dia), ":
        "Sinemet Plus (3 per day)",
    "Sinemet (3 al dia)":
        "Sinemet (3 per day)",
    "no sabe":
        "Does not know",
    "Sinemet (4 y medio al dia), Xadago (2 al dia), Pramipexol (1 al dia)":
        "Sinemet (4.5 per day), Xadago (2 per day), Pramipexole (1 per day)",
    "Dodopa (permanente al est\xf3mago), sinemet retard (1 al dia), Requip (4mg), Amantadina (2 al dia), Xadago (50mg), estimulacion cerebral profunda (DECP)":
        "Duodopa (permanent via stomach), Sinemet retard (1 per day), Requip (4mg), Amantadine (2 per day), Xadago (50mg), deep brain stimulation (DBS)",
    "Madopar (3/4 de pastilla al dia)":
        "Madopar (3/4 tablet per day)",
    "no recuerda (Miguel Buena\xf1o Pastor), preguntar a Aparkam":
        "Does not remember (ask Aparkam)",
    "Sinemet retard (1 al dia), Sinemet (3 al d\xeda), Xadago (1 al dia), Requip (2 al dia), una nueva que no recuerda":
        "Sinemet retard (1 per day), Sinemet (3 per day), Xadago (1 per day), Requip (2 per day), a new one they do not remember",
    "Sinemet (6 al dia), Sinemet retard (1 al dia), Xadago (1 al dia), Neupro (parche)":
        "Sinemet (6 per day), Sinemet retard (1 per day), Xadago (1 per day), Neupro (patch)",
    "Mirapexin (1 al dia), Sinemet (5'5 al dia), Sinemet retard (1 al dia), Xadago (1 al dia)":
        "Mirapexin (1 per day), Sinemet (5.5 per day), Sinemet retard (1 per day), Xadago (1 per day)",
    "Sinemet plus (6 al dia), Sinemet retard (1 al dia), ":
        "Sinemet Plus (6 per day), Sinemet retard (1 per day)",
    "Sinemet (3 al dia), ":
        "Sinemet (3 per day)",
    "Mirapexin (1 al dia), Sinemet plus (4 al dia), Ongentys (1 al dia)":
        "Mirapexin (1 per day), Sinemet Plus (4 per day), Ongentys (1 per day)",
    "Sinemet (4 al dia), Sienemt retard (1 al dia), estimulador cerebral, ":
        "Sinemet (4 per day), Sinemet retard (1 per day), brain stimulator",
    "Oprymea (1 al dia), Sinemet (6 al dia),  ":
        "Oprymea (1 per day), Sinemet (6 per day)",
    "Sinemet (5 al dia), Xadago (1 al dia), ":
        "Sinemet (5 per day), Xadago (1 per day)",
    "Luis Lopez Navas (no sabe etapa ni medicacion, preguntar Aparkam)":
        "Does not know stage or medication (ask Aparkam)",
    "Sinemet (6 al dia), no esta seguro y tampoco de la etapa":
        "Sinemet (6 per day), not sure about medication or stage",
    "Amlodopino 1 vez al d\xeda (desde hace 5 a\xf1os) y Atorvastatina 1 vez al d\xeda (desde hace +15 a\xf1os)":
        "Amlodipine 1 time per day (for 5 years) and Atorvastatin 1 time per day (for +15 years)",
    "Metformina 1 comprimido por comida cada 12 horas (desde hace 8 a\xf1os), Losartan 1 comprimido al d\xeda (desde hace 8 a\xf1os), aspirina con un 1 comprimido por la ma\xf1ana (desde hace 3 a\xf1os)":
        "Metformin 1 tablet per meal every 12 hours (for 8 years), Losartan 1 tablet per day (for 8 years), Aspirin 1 tablet in the morning (for 3 years)",
    "Paracetamol seg\xfan necesidad cada 8h (desde hace 2 a\xf1os), Vitaminas de Calcio y Vitamina D 2 veces al d\xeda con comida (desde hace 4 a\xf1os)":
        "Paracetamol as needed every 8h (for 2 years), Calcium vitamins and Vitamin D 2 times per day with food (for 4 years)",
    "Ibuprofeno 1 comprimido por la noche (desde hace 2 a\xf1os), medicina para el h\xedgado 1 vez al d\xeda (desde hace 5 a\xf1os)":
        "Ibuprofen 1 tablet at night (for 2 years), liver medication 1 time per day (for 5 years)",
    "Medicina para la hipertensi\xf3n 1 vez al d\xeda (desde hace 5 a\xf1os), medicina para la diabetes 2 veces al d\xeda (desde hace +5 a\xf1os)":
        "Hypertension medication 1 time per day (for 5 years), diabetes medication 2 times per day (for +5 years)",
    "Bisoprolol 1 comprimido al d\xeda (desde hace 6 a\xf1os), vitaminas 2 veces al d\xeda (desde hace +10 a\xf1os), donepezilo 1 vez al d\xeda (no recuerda, apr\xf3ximadamente desde hace +5 a\xf1os)":
        "Bisoprolol 1 tablet per day (for 6 years), vitamins 2 times per day (for +10 years), donepezil 1 time per day (does not remember, approximately for +5 years)",
    "-": "-",
    "Atorvastatina 1 vez al d\xeda (desde hace +5 a\xf1os), Ibuprofeno cada 6 horas (desde hace 10 a\xf1os apr\xf3ximadamente)":
        "Atorvastatin 1 time per day (for +5 years), Ibuprofen every 6 hours (for approximately 10 years)",
    "Omeprazol una vez al d\xeda (desde hace 3 a\xf1os), forosemida una vez al d\xeda (desde hace 5 a\xf1os)":
        "Omeprazole once per day (for 3 years), furosemide once per day (for 5 years)",
    "Deficiencia pulmonar y card\xedaca":
        "Pulmonary and cardiac deficiency",
    "No tiene definici\xf3n ocular (muy elevado)":
        "No ocular prescription (very high)",
    "Sorda ":
        "Deaf",
    "Artrosis y Artrititis":
        "Osteoarthritis and Arthritis",
    "Marcapasos - nueva v\xe1lvula":
        "Pacemaker - new valve",
    "Sordera":
        "Deafness",
    "Nada":
        "None",
    "Nada/bast\xf3n":
        "None / uses cane",
    "Fractura de pelvis/bast\xf3n":
        "Pelvis fracture / uses cane",
    "Artrosis/artritis":
        "Osteoarthritis / Arthritis",
    "Hipertensi\xf3n, problemas de movilidad":
        "Hypertension, mobility problems",
    "Artrosis":
        "Osteoarthritis",
    "Arritm\xeda (Sintr\xf3n)":
        "Arrhythmia (Sintrom / Warfarin)",
    "Ri\xf1ones":
        "Kidney issues",
}
df["Pharmacology_and_Dosage"] = df["Pharmacology_and_Dosage"].map(pharma_map).fillna(df["Pharmacology_and_Dosage"])

# ── Symptoms_related_to_Parkinsons_disease ────────────────────────────────────
symptoms_map = {
    "discinesias, perdida de estabilidad ":
        "Dyskinesia, loss of stability",
    "rigidez en la parte derecha":
        "Rigidity on the right side",
    "mucha rigidez, inestabilidad especialmente en los giros (lleva baston), discinesias":
        "Severe rigidity, instability especially when turning (uses cane), dyskinesia",
    "temblor de manos":
        "Hand tremor",
    "Inestabilidad":
        "Instability",
    "inestabilidad, discinesias en ambas manos y tronco":
        "Instability, dyskinesia in both hands and trunk",
    "Inestabilidad, temblor en lado izquierdo y algo de rigidez en la espalda":
        "Instability, tremor on the left side and some back rigidity",
    "Discinesia por las ma\xf1anas sobre todo, movilidad lado izquierdo en manos y pies":
        "Dyskinesia mainly in the mornings, left-side mobility issues in hands and feet",
    "Temblor y dificultad para manipular en las manos, temblor en las piernas":
        "Tremor and difficulty manipulating with hands, tremor in the legs",
    "temblor mano derecha, sudoraci\xf3n":
        "Right hand tremor, sweating",
    "bloqueos, mucha discinesia en todo el cuerpo":
        "Freezing episodes, severe dyskinesia throughout the body",
    "Temblor en ambas piernas, rigidez en cuello y tronco, rigidez en mano izquierda":
        "Tremor in both legs, rigidity in neck and trunk, rigidity in the left hand",
    "un poco de perdida de habla, discinesias en ambos lados, rigidez":
        "Some speech loss, dyskinesia on both sides, rigidity",
    "discinesia mano derecha, rigidez en gemelos, arrastra pie izquierdo, camina lento, a veces boca carraspera":
        "Right hand dyskinesia, calf rigidity, drags left foot, walks slowly, sometimes hoarse voice",
    "falta de olfato, estre\xf1imiento, problemas visuales (distinguir colores), discinesias en mano derecha y un poco pierna derecha, movimientos involuntarios (tirarse de la cama), calambres":
        "Loss of smell, constipation, visual problems (distinguishing colours), dyskinesia in right hand and slightly right leg, involuntary movements (falling out of bed), cramps",
    "mucha rigidez, mucha discinesia, ":
        "Severe rigidity, severe dyskinesia",
    "movimiento lento, tartamudeo, lado derecho (discinesia mano), movilidad de las piernas":
        "Slow movement, stuttering, right side (hand dyskinesia), leg mobility",
    "Inestabilidad al levantarse, discinesias en manos y piernas, dificultad de manipulacion con las manos":
        "Instability when standing up, dyskinesia in hands and legs, difficulty manipulating with hands",
    "malestar, dificultad para andar, se cae mucho":
        "Discomfort, difficulty walking, falls frequently",
    "dificultad en el habla, memoria, ":
        "Speech difficulty, memory issues",
    "dificultad para andar, dificultad para hablar, inestabilidad, ":
        "Difficulty walking, difficulty speaking, instability",
    "-": "-",
}
df["Symptoms_related_to_Parkinsons_disease"] = (
    df["Symptoms_related_to_Parkinsons_disease"]
    .map(symptoms_map)
    .fillna(df["Symptoms_related_to_Parkinsons_disease"])
)

# ── Are_you_currently_experiencing... ─────────────────────────────────────────
current_map = {
    "discinesia":
        "Dyskinesia",
    "no, por la medicaci\xf3n":
        "No, due to medication",
    "discinesia, inestabilidad":
        "Dyskinesia, instability",
    "temblor de manos":
        "Hand tremor",
    "Inestabilidad":
        "Instability",
    "leve discinesia":
        "Mild dyskinesia",
    "un poco de temblor en la pierna":
        "Slight leg tremor",
    "un poco de discinesia en mano y pierna":
        "Slight dyskinesia in hand and leg",
    "no":
        "No",
    "ninguno":
        "None",
    "levemente, son m\xe1s por la tarde":
        "Mildly, more in the afternoon",
    "los mismos":
        "The same",
    "discinesias":
        "Dyskinesia",
    "gemelos, discinesia mano y ronquera":
        "Calf issues, hand dyskinesia and hoarseness",
    "dice casi normal (pero tiene muchisima discinesia)":
        "Says almost normal (but has severe dyskinesia)",
    "movilidad de las piernas, discinesia mano":
        "Leg mobility issues, hand dyskinesia",
    "discinesias (mano derecha especialmente)":
        "Dyskinesia (right hand especially)",
    "no (dificultad para hablar)":
        "No (difficulty speaking)",
    "-": "-",
}
col_exp = "Are_you_currently_experiencing_any_of_these_symptoms_If_so_to_what_extent"
df[col_exp] = df[col_exp].map(current_map).fillna(df[col_exp])

df.to_csv(CSV_PATH, sep=";", index=False, encoding="utf-8-sig")
print("Saved:", CSV_PATH)

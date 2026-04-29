# Motion Insight Hub
## Dataset for Dyskinesia Assessment in Parkinson’s Disease

This repository provides the **anonymized motion datasets, metadata, and documentation**
associated with the PeerJ manuscript:

**A Technical Framework for Detecting Dyskinesia in Parkinson’s Disease  
Using Motion Tracking and Serious Games**

Carlos Rodrigo-Rivero, Nikola Hristov-Kalamov, Raúl Fernández-Ruiz,  
Carlos Garre del Olmo, Agustín Álvarez-Marquina, Rafael Martínez-Olalla,
Paulo Peixoto and Daniel Palacios‑Alonso.

---

## Project website

The official project website is:

👉 **https://motioninsighthub.web.app/**

The website provides:
- interactive visualisations of the recorded motion data,
- exploratory plots and summaries of kinematic variables,
- public dissemination material related to the project.


## Platform

Data were collected using **Motion Insight Hub**, a research web platform
developed by the **SA‑BIO Research Group** at **Universidad Rey Juan Carlos (URJC)**,
Madrid, Spain.

Motion Insight Hub is designed to support:
- acquisition of motion-derived digital biomarkers,
- analysis of motor performance in neurological disorders,
- reproducible research in Human–Computer Interaction and digital health.

---

## Dataset overview

### Participants
- **Total participants**: 46  
  - 20 Parkinson’s disease (PD) participants  
  - 26 normative participants  
- **Participants used for ML analysis**: 44  
  (two young reference users excluded from kinematic analysis)
- **PD severity**: Hoehn and Yahr stages 1.0–3.0

All participants provided informed consent.
The protocol was approved by the **Ethics Committee of Universidad Politécnica de Madrid (UPM)**.

---

## Data acquisition

Motion data were captured using the **Leap Motion Controller (v1.0)** integrated into
custom-designed serious games developed in **Unity 2022.3.10f1**.

Three exergames were designed to elicit distinct motor patterns:

1. **Baseline tremor assessment**  
   Static bilateral posture to quantify resting tremor and involuntary motion.
2. **Grip open/close task**  
   Repetitive hand opening and closing to assess bradykinesia, dexterity, and symmetry.
3. **Vertical obstacle-avoidance task**  
   Repetitive vertical arm motion to evaluate reaction time, coordination,
   and fatigue-induced dyskinesia (three visual variants).

The Leap Motion Controller records bilateral hand kinematics without physical contact.

---

## Recorded signals and features

Raw data consist of frame-level CSV recordings including:
- Hand palm position (x, y, z)
- Hand speed and velocity components
- Palm orientation vectors
- Fingertip positions
- Grip strength
- Frame timestamps and device metadata

From these signals, **62 kinematic, temporal, and spectral features** are extracted,
covering:
- tremor amplitude and frequency,
- movement speed and smoothness,
- grip dynamics,
- bilateral symmetry and variability.

---

## Machine learning use

The dataset supports classification between:
- **Normative participants**
- **Non-normative (Parkinson’s disease) participants**

Models evaluated in the paper include:
- Support Vector Machines (RBF kernel)
- Random Forests
- Grouped-feature Random Forest ensembles

Evaluation is performed using **Leave-One-Out Cross-Validation (LOOCV)**.

---

## Privacy and ethics

- No personally identifiable information (PII) is included.
- No video or image recordings of participants are shared.
- Only anonymized kinematic and derived feature data are provided.
- Data cannot be used to re-identify participants.

Use of this dataset must comply with ethical research standards and data‑protection laws.

---

## Intended use

This dataset is intended for:
- reproducibility of the PeerJ publication,
- research on dyskinesia and motor impairment in Parkinson’s disease,
- development and comparison of machine learning methods for motion analysis,
- HCI research using serious games and contactless tracking.

It is **not intended for clinical diagnosis**.

---

## Citation

If you use this dataset, please cite:

> Rodrigo‑Rivero, C., Hristov‑Kalamov, N., Fernández‑Ruiz, R., et al.  
> *A Technical Framework for Detecting Dyskinesia in Parkinson’s Disease Using Motion Tracking and Serious Games*.  
> PeerJ, submitted.

---

## Funding

This work was supported by the **State Research Agency of Spain (AEI)**  
under Grant **CarHaVoz: PID2023‑152984OB‑I00**.

---

## Contact

**SA‑BIO Research Group**  
Universidad Rey Juan Carlos (URJC), Madrid, Spain  

Corresponding author:  
**Daniel Palacios‑Alonso** – daniel.palacios@urjc.es

---

## License

This dataset is released for **non‑commercial research and educational use only**.
Refer to the LICENSE file for details.

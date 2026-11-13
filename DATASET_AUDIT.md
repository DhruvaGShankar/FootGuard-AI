# FootGuard AI — Comprehensive Dataset Audit Report
**Phase 1 Deliverable | Senior Machine Learning Engineering**  
**Date:** November 8, 2026  
**Working Directory:** `FootGuard AI/ML/`  
**Raw Data Directory:** `FootGuard AI/ML/raw/` (Read-Only)

---

## Executive Summary

A comprehensive, non-destructive audit of all raw datasets within `FootGuard AI/ML/raw/` was conducted. The repository contains **7,493 raw files** partitioned across two physical modalities:
- **Pressure & Shear Modality:** 5 distinct experimental collections (2,693 files) spanning mechanical bench-top calibration, gait-lab insole sensor streams, and dynamic 2D/3D pedobarography.
- **Thermal Modality:** 3 independent thermal collections (4,800 files) spanning calibrated thermal matrices (Celsius), multi-timepoint cold-stress rewarming sequences, and dual-modality (RGB + Thermal) screening cohorts.

### Paramount Findings & Data Integrity Guardrails
1. **Zero Diabetic Foot Ulcer (DFU) Presence / No Ulcer Labels:**  
   **None** of the 8 datasets contain active diabetic foot ulcers, wound progression labels, healing timecourses, or future ulcer onset annotations. 
   - *ThermoDataBase* and *STANDUP_Database* contain diabetes vs. non-diabetic controls (and IWGDF neuropathy/PAD risk grades in STANDUP).
   - *IAC-TecMed* contains **100% non-diabetic healthy individuals** undergoing cold-stress autonomic recovery.
   - *CAD_WALK* contains **100% healthy control individuals**.
   - *Gait Lab Validation* contains exactly **1 diabetic subject** and **1 healthy subject**.
   - *Rule Violation Prevention:* Under no circumstance will ulceration, ulcer healing duration, or tissue breakdown prediction be claimed or modeled.

2. **Distinction Between Calibrated Thermal Matrices vs. Pseudo-Color Images:**  
   - Calibrated Celsius float values exist in *ThermoDataBase* (as float64 CSV grids) and *IAC-TecMed* (as 32-bit floating-point TIFFs).
   - The PNG images in *STANDUP_Database* and *ThermoDataBase* are 8-bit palette-indexed or RGB false-color renderings. They do **not** represent linear calibrated physical temperatures.

3. **Participant-Level Splitting & Anti-Leakage Constraints:**  
   - In *ThermoDataBase* ($N=167$ subjects) and *CAD_WALK* ($N=51$ subjects), all trials and bilateral feet from the same subject must strictly reside within the same split (Train, Val, or Test).
   - In *Gait Lab Validation*, because $N=2$ (1 diabetic, 1 healthy), training a supervised binary classifier for diabetes is scientifically invalid, as the model would trivially overfit to individual biometric traits rather than pathology.

---

## 1. Global Dataset Inventory

| Modality | Dataset Name (Exact Path Relative to `ML/raw/`) | Primary File Formats | File Count | Subject / Sample Count | Primary Clinical / Experimental Attributes |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **Pressure** | `Pressure-Data/1-Calibration Investigation` | `.mat`, `.csv`, `.txt`, `.m` | 139 | Bench mechanical testing | Sensor area & position calibration (Instron vs sensor) |
| **Pressure** | `Pressure-Data/2 - Loading Profile Comparison` | `.csv`, `.mat`, `.txt`, `.xls`, `.m` | 49 | Bench indenter tests | Indenter diameters (15, 20, 30, 40 mm), normal stress profiles |
| **Pressure** | `Pressure-Data/3 - Bench Top Validation` | `.mat`, `.txt`, `.m` | 16 | Bench multi-angle tests | Multi-axial shear & normal loading (14 kPa, 70 kPa; Lat, Pos, Med, Ant, Angled) |
| **Pressure** | `Pressure-Data/4 - Gait Lab Validation` | `.csv`, `.mat`, `.xlsx`, `.m`, `.txt` | 24 | **2 subjects** (1 Diabetic, 1 Healthy) | Treadmill walking; 20-channel insole stream (normal + tri-axial shear) |
| **Pressure** | `Pressure-Data/CAD_WALK_Healthy_Controls_Dataset (1)` | `.nii`, `.xlsx`, `.pdf` | 2,465 | **51 subjects** (C01–C51); 2,413 dynamic trials | Dynamic vertical pedobarography (38×22 grid at 500 Hz); Walking speeds |
| **Thermal** | `Thermal-Data/IAC-TecMed Database_1` | `.png`, `.tiff`, `.csv`, `.txt` | 628 | **19 subjects** (All Non-Diabetic); 76 timepoints | Cold-stress autonomic rewarming (T0, T5, T10, T15); 32-bit float Celsius TIFFs + Depth + IR |
| **Thermal** | `Thermal-Data/STANDUP_Database` | `.png` | 830 | 415 paired RGB-Thermal captures | Diabetes risk stratification: Healthy ($n=125$), Diabetic R0 ($n=138$), R1 ($n=68$), R2 ($n=84$) |
| **Thermal** | `Thermal-Data/ThermoDataBase` | `.csv`, `.png`, `.xlsx` | 3,342 | **167 subjects** (45 Control, 122 Diabetic) | Calibrated Celsius CSV matrices (~148×57) & 4 angiosomes (LCA, LPA, MCA, MPA) |
| **Total** | **8 Datasets** | — | **7,493** | — | — |

---

## 2. In-Depth Audit: Pressure & Shear Data

### 2.1 CAD_WALK Healthy Controls Dataset
- **Directory:** `raw/Pressure-Data/CAD_WALK_Healthy_Controls_Dataset (1)/HealthyControls/`
- **Citation / Provenance:** Booth et al., June 2018, Zenodo (`doi:10.5281/zenodo.1265420`), Sint Maartenskliniek, Nijmegen.
- **Subjects:** 51 healthy Dutch participants (folders `C01` through `C51`).
  - Note: Metadata mentions C01–C55, but the filesystem contains exactly 51 folders (C01 to C51). C01–C50 have 48 trials each (24 left foot, 24 right foot); C51 has 13 trials. Total NIfTI trials = 2,413.
- **Data Format & Tensor Specifications:**
  - Standard NIfTI-1 format (`.nii`), uncompressed, 348-byte header.
  - Spatial Grid: $38 \times 22$ pressure cells.
  - Spatial Resolution: Non-square sensor elements, $\Delta x = 7.62\text{ mm}$ (AP) $\times \Delta y = 5.08\text{ mm}$ (ML).
  - Temporal Dimension: Variable sequence length $T \in [250, 500]$ frames per footstep stance phase.
  - Sampling Frequency: $500\text{ Hz}$ ($\Delta t = 2.0\text{ ms}$).
  - Precision: 64-bit double-precision floating-point (`datatype=64`, `bitpix=64`).
- **Metadata Available:**
  - `metadata.xlsx`: Subject ID, Height (cm), Weight (kg), Age (years), Sex (M/F), Shoe Size (EU), Handedness, Leg Dominance.
  - `walking_speeds_C##.xlsx`: Per-trial stance metrics (Walking speed, Stride length, Stride time).
- **Label Truth:** 100% Healthy Controls. **Zero diabetic or ulcer cases.**
- **Viable ML Tasks:**
  - 2D Spatial Pressure Pattern Feature Extraction (Peak Plantar Pressure, Pressure-Time Integral).
  - 1D/TCN Temporal Roll-Off Modeling: Stance phase temporal dynamics (Heel Strike $\rightarrow$ Midstance $\rightarrow$ Push-off).
  - Demographic / Gait Regression: Predicting gait speed or stance duration from spatiotemporal pressure sequences.
  - Self-Supervised / Autoencoding: Learning normative spatiotemporal plantar pressure representations for anomaly detection.

### 2.2 Gait Lab Validation
- **Directory:** `raw/Pressure-Data/4 - Gait Lab Validation/`
- **Citation / Provenance:** Athia Haron, University of Manchester (UoM), 2024 ("Happier Feet" custom sensor insole).
- **Subjects:** Exactly **2 participants**:
  - `Diabetic Participant`: Identified as `HFMD1` (2 walking runs: W1, W2; bilateral feet: LEFT, RIGHT).
  - `Healthy Participant`: Identified as `HFM2` (2 walking runs: W1, W2; bilateral feet: LEFT, RIGHT).
- **Data Format & Signal Channels:**
  - 8 raw CSV files, ~72,001 rows $\times$ 20 columns. Continuous sampling during treadmill walking.
  - Column 0: Foot designation (`LEFT` or `RIGHT`).
  - Column 1: Time in milliseconds ($t$).
  - Columns 2–9 ($F_1 \dots F_8$): Normal stress sensor voltages (V), specifically Metatarsal Head ($F_1$), Hallux ($F_2$), and Heel ($F_8$).
  - Columns 10–18 ($S_1 \dots S_9$): Tri-axial shear sensor signals (mV) corresponding to Metatarsal Head (MH), Hallux (HA), and Heel (HE) in anterior-posterior (AP) and medial-lateral (ML) axes.
  - Column 19: Hardware status flag.
- **Accompanying Artifacts:**
  - `Calibration Index.xlsx`: Calibration shear gains across LOW, MED, HIGH loading.
  - `*.mat`: Normal stress calibration profiles (`NormalData.FFgain`).
  - `FixJumpv24.m`: Artifact removal script handling baseline step jumps via changepoint analysis (`ischange`) and spike interpolation (`filloutliers`).
- **Critical Limitation & Negative Rule:**
  - **Supervised classification between diabetic and healthy is scientifically forbidden on this dataset.** With $N=1$ diabetic and $N=1$ control, any machine learning model would merely classify the individual walking idiosyncrasies (stride length, cadence, weight) of `HFMD1` vs `HFM2`.
- **Viable ML Tasks:**
  - Signal conditioning, artifact removal, and shear/normal cross-coupling filtering.
  - 1D CNN / TCN self-supervised step segmentation and stance phase extraction across multi-channel sensor signals.

### 2.3 Bench-Top Validation, Loading Profile Comparison, & Calibration Investigation
- **Directories:**
  - `raw/Pressure-Data/1-Calibration Investigation/` (139 files)
  - `raw/Pressure-Data/2 - Loading Profile Comparison/` (49 files)
  - `raw/Pressure-Data/3 - Bench Top Validation/` (16 files)
- **Nature of Data:** Pure electromechanical bench testing using Instron mechanical loading rigs.
  - Indenter testing across fixed circular diameters: 10mm, 15mm, 20mm, 25mm, 30mm, 40mm.
  - Directional shear validation: Pure Medial, Lateral, Anterior, Posterior, and 22.5° / 45° angled load vectors.
  - Calibrated normal loads: 14 kPa, 70 kPa, 140 kPa.
  - File formats: MATLAB v5 `.mat`, MATLAB v7.3 HDF5 `.mat`, Instron `.csv` time series.
- **Role in FootGuard AI:** Non-clinical physical reference standards. Used exclusively for physical unit validation, sensor response characterization, and transfer function benchmarking.

---

## 3. In-Depth Audit: Thermal Data

### 3.1 ThermoDataBase
- **Directory:** `raw/Thermal-Data/ThermoDataBase/ThermoDataBase/`
- **Provenance:** Standardized Clinical Plantar Thermogram Database.
- **Cohorts & Subjects:** Total $N=167$ distinct human participants:
  - **Control Group (CG):** 45 participants (`CG001_M` to `CG045_F`).
  - **Diabetic Group (DM):** 122 participants (`DM001_M` to `DM122_F`).
- **File Completeness:** Exactly **100% complete** across all 167 subjects:
  - Full foot: `_L.csv`, `_L.png`, `_R.csv`, `_R.png` ($167 \times 4 = 668$ files).
  - Angiosome sub-matrices: 4 anatomical angiosomes per foot (LCA: Lateral Calcaneal Artery, LPA: Lateral Plantar Artery, MCA: Medial Calcaneal Artery, MPA: Medial Plantar Artery) in both `.csv` and `.png` ($167 \times 16 = 2,672$ files).
- **Matrix Characteristics & Physical Temperature Verification:**
  - CSV files contain **true physical temperatures in degrees Celsius** as floating-point numbers.
  - Background non-foot pixels are zero-masked (`0.0`). Active plantar tissue pixels range strictly between $23.0^\circ\text{C}$ and $35.5^\circ\text{C}$.
  - Spatial Dimensions: **Variable bounding-box dimensions** per participant foot.
    - Height / Rows range: 103 to 200 pixels.
    - Width / Columns range: 41 to 88 pixels.
    - Typical aspect ratio: ~2.3:1 to 2.5:1.
  - Image files: 8-bit PNG false-color thermograms.
- **Tabular Metadata (`Plantar Thermogram Database.xlsx`):**
  - Columns: Subject, Gender (M/F), Age (years), Weight (kg), Height (m), IMC (BMI), regional angiosome temperature readings.
  - Data Quality Flag: Subject `DM003` contains missing values (`NaN`) for Weight, Height, and BMI.
- **Label Truth:** Binary grouping: **Control (Non-Diabetic) vs. Diabetic**.
  - **Zero ulcer annotations.**
- **Viable ML Tasks:**
  - **Thermal 2D CNN Baseline:** Supervised classification of Diabetes vs. Control using spatial temperature distribution and bilateral asymmetry ($\Delta T = |T_{\text{left}} - T_{\text{right}}|$).
  - **Spatial Heat-Transfer / Feature Extraction:** Angiosome-specific perfusion feature mapping.

### 3.2 IAC-TecMed Database
- **Directory:** `raw/Thermal-Data/IAC-TecMed Database_1/IAC-TecMed Database/`
- **Provenance:** IAC-TecMed Thermal Cold-Stress Challenge Database.
- **Subjects:** 19 distinct participants (`ADM001`, `CTA002`, `CTA003`, `ESP001`–`ESP010`, `EST004`, `EXT003`, `MED001`, `MED002`, `MED006`, `NRT001`–`NRT004`).
- **Cohort Pathology:** **100% Non-Diabetic Controls** (`Diabetic: No` confirmed across all 19 metadata records).
- **Temporal Structure:** Exactly **4 sequential observation timepoints** per participant:
  - $T_0$: Baseline resting thermal distribution.
  - $T_5$: 5 minutes post-thermal challenge (rewarming phase 1).
  - $T_{10}$: 10 minutes post-challenge (rewarming phase 2).
  - $T_{15}$: 15 minutes post-challenge (stabilization phase).
- **Multi-Modal Data Channels per Subject ($19 \times 4$ timepoints):**
  - `_Tmap_T*.tiff`: **Calibrated 32-bit floating point TIFF** ($288 \times 384$, `Mode: F`). Pixel values are direct Celsius temperatures (observed range: $19.0^\circ\text{C} \le T \le 36.2^\circ\text{C}$).
  - `_Depth_T*.png`: $288 \times 384$ distance/depth camera map.
  - `_IR_T*.png`: 8-bit infrared intensity rendering.
  - `_RGB_T*.png`: 24-bit color digital photograph.
  - `_RGB_Mask_L_T*.png` & `_RGB_Mask_R_T*.png`: Binary anatomical masks isolating left and right plantar surfaces.
- **Metadata Format:** `Metadata_<Subject>.csv`. Semicolon-delimited (`sep=';'`). Contains Age, Sex, Diabetic status, Foot size (EUR), Root temp, Left/Right foot mean temp, Frustum distance.
- **Crucial Clinical Guidance:**
  - $T_0 \rightarrow T_{15}$ represents **thermal recovery under controlled cold-stress challenge**, which measures microvascular autonomic vasomotor tone.
  - **It does NOT represent ulcer chronicity, wound healing time, or diabetic progression.**
- **Viable ML Tasks:**
  - **Thermal Temporal Analysis (TCN):** Modeling the nonlinear autonomic thermal recovery curve $T(t)$ across sequential timepoints.
  - Spatial-temporal feature encoding using combined CNN-TCN backbones.

### 3.3 STANDUP Database
- **Directory:** `raw/Thermal-Data/STANDUP_Database/STANDUP_Database/`
- **Provenance:** STANDUP Plantar Thermogram & RGB Screening Database.
- **File Structure & Counts:** Exactly 415 paired observations ($415 \times 2 = 830$ PNG files):
  - `healthy`: 125 thermal PNGs + 125 paired RGB PNGs.
  - `diabetic/R0`: 138 thermal PNGs + 138 paired RGB PNGs (Risk 0: No peripheral neuropathy or PAD).
  - `diabetic/R1`: 68 thermal PNGs + 68 paired RGB PNGs (Risk 1: Neuropathy or PAD present, no deformity).
  - `diabetic/R2`: 84 thermal PNGs + 84 paired RGB PNGs (Risk 2: Neuropathy + PAD, or deformity / previous ulcer history).
  - Pair matching: 100% symmetric name matching between RGB and thermal folders.
- **Data Format:**
  - Thermal: 8-bit palette-indexed PNG (`Mode: P` or RGB, $514 \times 385$).
  - RGB: 24-bit color PNG ($514 \times 385$).
  - **Uncalibrated:** No direct temperature matrix (CSV or TIFF) is supplied. Values are false-color colormap indices (0–251).
- **Label Truth:**
  - Diabetic vs. Healthy.
  - IWGDF Neuropathy/PAD Risk Category ($R_0, R_1, R_2$).
  - **No active ulcer segmentation or healing labels.**
- **Viable ML Tasks:**
  - Multi-class risk categorization ($H, R_0, R_1, R_2$) using 2D CNN on false-color thermograms.
  - Representation learning comparing diabetic risk tiers.

---

## 4. Unsupported ML Predictions & Prohibited Modeling Tasks

To guarantee clinical validity and ethical machine-learning engineering, the following tasks are formally deemed **unsupported** by the available data and are strictly barred:

```
+---------------------------------------------------------------------------------------------------+
|                                  STRICTLY PROHIBITED TASKS                                        |
+------------------------------------+--------------------------------------------------------------+
| Proposed Task                      | Scientific / Empirical Barrier                               |
+------------------------------------+--------------------------------------------------------------+
| 1. Active Ulcer Detection /        | ZERO datasets contain open wound imagery, ulcer bounding      |
|    Ulcer Segmentation              | boxes, or wound annotations.                                 |
+------------------------------------+--------------------------------------------------------------+
| 2. Ulcer Healing Duration /        | Neither IAC-TecMed nor any other dataset tracks wound healing|
|    Tissue Recovery Prediction      | over days/weeks. IAC-TecMed timepoints are minutes of         |
|                                    | autonomic cold rewarming, not clinical healing.              |
+------------------------------------+--------------------------------------------------------------+
| 3. Diabetic Classification on      | The dataset contains N=1 diabetic and N=1 healthy subject.   |
|    Gait Lab Insole Data            | Any classifier will achieve 100% accuracy by memorizing      |
|                                    | individual walking biometrics, yielding zero generalization. |
+------------------------------------+--------------------------------------------------------------+
| 4. Diabetic Classification on      | All 51 subjects (2,413 trials) are healthy controls. No     |
|    CAD_WALK Pedobarography         | pathological counter-class exists.                           |
+------------------------------------+--------------------------------------------------------------+
| 5. Direct Celsius Temperature      | PNG files in STANDUP are 8-bit false-color colormaps without |
|    Regression from STANDUP PNGs    | thermal calibration curves or radiometric metadata.          |
+---------------------------------------------------------------------------------------------------+
```

---

## 5. Supported Machine Learning Experiments

Based on verified ground truth and sample distributions, the following experimental pipelines are valid, reproducible, and supported:

### 5.1 Pipeline A: Thermal Analysis
1. **Experiment A1 — 2D CNN Plantar Diabetes Classification (ThermoDataBase):**
   - **Input:** Calibrated temperature matrices ($H \times W$ Celsius float arrays padded/resized to a uniform standard, e.g., $160 \times 80$, with zero background and standardized foreground).
   - **Bilateral Features:** Dual-channel input ($\text{Foot}_{\text{Left}}$, $\text{Foot}_{\text{Right}}$) or explicit asymmetry map $\Delta T(x, y) = |T_L(x, y) - \text{flip}(T_R(x, y))|$.
   - **Ground Truth:** Binary label (0 = Control, 1 = Diabetic).
   - **Partitioning:** Stratified subject-level split: 116 Train subjects (~70%), 25 Val subjects (~15%), 26 Test subjects (~15%). Zero subject overlap.
   - **Architecture:** Lightweight 2D ConvNet baseline (3–4 convolutional stages + BatchNorm + Dropout + GAP).

2. **Experiment A2 — Thermal Autonomic Temporal Analysis (IAC-TecMed):**
   - **Input:** Sequential calibrated thermal matrices across 4 rewarming timepoints ($T_0, T_5, T_{10}, T_{15}$) extracted from 32-bit float TIFFs ($288 \times 384$).
   - **Target / Task:** Temporal feature extraction and predictive rewarming trajectory modeling (autonomic thermal inertia).
   - **Architecture:** 2D CNN spatial encoder + 1D Temporal Convolutional Network (TCN) with causal dilated convolutions.
   - **Partitioning:** Participant-level split: 13 Train subjects, 3 Val subjects, 3 Test subjects.

3. **Experiment A3 — Exploratory Diabetes & Neuropathy Risk Classification (STANDUP):**
   - **Input:** 2D Thermal screening images ($514 \times 385 \times 3$).
   - **Target:** 4-class risk classification (Healthy, $R_0, R_1, R_2$) or binary Diabetes classification.
   - **Architecture:** Transfer-learning or scratch 2D CNN with rigorous cross-validation.

### 5.2 Pipeline B: Pressure & Shear Analysis
1. **Experiment B1 — Spatiotemporal Plantar Pressure 1D/2D Feature Extraction (CAD_WALK):**
   - **Input:** $38 \times 22 \times T$ dynamic vertical pressure sequences from 2,413 walking trials.
   - **Task:** 
     - 2D Peak Plantar Pressure (PPP) and Pressure-Time Integral (PTI) spatial feature extraction.
     - 1D CNN / TCN modeling of the temporal ground-reaction curve across stance phases (0% to 100% stance time).
     - Regression to walking speed, stride length, and stride cadence.
   - **Partitioning:** Participant-level split across 51 subjects: 35 Train subjects (~70%), 8 Val subjects (~15%), 8 Test subjects (~15%). All 48 trials for any individual stay in their assigned split.

2. **Experiment B2 — Multi-Channel Insole Shear & Normal Signal Modeling (Gait Lab Validation):**
   - **Input:** 20-channel continuous time-series stream (Normal stress voltages $F_1, F_2, F_8$ + Tri-axial shear mV $S_1 \dots S_9$).
   - **Task:** Unsupervised stance phase segmentation, step-cycle detection, and sensor cross-talk denoising via 1D TCN / Temporal autoencoders.

---

## 6. Required Preprocessing Strategies

### 6.1 Thermal Preprocessing Specifications
1. **ThermoDataBase:**
   - **Parser:** Pure numeric CSV loader with automatic dimension detection.
   - **Background Handling:** Background pixels are strictly $0.0^\circ\text{C}$. To avoid biasing neural networks, create an explicit binary foot mask $M(x, y) = \mathbb{I}(T(x, y) > 15.0)$. Compute mean and standard deviation exclusively across masked plantar pixels ($M = 1$).
   - **Spatial Resizing / Padding:** Standardize variable shapes ($103\text{--}200 \times 41\text{--}88$) to fixed canonical tensor dimension (e.g., $160 \times 80$) using bilinear interpolation for continuous Celsius maps, or zero-padding maintaining physical aspect ratio.
   - **Bilateral Pairing:** Align Left and Right foot thermograms; horizontally reflect the Right foot to achieve anatomical symmetry with the Left foot for direct differencing ($\Delta T$).

2. **IAC-TecMed:**
   - **Parser:** Read 32-bit floating-point TIFFs using `PIL` or `imageio` (`mode='F'`).
   - **Metadata Parser:** Semicolon-delimited parser (`sep=';'`) with regex field sanitization to accommodate non-standard fields like `Frustum (mm);750, 850`.
   - **Temporal Alignment:** Stack chronologically sorted timepoints $[T_0, T_5, T_{10}, T_{15}]$ into tensor shape $(B, 4, H, W)$.

3. **STANDUP:**
   - Normalize uint8 pixel intensities to $[0, 1]$.
   - Image augmentation (minor rotation $\pm 5^\circ$, jitter) strictly preserving left/right foot orientation.

### 6.2 Pressure Preprocessing Specifications
1. **CAD_WALK NIfTI:**
   - **Parser:** Native NIfTI-1 reader extracting 64-bit float arrays $(38, 22, T)$.
   - **Spatial Aspect Ratio:** Account for non-square pixel dimensions ($\Delta x = 7.62\text{ mm}$, $\Delta y = 5.08\text{ mm}$, ratio $1.5:1$).
   - **Temporal Normalization:** Resample stance phases (variable $T \approx 300\text{--}450$ frames) to fixed 100 percentage points (0% to 100% of gait stance phase) using cubic spline interpolation.

2. **Gait Lab Validation CSV:**
   - **Parser:** Headerless CSV loader (`header=None`, 20 columns).
   - **Filtering:** Implement Python equivalent of `FixJumpv24.m`: changepoint detection for voltage baseline shifts, median filter for spike removal, and zero-phase Butterworth low-pass filter ($f_c = 15\text{ Hz}$).
   - **Calibration Gains:** Apply calibration formulas from `Calibration Index.xlsx` to convert raw sensor voltages to physical units ($\text{kPa}$ for normal stress, $\text{kPa}$ for AP/ML shear).

---

## 7. Next Steps for Phase 2 (Preprocessing Implementation)

Upon approval of this Phase 1 audit:
1. Establish `ML/processed/` directory trees for thermal and pressure data.
2. Generate deterministic, subject-independent split indices in `ML/splits/` with fixed random seed ($42$).
3. Implement `ML/thermal/preprocessing.py` and `ML/pressure/preprocessing.py` satisfying all unit preservation and anti-leakage rules.
4. Execute preprocessing pipelines and compile automated data quality reports verifying tensor shapes, non-zero gradients, and metric distributions.


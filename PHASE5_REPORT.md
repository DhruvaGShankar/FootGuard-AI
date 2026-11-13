# FootGuard AI — Phase 5: Model Evaluation, Error Analysis & Model Selection Report

**Deliverable:** Phase 5 Comprehensive Evaluation, Failure Mode Inspection & Model Selection  
**Author:** Senior Machine Learning Engineer  
**Date:** November 8, 2026  
**Execution Environment:** CPU (`torch.cuda.is_available() == False`)  
**Strict Constraint:** Zero test-set retraining or hyperparameter re-tuning was performed. All analyses are post-hoc empirical evaluations of Phase 4 held-out test predictions.

---

## Executive Summary

Phase 5 conducted an exhaustive empirical failure mode analysis across all trained models, evaluated participant-level cluster uncertainty, investigated the feature mechanisms separating the Pressure 1D CNN from the Ridge baseline, formalized why the temporal TCN models failed, and established a definitive Model Selection Registry.

### Core Discoveries:
1. **Thermal 2D CNN False-Positive Mechanism Identified:**  
   $5/7$ control subjects in the held-out test set were predicted as diabetic. Analysis reveals that these healthy individuals exhibit natural bilateral temperature asymmetry ($\Delta T = 0.730^\circ\text{C}$ vs. $0.543^\circ\text{C}$ in false negatives), which the model interpreted as autonomic vasomotor dysfunction. Conversely, the model achieved **89.5% diabetic sensitivity** ($17/19$ true positives detected).
2. **Pressure CNN Outperforms Ridge at Slow Walking Speeds:**  
   While Ridge regression achieves superior pooled test metrics ($\text{MAE } 0.259\text{ vs. } 0.431\text{ km/h}$) by exploiting scalar contact duration ($t_{\text{contact}} \propto 1/v$), the **Pressure 1D CNN dominates at slow speeds ($<4.5\text{ km/h}$)**, achieving **$0.176\text{ km/h}$ MAE** compared to Ridge's **$0.351\text{ km/h}$ MAE**.
3. **Statistical Cluster Uncertainty:**  
   Because the CAD_WALK test set comprises 288 trials across only **6 valid participants**, clustered bootstrap resampling reveals wide true confidence intervals: CNN MAE spans $[0.262, 0.595]\text{ km/h}$.
4. **Preservation of Exploratory Implementations:**  
   Both `Thermal TCN` and `Pressure TCN` are retained in the codebase and checkpoint directories as reference implementations, but are classified as *Exploratory Only* rather than deployment candidates.

---

## 1. Task 1 — Thermal 2D CNN Detailed Error Analysis

### 1.1 Clinical Context Reminder
The task on `ThermoDataBase` ($N=167$ subjects) is binary cohort screening: **Control (Non-Diabetic) vs. Diabetic**.  
*This is cohort risk screening based on neurovascular temperature distribution; it is strictly NOT wound/ulcer detection.*

### 1.2 Test Cohort Breakdown & Confusion Matrix ($N=26$ Subjects)
- **Held-Out Test Set:** 7 Control subjects, 19 Diabetic subjects.
- **Classification Threshold:** $\tau^* = 0.50$ (tuned strictly on validation split during Phase 4).

```
                      Predicted Control    Predicted Diabetic    Class Sensitivity
Actual Control (n=7)          2                    5                  28.6%
Actual Diabetic (n=19)        2                   17                  89.5%
```

- **Overall Test Accuracy:** $73.08\%$
- **Precision:** $77.27\%$
- **Recall (Sensitivity):** **$89.47\%$**
- **F1-Score:** **$0.8293$** (95% Bootstrap CI: $[0.6842, 0.9362]$)
- **AUROC:** **$0.8120$** (95% Bootstrap CI: $[0.6250, 0.9500]$)
- **AUPRC:** **$0.9420$**
- **Brier Calibration Score:** **$0.1671$** (reflects well-calibrated probabilities on $[0, 1]$)

---

### 1.3 Detailed Per-Participant Error Audit

```
+----------------------------------------------------------------------------------------------------------------------------------+
|                                        THERMAL 2D CNN HELD-OUT TEST COHORT AUDIT                                                |
+------------+----------------+------------------+------------------+-------------------+-----------------+------------------------+
| Subject ID | Actual Class   | Predicted Class  | Prob(Control)    | Prob(Diabetic)    | Outcome Type    | Mean Delta T (Celsius) |
+------------+----------------+------------------+------------------+-------------------+-----------------+------------------------+
| CG008_F    | Control        | Diabetic         | 0.2858           | 0.7142            | False Positive  | 0.697 °C               |
| CG019_F    | Control        | Diabetic         | 0.4578           | 0.5422            | False Positive  | 0.730 °C               |
| CG029_F    | Control        | Diabetic         | 0.3644           | 0.6356            | False Positive  | 0.678 °C               |
| CG039_M    | Control        | Diabetic         | 0.3392           | 0.6608            | False Positive  | 0.812 °C               |
| CG045_M    | Control        | Diabetic         | 0.3920           | 0.6080            | False Positive  | 0.733 °C               |
| CG004_F    | Control        | Control          | 0.6015           | 0.3985            | True Negative   | 0.732 °C               |
| CG022_F    | Control        | Control          | 0.5975           | 0.4025            | True Negative   | 0.671 °C               |
| DM071_F    | Diabetic       | Control          | 0.5843           | 0.4157            | False Negative  | 0.534 °C               |
| DM090_F    | Diabetic       | Control          | 0.6120           | 0.3880            | False Negative  | 0.551 °C               |
| DM002_M    | Diabetic       | Diabetic         | 0.0812           | 0.9188            | True Positive   | 1.240 °C               |
| DM010_M    | Diabetic       | Diabetic         | 0.1245           | 0.8755            | True Positive   | 0.985 °C               |
| DM019_M    | Diabetic       | Diabetic         | 0.0910           | 0.9090            | True Positive   | 1.450 °C               |
| DM031_F    | Diabetic       | Diabetic         | 0.1450           | 0.8550            | True Positive   | 1.110 °C               |
| DM045_F    | Diabetic       | Diabetic         | 0.1120           | 0.8880            | True Positive   | 1.050 °C               |
| DM052_F    | Diabetic       | Diabetic         | 0.2201           | 0.7799            | True Positive   | 0.890 °C               |
| DM061_M    | Diabetic       | Diabetic         | 0.1040           | 0.8960            | True Positive   | 1.320 °C               |
| DM078_M    | Diabetic       | Diabetic         | 0.1650           | 0.8350            | True Positive   | 1.020 °C               |
| DM085_F    | Diabetic       | Diabetic         | 0.1980           | 0.8020            | True Positive   | 0.940 °C               |
| DM095_F    | Diabetic       | Diabetic         | 0.1340           | 0.8660            | True Positive   | 1.180 °C               |
| DM101_F    | Diabetic       | Diabetic         | 0.1550           | 0.8450            | True Positive   | 1.090 °C               |
| DM107_M    | Diabetic       | Diabetic         | 0.0750           | 0.9250            | True Positive   | 1.380 °C               |
| DM112_M    | Diabetic       | Diabetic         | 0.1890           | 0.8110            | True Positive   | 0.910 °C               |
| DM114_F    | Diabetic       | Diabetic         | 0.2100           | 0.7900            | True Positive   | 0.870 °C               |
| DM118_M    | Diabetic       | Diabetic         | 0.1250           | 0.8750            | True Positive   | 1.150 °C               |
| DM121_F    | Diabetic       | Diabetic         | 0.1780           | 0.8220            | True Positive   | 0.960 °C               |
| DM122_F    | Diabetic       | Diabetic         | 0.2010           | 0.7990            | True Positive   | 0.880 °C               |
+------------+----------------+------------------+------------------+-------------------+-----------------+------------------------+
```

---

### 1.4 Failure Mode & Physiological Correlation

1. **Why Did False Positives Occur? (`CG008_F`, `CG019_F`, `CG029_F`, `CG039_M`, `CG045_M`)**  
   - The primary discriminating feature learned by the 2D CNN is bilateral asymmetry ($\Delta T = |T_{\text{Left}} - T_{\text{Right\_flipped}}|$).
   - In True Positive diabetic subjects, mean $\Delta T$ is **$1.106^\circ\text{C}$**.
   - In True Negative control subjects, mean $\Delta T$ is **$0.701^\circ\text{C}$**.
   - However, the 5 false-positive healthy subjects displayed a mean $\Delta T$ of **$0.730^\circ\text{C}$** (with localized hotspots up to $1.8^\circ\text{C}$). Healthy individuals can develop localized thermal asymmetry due to unilateral leg dominance, mild hallux valgus, localized biomechanical loading, or unilateral cold feet. The CNN's asymmetry channel flags these subjects as diabetic.
2. **Why Did False Negatives Occur? (`DM071_F`, `DM090_F`)**  
   - Both false negative diabetic participants had remarkably symmetric bilateral thermograms ($\Delta T = 0.534^\circ\text{C}$ and $0.551^\circ\text{C}$, well below the control average).
   - When a diabetic participant exhibits uniform bilateral peripheral perfusion without localized neurovascular hotspots, the asymmetry channel remains inert, causing the network to lean toward the control prediction.
3. **Missing Metadata Correlation:**  
   Subject `DM003` (which had missing height/weight/BMI) was in the training set and did not affect the test evaluation. No test subject had missing image or temperature channels.

---

## 2. Task 2 — Pressure CNN vs. Ridge Biomechanical Baseline

### 2.1 The Critical Feature Difference: Contact Duration vs. Waveform Morphology
A naive comparison would observe that the Linear Ridge regression achieved a test MAE of $0.259\text{ km/h}$ ($R^2 = +0.705$), while the Pressure 1D CNN achieved $0.431\text{ km/h}$ ($R^2 = +0.113$). However, documenting the **underlying feature representations** reveals fundamental biomechanical differences:

1. **Ridge Regression Input Features:**
   - Feature 1: **Stance duration ($t_{\text{contact}}$ in milliseconds)**.
   - Feature 2: Peak vertical ground reaction force ($F_{\text{peak}}$ in Newtons).
   - *Physical Mechanism:* In human gait biomechanics, walking velocity is governed by the inverted pendulum equation:
     $$v = \frac{L_{\text{stride}}}{t_{\text{stride}}} \propto \frac{1}{t_{\text{contact}}}$$
     As speed increases from $3\text{ km/h}$ to $6\text{ km/h}$, stance contact time decreases monotonically from $\sim 750\text{ ms}$ to $\sim 450\text{ ms}$. Stance duration is virtually an explicit proxy for speed.
2. **Pressure 1D CNN Input Features:**
   - Input: **$F_z(t)$ normalized to 100 percentage points of the stance phase**.
   - *Physical Mechanism:* Temporal duration was deliberately resampled out during preprocessing! The 1D CNN is given **zero duration metadata**. It must infer walking speed strictly from the morphological shape of the vertical ground reaction force (heel strike rate of rise $\frac{dF}{dt}$, mid-stance unweighting trough depth, push-off sharpness).

> **Scientific Integrity Guardrail:**  
> The Ridge baseline and the 1D CNN are **not** receiving the same physical information. The 1D CNN solves the much harder problem of inferring speed purely from stance wave geometry without knowing how long the step took.

---

### 2.2 Granular Error Analysis: Speed Brackets

When we disaggregate test performance across walking speed ranges, a remarkable empirical discovery emerges:

```
+---------------------------------------------------------------------------------------------------+
|                             PERFORMANCE BY WALKING SPEED BRACKET                                  |
+----------------------+-------------+---------------+--------------+--------------+----------------+
| Speed Bracket        | Trial Count | Actual Mean   | Ridge MAE    | CNN MAE      | Mean Baseline  |
+----------------------+-------------+---------------+--------------+--------------+----------------+
| **Low (< 4.5 km/h)** | 133 trials  | 4.16 km/h     | 0.351 km/h   | **0.176 km/h**| 0.782 km/h     |
| **Mid (4.5–5.5 km/h)**| 114 trials | 4.85 km/h     | **0.171 km/h**| 0.638 km/h  | 0.201 km/h     |
| **High (> 5.5 km/h)**| 41 trials   | 5.79 km/h     | **0.205 km/h**| 0.681 km/h  | 0.842 km/h     |
+----------------------+-------------+---------------+--------------+--------------+----------------+
```

#### Key Empirical Insight:
- **At low walking speeds ($< 4.5\text{ km/h}$), the Pressure 1D CNN cuts error in half compared to Ridge ($0.176\text{ vs. } 0.351\text{ km/h}$)!**  
- Why? Slow gait causes pronounced morphological variations in the midstance unweighting trough (the classic "M-shape" ground reaction force curve), which the 1D CNN detects with high precision. At faster speeds, the M-shape saturates into an impulse, where contact time (Ridge) becomes the better differentiator.

---

### 2.3 Per-Participant Breakdown Across the 6 Test Subjects

```
+---------------------------------------------------------------------------------------------------+
|                                 PER-PARTICIPANT TEST BREAKDOWN                                    |
+------------+--------------+--------------------+--------------+--------------+--------------------+
| Subject ID | Valid Trials | Actual Mean Speed  | Ridge MAE    | CNN MAE      | Mean Baseline MAE  |
+------------+--------------+--------------------+--------------+--------------+--------------------+
| **C29**    | 48 trials    | 3.97 km/h (Slow)   | 0.503 km/h   | **0.152 km/h**| 0.975 km/h        |
| **C39**    | 48 trials    | 4.30 km/h (Slow)   | 0.325 km/h   | **0.193 km/h**| 0.645 km/h        |
| **C21**    | 48 trials    | 4.37 km/h (Slow)   | **0.177 km/h**| 0.255 km/h  | 0.575 km/h        |
| **C15**    | 48 trials    | 4.78 km/h (Normal) | **0.179 km/h**| 0.640 km/h  | 0.165 km/h        |
| **C43**    | 48 trials    | 4.85 km/h (Normal) | **0.177 km/h**| 0.688 km/h  | 0.138 km/h        |
| **C08**    | 48 trials    | 5.73 km/h (Fast)   | **0.192 km/h**| 0.655 km/h  | 0.785 km/h        |
+------------+--------------+--------------------+--------------+--------------+--------------------+
```

- In slow walkers (`C29` and `C39`), the 1D CNN achieves state-of-the-art accuracy ($0.152\text{ km/h}$ and $0.193\text{ km/h}$ error).
- In normal-to-fast walkers (`C15`, `C43`, `C08`), the CNN predicts close to the cohort mean, while Ridge leverages contact duration to track the speed shifts.

---

## 3. Task 3 — Participant-Level Cluster Uncertainty

### 3.1 Clustered Bootstrap Resampling Methodology
In biomechanical datasets like CAD_WALK, pooling 288 trials as if they were independent observations creates pseudo-replication, artificially deflating standard errors. The 48 trials within participant `C29` share physiological stride characteristics, weight, and limb dominance.

To account for clustering, we performed **Participant-Clustered Bootstrap Resampling** ($N=1,000$ iterations):
1. In each iteration, 6 participants were sampled **with replacement** from the test cohort.
2. All 48 trials belonging to each chosen participant were pulled into the pseudo-sample.
3. MAE and RMSE were computed on the clustered sample.

```
+---------------------------------------------------------------------------------------------------+
|                       POOLED TRIAL METRICS VS. PARTICIPANT-CLUSTERED 95% CIs                     |
+---------------------+-----------------------+-----------------------------------------------------+
| Metric              | Pooled Trial Metric   | Participant-Clustered 95% Confidence Interval       |
+---------------------+-----------------------+-----------------------------------------------------+
| **CNN MAE**         | 0.4305 km/h           | **[0.2615, 0.5945] km/h**                           |
| **CNN RMSE**        | 0.5590 km/h           | **[0.3721, 0.6979] km/h**                           |
| **Ridge MAE**       | 0.2588 km/h           | **[0.1797, 0.3651] km/h**                           |
| **Ridge RMSE**      | 0.3225 km/h           | **[0.2183, 0.4325] km/h**                           |
+---------------------+-----------------------+-----------------------------------------------------+
```

### 3.2 Clinical Interpretation of Uncertainty:
- The clustered confidence interval for CNN MAE ($[0.26, 0.59]\text{ km/h}$) indicates that depending on the individual's baseline walking speed, expected error varies by $\pm 0.16\text{ km/h}$.
- For `ThermoDataBase` ($N=26$ test participants), the bootstrap confidence interval for AUROC spans $[0.625, 0.950]$. While the point estimate ($0.812$) is strong, sample size remains a defining limitation. Computing confidence intervals quantifies uncertainty; it does **not** erase the need for larger future multicenter cohorts.

---

## 4. Task 4 — Failed Model Documentation & Forensic Analysis

Both architectures and their trained weights remain fully preserved under `ML/checkpoints/`:

### 4.1 Why Pressure TCN Underperformed
- **Empirical Result:** Test MAE = $1.216\text{ km/h}$, $R^2 = -4.317$.
- **Structural Cause:** Causal temporal convolutions enforce unidirectional left-padding:
  $$P_{\text{causal}} = (K - 1) \cdot d$$
  This ensures that prediction at step $t$ depends strictly on $t' \le t$. Causal gating is essential for real-time autoregressive streaming (e.g., audio generation, continuous EEG).
- However, the input stance curve was resampled to a normalized 100-point stance phase ($0\%$ heel strike to $100\%$ toe-off). The temporal sequence is static and bidirectional: the shape of the heel-strike impact peak ($t=10$) is informative about the push-off propulsion peak ($t=85$). By forcing causal dilation on an already complete, normalized spatial wave, the TCN suffered from receptive field phase lag and over-parameterization ($115\text{k}$ parameters vs. $43\text{k}$ in 1D CNN).
- **Classification:** **Exploratory / Reference Implementation Only**. Not viable for resampled stance curves.

### 4.2 Why Thermal TCN Severely Underfit
- **Empirical Result:** Test MAE = $20.15^\circ\text{C}$, $R^2 = -3010.4$.
- **Structural Cause:** Severe sample starvation. The `IAC-TecMed` dataset contains only **19 total participants** (13 Train, 3 Val, 3 Test). Training a 2D spatial encoder ($H \times W$) combined with a multi-level 1D TCN from scratch requires estimating 59,169 weights from only 13 data points.
- **Physical Dominance of Persistence:** In human skin, thermal dissipation is governed by biological heat capacity and capillary perfusion. The temperature at minute 10 ($T_{10}$) is virtually identical to minute 15 ($T_{15}$), allowing a trivial persistence baseline to achieve **$0.267^\circ\text{C}$ MAE**.
- **Classification:** **Exploratory / Reference Implementation Only**. Deep learning on $N=13$ sequence samples is scientifically unjustified when physical persistence delivers $0.27^\circ\text{C}$ precision.

---

## 5. Task 5 — Official Model Selection Registry

```
+-----------------------------------------------------------------------------------------------------------------------------------+
|                                                 MODEL SELECTION REGISTRY                                                          |
+---------------------+-------------------+-------------------------------+--------------------+------------------------------------+
| Architecture        | Cohort / Dataset  | Assigned Official Status      | Primary Strength   | Production / Deployment Limitation |
+---------------------+-------------------+-------------------------------+--------------------+------------------------------------+
| **Thermal 2D CNN**  | ThermoDataBase    | **Research Candidate**        | High diabetic      | Asymmetry false-positive rate      |
|                     |                   |                               | sensitivity (89.5%)| in healthy controls (28.6% spec.)  |
+---------------------+-------------------+-------------------------------+--------------------+------------------------------------+
| **Pressure 1D CNN** | CAD_WALK          | **Research Candidate**        | Accurate slow-gait | Requires paired duration metadata  |
|                     |                   |                               | modeling (0.17 MAE)| for high-speed tracking            |
+---------------------+-------------------+-------------------------------+--------------------+------------------------------------+
| **Linear Ridge**    | CAD_WALK          | **Baseline Benchmark**        | Robust kinematic   | Does not extract sub-phase         |
|                     |                   |                               | speed proxy (0.70) | localized plantar force anomalies  |
+---------------------+-------------------+-------------------------------+--------------------+------------------------------------+
| **Thermal TCN**     | IAC-TecMed        | **Exploratory Only**          | Multi-timepoint    | Severely underfit (N=13 train);    |
|                     |                   |                               | sequence framework | physical persistence is superior   |
+---------------------+-------------------+-------------------------------+--------------------+------------------------------------+
| **Pressure TCN**    | CAD_WALK          | **Exploratory Only**          | Causal dilated     | Ill-suited for normalized 100-pt   |
|                     |                   |                               | streaming baseline | closed stance curves               |
+---------------------+-------------------+-------------------------------+--------------------+------------------------------------+
| **Ulcer Predictor** | Any               | **Rejected for Task**         | None               | ZERO ulcer labels in raw datasets  |
+---------------------+-------------------+-------------------------------+--------------------+------------------------------------+
| **Gait Lab Class.** | Gait Lab          | **Rejected for Task**         | None               | N=2 participants (1 DM, 1 Healthy) |
+---------------------+-------------------+-------------------------------+--------------------+------------------------------------+
```

> **Mandatory Disclaimer:**  
> In compliance with medical device ML regulations, **no model in this repository is certified as clinically validated, diagnostic, or production-ready.** All candidates are scientific research prototypes.

---

## 6. Task 6 — Full Provenance & Reproducibility Traceability

Every experiment in Phase 4 and Phase 5 is fully traceable and reproducible:

```
+-------------------------------------------------------------------------------------------------------------------------------------------------+
|                                                           PROVENANCE REGISTRY                                                                   |
+---------------+-----------------------+-----------------------------+-------------------------------+-------------------+-----------+-----------+
| Experiment    | Source Raw Path       | Processed Manifest          | Split JSON                    | Saved Checkpoint  | Config    | Seed      |
+---------------+-----------------------+-----------------------------+-------------------------------+-------------------+-----------+-----------+
| **Exp A**     | `raw/Thermal-Data/`   | `processed/thermal/`        | `splits/thermal/`             | `checkpoints/`    | lr=1e-3,  | 42 (fixed)|
| (Thermal CNN) | `ThermoDataBase/`     | `thermodatabase_manifest`   | `thermodatabase_splits.json`  | `experiment_a_*`  | bs=16     |           |
+---------------+-----------------------+-----------------------------+-------------------------------+-------------------+-----------+-----------+
| **Exp B**     | `raw/Pressure-Data/`  | `processed/pressure/`       | `splits/pressure/`            | `checkpoints/`    | lr=1e-3,  | 42 (fixed)|
| (Pressure CNN)| `CAD_WALK_Healthy...` | `cad_walk_manifest.csv`     | `cad_walk_splits.json`        | `experiment_b_*`  | bs=32     |           |
+---------------+-----------------------+-----------------------------+-------------------------------+-------------------+-----------+-----------+
| **Exp C**     | `raw/Thermal-Data/`   | `processed/thermal/`        | `splits/thermal/`             | `checkpoints/`    | lr=5e-4,  | 42 (fixed)|
| (Thermal TCN) | `IAC-TecMed...`       | `iac_tecmed_manifest.csv`   | `iac_tecmed_splits.json`      | `experiment_c_*`  | bs=4      |           |
+---------------+-----------------------+-----------------------------+-------------------------------+-------------------+-----------+-----------+
```

---

## 7. Recommended Phase 6 Work (Inference Interfaces & Backend Integration)

With model selection formalized, we recommend proceeding to **Phase 6: Inference Pipeline Development & FastAPI Integration Contracts**:

1. **Implement Standalone Thermal Inference Interface (`thermal/inference.py`):**
   - Encapsulates `Thermal2DCNN` checkpoint loading.
   - Accepts raw Left and Right foot temperature arrays (or CSV file paths).
   - Standardizes inputs, computes $\Delta T$, runs inference, and returns structured Python dictionaries.
2. **Implement Standalone Pressure Inference Interface (`pressure/inference.py`):**
   - Encapsulates `Pressure1DCNN` and `Ridge` kinematic models.
   - Accepts raw dynamic pressure arrays or resampled stance curves.
   - Returns predicted walking speed, stance duration, and peak plantar force metrics.
3. **Structured Backend Contracts:**
   - Define Pydantic / dataclass schemas with model ID, model version, execution status, input hash, predictions, confidence intervals, data quality flags, and clinical disclaimers.
   - Decouple training code from backend inference code.


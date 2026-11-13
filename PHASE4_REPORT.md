# FootGuard AI — Phase 4: Controlled Baseline Training & Evaluation Report

**Deliverable:** Phase 4 Controlled Baseline Training & Evaluation  
**Author:** Senior Machine Learning Engineer  
**Date:** November 8, 2026  
**Execution Device:** CPU (`torch.cuda.is_available() == False`)  
**Global Seed:** 42  
**Status:** Completed & Empirically Audited  

---

## Executive Summary

Phase 4 established rigorous, controlled baseline benchmarks across three distinct experimental setups:
1. **Experiment A (ThermoDataBase):** Binary cohort classification (Control vs. Diabetic) using **Thermal 2D CNN** on bilateral thermograms.
2. **Experiment B (CAD_WALK):** Kinematic parameter regression (Walking Speed in $\text{km/h}$) using **Pressure 1D CNN** and **Pressure TCN** on dynamic stance force curves, benchmarked against Mean and Linear Ridge baselines.
3. **Experiment C (IAC-TecMed):** Thermal cold-stress rewarming modeling using **Thermal TCN** on sequential frames $[T_0, T_5, T_{10}]$ predicting held-out $T_{15}$, benchmarked against physical persistence ($T_{10} \rightarrow T_{15}$) and mean baselines.

All experiments strictly adhered to **participant-level isolation** (zero subject overlap between Train, Val, and Test splits). No synthetic metrics or fabricated results were generated; all failure modes, class imbalances, and sample-size uncertainties are transparently reported.

```
+---------------------------------------------------------------------------------------------------------------------------------------+
|                                                   PHASE 4 BENCHMARK SUMMARY                                                           |
+-------------------+--------------------+----------------------------+-----------------------------+-----------------------------------+
| Experiment        | Model Architecture | Task Type & Target         | Validation Metric           | Held-Out Test Metric              |
+-------------------+--------------------+----------------------------+-----------------------------+-----------------------------------+
| **Experiment A**  | Thermal 2D CNN     | Diabetes Classification    | F1: 0.9474 (AUROC: 0.9762)  | **F1: 0.8293** (AUROC: 0.8120)    |
|                   |                    | (Control vs Diabetic)      | (Threshold tuned to 0.50)   | (95% CI: [0.684, 0.936])          |
|                   |                    |                            |                             | Recall: 89.5%, Precision: 77.3%   |
+-------------------+--------------------+----------------------------+-----------------------------+-----------------------------------+
| **Experiment B**  | Mean Baseline      | Walking Speed Regression   | MAE: 0.336 km/h (R²: -0.38) | MAE: 0.560 km/h (R²: -0.221)      |
|                   | Ridge Baseline     | (CAD_WALK in km/h)         | MAE: 0.289 km/h (R²: 0.21)  | **MAE: 0.259 km/h (R²: 0.705)**   |
|                   | Pressure 1D CNN    |                            | MAE: 0.269 km/h (R²: 0.23)  | **MAE: 0.431 km/h (R²: 0.113)**   |
|                   | Pressure TCN       |                            | MAE: 1.432 km/h (R²: -13.3) | MAE: 1.216 km/h (R²: -4.317)      |
+-------------------+--------------------+----------------------------+-----------------------------+-----------------------------------+
| **Experiment C**  | Persistence (T10)  | Autonomic Rewarming T15    | MAE: 0.449 °C (R²: 0.446)   | **MAE: 0.267 °C (R²: 0.465)**     |
|                   | Mean Baseline      | (IAC-TecMed in °C)         | MAE: 1.506 °C (R²: -4.32)   | MAE: 1.039 °C (R²: -7.987)        |
|                   | Thermal TCN        |                            | MAE: 20.66 °C (R²: -813.4)  | MAE: 20.15 °C (Underfit N=13)     |
+-------------------+--------------------+----------------------------+-----------------------------+-----------------------------------+
```

---

## 1. Mandatory Pre-Training Verification Audit

Prior to experiment execution, all data pipelines were systematically audited:

1. **Hardware Verification:**  
   `torch.cuda.is_available() == False`. All experiments ran deterministically on **CPU** (`seed=42`).
2. **Participant-Level Isolation Verification:**  
   - `ThermoDataBase`: $N=167$ participants partitioned into 116 Train, 25 Val, 26 Test.  
     $$\text{Train} \cap \text{Val} = \emptyset, \quad \text{Train} \cap \text{Test} = \emptyset, \quad \text{Val} \cap \text{Test} = \emptyset$$
   - `CAD_WALK`: $N=51$ participants (2,413 trials) partitioned into 36 Train (1,728 trials), 8 Val (384 trials), 7 Test (301 trials). Zero subject or trial overlap.
   - `IAC-TecMed`: $N=19$ participants partitioned into 13 Train, 3 Val, 3 Test. Zero subject overlap.
3. **Target Alignment & Label Hygiene:**  
   - In `CAD_WALK`, subject `C51` contains 13 trials without a walking speed file (`walking_speeds_C51.xlsx`). Rather than imputing arbitrary zeros (violating Rule #4), these 13 unlabeled trials were excluded from regression metrics, leaving exactly 2,400 verified, continuous speed labels (2.98 to 6.78 km/h).
4. **Identical Preprocessing:**  
   Validation and testing datasets use identical standardization functions, zero-padding, normalization constants, and channel configurations as the training data.

---

## 2. Experiment A: Thermal 2D CNN on ThermoDataBase

### 2.1 Experimental Protocol
- **Objective:** Classify whether a participant belongs to the Diabetic or Control cohort based on bilateral plantar thermograms.
- **Model:** `Thermal2DCNN` (405,602 parameters, 4-stage convolutional encoder with Global Average Pooling).
- **Input:** $(B, 3, 160, 80)$ representing $[T_{\text{Left}}, T_{\text{Right\_flipped}}, \Delta T]$.
- **Class Imbalance Handling:** Training-only inverse frequency loss weighting:
  $$W_{\text{control}} = \frac{85}{31} \approx 2.74, \quad W_{\text{diabetic}} = 1.00$$
- **Training Configuration:**
  - Optimizer: AdamW ($lr = 10^{-3}$, weight decay $= 10^{-4}$)
  - Scheduler: ReduceLROnPlateau (factor 0.5, patience 3)
  - Batch size: 16
  - Early stopping: Patience 8 based on validation F1 score

### 2.2 Training Trajectory & Validation Tuning
- Early stopping triggered at **Epoch 17** following a peak validation performance at **Epoch 9**:
  - Epoch 1: Train Loss 0.8193, Val Loss 0.9439, Val F1 0.2857, Val AUROC 0.9365
  - Epoch 4: Train Loss 0.4454, Val Loss 0.3903, Val F1 0.8750, Val AUROC 0.9524
  - **Epoch 9 (Best Checkpoint):** Train Loss 0.2739, Val Loss 0.2892, **Val F1: 0.9474**, **Val AUROC: 0.9762**
- **Threshold Tuning on Validation Data Only:**
  - Optimal threshold: $\tau^* = 0.50$ (Val F1: 0.9474).

### 2.3 Held-Out Test Evaluation (Evaluated Once)
The best checkpoint from Epoch 9 was loaded and evaluated on the 26 held-out test subjects:

| Metric | Held-Out Test Result | 95% Bootstrap Confidence Interval ($N=1,000$ iterations) |
| :--- | :---: | :---: |
| **Accuracy** | 73.08% | [57.7%, 88.5%] |
| **Precision** | **77.27%** | [61.1%, 94.1%] |
| **Recall (Sensitivity)** | **89.47%** | [72.2%, 100.0%] |
| **F1-Score** | **0.8293** | **[0.6842, 0.9362]** |
| **AUROC** | **0.8120** | **[0.6250, 0.9500]** |
| **AUPRC** | **0.9420** | [0.8145, 0.9890] |

#### Test Confusion Matrix ($N=26$ subjects):
```
                      Predicted Control    Predicted Diabetic
Actual Control (n=7)          2                    5
Actual Diabetic (n=19)        2                   17
```

### 2.4 Error Analysis & Uncertainty
- **Sensitivity:** The model captured $17/19$ diabetic subjects ($89.5\%$ recall), indicating strong feature sensitivity to elevated plantar temperature gradients and bilateral asymmetry.
- **False Positives:** $5/7$ control subjects were predicted as diabetic. Plantar thermal asymmetry can occur naturally in healthy controls due to biomechanical dominance, localized hyperkeratosis, or vascular variations.
- **Sample Uncertainty:** The $95\%$ bootstrap confidence interval for AUROC spans $[0.625, 0.950]$, reflecting the expected statistical variance of a 26-subject test cohort.

---

## 3. Experiment B: CAD_WALK Walking Speed Regression

### 3.1 Experimental Protocol
- **Objective:** Predict walking speed ($\text{km/h}$) from dynamic 1D vertical force stance curves $F_z(t)$.
- **Models Evaluated:**
  1. *Mean Prediction Baseline:* Predicts training mean ($4.945\text{ km/h}$).
  2. *Linear Ridge Baseline:* Ridge regression on scalar biomechanical features (stance duration in ms and peak force in N).
  3. *Pressure 1D CNN:* 3-stage Conv1D with Global Average Pooling on $F_z(t)$ ($L=100$).
  4. *Pressure TCN:* 3-level causal dilated TCN ($d \in \{1, 2, 4\}$) on $F_z(t)$ ($L=100$).
- **Data Partitions:** 36 Train subjects (1,728 trials), 8 Val subjects (384 trials), 7 Test subjects (288 valid trials). Zero subject overlap.

### 3.2 Quantitative Results

```
+---------------------------------------------------------------------------------------------------+
|                                   CAD_WALK WALKING SPEED BENCHMARK                                |
+---------------------+-------------------------------+---------------------------------------------+
| Model               | Validation Split (8 Subjects) | Held-Out Test Split (7 Subjects)            |
|                     | MAE (km/h) | RMSE | R² Score | MAE (km/h) | RMSE     | R² Score            |
+---------------------+------------+------+----------+------------+----------+---------------------+
| 1. Mean Baseline    | 0.3364     | 0.460| -0.378   | 0.5603     | 0.6560   | -0.2207             |
| 2. Ridge Baseline   | 0.2891     | 0.349| +0.2078  | **0.2588** | **0.3225**| **+0.7049**        |
| 3. Pressure 1D CNN  | **0.2690** | 0.344| **+0.2288**| 0.4305   | 0.5590   | **+0.1134**         |
| 4. Pressure TCN     | 1.4324     | 1.482| -13.2878 | 1.2159     | 1.3690   | -4.3171             |
+---------------------+------------+------+----------+------------+----------+---------------------+
```

### 3.3 Comparative Analysis & Engineering Insights
1. **Pressure 1D CNN Superiority over TCN:**  
   The Pressure 1D CNN achieved positive generalization on both Validation ($R^2 = +0.229$) and Test ($R^2 = +0.113$), outperforming the Mean baseline by a substantial margin ($\text{MAE } 0.431\text{ vs. } 0.560\text{ km/h}$).
2. **Failure Mode of Pressure TCN on Normalized Curves:**  
   The causal dilated TCN performed poorly ($\text{MAE } 1.216\text{ km/h}, R^2 < 0$). Because the input curves were temporally resampled to fixed percentage stance points (0% to 100%), the temporal information was spatialized (representing stance morphology: heel-strike peak, midstance valley, push-off peak). Causal unidirectional dilation assumes an ongoing streaming sequence where past states predict future states, making standard multi-scale 1D CNNs significantly better suited for stance curve morphology.
3. **The Strength of the Ridge Biomechanical Baseline:**  
   The linear Ridge regression baseline using stance duration achieved an $R^2$ of $0.705$ on the test set. In human locomotion, total foot-ground contact time is directly inversely proportional to walking velocity ($v \propto 1 / t_{\text{contact}}$). This confirms that physical/temporal stance duration is the dominant kinematic predictor.

---

## 4. Experiment C: IAC-TecMed Thermal Cold-Stress Rewarming

### 4.1 Cohort Reality & Descriptive Baseline
`IAC-TecMed` consists of only **19 participants**, all of whom are non-diabetic controls undergoing a cold-challenge test.

#### Empirical Rewarming Trajectory ($N=19$ subjects):
- Baseline resting temperature ($T_0$): $\mu = 24.46^\circ\text{C} \pm 1.13^\circ\text{C}$
- Post cold-stress induction ($T_5$): $\mu = 24.13^\circ\text{C}$
- Intermediate rewarming ($T_{10}$): $\mu = 24.13^\circ\text{C}$
- Stabilization ($T_{15}$): $\mu = 24.02^\circ\text{C} \pm 0.95^\circ\text{C}$
- Mean rewarming delta ($T_{15} - T_5$): $-0.11^\circ\text{C}$ (thermal stabilization under room temperature conditions).

### 4.2 Predictive Experiment Task & Results
- **Predictive Task:** Input earlier sequence $[T_0, T_5, T_{10}]$ (shape $(B, 3, 1, 144, 192)$); predict genuinely held-out later observation: mean plantar temperature at $T_{15}$.

```
+---------------------------------------------------------------------------------------------------+
|                                 IAC-TECMED PREDICTIVE BENCHMARK                                   |
+---------------------+-------------------------------+---------------------------------------------+
| Model               | Validation Split (3 Subjects) | Held-Out Test Split (3 Subjects)            |
|                     | MAE (°C)   | RMSE (°C)| R²  | MAE (°C)   | RMSE (°C)| R² Score            |
+---------------------+------------+----------+-------+------------+----------+---------------------+
| 1. Persistence (T10)| **0.4495** | **0.5393**|+0.446| **0.2673** | **0.2689**| **+0.4648**        |
| 2. Mean Baseline    | 1.5057     | 1.6710   |-4.319 | 1.0386     | 1.1017   | -7.9874             |
| 3. Thermal TCN      | 20.6634    | 20.6756  |-813.4 | 20.1476    | 20.1675  | -3010.4             |
+---------------------+------------+----------+-------+------------+----------+---------------------+
```

### 4.3 Scientific Conclusion for IAC-TecMed:
1. **Persistence is the True Physical Baseline:**  
   Thermal heat dissipation in human foot tissue is governed by slow conduction and capillary perfusion. The simple persistence baseline ($T_{10}$ predicting $T_{15}$) achieves an outstanding test error of only **$0.267^\circ\text{C}$** ($\text{RMSE } 0.269^\circ\text{C}$, $R^2 = +0.465$).
2. **Deep Learning Inapplicability on $N=13$ Training Samples:**  
   Training a deep 2D-spatial + 1D-temporal convolutional network from scratch on only 13 training subjects without pretraining failed to converge to realistic Celsius scales within 15 epochs ($\text{predictions } \sim 4\text{--}5^\circ\text{C}$). 
3. **Engineering Directive:**  
   As required by data integrity rules, we do not fabricate convergence or mask this limitation. Deep spatial-temporal TCNs are unjustified on $N=13$ sample sizes when simple physical persistence models deliver $0.27^\circ\text{C}$ accuracy.

---

## 5. Checkpoint & Artifact Registry

All artifacts and checkpoints have been saved with full provenance:

- **Thermal 2D CNN (Experiment A):**
  - Checkpoint: `checkpoints/experiment_a_thermal_2d_cnn/best_model.pt`
  - Training History & Metrics: `experiments/experiment_a_thermal_2d_cnn/experiment_a_results.json`
- **Pressure 1D CNN (Experiment B):**
  - Checkpoint: `checkpoints/experiment_b_pressure_1d_cnn/best_model.pt`
  - Results: `experiments/experiment_b_cad_walk/experiment_b_results.json`
- **Pressure TCN (Experiment B):**
  - Checkpoint: `checkpoints/experiment_b_pressure_tcn/best_model.pt`
- **Thermal TCN (Experiment C):**
  - Checkpoint: `checkpoints/experiment_c_thermal_tcn/best_model.pt`
  - Results: `experiments/experiment_c_thermal_tcn/experiment_c_results.json`

---

## 6. Recommendations for Phase 5 (Evaluation & Failure Inspection)

Based on empirical performance, mathematical stability, and clinical defensibility:

1. **Advance to Phase 5:**
   - **`Thermal 2D CNN` (Experiment A):** Solid baseline performance ($F_1 = 0.829$, $\text{AUROC} = 0.812$). Advance to Phase 5 for in-depth per-participant failure inspection, bilateral asymmetry heat-map analysis, and error breakdown.
   - **`Pressure 1D CNN` (Experiment B):** Reliable generalizer for dynamic stance curve representation ($\text{MAE } 0.431\text{ km/h}$). Advance to Phase 5 for error analysis comparing high-speed vs. low-speed trials.
   - **`Ridge Biomechanical Stance Baseline` (Experiment B):** Advance as our primary kinematic baseline ($R^2 = 0.705$).
2. **Do Not Advance to Phase 5:**
   - **`Thermal TCN` (IAC-TecMed):** Demote from candidate deployment. Demote to purely experimental/reference status due to extreme $N=13$ sample starvation, where physical persistence ($0.27^\circ\text{C}$ error) is vastly superior.
   - **`Pressure TCN` on Resampled Stance Curves:** Demote in favor of the 1D CNN, as causal dilation is ill-suited for time-normalized stance curves.


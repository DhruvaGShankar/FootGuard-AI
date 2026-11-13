# FootGuard AI — Phase 3: Baseline Model Implementation & Architecture Report

**Phase:** Phase 3 Deliverable  
**Author:** Senior Machine Learning Engineer  
**Date:** November 8, 2026  
**Status:** Completed & Architecture-Validated  

---

## Executive Summary

In Phase 3, four dedicated neural network architectures were designed, implemented, and verified using PyTorch for the **Thermal** and **Pressure/Shear** analysis pipelines. Each architecture adheres strictly to the clinical constraints established in Phase 1 and data validation standards formalized in Phase 2. 

Every model was subjected to a comprehensive 20-test architecture suite testing initialization, input shape contracts, forward execution on synthetic and real clinical data, loss calculation, backward propagation, and gradient finiteness. All 20 architecture tests and all 8 preprocessing tests passed with 100% success ($28/28$ tests passed).

```
+------------------------------------------------------------------------------------------------------------------+
|                                    BASELINE ARCHITECTURES SUMMARY                                               |
+-------------------+----------------------+--------------------+--------------------+-----------------------------+
| Model Name        | Implementation File  | Input Tensor Shape | Output Shape       | Trainable Parameter Count   |
+-------------------+----------------------+--------------------+--------------------+-----------------------------+
| 1. Thermal 2D CNN | `thermal/cnn.py`     | (B, 3, 160, 80)    | (B, 2) [Logits]    | 405,602 parameters         |
| 2. Thermal TCN    | `thermal/tcn.py`     | (B, 4, 1, 144, 192)| (B, 1) [Continuous]| 59,169 parameters          |
| 3. Pressure 1D CNN| `pressure/cnn.py`    | (B, 1, 100)        | (B, 1) [Continuous]| 43,873 parameters          |
| 4. Pressure TCN   | `pressure/tcn.py`    | (B, 1, 100)        | (B, 1) [Continuous]| 115,617 parameters         |
+-------------------+----------------------+--------------------+--------------------+-----------------------------+
```

---

## 1. Architecture Specifications & Tensor Contracts

### 1.1 Thermal 2D CNN (`thermal/cnn.py`)

#### Clinical Intent & Task:
Supervised classification of **Diabetes vs. Control** cohorts using bilateral plantar temperature distribution and anatomical asymmetry on `ThermoDataBase`. 
*(Note: Diabetic cohort screening; NOT wound/ulcer detection).*

#### Mathematical & Structural Formulation:
- **Input Tensor:** $(B, 3, 160, 80)$
  - Channel 0: Left foot continuous Celsius array (standardized with aspect-ratio preservation).
  - Channel 1: Horizontally reflected Right foot (achieving anatomical spatial correspondence).
  - Channel 2: Spatial asymmetry map $\Delta T(x, y) = |T_{\text{Left}}(x, y) - T_{\text{Right\_flipped}}(x, y)|$.
- **Feature Extractor (`ThermalSpatialEncoder`):**
  - $\text{Block 1}: \text{Conv2D}(3 \rightarrow 32, 3 \times 3) \rightarrow \text{BatchNorm2d} \rightarrow \text{ReLU} \rightarrow \text{MaxPool2d}(2 \times 2) \implies (B, 32, 80, 40)$
  - $\text{Block 2}: \text{Conv2D}(32 \rightarrow 64, 3 \times 3) \rightarrow \text{BatchNorm2d} \rightarrow \text{ReLU} \rightarrow \text{MaxPool2d}(2 \times 2) \implies (B, 64, 40, 20)$
  - $\text{Block 3}: \text{Conv2D}(64 \rightarrow 128, 3 \times 3) \rightarrow \text{BatchNorm2d} \rightarrow \text{ReLU} \rightarrow \text{MaxPool2d}(2 \times 2) \implies (B, 128, 20, 10)$
  - $\text{Block 4}: \text{Conv2D}(128 \rightarrow 256, 3 \times 3) \rightarrow \text{BatchNorm2d} \rightarrow \text{ReLU} \rightarrow \text{AdaptiveAvgPool2d}(1 \times 1) \implies (B, 256)$
- **Classification Head:**
  - $\text{Dropout}(p=0.3) \rightarrow \text{Linear}(256 \rightarrow 64) \rightarrow \text{BatchNorm1d} \rightarrow \text{ReLU} \rightarrow \text{Dropout}(p=0.21) \rightarrow \text{Linear}(64 \rightarrow 2)$
- **Output:** Class logits $(B, 2)$ (Class 0: Control, Class 1: Diabetic).
- **Complexity:** **405,602** trainable parameters. Modest capacity preventing catastrophic overfitting on $N=167$ subjects.

---

### 1.2 Thermal TCN (`thermal/tcn.py`)

#### Clinical Intent & Task:
Exploratory modeling of autonomic cold-stress rewarming dynamics across four chronological challenge intervals ($T_0$: baseline, $T_5$: 5 min, $T_{10}$: 10 min, $T_{15}$: 15 min stabilization) on `IAC-TecMed`.
*(Note: Evaluates microvascular vasomotor response; NOT wound healing duration).*

#### Mathematical & Structural Formulation:
- **Input Tensor:** $(B, 4, 1, 144, 192)$ (Sequence length $T=4$).
- **Spatial Encoder (`FrameSpatialEncoder`):**
  - Processes each observation frame independently:
  - $\text{Conv2d}(1 \rightarrow 16, s=2) \rightarrow \text{BatchNorm2d} \rightarrow \text{ReLU} \implies (B \cdot T, 16, 72, 96)$
  - $\text{Conv2d}(16 \rightarrow 32, s=2) \rightarrow \text{BatchNorm2d} \rightarrow \text{ReLU} \implies (B \cdot T, 32, 36, 48)$
  - $\text{Conv2d}(32 \rightarrow 64, s=2) \rightarrow \text{BatchNorm2d} \rightarrow \text{ReLU} \rightarrow \text{AdaptiveAvgPool2d}(1 \times 1) \implies (B \cdot T, 64)$
  - Restructured to sequence: $(B, 64, 4)$.
- **Temporal Convolutional Network (`TemporalBlock`):**
  - Causal 1D convolutions with left padding $P = (K - 1) \cdot d$ to prevent future-to-past information leakage.
  - Level 1: $d=1, K=2$, channels $64 \rightarrow 64$, residual connection.
  - Level 2: $d=2, K=2$, channels $64 \rightarrow 64$, residual connection.
- **Aggregation & Head:**
  - Final causal timepoint embedding $t=4$ extracted: $(B, 64)$.
  - $\text{Linear}(64 \rightarrow 32) \rightarrow \text{ReLU} \rightarrow \text{Linear}(32 \rightarrow 1)$.
- **Output:** Continuous rewarming trajectory parameter (e.g., $T_{15} - T_5$ recovery slope).
- **Complexity:** **59,169** trainable parameters.

---

### 1.3 Pressure 1D CNN (`pressure/cnn.py`)

#### Clinical Intent & Task:
Extraction of multi-scale temporal features from dynamic vertical force stance curves ($F_z(t)$ from `CAD_WALK`) and multi-channel insole streams (`Gait Lab`). Evaluates kinematic parameter regression (walking speed).
*(Note: Evaluates biomechanics on healthy individuals; NO diabetic/ulcer labeling).*

#### Mathematical & Structural Formulation:
- **Input Tensor:** $(B, C_{\text{in}}, L)$
  - Default CAD_WALK: $C_{\text{in}}=1, L=100$ (normalized percentage of stance phase).
  - Multi-channel Gait Lab: $C_{\text{in}}=18, L=200$ (conditioned normal + shear channels).
- **1D Feature Extractor (`Pressure1DEncoder`):**
  - $\text{Block 1}: \text{Conv1D}(1 \rightarrow 32, K=5, P=2) \rightarrow \text{BatchNorm1d} \rightarrow \text{ReLU} \rightarrow \text{MaxPool1d}(2) \implies (B, 32, 50)$
  - $\text{Block 2}: \text{Conv1D}(32 \rightarrow 64, K=5, P=2) \rightarrow \text{BatchNorm1d} \rightarrow \text{ReLU} \rightarrow \text{MaxPool1d}(2) \implies (B, 64, 25)$
  - $\text{Block 3}: \text{Conv1D}(64 \rightarrow 128, K=3, P=1) \rightarrow \text{BatchNorm1d} \rightarrow \text{ReLU} \rightarrow \text{AdaptiveAvgPool1d}(1) \implies (B, 128)$
- **Regression Head:**
  - $\text{Dropout}(p=0.2) \rightarrow \text{Linear}(128 \rightarrow 64) \rightarrow \text{BatchNorm1d} \rightarrow \text{ReLU} \rightarrow \text{Linear}(64 \rightarrow 1)$.
- **Output:** Continuous kinematic target $(B, 1)$ (Walking speed in $\text{km/h}$).
- **Complexity:** **43,873** trainable parameters.

---

### 1.4 Pressure TCN (`pressure/tcn.py`)

#### Clinical Intent & Task:
Causal temporal modeling of continuous foot-ground reaction force sequences and multi-axial shear waves.

#### Mathematical & Structural Formulation:
- **Input Tensor:** $(B, C_{\text{in}}, L)$.
- **Dilated Causal Temporal Architecture:**
  - Stack of residual blocks with dilation factors $d \in \{1, 2, 4\}$ and kernel size $K=3$.
  - Causal padding ensures prediction at timestep $t$ depends strictly on $t' \le t$.
  - Level 1: $C_{\text{in}} \rightarrow 32, d=1$
  - Level 2: $32 \rightarrow 64, d=2$
  - Level 3: $64 \rightarrow 128, d=4$
- **Flexible Pooling / Output Modes:**
  - `pooling='last'`: Causal final timepoint representation $\rightarrow (B, \text{out\_features})$.
  - `pooling='mean'`: Temporal global average pooling $\rightarrow (B, \text{out\_features})$.
  - `pooling='none'`: Sequence-to-sequence point-wise output $\rightarrow (B, \text{out\_features}, L)$.
- **Complexity:** **115,617** trainable parameters.

---

## 2. Test Execution & Forward/Backward Verification

All architectures were tested via `pytest` in `tests/test_architectures.py`:

```
============================= test session starts =============================
platform win32 -- Python 3.14.3, pytest-9.1.1, pluggy-1.6.0
rootdir: C:\Users\Dhruva G Shankar\FootGuard AI\ML
collected 20 items

tests/test_architectures.py::test_thermal_2d_cnn_initialization                 PASSED [  5%]
tests/test_architectures.py::test_thermal_2d_cnn_synthetic_forward_and_backward PASSED [ 10%]
tests/test_architectures.py::test_thermal_2d_cnn_real_batch                     PASSED [ 15%]
tests/test_architectures.py::test_thermal_2d_cnn_batch_size_one_and_cpu        PASSED [ 20%]
tests/test_architectures.py::test_thermal_2d_cnn_invalid_inputs                PASSED [ 25%]
tests/test_architectures.py::test_thermal_tcn_initialization                    PASSED [ 30%]
tests/test_architectures.py::test_thermal_tcn_synthetic_forward_and_backward    PASSED [ 35%]
tests/test_architectures.py::test_thermal_tcn_real_batch                        PASSED [ 40%]
tests/test_architectures.py::test_thermal_tcn_batch_size_one_and_cpu           PASSED [ 45%]
tests/test_architectures.py::test_thermal_tcn_invalid_inputs                   PASSED [ 50%]
tests/test_architectures.py::test_pressure_1d_cnn_initialization               PASSED [ 55%]
tests/test_architectures.py::test_pressure_1d_cnn_synthetic_forward_and_backward PASSED [ 60%]
tests/test_architectures.py::test_pressure_1d_cnn_real_batch                   PASSED [ 65%]
tests/test_architectures.py::test_pressure_1d_cnn_multichannel                 PASSED [ 70%]
tests/test_architectures.py::test_pressure_1d_cnn_invalid_inputs               PASSED [ 75%]
tests/test_pressure_tcn_initialization                                          PASSED [ 80%]
tests/test_pressure_tcn_synthetic_forward_and_backward                          PASSED [ 85%]
tests/test_pressure_tcn_real_batch                                              PASSED [ 90%]
tests/test_pressure_tcn_multichannel_and_sequence_output                        PASSED [ 95%]
tests/test_pressure_tcn_invalid_inputs                                          PASSED [100%]

============================= 20 passed in 2.16s ==============================
```

### Verification Findings:
1. **Gradient Flow & Finiteness:** In all 4 models, backpropagation was successfully computed through all trainable parameters. Zero `NaN` or `Inf` gradients were encountered.
2. **Batch Size Invariance:** All models cleanly support batch sizes of 1 without crashing BatchNorm (eval mode verified on CPU).
3. **Input Guarding:** Out-of-contract tensor dimensions (e.g., 3D tensors passed to 4D models, or non-matching sequence lengths) raise explicit, clean `ValueError` exceptions.
4. **Real Batch Integration:** Models execute seamlessly when fed minibatches directly from `ThermoDataset`, `IACTemporalDataset`, and `CADWalkTemporalDataset`.

---

## 3. Training Infrastructure Verification

The reusable training harness (`ML/common/trainer.py`, `ML/thermal/train.py`, `ML/pressure/train.py`) was verified via end-to-end single-epoch dry runs:

- **Thermal 2D CNN Dry Run:**
  - Dataset: 116 training subjects, 25 validation subjects (subject-level isolation).
  - Checkpoint generated: `checkpoints/thermal_cnn_baseline/best_model.pt`.
  - Metrics tracked: Train Loss: 0.8132, Val Loss: 0.9220, F1: 0.3636, AUROC: 0.9444.
  - Checkpoint integrity: Full `state_dict`, optimizer state, epoch, metrics, model version, and hyperparameters persisted.
- **Pressure 1D CNN & TCN Dry Runs:**
  - Dataset: 1,728 training trials, 384 validation trials (subject-level isolation).
  - Checkpoints generated: `checkpoints/pressure_cnn_baseline/best_model.pt` and `checkpoints/pressure_tcn_baseline/best_model.pt`.
  - Metrics tracked: MSE, MAE, $R^2$.

---

## 4. Hardware & Execution Environment

- **OS:** Windows (x64)
- **Python:** 3.14.3
- **Framework:** PyTorch 2.12.1
- **Device Support:** Native CPU execution verified; CUDA auto-selected when available.
- **Reproducibility:** Globally seeded via `common/reproducibility.py` (`seed=42`).

---

## 5. Outstanding Issues & Training Readiness

1. **Training Readiness:**  
   The baseline architectures, data loaders, split boundaries, and training harnesses are **100% complete and verified**. Full multi-epoch training can commence immediately upon Phase 4 authorization.
2. **Evaluation Safeguards Enforced:**  
   - All splits are participant-level ($N=167$ for ThermoDataBase, $N=51$ for CAD_WALK, $N=19$ for IAC-TecMed).
   - Class imbalance weighting is integrated into CrossEntropyLoss for ThermoDataBase ($122\text{ Diabetic}:45\text{ Control}$).
   - No clinical efficacy or ulcer prediction claims will be made.


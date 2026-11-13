# FootGuard AI — Phase 2 Preprocessing & Data Validation Notes

**Document:** Engineering & Clinical Validation Notes  
**Author:** Senior Machine Learning Engineer  
**Date:** November 8, 2026  
**Status:** Approved & Formally Documented  

---

## 1. STANDUP Database: Partitioning & Leakage Safeguards

### Empirical Finding:
In `raw/Thermal-Data/STANDUP_Database/STANDUP_Database/`, there are 415 thermal and 415 RGB images named uniformly according to group and sequence number:
- `healthy/thermal/H_0001.png` to `H_0125.png`
- `diabetic/R0/thermal/R0_0001.png` to `R0_0138.png`
- `diabetic/R1/thermal/R1_0001.png` to `R1_0068.png`
- `diabetic/R2/thermal/R2_0001.png` to `R2_0084.png`

No patient identifier registry, clinical log, or demographic lookup sheet accompanies this dataset. 

### Data Integrity Decision & Limitation:
1. **No Patient-Level Isolation Claim:**  
   Because individual subject identities cannot be authenticated from the file headers or naming schemes, images may represent either distinct individuals or repeated captures/bilateral observations of the same subjects. Therefore, **we do NOT claim patient-independent or patient-leakage-free validation on STANDUP.**
2. **Role in FootGuard AI:**  
   STANDUP is relegated strictly to an **exploratory screening baseline**.
3. **Primary Evaluation Standard:**  
   `ThermoDataBase` ($N=167$ authenticated subjects) remains our primary clinical benchmark for thermal diabetes classification, where patient-level isolation is 100% mathematically verified.

---

## 2. CAD_WALK: Spatial Transformation (38 × 22 to Canonical 48 × 32)

### Empirical Finding:
`CAD_WALK` exports dynamic NIfTI files where spatial dimensions vary slightly per trial (e.g., $35 \times 23$, $36 \times 22$, $38 \times 22$, $40 \times 21$, up to a maximum observed bounding box of $42 \times 28$). This occurs because the acquisition software exported cropped bounding boxes around the active footprint on the 0.5m footscan plate.

### Transformation Strategy:
1. **Preservation of Raw Physical Calibration:**  
   Instead of using spatial bilinear or bicubic interpolation—which would blur acute local pressure peaks (e.g., under the metatarsal heads) and distort physical pressure values ($\text{kPa}$)—we employ **symmetric zero-padding**.
2. **Canonical Dimensions $(48, 32)$:**  
   The target dimension of $(48, 32)$ was chosen because:
   - It comfortably encloses the maximum observed foot dimension ($42 \times 28$) with a uniform margin.
   - Both 48 and 32 are multiples of 16 (and powers of 2 factors), allowing standard convolutional downsampling strides ($/2, /4, /8$) without fractional padding artifacts.
3. **Physical Pitch:**  
   The physical sensor pitch remains intact:
   $$\Delta x = 7.62\text{ mm (AP axis)}, \quad \Delta y = 5.08\text{ mm (ML axis)}$$
   No geometric warping or aspect-ratio distortion is introduced. Background padding values are strictly $0.0\text{ kPa}$, matching the physical reality of unloaded sensor cells outside the plantar boundary.

---

## 3. Gait Lab Validation: Source (20 Columns) to Output (18 Channels) Mapping

### Source Data Structure:
Each continuous walking recording (e.g., `HFMD1_LEFT_W1.csv`, `HFM2_LEFT_W1.csv`) contains 72,001 rows and exactly **20 columns**:

| Column Index | Source Designation | Data Type | Physical Meaning | Included in Signal Tensor? |
| :---: | :---: | :---: | :---: | :---: |
| **0** | Shoe / Foot | String (`LEFT` / `RIGHT`) | Anatomical designation | ❌ No (stored in sample metadata) |
| **1** | Time ($t$) | Integer (milliseconds) | Hardware acquisition timestamp | ❌ No (used to derive continuous time vector `time_sec`) |
| **2** | $F_1$ | Float (Volts) | Normal stress sensor: 1st Metatarsal Head (MH) | ✅ **Channel 0** |
| **3** | $F_2$ | Float (Volts) | Normal stress sensor: Hallux (HA) | ✅ **Channel 1** |
| **4–9** | $F_3 \dots F_8$ | Float (Volts) | Normal stress sensors: Midfoot & Heel ($F_8$) | ✅ **Channels 2–7** |
| **10** | $S_3$ | Float (mV) | Shear sensor: Metatarsal Head (MH) axis | ✅ **Channel 8** |
| **11** | $S_1$ | Float (mV) | Shear sensor: Metatarsal Head (MH) axis | ✅ **Channel 9** |
| **12** | $S_2$ | Float (mV) | Shear sensor: Metatarsal Head (MH) axis | ✅ **Channel 10** |
| **13** | $S_9$ | Float (mV) | Shear sensor: Heel (HE) axis | ✅ **Channel 11** |
| **14** | $S_7$ | Float (mV) | Shear sensor: Heel (HE) axis | ✅ **Channel 12** |
| **15** | $S_8$ | Float (mV) | Shear sensor: Heel (HE) axis | ✅ **Channel 13** |
| **16** | $S_6$ | Float (mV) | Shear sensor: Hallux (HA) axis | ✅ **Channel 14** |
| **17** | $S_4$ | Float (mV) | Shear sensor: Hallux (HA) axis | ✅ **Channel 15** |
| **18** | $S_5$ | Float (mV) | Shear sensor: Hallux (HA) axis | ✅ **Channel 16** |
| **19** | Flag / Sync | Integer / Float | Acquisition frame flag / hardware status | ✅ **Channel 17** |

### Summary of Mapping:
- Column 0 (Foot string) and Column 1 (Time ms) are parsed into trial metadata and relative time coordinates.
- Columns 2 through 19 represent the **18 continuous physical and hardware channels** ($8\text{ normal stress} + 9\text{ tri-axial shear} + 1\text{ sync flag} = 18\text{ channels}$).
- Hence, the conditioned tensor shape across $L$ timesteps is strictly $(18, L)$ (Channels-first in PyTorch).

---

## 4. IAC-TecMed: Sample Size Constraints & Clinical Interpretation

### Empirical Reality:
- Total Participants: Exactly **19 human subjects**.
- Pathology: **100% Non-Diabetic Controls** (`Diabetic: No` across all 19 metadata sheets).
- Observations: 4 sequential cold-stress recovery timepoints ($T_0$: baseline, $T_5$: 5 min rewarming, $T_{10}$: 10 min rewarming, $T_{15}$: 15 min stabilization).

### Engineering & Clinical Constraints:
1. **Strictly Exploratory / Architectural Proof-of-Concept:**  
   With $N=19$ subjects (13 Train, 3 Val, 3 Test in participant-level split), the IAC-TecMed temporal model is an architectural validation tool for spatial-temporal encoding (CNN-TCN). It is **not** suitable for high-confidence clinical inference or generalized population claims.
2. **Prohibition of "Wound Healing" Nomenclature:**  
   The 15-minute thermal rewarming trajectory reflects microvascular vasomotor tone and sympathetic response to an acute thermal shock. It **must not** be described as tissue healing, ulcer duration, or long-term therapeutic recovery.
3. **Evaluation Protocol:**  
   Evaluations focus on feature extraction quality, temporal smoothness, and rewarming trend reconstruction (e.g., predicting $T_{15} - T_5$ rewarming slope), with full reporting of sample size uncertainty.


"""
FootGuard AI - ML Common Configuration
Central repository paths, preprocessing parameters, and constants.
"""

from pathlib import Path

# Base ML Root
ML_ROOT = Path(__file__).resolve().parent.parent

# Raw data directories (READ ONLY)
RAW_DIR = ML_ROOT / "raw"
PRESSURE_RAW_DIR = RAW_DIR / "Pressure-Data"
THERMAL_RAW_DIR = RAW_DIR / "Thermal-Data"

# Specific raw datasets
PRESSURE_CALIB_DIR = PRESSURE_RAW_DIR / "1-Calibration Investigation"
PRESSURE_LOADING_DIR = PRESSURE_RAW_DIR / "2 - Loading Profile Comparison"
PRESSURE_BENCH_DIR = PRESSURE_RAW_DIR / "3 - Bench Top Validation"
PRESSURE_GAIT_DIR = PRESSURE_RAW_DIR / "4 - Gait Lab Validation"
PRESSURE_CAD_WALK_DIR = PRESSURE_RAW_DIR / "CAD_WALK_Healthy_Controls_Dataset (1)" / "HealthyControls"

THERMAL_IAC_DIR = THERMAL_RAW_DIR / "IAC-TecMed Database_1" / "IAC-TecMed Database"
THERMAL_STANDUP_DIR = THERMAL_RAW_DIR / "STANDUP_Database" / "STANDUP_Database"
THERMAL_TDB_DIR = THERMAL_RAW_DIR / "ThermoDataBase" / "ThermoDataBase"

# Processed output directories
PROCESSED_DIR = ML_ROOT / "processed"
PROCESSED_THERMAL_DIR = PROCESSED_DIR / "thermal"
PROCESSED_PRESSURE_DIR = PROCESSED_DIR / "pressure"

# Splits and experiment artifacts
SPLITS_DIR = ML_ROOT / "splits"
SPLITS_THERMAL_DIR = SPLITS_DIR / "thermal"
SPLITS_PRESSURE_DIR = SPLITS_DIR / "pressure"

CHECKPOINTS_DIR = ML_ROOT / "checkpoints"
EVALUATION_DIR = ML_ROOT / "evaluation"
EXPERIMENTS_DIR = ML_ROOT / "experiments"

# Preprocessing parameters
SEED = 42

# Thermal 2D CNN preprocessing canonical resolution (H, W)
THERMAL_CANONICAL_SHAPE = (160, 80)
# Temperature threshold below which pixels are considered masked ambient background
THERMAL_BACKGROUND_THRESHOLD_C = 15.0
# Physiological valid temperature bounds for human plantar skin (°C)
THERMAL_MIN_VALID_TEMP_C = 18.0
THERMAL_MAX_VALID_TEMP_C = 42.0

# Pressure temporal resampling (fixed percentage of stance phase)
PRESSURE_STANCE_RESAMPLED_LEN = 100
# CAD_WALK nominal spatial grid
PRESSURE_GRID_SHAPE = (38, 22)
# CAD_WALK canonical uniform padded grid (H, W) zero-padded to accommodate all foot sizes
PRESSURE_CANONICAL_GRID = (48, 32)
# Sensor cell physical pitch (mm)
PRESSURE_SENSOR_PITCH_MM = (7.62, 5.08)

# Split ratios (Participant-level)
TRAIN_RATIO = 0.70
VAL_RATIO = 0.15
TEST_RATIO = 0.15

# Ensure output directories exist
for p in [PROCESSED_THERMAL_DIR, PROCESSED_PRESSURE_DIR, SPLITS_THERMAL_DIR, 
         SPLITS_PRESSURE_DIR, CHECKPOINTS_DIR, EVALUATION_DIR, EXPERIMENTS_DIR]:
    p.mkdir(parents=True, exist_ok=True)

"""
FootGuard AI - Dataset Loading & Inspection Service
Supports real research datasets (ThermoDataBase, IAC-TecMed, STANDUP, CAD_WALK, Gait Lab)
and synthetic datasets (footguard_synthetic_paired_readings.csv + local synthetic CSVs).
Uses chunked reads and session-level filtering for the 258.9 MB synthetic dataset.
Raw datasets remain strictly immutable.
"""

from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Dict, Any, List, Optional
import json
import os
import numpy as np
import pandas as pd
from PIL import Image

from services.config_service import ConfigService


@dataclass
class DatasetDescriptor:
    dataset_id: str
    name: str
    origin: str  # "real" or "synthetic"
    provenance_label: str  # "CLINICAL_RESEARCH_DATASET" or "SYNTHETIC_NOT_CLINICAL_GROUND_TRUTH"
    modality: str  # "thermal", "pressure", "multimodal"
    description: str
    path: str
    file_size_mb: float
    num_samples: Optional[int]
    num_participants_or_sessions: Optional[int]
    split_methodology: str
    is_calibrated_celsius: Optional[bool]
    compatible_models: List[str]
    limitations: str

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


class DatasetService:
    """Manages dataset discovery, chunked loading, metadata inspection, and split summaries."""

    def __init__(self, config_service: Optional[ConfigService] = None):
        self.cfg = config_service or ConfigService()
        self.ml_root = self.cfg.ml_root
        self.processed_dir = self.cfg.get_path("processed_dir")
        self.splits_dir = self.cfg.get_path("splits_dir")
        self.raw_dir = self.cfg.get_path("raw_real_dir")
        self.synthetic_csv = self.cfg.get_path("synthetic_csv")
        self.data_synthetic_dir = self.cfg.get_path("data_synthetic_dir")

    def _safe_manifest_stats(self, manifest_path: Path, id_col: str) -> Dict[str, Optional[int]]:
        if not manifest_path.exists():
            return {"num_samples": None, "num_ids": None}
        df = pd.read_csv(manifest_path)
        num_samples = int(len(df))
        num_ids = int(df[id_col].nunique()) if id_col in df.columns else None
        return {"num_samples": num_samples, "num_ids": num_ids}

    def list_datasets(self) -> List[DatasetDescriptor]:
        """Returns descriptors for all available real and synthetic datasets without loading large files."""
        descriptors: List[DatasetDescriptor] = []

        # 1. ThermoDataBase (Real, Thermal, Calibrated Celsius)
        tdb_manifest = self.processed_dir / "thermal" / "thermodatabase_manifest.csv"
        tdb_stats = self._safe_manifest_stats(tdb_manifest, "subject_id")
        descriptors.append(
            DatasetDescriptor(
                dataset_id="thermodatabase",
                name="ThermoDataBase (Plantar Thermography)",
                origin="real",
                provenance_label="CLINICAL_RESEARCH_DATASET",
                modality="thermal",
                description="Calibrated Celsius bilateral plantar thermograms (Control vs. Diabetic cohort).",
                path=str(tdb_manifest if tdb_manifest.exists() else self.raw_dir / "Thermal-Data" / "ThermoDataBase"),
                file_size_mb=round(tdb_manifest.stat().st_size / (1024 * 1024), 3) if tdb_manifest.exists() else 0.0,
                num_samples=tdb_stats["num_samples"],
                num_participants_or_sessions=tdb_stats["num_ids"],
                split_methodology="Participant-level 70/15/15 (116 train / 25 val / 26 test)",
                is_calibrated_celsius=True,
                compatible_models=["thermal_2d_cnn"],
                limitations="Diabetes cohort stratification only (NOT ulcer detection). Small test cohort (N=26).",
            )
        )

        # 2. IAC-TecMed (Real, Thermal TCN, Calibrated Celsius)
        iac_manifest = self.processed_dir / "thermal" / "iac_tecmed_manifest.csv"
        iac_stats = self._safe_manifest_stats(iac_manifest, "subject_id")
        descriptors.append(
            DatasetDescriptor(
                dataset_id="iac_tecmed",
                name="IAC-TecMed (Dynamic Cold-Rewarming)",
                origin="real",
                provenance_label="CLINICAL_RESEARCH_DATASET",
                modality="thermal",
                description="Multi-timepoint (T0, T5, T10, T15) calibrated TIFF rewarming sequences.",
                path=str(iac_manifest if iac_manifest.exists() else self.raw_dir / "Thermal-Data" / "IAC-TecMed Database_1"),
                file_size_mb=round(iac_manifest.stat().st_size / (1024 * 1024), 3) if iac_manifest.exists() else 0.0,
                num_samples=iac_stats["num_samples"],
                num_participants_or_sessions=iac_stats["num_ids"],
                split_methodology="Participant-level (13 train / 3 val / 3 test)",
                is_calibrated_celsius=True,
                compatible_models=["thermal_tcn"],
                limitations="Severely limited sample size (N=19). Experimental trajectory modeling only; NOT wound healing.",
            )
        )

        # 3. STANDUP (Real, Thermal Screening PNGs - Uncalibrated)
        st_manifest = self.processed_dir / "thermal" / "standup_manifest.csv"
        st_stats = self._safe_manifest_stats(st_manifest, "image_id")
        descriptors.append(
            DatasetDescriptor(
                dataset_id="standup",
                name="STANDUP Database (Uncalibrated PNG Screening)",
                origin="real",
                provenance_label="CLINICAL_RESEARCH_DATASET",
                modality="thermal",
                description="False-color PNG plantar screening images with IWGDF risk categories.",
                path=str(st_manifest if st_manifest.exists() else self.raw_dir / "Thermal-Data" / "STANDUP_Database"),
                file_size_mb=round(st_manifest.stat().st_size / (1024 * 1024), 3) if st_manifest.exists() else 0.0,
                num_samples=st_stats["num_samples"],
                num_participants_or_sessions=None,
                split_methodology="No participant IDs available; leakage-free patient splitting impossible",
                is_calibrated_celsius=False,
                compatible_models=[],
                limitations="Uncalibrated 8-bit pseudo-color PNGs (blocked from Celsius Thermal2DCNN) and lacks participant IDs.",
            )
        )

        # 4. CAD_WALK (Real, Pressure, Healthy Controls)
        cad_manifest = self.processed_dir / "pressure" / "cad_walk_manifest.csv"
        cad_stats = self._safe_manifest_stats(cad_manifest, "subject_id")
        descriptors.append(
            DatasetDescriptor(
                dataset_id="cad_walk",
                name="CAD_WALK (Healthy Controls Pressure & Speed)",
                origin="real",
                provenance_label="CLINICAL_RESEARCH_DATASET",
                modality="pressure",
                description="Dynamic plantar pressure & 100-pt normalized vGRF stance curves with walking speed labels.",
                path=str(cad_manifest if cad_manifest.exists() else self.raw_dir / "Pressure-Data"),
                file_size_mb=round(cad_manifest.stat().st_size / (1024 * 1024), 3) if cad_manifest.exists() else 0.0,
                num_samples=cad_stats["num_samples"],
                num_participants_or_sessions=cad_stats["num_ids"],
                split_methodology="Participant-level 70/15/15 (36 train / 8 val / 7 test; C51 unlabeled excluded)",
                is_calibrated_celsius=None,
                compatible_models=["pressure_1d_cnn", "pressure_tcn", "pressure_ridge_baseline"],
                limitations="100% healthy Dutch controls. Cannot be used for diabetic or ulcer classification.",
            )
        )

        # 5. Gait Lab (Real, Multi-Channel Insole, N=2)
        gl_manifest = self.processed_dir / "pressure" / "gait_lab_manifest.csv"
        gl_stats = self._safe_manifest_stats(gl_manifest, "subject_id")
        descriptors.append(
            DatasetDescriptor(
                dataset_id="gait_lab",
                name="Gait Lab Validation (18-Channel Insole)",
                origin="real",
                provenance_label="CLINICAL_RESEARCH_DATASET",
                modality="pressure",
                description="Continuous multi-channel pressure/shear insole runs with FixJumpv24 detrending.",
                path=str(gl_manifest if gl_manifest.exists() else self.raw_dir / "Pressure-Data" / "4 - Gait Lab Validation"),
                file_size_mb=round(gl_manifest.stat().st_size / (1024 * 1024), 3) if gl_manifest.exists() else 0.0,
                num_samples=gl_stats["num_samples"],
                num_participants_or_sessions=gl_stats["num_ids"],
                split_methodology="Exploratory only (N=2 participants: 1 diabetic, 1 control)",
                is_calibrated_celsius=None,
                compatible_models=[],
                limitations="Severe participant identity confounding (N=2). Supervised classification invalid.",
            )
        )

        # 6. Synthetic Paired Readings (258.9 MB CSV, 1M rows, 3,334 sessions)
        if self.synthetic_csv.exists():
            size_mb = round(self.synthetic_csv.stat().st_size / (1024 * 1024), 2)
            descriptors.append(
                DatasetDescriptor(
                    dataset_id="synthetic_paired_readings",
                    name="FootGuard Synthetic Paired Readings (CSV)",
                    origin="synthetic",
                    provenance_label="SYNTHETIC_NOT_CLINICAL_GROUND_TRUTH",
                    modality="multimodal",
                    description="1M-row paired synthetic thermal zone, 6-site pressure, and 6-axis IMU readings across 3,334 sessions.",
                    path=str(self.synthetic_csv),
                    file_size_mb=size_mb,
                    num_samples=1_000_000,
                    num_participants_or_sessions=3334,
                    split_methodology="Session-level isolation (session_id) for synthetic experiments",
                    is_calibrated_celsius=True,
                    compatible_models=["pressure_1d_cnn", "pressure_tcn"],
                    limitations="SYNTHETIC_NOT_CLINICAL_GROUND_TRUTH. Tabular/zone features; never mix with real clinical evaluations.",
                )
            )

        # 7. Any additional CSVs placed in data/synthetic/
        for extra_csv in sorted(self.data_synthetic_dir.glob("*.csv")):
            size_mb = round(extra_csv.stat().st_size / (1024 * 1024), 3)
            descriptors.append(
                DatasetDescriptor(
                    dataset_id=f"synthetic_custom::{extra_csv.stem}",
                    name=f"Custom Synthetic CSV ({extra_csv.name})",
                    origin="synthetic",
                    provenance_label="SYNTHETIC_NOT_CLINICAL_GROUND_TRUTH",
                    modality="multimodal",
                    description=f"Local synthetic CSV stored in data/synthetic/{extra_csv.name}.",
                    path=str(extra_csv),
                    file_size_mb=size_mb,
                    num_samples=None,
                    num_participants_or_sessions=None,
                    split_methodology="User-defined / session-level",
                    is_calibrated_celsius=None,
                    compatible_models=[],
                    limitations="Synthetic user dataset; not clinical ground truth.",
                )
            )

        return descriptors

    def get_descriptor(self, dataset_id: str) -> DatasetDescriptor:
        for d in self.list_datasets():
            if d.dataset_id == dataset_id:
                return d
        raise ValueError(f"Unknown dataset_id: '{dataset_id}'")

    def inspect_metadata(self, dataset_id: str) -> Dict[str, Any]:
        """Inspects dataset schema, columns, and metadata without loading the full dataset into memory."""
        desc = self.get_descriptor(dataset_id)
        meta: Dict[str, Any] = {
            "descriptor": desc.to_dict(),
            "columns": [],
            "dtypes": {},
            "split_summary": self.get_split_summary(dataset_id),
        }

        if dataset_id == "synthetic_paired_readings":
            header_df = pd.read_csv(self.synthetic_csv, nrows=5)
            meta["columns"] = list(header_df.columns)
            meta["dtypes"] = {col: str(dtype) for col, dtype in header_df.dtypes.items()}
            meta["sample_session_ids"] = self.list_synthetic_sessions(max_sessions=20)
        elif dataset_id.startswith("synthetic_custom::"):
            csv_path = Path(desc.path)
            header_df = pd.read_csv(csv_path, nrows=5)
            meta["columns"] = list(header_df.columns)
            meta["dtypes"] = {col: str(dtype) for col, dtype in header_df.dtypes.items()}
        else:
            manifest_path = Path(desc.path)
            if manifest_path.is_file() and manifest_path.suffix == ".csv":
                df_head = pd.read_csv(manifest_path, nrows=5)
                meta["columns"] = list(df_head.columns)
                meta["dtypes"] = {col: str(dtype) for col, dtype in df_head.dtypes.items()}

        return meta

    def list_synthetic_sessions(self, max_sessions: int = 50) -> List[str]:
        """Lists up to `max_sessions` unique session IDs using chunked CSV reads."""
        if not self.synthetic_csv.exists():
            raise FileNotFoundError(f"Synthetic CSV not found: {self.synthetic_csv}")

        chunk_size = self.cfg.get_synthetic_loader_config().get("chunk_size", 50000)
        seen: List[str] = []
        seen_set = set()

        for chunk in pd.read_csv(self.synthetic_csv, usecols=["session_id"], chunksize=chunk_size):
            for sid in chunk["session_id"].dropna().unique():
                if sid not in seen_set:
                    seen_set.add(sid)
                    seen.append(str(sid))
                    if len(seen) >= max_sessions:
                        return seen
        return seen

    def load_synthetic_session(self, session_id: str, max_rows: int = 2000) -> pd.DataFrame:
        """
        Loads all rows (up to max_rows) for a single synthetic session via chunked reading.
        Never loads the full 258.9 MB CSV into memory.
        """
        if not self.synthetic_csv.exists():
            raise FileNotFoundError(f"Synthetic CSV not found: {self.synthetic_csv}")

        chunk_size = self.cfg.get_synthetic_loader_config().get("chunk_size", 50000)
        matched_chunks: List[pd.DataFrame] = []
        collected = 0
        found_in_previous = False

        for chunk in pd.read_csv(self.synthetic_csv, chunksize=chunk_size):
            sub = chunk[chunk["session_id"] == session_id]
            if not sub.empty:
                matched_chunks.append(sub)
                collected += len(sub)
                found_in_previous = True
                if collected >= max_rows:
                    break
            elif found_in_previous:
                # Sessions in synthetic CSV are contiguous (SYN-0001 .. SYN-3334)
                break

        if not matched_chunks:
            raise ValueError(f"Synthetic session '{session_id}' not found in {self.synthetic_csv.name}.")

        df = pd.concat(matched_chunks, ignore_index=True).iloc[:max_rows]
        return df

    def sample_synthetic_rows(self, n_rows: int = 500, seed: int = 42) -> pd.DataFrame:
        """
        Samples a representative subset from the 1M-row synthetic dataset using strided chunk reads.
        Does not load the entire file into memory.
        """
        if not self.synthetic_csv.exists():
            raise FileNotFoundError(f"Synthetic CSV not found: {self.synthetic_csv}")
        if n_rows <= 0:
            raise ValueError("n_rows must be > 0")

        chunk_size = self.cfg.get_synthetic_loader_config().get("chunk_size", 50000)
        rng = np.random.default_rng(seed)
        # Collect small deterministic samples across the first 5 chunks (250k rows) for fast interactive responsiveness
        max_chunks = 5
        per_chunk = max(1, int(np.ceil(n_rows / max_chunks)))
        samples: List[pd.DataFrame] = []

        for idx, chunk in enumerate(pd.read_csv(self.synthetic_csv, chunksize=chunk_size)):
            take = min(per_chunk, len(chunk))
            sample_idx = rng.choice(len(chunk), size=take, replace=False)
            samples.append(chunk.iloc[np.sort(sample_idx)])
            if idx + 1 >= max_chunks or sum(len(s) for s in samples) >= n_rows:
                break

        if not samples:
            raise ValueError("Synthetic dataset is empty.")

        df = pd.concat(samples, ignore_index=True).iloc[:n_rows].reset_index(drop=True)
        return df

    def load_dataset_preview(
        self,
        dataset_id: str,
        limit: int = 100,
        session_id: Optional[str] = None,
        seed: int = 42,
    ) -> pd.DataFrame:
        """Loads a preview DataFrame for either real manifests or synthetic CSVs."""
        desc = self.get_descriptor(dataset_id)

        if dataset_id == "synthetic_paired_readings":
            if session_id:
                return self.load_synthetic_session(session_id=session_id, max_rows=limit)
            return self.sample_synthetic_rows(n_rows=limit, seed=seed)

        if dataset_id.startswith("synthetic_custom::"):
            return pd.read_csv(desc.path, nrows=limit)

        manifest_path = Path(desc.path)
        if not manifest_path.exists():
            raise FileNotFoundError(f"Manifest not found for '{dataset_id}': {manifest_path}")

        df = pd.read_csv(manifest_path, nrows=limit)
        df["provenance"] = desc.provenance_label
        return df

    def get_real_sample(self, dataset_id: str, index: int = 0) -> Dict[str, Any]:
        """Loads a single processed sample (arrays + metadata) from a real dataset."""
        desc = self.get_descriptor(dataset_id)
        if desc.origin != "real":
            raise ValueError(f"Dataset '{dataset_id}' is not a real dataset.")

        manifest_path = Path(desc.path)
        if not manifest_path.exists():
            raise FileNotFoundError(f"Manifest not found for '{dataset_id}': {manifest_path}")

        df = pd.read_csv(manifest_path)
        if index < 0 or index >= len(df):
            raise IndexError(f"Sample index {index} out of range [0, {len(df) - 1}] for '{dataset_id}'.")

        row = df.iloc[index].to_dict()
        file_rel = row.get("file_path")
        if not file_rel:
            raise ValueError(f"Manifest row {index} missing 'file_path'.")

        file_path = self.processed_dir / str(file_rel)
        if not file_path.exists():
            alt_path = self.ml_root / str(file_rel)
            if alt_path.exists():
                file_path = alt_path
            else:
                raise FileNotFoundError(f"Sample artifact not found: {file_path}")

        payload: Dict[str, Any] = {
            "dataset_id": dataset_id,
            "origin": "real",
            "provenance_label": desc.provenance_label,
            "is_calibrated_celsius": desc.is_calibrated_celsius,
            "metadata": row,
            "file_path": str(file_path),
        }

        if file_path.suffix == ".npz":
            data = np.load(file_path)
            payload["arrays"] = {k: data[k] for k in data.files}
        elif file_path.suffix.lower() in [".png", ".jpg", ".jpeg"]:
            img = Image.open(file_path)
            payload["arrays"] = {"image_rgb": np.array(img)}
        else:
            payload["arrays"] = {}

        return payload

    def get_split_summary(self, dataset_id: str) -> Dict[str, Any]:
        """Returns participant-level or session-level split counts."""
        split_map = {
            "thermodatabase": self.splits_dir / "thermal" / "thermodatabase_splits.json",
            "iac_tecmed": self.splits_dir / "thermal" / "iac_tecmed_splits.json",
            "cad_walk": self.splits_dir / "pressure" / "cad_walk_splits.json",
            "gait_lab": self.splits_dir / "pressure" / "gait_lab_splits.json",
        }

        if dataset_id in split_map and split_map[dataset_id].exists():
            with open(split_map[dataset_id], "r", encoding="utf-8") as f:
                data = json.load(f)
            summary: Dict[str, Any] = {"split_file": str(split_map[dataset_id])}
            for k, v in data.items():
                summary[k] = len(v) if isinstance(v, list) else v
            return summary

        if dataset_id == "standup":
            return {"note": "Participant IDs unavailable; split not leakage-safe."}

        if dataset_id == "synthetic_paired_readings":
            return {
                "note": "Session-level isolation across synthetic session_id values (SYN-0001..SYN-3334).",
                "total_sessions": 3334,
            }

        return {"note": "No predefined split manifest."}

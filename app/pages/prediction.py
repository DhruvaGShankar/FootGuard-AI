"""
FootGuard AI - Local Prediction & Inference Page
Executes ThermalInferenceEngine and PressureInferenceEngine via InferenceService.
Enforces calibration, scalar/DEMO safety separation, and clear error reporting.
"""

from pathlib import Path
import tempfile
import numpy as np
import pandas as pd
import streamlit as st

from app.state import (
    get_config_service,
    get_dataset_service,
    get_model_service,
    get_inference_service,
)
from app.components.metric_cards import render_research_banner, render_kpi_row
from app.components.thermal_viewer import render_bilateral_thermal_maps
from app.components.pressure_viewer import render_force_curve
from common.schemas import PressureInferenceRequest


def _render_thermal_prediction() -> None:
    ds_svc = get_dataset_service()
    model_svc = get_model_service()
    inf_svc = get_inference_service()

    st.subheader("Thermal Cohort Screening (Control vs. Diabetic Cohort)")
    st.caption("Requires calibrated physical Celsius thermogram matrices (Left & Right feet). Uncalibrated PNGs (e.g. STANDUP) are blocked.")

    ckpts = model_svc.list_checkpoints("thermal_2d_cnn")
    if not ckpts:
        st.error("No Thermal 2D CNN checkpoints found in ML/checkpoints/.")
        return

    ckpt_map = {c.checkpoint_id: c.file_path for c in ckpts}
    selected_ckpt = st.selectbox("Select Thermal Checkpoint", options=list(ckpt_map.keys()))
    threshold = st.slider("Decision Threshold (τ)", min_value=0.10, max_value=0.90, value=0.50, step=0.05)

    input_mode = st.radio("Thermal Input Source", ["Real ThermoDataBase Sample", "Upload Calibrated CSV Pair"], horizontal=True)

    left_input = None
    right_input = None
    ground_truth_label = None
    data_origin = "real"

    if input_mode == "Real ThermoDataBase Sample":
        tdb_desc = ds_svc.get_descriptor("thermodatabase")
        max_idx = max(0, (tdb_desc.num_samples or 1) - 1)
        idx = st.number_input("ThermoDataBase Subject Index", min_value=0, max_value=max_idx, value=0, step=1)
        try:
            sample = ds_svc.get_real_sample("thermodatabase", index=int(idx))
            tensor = sample["arrays"]["tensor"]  # (3, 160, 80): [Left, Right_flipped, Delta_T]
            left_input = tensor[0]
            right_input = np.fliplr(tensor[1]).copy()
            ground_truth_label = sample["metadata"].get("class_name", "Unknown")
            st.write(f"Loaded Subject: `{sample['metadata'].get('subject_id')}` | Dataset Label: **{ground_truth_label}**")
        except Exception as exc:
            st.error(f"Could not load ThermoDataBase sample: {exc}")
            return
    else:
        c1, c2 = st.columns(2)
        left_file = c1.file_uploader("Upload Left Foot CSV (Celsius)", type=["csv", "png", "jpg"], key="left_csv_up")
        right_file = c2.file_uploader("Upload Right Foot CSV (Celsius)", type=["csv", "png", "jpg"], key="right_csv_up")
        data_origin = "uploaded"
        if left_file is not None and right_file is not None:
            if not left_file.name.lower().endswith(".csv") or not right_file.name.lower().endswith(".csv"):
                st.error("Uncalibrated image format rejected: please upload calibrated Celsius `.csv` matrices only.")
                return
            try:
                left_input = pd.read_csv(left_file, header=None).to_numpy(dtype=np.float32)
                right_input = pd.read_csv(right_file, header=None).to_numpy(dtype=np.float32)
            except Exception as exc:
                st.error(f"Failed to parse uploaded CSV matrices: {exc}")
                return

    if st.button("Run Thermal Inference", type="primary", disabled=(left_input is None or right_input is None)):
        try:
            res = inf_svc.run_thermal_inference(
                (left_input, right_input),
                checkpoint_path=ckpt_map[selected_ckpt],
                decision_threshold=float(threshold),
                data_origin=data_origin,
            )
            pred = res["prediction"]
            bm = res["bilateral_metrics"]

            render_kpi_row([
                ("Predicted Cohort", pred["predicted_label"], f"GT: {ground_truth_label}" if ground_truth_label else None),
                ("Raw Softmax Score (Uncalibrated)", f"{pred['confidence']:.3f}", None),
                ("Mean |L - R| Asymmetry", f"{bm['mean_abs_asymmetry_c']:.2f} °C", None),
                ("Area ΔT > 2.2°C", f"{bm['hyperthermic_asymmetry_fraction'] * 100:.1f}%", None),
            ])
            st.caption(f"**Statistical Note:** {res['softmax_calibration_note']}")
            st.caption(f"**Input SHA-256:** `{res['input_hash']}` | **Latency:** `{res['latency_ms']} ms`")

            if res["quality_flags"]["warnings"]:
                for w in res["quality_flags"]["warnings"]:
                    st.warning(w)

            render_bilateral_thermal_maps(left_input, np.fliplr(right_input), np.abs(left_input - np.fliplr(right_input)))
            st.info(res["disclaimer"])
        except Exception as exc:
            st.error(f"Thermal inference validation failed: {exc}")


def _render_pressure_prediction() -> None:
    ds_svc = get_dataset_service()
    model_svc = get_model_service()
    inf_svc = get_inference_service()

    st.subheader("Pressure & Gait Walking Speed Regression (Healthy Controls Domain)")
    st.caption("Supports real 1D vertical ground reaction force (vGRF) curves or scalar Ridge baseline parameters.")

    ckpts = model_svc.list_checkpoints("pressure_1d_cnn")
    if not ckpts:
        st.error("No Pressure 1D CNN checkpoints found in ML/checkpoints/.")
        return

    ckpt_map = {c.checkpoint_id: c.file_path for c in ckpts}
    selected_ckpt = st.selectbox("Select Pressure CNN Checkpoint", options=list(ckpt_map.keys()))

    input_mode = st.radio(
        "Pressure Input Source",
        ["Real CAD_WALK Trial", "Upload 1D Force Curve (.csv/.npy)", "Scalar Stance Parameters (Ridge Baseline)"],
        horizontal=True,
    )

    payload = None
    plot_curve = None
    actual_speed = None
    data_origin = "real"

    if input_mode == "Real CAD_WALK Trial":
        cad_desc = ds_svc.get_descriptor("cad_walk")
        max_idx = max(0, (cad_desc.num_samples or 1) - 1)
        idx = st.number_input("CAD_WALK Trial Index", min_value=0, max_value=max_idx, value=0, step=1)
        try:
            sample = ds_svc.get_real_sample("cad_walk", index=int(idx))
            meta = sample["metadata"]
            actual_speed = meta.get("walking_speed")
            curve = sample["arrays"]["force_curve"]
            plot_curve = curve
            if actual_speed and float(actual_speed) > 0.5:
                st.write(f"Loaded Trial: `{meta.get('trial_id')}` | Recorded Speed: **{float(actual_speed):.2f} km/h**")
            else:
                st.warning(f"Loaded Trial `{meta.get('trial_id')}` has no recorded walking speed label (e.g. subject C51).")
            payload = PressureInferenceRequest(
                patient_id=str(meta.get("subject_id", "")),
                trial_id=str(meta.get("trial_id", "")),
                foot_side=str(meta.get("foot", "unspecified")).lower(),
                vertical_force_curve=curve.tolist(),
                stance_duration_ms=float(meta["duration_ms"]) if meta.get("duration_ms") else None,
                peak_force_n=float(meta["peak_force_n"]) if meta.get("peak_force_n") else None,
            )
        except Exception as exc:
            st.error(f"Could not load CAD_WALK sample: {exc}")
            return

    elif input_mode == "Upload 1D Force Curve (.csv/.npy)":
        data_origin = "uploaded"
        up_file = st.file_uploader("Upload 1D vGRF Curve", type=["csv", "npy"], key="pressure_up")
        duration_opt = st.number_input("Stance Duration (ms)", min_value=150.0, max_value=3000.0, value=620.0, step=10.0)
        if up_file is not None:
            try:
                if up_file.name.lower().endswith(".npy"):
                    arr = np.load(up_file).astype(np.float32).ravel()
                else:
                    arr = pd.read_csv(up_file, header=None).to_numpy(dtype=np.float32).ravel()
                plot_curve = arr
                payload = PressureInferenceRequest(
                    vertical_force_curve=arr.tolist(),
                    stance_duration_ms=float(duration_opt),
                )
            except Exception as exc:
                st.error(f"Failed to parse uploaded force curve: {exc}")
                return
    else:
        data_origin = "scalar_input"
        c1, c2 = st.columns(2)
        dur = c1.number_input("Stance Duration (ms)", min_value=50.0, max_value=3000.0, value=615.0, step=5.0)
        peak = c2.number_input("Peak Vertical Force (N)", min_value=50.0, max_value=5000.0, value=840.0, step=10.0)
        allow_demo = st.checkbox(
            "Enable DEMO Mode (synthesize artificial bimodal curve for 1D CNN — NOT a real sensor measurement)",
            value=False,
        )
        payload = PressureInferenceRequest(
            stance_duration_ms=float(dur),
            peak_force_n=float(peak),
            allow_synthetic_demo=bool(allow_demo),
        )

    if st.button("Run Pressure Inference", type="primary", disabled=(payload is None)):
        try:
            res = inf_svc.run_pressure_inference(
                payload,
                checkpoint_path=ckpt_map[selected_ckpt],
                data_origin=data_origin,
            )
            pred = res["prediction"]
            dyn = res["stance_dynamics"]
            unc = res["uncertainty"]

            if res["is_demo"]:
                st.warning("DEMO MODE ACTIVE: Output includes synthesized waveform data. Not a real sensor measurement.")

            cnn_str = f"{pred['predicted_walking_speed_kmh']:.2f} km/h" if pred["predicted_walking_speed_kmh"] is not None else "Omitted (Scalar Mode)"
            ridge_str = f"{pred['baseline_ridge_speed_kmh']:.2f} km/h" if pred["baseline_ridge_speed_kmh"] is not None else "N/A"
            render_kpi_row([
                ("1D CNN Speed", cnn_str, f"Actual: {float(actual_speed):.2f}" if actual_speed and float(actual_speed) > 0.5 else None),
                ("Ridge Baseline Speed", ridge_str, None),
                ("Primary Model", pred["primary_model_used"], None),
                ("95% CI (km/h)", f"[{unc['confidence_interval_95'][0]}, {unc['confidence_interval_95'][1]}]", f"MAE {unc['estimated_mae_kmh']}"),
            ])

            st.write(
                f"**Speed Bracket:** {pred['speed_bracket']} | **Peak Force:** {dyn['peak_force_n']} N | "
                f"**Stance Duration:** {dyn['stance_duration_ms']} ms | "
                f"**Impulse:** {dyn['impulse_n_s'] if dyn['impulse_n_s'] is not None else 'N/A (no curve)'} N·s"
            )
            st.caption(f"**Input SHA-256:** `{res['input_hash']}` | **Latency:** `{res['latency_ms']} ms`")

            for w in res["quality_flags"]["warnings"]:
                st.warning(w)

            if plot_curve is not None:
                render_force_curve(plot_curve)

            st.info(res["disclaimer"])
        except Exception as exc:
            st.error(f"Pressure inference validation failed: {exc}")


def render() -> None:
    cfg = get_config_service()
    st.title("Local Model Prediction")
    render_research_banner(cfg.get_disclaimer())

    modality = st.radio(
        "Select Prediction Pipeline",
        ["Thermal Cohort Screening (Thermal 2D CNN)", "Pressure & Gait Regression (Pressure 1D CNN / Ridge)"],
        horizontal=True,
    )
    st.divider()
    if modality.startswith("Thermal"):
        _render_thermal_prediction()
    else:
        _render_pressure_prediction()


render()


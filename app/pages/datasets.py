"""
FootGuard AI - Datasets Inspection Page
Allows discovering real and synthetic datasets, inspecting metadata and splits,
validating compatibility, and visualizing thermal/pressure samples safely.
"""

import numpy as np
import pandas as pd
import streamlit as st

from app.state import (
    get_config_service,
    get_dataset_service,
    get_validation_service,
    get_model_service,
    create_ingestion_service,
)
from services.ingestion_service import IngestionSpec
from app.components.metric_cards import render_research_banner, render_provenance_badge
from app.components.thermal_viewer import render_bilateral_thermal_maps, render_uncalibrated_image
from app.components.pressure_viewer import render_force_curve, render_synthetic_session_signals


def render() -> None:
    cfg = get_config_service()
    ds_svc = get_dataset_service()
    val_svc = get_validation_service()
    model_svc = get_model_service()

    st.title("Datasets & Provenance Explorer")
    render_research_banner(cfg.get_disclaimer())

    datasets = ds_svc.list_datasets()
    ds_map = {d.dataset_id: d for d in datasets}

    selected_id = st.selectbox(
        "Select Dataset",
        options=list(ds_map.keys()),
        format_func=lambda k: f"{ds_map[k].name} [{ds_map[k].origin.upper()}]",
    )
    desc = ds_map[selected_id]
    render_provenance_badge(desc.origin, desc.provenance_label)

    # Metadata inspection (memory-safe, no full load)
    try:
        meta = ds_svc.inspect_metadata(selected_id)
    except Exception as exc:
        st.error(f"Could not inspect dataset metadata: {exc}")
        return

    c1, c2, c3 = st.columns(3)
    c1.metric("Modality", desc.modality)
    c2.metric("Samples", desc.num_samples if desc.num_samples is not None else "Unknown")
    c3.metric("File Size (MB)", desc.file_size_mb)

    st.markdown(f"**Description:** {desc.description}")
    st.markdown(f"**Split Methodology:** {desc.split_methodology}")
    st.markdown(f"**Known Limitations:** {desc.limitations}")
    if desc.is_calibrated_celsius is False:
        st.error("Calibration Status: **UNCALIBRATED** (8-bit pseudo-color PNGs; Celsius calculations blocked).")
    elif desc.is_calibrated_celsius is True:
        st.success("Calibration Status: **Calibrated Celsius (°C)**")

    with st.expander("Split Summary & Schema Columns", expanded=False):
        st.json(meta.get("split_summary", {}))
        if meta.get("columns"):
            st.write("Columns:", meta["columns"])

    # Model Compatibility Checker
    st.subheader("Model Compatibility Check")
    model_ids = [m.model_id for m in model_svc.list_models()]
    check_model_id = st.selectbox("Check compatibility with model", options=model_ids)
    compat = val_svc.check_model_compatibility(selected_id, check_model_id)
    if compat.is_compatible:
        st.success(f"Compatible: {compat.reason}")
        for w in compat.warnings:
            st.warning(w)
    else:
        st.error(f"Incompatible / Blocked: {compat.reason}")

    st.divider()
    st.subheader("Data Preview & Validation")

    if selected_id == "synthetic_paired_readings":
        st.info("Chunked loader active: only the selected session or sampled subset is read from the 258.9 MB CSV.")
        mode = st.radio("Load Mode", ["Single Synthetic Session", "Strided Sample Subset"], horizontal=True)
        try:
            if mode == "Single Synthetic Session":
                session_opts = meta.get("sample_session_ids", ["SYN-0001"])
                sid = st.selectbox("Select Session ID", options=session_opts)
                df_preview = ds_svc.load_synthetic_session(session_id=sid, max_rows=500)
                st.dataframe(df_preview.head(50), use_container_width=True)
                report = val_svc.validate_dataframe(selected_id, df_preview)
                render_synthetic_session_signals(df_preview, sid)
            else:
                n_rows = st.slider("Sample Size (rows)", min_value=50, max_value=1000, value=200, step=50)
                df_preview = ds_svc.sample_synthetic_rows(n_rows=n_rows, seed=42)
                st.dataframe(df_preview.head(50), use_container_width=True)
                report = val_svc.validate_dataframe(selected_id, df_preview)
        except Exception as exc:
            st.error(f"Failed to load synthetic preview: {exc}")
            return
    else:
        try:
            df_preview = ds_svc.load_dataset_preview(selected_id, limit=100)
            st.dataframe(df_preview.head(30), use_container_width=True)
            report = val_svc.validate_dataframe(selected_id, df_preview)
        except Exception as exc:
            st.error(f"Failed to load dataset preview: {exc}")
            return

    with st.expander("Validation Report", expanded=True):
        st.write(f"**Valid:** {report.is_valid} | **Preview Shape:** {report.shape} | **Missing Values:** {report.missing_values_count}")
        if report.label_summary:
            st.write("**Label Distribution Summary:**")
            st.json(report.label_summary)
        for w in report.warnings:
            st.warning(w)
        for err in report.errors:
            st.error(err)

    # Sample Signal / Image Visualization for Real Datasets
    if desc.origin == "real" and desc.num_samples:
        st.divider()
        st.subheader("Sample Visualization")
        max_idx = max(0, int(desc.num_samples) - 1)
        sample_idx = st.number_input("Sample Index", min_value=0, max_value=max_idx, value=0, step=1)
        try:
            sample = ds_svc.get_real_sample(selected_id, index=int(sample_idx))
            st.json(sample["metadata"])
            arrays = sample.get("arrays", {})
            if selected_id == "thermodatabase" and "tensor" in arrays:
                tensor = arrays["tensor"]  # (3, 160, 80)
                render_bilateral_thermal_maps(tensor[0], tensor[1], tensor[2])
            elif selected_id == "iac_tecmed" and "sequence" in arrays:
                seq = arrays["sequence"]  # (4, 1, H, W)
                st.write(f"4-Timepoint Sequence Shape: `{seq.shape}` (T0, T5, T10, T15)")
                render_bilateral_thermal_maps(seq[0, 0], seq[-1, 0], np.abs(seq[-1, 0] - seq[0, 0]), title="IAC-TecMed T0 vs T15 (°C)")
            elif selected_id == "standup" and "image_rgb" in arrays:
                render_uncalibrated_image(arrays["image_rgb"], caption=f"STANDUP {sample['metadata'].get('image_id', '')}")
            elif selected_id == "cad_walk" and "force_curve" in arrays:
                render_force_curve(arrays["force_curve"], title="CAD_WALK Normalized Stance Force Curve (N)")
            elif selected_id == "gait_lab" and "window" in arrays:
                win = arrays["window"]
                st.write(f"Gait Lab Window Shape: `{win.shape}` (18 sensor channels x 200 steps)")
                render_force_curve(win[0], title="Gait Lab Channel 0 Stream", x_label="Sample Index")
        except Exception as exc:
            st.error(f"Could not load sample visualization: {exc}")

    st.divider()
    with st.expander("Controlled CSV Dataset Ingestion", expanded=False):
        st.caption(
            "Upload a CSV dataset to preview, map features/target/session ID, define units and provenance, "
            "and save an isolated copy locally without modifying the original file."
        )
        uploaded_csv = st.file_uploader("Upload CSV Dataset", type=["csv"], key="ingest_csv_uploader")
        if uploaded_csv is not None:
            try:
                up_df = pd.read_csv(uploaded_csv)
                st.dataframe(up_df.head(20), use_container_width=True)
                cols = list(up_df.columns)
                num_cols = list(up_df.select_dtypes(include=[np.number]).columns)

                c_name, c_prov = st.columns(2)
                ds_name = c_name.text_input("Dataset Name", value=uploaded_csv.name.replace(".csv", ""))
                provenance = c_prov.selectbox("Provenance", options=["synthetic", "real"])

                c_sess, c_tgt = st.columns(2)
                sess_col = c_sess.selectbox("Session / Participant ID Column", options=cols)
                tgt_col = c_tgt.selectbox("Target Column", options=num_cols or cols)

                feat_cols = st.multiselect(
                    "Input Feature Columns",
                    options=[c for c in num_cols if c != tgt_col],
                    default=[c for c in num_cols if c != tgt_col][:4],
                )
                c_u1, c_u2, c_cal = st.columns(3)
                temp_unit = c_u1.text_input("Thermal Unit", value="celsius")
                press_unit = c_u2.text_input("Pressure Unit", value="newton")
                is_cal = c_cal.checkbox("Calibrated Celsius", value=(provenance == "synthetic"))

                if st.button("Validate & Save Local Copy", key="btn_ingest_save"):
                    ing_svc = create_ingestion_service()
                    spec = IngestionSpec(
                        dataset_name=ds_name,
                        feature_columns=feat_cols,
                        target_column=tgt_col,
                        session_id_column=sess_col,
                        units={"temperature": temp_unit, "pressure": press_unit},
                        provenance=provenance,
                        is_calibrated_celsius=is_cal,
                    )
                    manifest = ing_svc.ingest_dataframe(up_df, spec)
                    st.success(f"Ingested `{manifest['dataset_id']}` ({manifest['num_rows']} rows, {manifest['num_sessions']} sessions) -> `{manifest['csv_path']}`")
                    for w in manifest.get("warnings", []):
                        st.warning(w)
            except Exception as exc:
                st.error(f"CSV Ingestion blocked/failed: {exc}")


render()


"""
FootGuard AI - Models & Training Page
Discover models and checkpoints, verify dataset compatibility, configure hyperparameters,
execute training via TrainingService, and save new checkpoints safely.
"""

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from app.state import (
    get_config_service,
    get_dataset_service,
    get_validation_service,
    get_model_service,
    create_training_service,
    create_synthetic_generator_service,
    create_synthetic_training_service,
)
from app.components.metric_cards import render_research_banner, render_provenance_badge
from services.synthetic_generator_service import SyntheticGenerationConfig, SYNTHETIC_PROVENANCE_TAG


def render() -> None:
    cfg = get_config_service()
    ds_svc = get_dataset_service()
    val_svc = get_validation_service()
    model_svc = get_model_service()

    st.title("Models & Controlled Training")
    render_research_banner(cfg.get_disclaimer())

    st.subheader("Model & Checkpoint Registry")
    models = model_svc.list_models()
    model_map = {m.model_id: m for m in models}
    checkpoints = model_svc.list_checkpoints()

    with st.expander(f"Discovered Checkpoints ({len(checkpoints)})", expanded=False):
        if checkpoints:
            ckpt_rows = [
                {
                    "Checkpoint": c.checkpoint_id,
                    "Model": c.model_id,
                    "Epoch": str(c.epoch) if c.epoch is not None else "-",
                    "Size (KB)": c.file_size_kb,
                    "Version": c.model_version,
                }
                for c in checkpoints
            ]
            st.dataframe(pd.DataFrame(ckpt_rows), use_container_width=True, hide_index=True)
        else:
            st.write("No checkpoints found.")

    st.divider()
    tab_real, tab_syn = st.tabs(["Real Dataset Training", "Synthetic Data Experiments"])
    with tab_real:
        _render_real_training(cfg, ds_svc, val_svc, model_map)
    with tab_syn:
        _render_synthetic_experiments()


def _render_real_training(cfg, ds_svc, val_svc, model_map) -> None:
    st.subheader("Configure Training Run")

    datasets = ds_svc.list_datasets()
    ds_map = {d.dataset_id: d for d in datasets}

    col1, col2 = st.columns(2)
    with col1:
        selected_model_id = st.selectbox(
            "Select Model Architecture",
            options=list(model_map.keys()),
            format_func=lambda k: f"{model_map[k].name} ({model_map[k].registry_status})",
        )
    with col2:
        selected_dataset_id = st.selectbox(
            "Select Dataset",
            options=list(ds_map.keys()),
            format_func=lambda k: f"{ds_map[k].name} [{ds_map[k].origin.upper()}]",
        )

    spec = model_map[selected_model_id]
    st.caption(
        f"**Task:** {spec.task_type} | **Input:** `{spec.input_contract}` | "
        f"**Output:** `{spec.output_contract}` | **Limitations:** {spec.limitations}"
    )

    compat = val_svc.check_model_compatibility(selected_dataset_id, selected_model_id)
    if compat.is_compatible:
        st.success(f"Approved for Training: {compat.reason}")
        for w in compat.warnings:
            st.warning(w)
    else:
        st.error(f"Training Blocked: {compat.reason}")

    defaults = cfg.get_training_defaults()
    with st.form("training_config_form"):
        c1, c2, c3 = st.columns(3)
        if selected_model_id == "pressure_ridge_baseline":
            alpha = c1.number_input("Ridge Alpha (L2)", min_value=0.01, max_value=100.0, value=1.0, step=0.1)
            seed = c2.number_input("Random Seed", min_value=0, max_value=99999, value=int(defaults.get("seed", 42)), step=1)
            epochs = 1
            batch_size = 16
            lr = 1e-3
            patience = 5
        else:
            epochs = c1.number_input("Epochs", min_value=1, max_value=100, value=int(defaults.get("epochs", 10)), step=1)
            batch_size = c2.selectbox("Batch Size", options=[4, 8, 16, 32], index=2)
            lr = c3.number_input("Learning Rate", min_value=1e-5, max_value=1e-1, value=float(defaults.get("learning_rate", 0.001)), format="%.5f")
            c4, c5 = st.columns(2)
            patience = c4.number_input("Early Stopping Patience", min_value=1, max_value=30, value=int(defaults.get("patience", 5)), step=1)
            seed = c5.number_input("Random Seed", min_value=0, max_value=99999, value=int(defaults.get("seed", 42)), step=1)
            alpha = 1.0

        custom_exp_id = st.text_input(
            "Custom Experiment ID (optional; leave blank for timestamped ID)",
            value="",
            help="Protected Phase 4/5 checkpoints cannot be overwritten.",
        )

        submitted = st.form_submit_button(
            "Start Training",
            disabled=not compat.is_compatible,
            type="primary",
        )

    if submitted and compat.is_compatible:
        ts = create_training_service()
        progress_bar = st.progress(0.0, text="Initializing training...")
        status_text = st.empty()

        def on_progress(ep: int, total: int, metrics: dict) -> None:
            progress_bar.progress(ep / max(1, total), text=f"Epoch {ep}/{total}")
            status_text.write(f"Epoch {ep}/{total} — train_loss: `{metrics['train_loss']:.4f}` | val_loss: `{metrics['val_loss']:.4f}`")

        hp = {
            "epochs": int(epochs),
            "batch_size": int(batch_size),
            "learning_rate": float(lr),
            "patience": int(patience),
            "seed": int(seed),
            "alpha": float(alpha),
            "device": "cpu",
        }
        exp_id_arg = custom_exp_id.strip() or None

        try:
            with st.spinner("Running controlled training..."):
                result = ts.start_training(
                    model_id=selected_model_id,
                    dataset_id=selected_dataset_id,
                    hyperparameters=hp,
                    experiment_id=exp_id_arg,
                    progress_callback=on_progress,
                )
            progress_bar.progress(1.0, text="Training complete")
            st.success(f"Training completed! Experiment `{result['experiment_id']}` saved.")
            st.write(f"**Checkpoint:** `{result['checkpoint_location']}`")
            st.subheader("Evaluation Metrics (Test Split)")
            st.json(result.get("evaluation_metrics", {}))

            hist = result.get("history", {})
            if hist.get("train_loss"):
                fig = go.Figure()
                epochs_x = list(range(1, len(hist["train_loss"]) + 1))
                fig.add_trace(go.Scatter(x=epochs_x, y=hist["train_loss"], mode="lines+markers", name="Train Loss"))
                if hist.get("val_loss"):
                    fig.add_trace(go.Scatter(x=epochs_x, y=hist["val_loss"], mode="lines+markers", name="Val Loss"))
                fig.update_layout(title="Training & Validation Loss", xaxis_title="Epoch", yaxis_title="Loss", height=320)
                st.plotly_chart(fig, use_container_width=True)
        except Exception as exc:
            st.error(f"Training failed safely: {exc}")


def _render_synthetic_experiments() -> None:
    st.subheader("Synthetic Data Experiments")
    render_provenance_badge("synthetic", SYNTHETIC_PROVENANCE_TAG)
    st.caption(
        "Controlled synthetic experimentation only. Labels are derived from hidden simulator latent states "
        "(never algebraic pressure sums). Checkpoints are isolated in `checkpoints/synthetic_experiments/`."
    )

    gen_svc = create_synthetic_generator_service()
    syn_train_svc = create_synthetic_training_service()

    with st.expander("1. Generate Synthetic Multimodal Dataset", expanded=True):
        with st.form("syn_gen_form"):
            c1, c2, c3 = st.columns(3)
            ds_name = c1.text_input("Synthetic Dataset Name", value="sim_multimodal_v1")
            num_sessions = c2.number_input("Simulated Sessions", min_value=6, max_value=100, value=12, step=2)
            steps_per_sess = c3.number_input("Steps per Session", min_value=20, max_value=500, value=60, step=10)

            c4, c5, c6 = st.columns(3)
            sampling_hz = c4.number_input("Sampling Frequency (Hz)", min_value=1.0, max_value=100.0, value=10.0, step=1.0)
            noise_std = c5.number_input("Sensor Noise Std", min_value=0.0, max_value=1.0, value=0.05, step=0.01)
            variation = c6.number_input("Measurement Variation", min_value=0.0, max_value=1.0, value=0.15, step=0.05)

            c7, c8, c9 = st.columns(3)
            thermal_base = c7.number_input("Thermal Base (°C)", min_value=24.0, max_value=36.0, value=30.8, step=0.2)
            load_scale = c8.number_input("Pressure Load Scale (N)", min_value=100.0, max_value=900.0, value=400.0, step=25.0)
            gen_seed = c9.number_input("Generator Seed", min_value=0, max_value=99999, value=42, step=1)

            if st.form_submit_button("Generate Synthetic Dataset", type="primary"):
                try:
                    cfg_gen = SyntheticGenerationConfig(
                        dataset_name=ds_name,
                        num_sessions=int(num_sessions),
                        steps_per_session=int(steps_per_sess),
                        sampling_hz=float(sampling_hz),
                        thermal_base_c=float(thermal_base),
                        pressure_load_scale_n=float(load_scale),
                        sensor_noise_std=float(noise_std),
                        measurement_variation=float(variation),
                        seed=int(gen_seed),
                    )
                    manifest = gen_svc.generate(cfg_gen)
                    st.success(f"Generated `{manifest['dataset_id']}` ({manifest['num_rows']} rows across {manifest['num_sessions']} sessions) [{manifest['data_status']}]")
                except Exception as exc:
                    st.error(f"Synthetic generation failed: {exc}")

    generated = gen_svc.list_generated_datasets()
    if not generated:
        st.info("No generated synthetic datasets found yet. Generate one above to inspect and train.")
        return

    ds_ids = [g["dataset_id"] for g in generated]
    selected_syn_ds = st.selectbox("2. Select Generated Synthetic Dataset", options=ds_ids)

    try:
        syn_df, syn_manifest, _ = gen_svc.load_generated_dataset(selected_syn_ds)
        with st.expander("Inspect Generated Samples, Labels & Features", expanded=False):
            st.write(f"**Provenance:** `{syn_manifest.get('data_status')}` | **Rows:** {len(syn_df)} | **Sessions:** {syn_df['session_id'].nunique()}")
            st.dataframe(syn_df.head(20), use_container_width=True)
            st.write("**Latent Target Semantics:**")
            st.json(syn_manifest.get("target_semantics", {}))
            st.write("**Available Feature Groups:**")
            st.json(syn_manifest.get("feature_groups", {}))
    except Exception as exc:
        st.error(f"Could not load generated synthetic dataset: {exc}")
        return

    st.subheader("3. Configure & Train Synthetic Experiment")
    c_m, c_t = st.columns(2)
    syn_model_id = c_m.selectbox(
        "Compatible Model",
        options=list(syn_train_svc.SUPPORTED_MODELS.keys()),
        key="syn_model_select",
    )

    if syn_model_id == "thermal_2d_cnn":
        target_opts = ["latent_thermal_regime"]
        feat_opts = ["thermal_tensors(3,160,80)"]
        default_feats = feat_opts
    elif syn_model_id in ("pressure_1d_cnn", "pressure_tcn", "pressure_ridge_baseline"):
        target_opts = ["latent_gait_speed_kmh"]
        feat_opts = gen_svc.PRESSURE_FEATURE_COLS
        default_feats = feat_opts
    else:
        target_opts = ["latent_gait_speed_kmh", "latent_thermal_regime"]
        feat_opts = gen_svc.THERMAL_FEATURE_COLS + gen_svc.PRESSURE_FEATURE_COLS + gen_svc.IMU_FEATURE_COLS
        default_feats = feat_opts

    syn_target = c_t.selectbox("Latent Target", options=target_opts, key="syn_target_select")
    selected_feats = st.multiselect("Feature Mapping", options=feat_opts, default=default_feats, key="syn_feat_select")

    with st.form("syn_train_form"):
        sc1, sc2, sc3, sc4 = st.columns(4)
        syn_epochs = sc1.number_input("Epochs", min_value=1, max_value=50, value=5, step=1, key="syn_epochs")
        syn_bs = sc2.selectbox("Batch Size", options=[4, 8, 16, 32], index=1, key="syn_bs")
        syn_lr = sc3.number_input("Learning Rate", min_value=1e-5, max_value=1e-1, value=0.001, format="%.5f", key="syn_lr")
        syn_seed = sc4.number_input("Training Seed", min_value=0, max_value=99999, value=42, step=1, key="syn_seed")

        if st.form_submit_button("Start Synthetic Training", type="primary"):
            try:
                feat_arg = None if syn_model_id == "thermal_2d_cnn" else selected_feats
                res = syn_train_svc.start_synthetic_training(
                    dataset_id=selected_syn_ds,
                    model_id=syn_model_id,
                    target_col=syn_target,
                    feature_cols=feat_arg,
                    hyperparameters={
                        "epochs": int(syn_epochs),
                        "batch_size": int(syn_bs),
                        "learning_rate": float(syn_lr),
                        "seed": int(syn_seed),
                    },
                )
                st.success(f"Synthetic experiment `{res['experiment_id']}` completed! [{SYNTHETIC_PROVENANCE_TAG}]")
                st.write(f"**Isolated Checkpoint:** `{res['checkpoint_location']}`")
                st.json(res.get("evaluation_metrics", {}))
                hist = res.get("history", {})
                if hist.get("train_loss"):
                    fig = go.Figure()
                    ep_x = list(range(1, len(hist["train_loss"]) + 1))
                    fig.add_trace(go.Scatter(x=ep_x, y=hist["train_loss"], mode="lines+markers", name="Train Loss"))
                    if hist.get("val_loss"):
                        fig.add_trace(go.Scatter(x=ep_x, y=hist["val_loss"], mode="lines+markers", name="Val Loss"))
                    fig.update_layout(title="Synthetic Training & Validation Loss", xaxis_title="Epoch", yaxis_title="Loss", height=300)
                    st.plotly_chart(fig, use_container_width=True)
            except Exception as exc:
                st.error(f"Synthetic training blocked/failed: {exc}")


render()


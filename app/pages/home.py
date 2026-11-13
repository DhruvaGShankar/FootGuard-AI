"""
FootGuard AI - Home Page
Overview of available datasets, models, checkpoints, recent experiments, and quick navigation.
"""

import pandas as pd
import streamlit as st

from app.state import (
    get_config_service,
    get_dataset_service,
    get_model_service,
    get_experiment_service,
)
from app.components.metric_cards import render_research_banner, render_kpi_row


def render() -> None:
    cfg = get_config_service()
    ds_svc = get_dataset_service()
    model_svc = get_model_service()
    exp_svc = get_experiment_service()

    st.title("FootGuard AI — Local ML Research Studio")
    render_research_banner(cfg.get_disclaimer())

    datasets = ds_svc.list_datasets()
    models = model_svc.list_models()
    checkpoints = model_svc.list_checkpoints()
    experiments = exp_svc.list_experiments(include_legacy=True)

    render_kpi_row([
        ("Available Datasets", len(datasets), f"{sum(d.origin == 'real' for d in datasets)} real / {sum(d.origin == 'synthetic' for d in datasets)} synthetic"),
        ("Model Architectures", len(models), None),
        ("Saved Checkpoints", len(checkpoints), None),
        ("Recorded Experiments", len(experiments), None),
    ])

    st.subheader("Quick Actions")
    c1, c2, c3, c4 = st.columns(4)
    with c1:
        if st.button("Inspect Datasets", use_container_width=True):
            st.switch_page("pages/datasets.py")
    with c2:
        if st.button("Run Prediction", use_container_width=True):
            st.switch_page("pages/prediction.py")
    with c3:
        if st.button("Models & Training", use_container_width=True):
            st.switch_page("pages/training.py")
    with c4:
        if st.button("Experiments & Results", use_container_width=True):
            st.switch_page("pages/experiments.py")

    st.divider()
    col_left, col_right = st.columns(2)

    with col_left:
        st.subheader("Datasets Summary")
        ds_rows = [
            {
                "Dataset": d.dataset_id,
                "Origin": d.origin.upper(),
                "Modality": d.modality,
                "Samples": d.num_samples if d.num_samples is not None else "N/A",
                "Calibrated °C": "Yes" if d.is_calibrated_celsius is True else ("No" if d.is_calibrated_celsius is False else "N/A"),
            }
            for d in datasets
        ]
        st.dataframe(pd.DataFrame(ds_rows), use_container_width=True, hide_index=True)

    with col_right:
        st.subheader("Models & Checkpoints")
        m_rows = [
            {
                "Model": m.model_id,
                "Task": m.task_type,
                "Status": m.registry_status,
                "Default Checkpoint": m.default_checkpoint,
            }
            for m in models
        ]
        st.dataframe(pd.DataFrame(m_rows), use_container_width=True, hide_index=True)

    st.subheader("Recent Experiments")
    if experiments:
        exp_rows = [
            {
                "Experiment ID": e.get("experiment_id"),
                "Model": e.get("model_name"),
                "Dataset": e.get("dataset_name"),
                "Origin": str(e.get("dataset_origin", "")).upper(),
                "Timestamp": e.get("timestamp", ""),
            }
            for e in experiments[:8]
        ]
        st.dataframe(pd.DataFrame(exp_rows), use_container_width=True, hide_index=True)
    else:
        st.write("No experiments recorded yet.")


render()


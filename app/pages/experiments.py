"""
FootGuard AI - Experiments & Results Page
Displays experiment history, classification/regression metrics, confusion matrices,
training curves, model comparisons, and JSON/Markdown report exports.
"""

import pandas as pd
import plotly.figure_factory as ff
import plotly.graph_objects as go
import streamlit as st

from app.state import (
    get_config_service,
    get_experiment_service,
    get_evaluation_service,
)
from app.components.metric_cards import render_research_banner, render_provenance_badge


def render() -> None:
    cfg = get_config_service()
    exp_svc = get_experiment_service()
    eval_svc = get_evaluation_service()

    st.title("Experiments & Evaluation Results")
    render_research_banner(cfg.get_disclaimer())

    experiments = exp_svc.list_experiments(include_legacy=True)
    if not experiments:
        st.info("No experiment records found yet.")
        return

    origin_filter = st.radio("Filter by Dataset Origin", ["All", "real", "synthetic"], horizontal=True)
    filtered = [
        e for e in experiments
        if origin_filter == "All" or e.get("dataset_origin") == origin_filter
    ]

    if not filtered:
        st.warning(f"No experiments found for origin '{origin_filter}'.")
        return

    st.subheader("Model Comparison Table")
    comp_rows = eval_svc.build_comparison_table(filtered)
    st.dataframe(pd.DataFrame(comp_rows), use_container_width=True, hide_index=True)

    st.divider()
    st.subheader("Inspect Experiment Details")

    exp_map = {e.get("experiment_id", f"unknown_{idx}"): e for idx, e in enumerate(filtered)}
    selected_exp_id = st.selectbox("Select Experiment", options=list(exp_map.keys()))
    exp = exp_map[selected_exp_id]

    render_provenance_badge(
        exp.get("dataset_origin", "real"),
        "SYNTHETIC_NOT_CLINICAL_GROUND_TRUTH" if exp.get("dataset_origin") == "synthetic" else "CLINICAL_RESEARCH_DATASET",
    )

    c1, c2, c3 = st.columns(3)
    c1.metric("Model", exp.get("model_name", "unknown"))
    c2.metric("Dataset", exp.get("dataset_name", "unknown"))
    c3.metric("Timestamp", str(exp.get("timestamp", ""))[:19])

    st.markdown(f"**Split Methodology:** {exp.get('split_methodology', 'N/A')}")
    st.markdown(f"**Checkpoint Location:** `{exp.get('checkpoint_location', 'N/A')}`")

    metrics = exp.get("evaluation_metrics", {})
    col_m, col_cm = st.columns(2)
    with col_m:
        st.write("**Evaluation Metrics**")
        st.json(metrics)
        with st.expander("Hyperparameters & Training Configuration", expanded=False):
            st.json({
                "hyperparameters": exp.get("hyperparameters", {}),
                "training_configuration": exp.get("training_configuration", {}),
            })

    with col_cm:
        cm = metrics.get("confusion_matrix")
        if cm and isinstance(cm, list) and len(cm) == 2:
            st.write("**Confusion Matrix (Test Set)**")
            fig_cm = ff.create_annotated_heatmap(
                z=cm,
                x=["Pred Control", "Pred Diabetic"],
                y=["True Control", "True Diabetic"],
                colorscale="Blues",
                showscale=True,
            )
            fig_cm.update_layout(height=320, margin=dict(l=10, r=10, t=30, b=10))
            st.plotly_chart(fig_cm, use_container_width=True)

    hist = exp.get("history", {})
    if isinstance(hist, dict) and hist.get("train_loss"):
        st.subheader("Training Curves")
        fig_loss = go.Figure()
        epochs_x = list(range(1, len(hist["train_loss"]) + 1))
        fig_loss.add_trace(go.Scatter(x=epochs_x, y=hist["train_loss"], mode="lines+markers", name="Train Loss"))
        if hist.get("val_loss"):
            fig_loss.add_trace(go.Scatter(x=epochs_x, y=hist["val_loss"], mode="lines+markers", name="Val Loss"))
        fig_loss.update_layout(title="Loss vs. Epoch", xaxis_title="Epoch", yaxis_title="Loss", height=320)
        st.plotly_chart(fig_loss, use_container_width=True)

    # Phase 5 Uncertainty Summary if legacy Phase 4 experiment is selected
    p5 = exp_svc.load_phase5_analysis()
    if p5 and exp.get("is_legacy"):
        with st.expander("Phase 5 Participant-Clustered Uncertainty & Error Analysis", expanded=False):
            st.json(p5)

    st.divider()
    st.subheader("Export Experiment Report")
    d1, d2 = st.columns(2)
    try:
        json_str = exp_svc.export_report(selected_exp_id, fmt="json")
        md_str = exp_svc.export_report(selected_exp_id, fmt="md")
        with d1:
            st.download_button(
                "Download JSON Report",
                data=json_str,
                file_name=f"{selected_exp_id}_report.json",
                mime="application/json",
                use_container_width=True,
            )
        with d2:
            st.download_button(
                "Download Markdown Report",
                data=md_str,
                file_name=f"{selected_exp_id}_report.md",
                mime="text/markdown",
                use_container_width=True,
            )
    except Exception as exc:
        st.error(f"Could not export report: {exc}")


render()


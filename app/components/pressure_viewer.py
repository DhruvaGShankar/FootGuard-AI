"""
FootGuard AI - Pressure & Gait Visualization Components
"""

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st


def render_force_curve(
    curve: np.ndarray,
    title: str = "Vertical Ground Reaction Force (vGRF) Stance Curve",
    x_label: str = "Normalized Stance (%)",
) -> None:
    arr = np.asarray(curve, dtype=float).ravel()
    x = np.linspace(0, 100, len(arr))
    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=x,
            y=arr,
            mode="lines",
            fill="tozeroy",
            name="vGRF (N)",
        )
    )
    fig.update_layout(
        title=title,
        xaxis_title=x_label,
        yaxis_title="Vertical Force (N)",
        height=320,
        margin=dict(l=10, r=10, t=45, b=10),
    )
    st.plotly_chart(fig, use_container_width=True)


def render_synthetic_session_signals(df_session: pd.DataFrame, session_id: str) -> None:
    """Plots 6-site synthetic pressure and thermal zone means for a single synthetic session."""
    p_cols = [
        "heel_pressure_n",
        "mid_pressure_n",
        "mth1_pressure_n",
        "mth3_pressure_n",
        "mth5_pressure_n",
        "toe_hallux_pressure_n",
    ]
    available_p = [c for c in p_cols if c in df_session.columns]
    if available_p:
        fig_p = go.Figure()
        x = np.arange(len(df_session))
        for col in available_p:
            fig_p.add_trace(go.Scatter(x=x, y=df_session[col], mode="lines", name=col))
        fig_p.update_layout(
            title=f"Synthetic 6-Site Plantar Pressure — {session_id} (SYNTHETIC_NOT_CLINICAL_GROUND_TRUTH)",
            xaxis_title="Sample Index (10s interval)",
            yaxis_title="Force (N)",
            height=320,
            margin=dict(l=10, r=10, t=45, b=10),
        )
        st.plotly_chart(fig_p, use_container_width=True)

    t_cols = [
        "thermal_heel_zone_mean_c",
        "thermal_midfoot_zone_mean_c",
        "thermal_forefoot_zone_mean_c",
    ]
    available_t = [c for c in t_cols if c in df_session.columns]
    if available_t:
        fig_t = go.Figure()
        x = np.arange(len(df_session))
        for col in available_t:
            fig_t.add_trace(go.Scatter(x=x, y=df_session[col], mode="lines", name=col))
        fig_t.update_layout(
            title=f"Synthetic Thermal Zone Means — {session_id} (SYNTHETIC_NOT_CLINICAL_GROUND_TRUTH)",
            xaxis_title="Sample Index (10s interval)",
            yaxis_title="Temperature (°C)",
            height=300,
            margin=dict(l=10, r=10, t=45, b=10),
        )
        st.plotly_chart(fig_t, use_container_width=True)


"""
FootGuard AI - Thermal Visualization Components
"""

from typing import Optional
import numpy as np
import plotly.express as px
from plotly.subplots import make_subplots
import plotly.graph_objects as go
import streamlit as st


def render_bilateral_thermal_maps(
    left_c: np.ndarray,
    right_flipped_c: np.ndarray,
    delta_t_c: Optional[np.ndarray] = None,
    title: str = "Bilateral Plantar Thermograms (°C)",
) -> None:
    """Renders Left, anatomically aligned Right, and optional Delta_T heatmaps."""
    n_cols = 3 if delta_t_c is not None else 2
    subtitles = ["Left Foot (°C)", "Right Foot Aligned (°C)"]
    if delta_t_c is not None:
        subtitles.append("|L - R| Asymmetry (°C)")

    fig = make_subplots(rows=1, cols=n_cols, subplot_titles=subtitles, horizontal_spacing=0.06)

    fig.add_trace(
        go.Heatmap(z=left_c, colorscale="Inferno", zmin=15.0, zmax=36.0, showscale=False),
        row=1,
        col=1,
    )
    fig.add_trace(
        go.Heatmap(
            z=right_flipped_c,
            colorscale="Inferno",
            zmin=15.0,
            zmax=36.0,
            colorbar=dict(title="°C", x=0.64 if n_cols == 3 else 1.02),
        ),
        row=1,
        col=2,
    )
    if delta_t_c is not None:
        fig.add_trace(
            go.Heatmap(
                z=delta_t_c,
                colorscale="Reds",
                zmin=0.0,
                zmax=4.0,
                colorbar=dict(title="ΔT °C", x=1.02),
            ),
            row=1,
            col=3,
        )

    for c in range(1, n_cols + 1):
        fig.update_yaxes(autorange="reversed", showticklabels=False, row=1, col=c)
        fig.update_xaxes(showticklabels=False, row=1, col=c)

    fig.update_layout(title=title, height=380, margin=dict(l=10, r=10, t=50, b=10))
    st.plotly_chart(fig, use_container_width=True)


def render_uncalibrated_image(rgb_arr: np.ndarray, caption: str = "Uncalibrated PNG (Not Physical Celsius)") -> None:
    st.warning("Uncalibrated 8-bit image: pixel colors do NOT represent physical Celsius temperatures.")
    st.image(rgb_arr, caption=caption, use_container_width=False, width=260)


"""
FootGuard AI - Metric & Provenance UI Components
"""

from typing import List, Tuple, Any, Optional
import streamlit as st


def render_research_banner(disclaimer: str) -> None:
    st.info(disclaimer)


def render_provenance_badge(origin: str, provenance_label: str) -> None:
    if origin == "synthetic":
        st.warning(f"Provenance: **{provenance_label}** (`synthetic` — not clinical ground truth)")
    else:
        st.success(f"Provenance: **{provenance_label}** (`real` research dataset)")


def render_kpi_row(items: List[Tuple[str, Any, Optional[str]]]) -> None:
    cols = st.columns(len(items))
    for col, (label, value, delta) in zip(cols, items):
        with col:
            st.metric(label=label, value=value, delta=delta)


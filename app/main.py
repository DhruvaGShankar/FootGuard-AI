"""
FootGuard AI — Local-First Python ML Research Studio
Launch command:
    cd ML
    streamlit run app/main.py
"""

import sys
from pathlib import Path
import streamlit as st

ML_ROOT = Path(__file__).resolve().parent.parent
if str(ML_ROOT) not in sys.path:
    sys.path.insert(0, str(ML_ROOT))

st.set_page_config(
    page_title="FootGuard AI — Local ML Research Studio",
    layout="wide",
    initial_sidebar_state="expanded",
)

pages = [
    st.Page("pages/home.py", title="Home", icon=":material/home:", default=True),
    st.Page("pages/datasets.py", title="Datasets", icon=":material/dataset:"),
    st.Page("pages/training.py", title="Models & Training", icon=":material/model_training:"),
    st.Page("pages/prediction.py", title="Prediction", icon=":material/online_prediction:"),
    st.Page("pages/experiments.py", title="Experiments & Results", icon=":material/analytics:"),
]

with st.sidebar:
    st.caption("FootGuard AI v2.0 (Local-First)")
    st.caption("Research prototype — not for clinical diagnosis or ulcer detection.")

nav = st.navigation(pages)
nav.run()


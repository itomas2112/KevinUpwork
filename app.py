"""
Trading Analysis Platform - Main Application
"""
import streamlit as st

from utils.session_state import initialize_session_state
from ui.sidebar import render_sidebar
from ui.charting_tab import render_charting_tab
from ui.strategy_builder_tab import render_strategy_builder_tab

# -------------------------------------------------
# Configuration
# -------------------------------------------------
st.set_page_config(
    layout="wide",
    initial_sidebar_state="expanded"
)

# -------------------------------------------------
# Initialize
# -------------------------------------------------
initialize_session_state()

# -------------------------------------------------
# Main App
# -------------------------------------------------
st.title("Trading Analysis Platform")

# Create tabs
tab1, tab2 = st.tabs(["📊 Charting", "🔧 Strategy Builder"])

# Render sidebar and get configuration
sidebar_config = render_sidebar()

# Render tabs
with tab1:
    render_charting_tab(sidebar_config)

with tab2:
    render_strategy_builder_tab()
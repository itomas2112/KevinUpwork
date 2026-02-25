"""
Trading Analysis Platform - Main Application
"""
import streamlit as st

from utils.session_state import initialize_session_state
from ui.sidebar import render_sidebar
from ui.charting_tab import render_charting_tab
from ui.strategy_builder_tab import render_strategy_builder_tab
from ui.performance_tab import render_performance_tab

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
# Global table styling: compact, centered, auto-fit
# -------------------------------------------------
st.markdown("""
<style>
/* Auto-fit width instead of stretching to container */
.stTable table {
    width: auto !important;
}
/* Compact cells, left-aligned */
.stTable table th,
.stTable table td {
    text-align: left !important;
    padding: 4px 10px !important;
    font-size: 0.82rem !important;
    white-space: nowrap !important;
}
/* Remove full-width border stretching */
.stTable {
    width: fit-content !important;
}
</style>
""", unsafe_allow_html=True)

# -------------------------------------------------
# Main App
# -------------------------------------------------
st.title("Trading Analysis Platform")

# Create tabs
tab1, tab2, tab3 = st.tabs(["📊 Charting", "🔧 Strategy Builder", "📈 Performance"])

# Render sidebar and get configuration
sidebar_config = render_sidebar()

# Render tabs
with tab1:
    render_charting_tab(sidebar_config)

with tab2:
    render_strategy_builder_tab()

with tab3:
    render_performance_tab(sidebar_config)
"""
Trading Analysis Platform - Main Application
"""
import streamlit as st

from utils.session_state import initialize_session_state
from ui.sidebar import render_sidebar
from ui.charting_tab import render_charting_tab
from ui.strategy_builder_tab import render_strategy_builder_tab
from ui.performance_tab import render_performance_tab
from ui.strategy_testing_tab import render_strategy_testing_tab
from ui.monte_carlo_tab import render_monte_carlo_tab
from ui.grid_search_tab import render_grid_search_tab
from ui.correlation_tab import render_correlation_tab
from ui.wave_analysis_tab import render_wave_analysis_tab

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
col_title, col_tf = st.columns([4, 1])
with col_title:
    st.title("Trading Analysis Platform")
with col_tf:
    st.radio("Base Timeframe", ["15m", "1H"], horizontal=True, key="base_timeframe")

# Clear data when base timeframe changes
base_tf = st.session_state.get("base_timeframe", "15m")
prev_base = st.session_state.get("_prev_base_timeframe", "15m")
if base_tf != prev_base:
    st.session_state["_prev_base_timeframe"] = base_tf
    for k in ["df_raw", "df_ohlc", "df_features", "_indicator_params",
              "_indicator_params_data_fp", "_agg_timeframe"]:
        st.session_state.pop(k, None)
    for k in list(st.session_state.keys()):
        if k.startswith(("_bt_cache_", "_perf_", "_gs_", "_cv_", "_test_", "_corr_", "_wa_")):
            st.session_state.pop(k, None)
    st.rerun()

# Create tabs
tab1, tab2, tab3, tab4, tab5, tab6, tab7, tab8 = st.tabs(["📊 Charting", "🔧 Strategy Builder", "📈 Performance", "🧪 Strategy Testing", "🎲 Monte Carlo", "🔍 Grid Search", "🔗 Correlation", "🌊 Wave Analysis"])

# Render sidebar and get configuration
sidebar_config = render_sidebar()

# Render tabs
with tab1:
    render_charting_tab(sidebar_config)

with tab2:
    render_strategy_builder_tab()

with tab3:
    render_performance_tab(sidebar_config)

with tab4:
    render_strategy_testing_tab(sidebar_config)

with tab5:
    render_monte_carlo_tab()

with tab6:
    render_grid_search_tab(sidebar_config)

with tab7:
    render_correlation_tab(sidebar_config)

with tab8:
    render_wave_analysis_tab(sidebar_config)
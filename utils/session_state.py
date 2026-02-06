"""
Session state initialization and management
"""
import streamlit as st
import json
import os
from config.constants import STRATEGIES_FILE


def initialize_session_state():
    """Initialize all session state variables"""

    # Saved strategies
    if 'saved_strategies' not in st.session_state:
        if os.path.exists(STRATEGIES_FILE):
            with open(STRATEGIES_FILE, 'r') as f:
                st.session_state['saved_strategies'] = json.load(f)
        else:
            st.session_state['saved_strategies'] = []

    # Selected strategies
    if 'selected_strategies' not in st.session_state:
        st.session_state['selected_strategies'] = {}

    # Strategy builder state
    if 'strategy_started' not in st.session_state:
        st.session_state['strategy_started'] = False

    if 'strategy_direction' not in st.session_state:
        st.session_state['strategy_direction'] = None

    if 'entry_conditions_count' not in st.session_state:
        st.session_state['entry_conditions_count'] = 0

    if 'exit_conditions_count' not in st.session_state:
        st.session_state['exit_conditions_count'] = 0

    if 'selected_custom_strategy_idx' not in st.session_state:
        st.session_state['selected_custom_strategy_idx'] = 0

    if 'strategy_name_input' not in st.session_state:
        st.session_state['strategy_name_input'] = ""
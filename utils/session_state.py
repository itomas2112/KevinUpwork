"""
Session state initialization and management
"""
import streamlit as st
import json
import os
from config.constants import STRATEGIES_FILE
from strategies.group_set_manager import load_group_sets
from strategies.strategy_validator import validate_strategy
from indicators.calculate_indicators import migrate_indicator_settings


def initialize_session_state():
    """Initialize all session state variables"""

    # Saved strategies — validate on load, skip invalid ones
    if 'saved_strategies' not in st.session_state:
        if os.path.exists(STRATEGIES_FILE):
            try:
                with open(STRATEGIES_FILE, 'r') as f:
                    raw_strategies = json.load(f)

                valid_strategies = []
                invalid_count = 0
                for strategy in raw_strategies:
                    # Migrate old indicator settings before validation
                    ind = strategy.get('indicator_settings')
                    if isinstance(ind, dict):
                        strategy['indicator_settings'] = migrate_indicator_settings(ind)
                    is_valid, errors = validate_strategy(strategy)
                    if is_valid:
                        valid_strategies.append(strategy)
                    else:
                        invalid_count += 1

                st.session_state['saved_strategies'] = valid_strategies

                if invalid_count > 0:
                    st.session_state['_invalid_strategies_on_load'] = invalid_count

            except (json.JSONDecodeError, ValueError):
                st.session_state['saved_strategies'] = []
        else:
            st.session_state['saved_strategies'] = []

    # Show one-time warning for invalid strategies skipped on load
    invalid_count = st.session_state.pop('_invalid_strategies_on_load', 0)
    if invalid_count > 0:
        st.warning(f"⚠️ {invalid_count} invalid strategy(ies) were skipped during load. "
                   f"They had missing or incorrect fields.")

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

    # Exit groups management
    if 'exit_groups' not in st.session_state:
        st.session_state['exit_groups'] = []

    if 'initial_stop' not in st.session_state:
        st.session_state['initial_stop'] = None

    # DRM data for both patterns (used by Performance tab)
    if 'drm_bullish' not in st.session_state:
        st.session_state['drm_bullish'] = None

    if 'drm_bearish' not in st.session_state:
        st.session_state['drm_bearish'] = None

    # Charting tab multi-pattern selections
    if 'charting_selections' not in st.session_state:
        st.session_state['charting_selections'] = [{
            "mode": "Specified Secondary",
            "pattern_type": "Bullish",
            "primary": None,
            "secondary": None,
        }]

    # Performance tab selections
    if 'perf_selections' not in st.session_state:
        st.session_state['perf_selections'] = [{
            "mode": "All Patterns",
            "pattern_type": "Bullish",
            "primary": None,
            "secondary": None,
        }]

    # Monte Carlo results cache
    if 'mc_results' not in st.session_state:
        st.session_state['mc_results'] = None

    # Grid Search tab
    if 'gs_selections' not in st.session_state:
        st.session_state['gs_selections'] = [{
            "mode": "All Patterns",
            "pattern_type": "Bullish",
            "primary": None,
            "secondary": None,
        }]
    if 'saved_group_sets' not in st.session_state:
        st.session_state['saved_group_sets'] = load_group_sets()

    # Strategy Testing tab selections
    if 'testing_selections' not in st.session_state:
        st.session_state['testing_selections'] = [{
            "mode": "All Patterns",
            "pattern_type": "Bullish",
            "primary": None,
            "secondary": None,
        }]

    # Correlation Analysis tab selections
    if 'corr_selections' not in st.session_state:
        st.session_state['corr_selections'] = [{
            "mode": "All Patterns",
            "pattern_type": "Bullish",
            "primary": None,
            "secondary": None,
        }]
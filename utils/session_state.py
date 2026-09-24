"""
Session state initialization and management
"""
import streamlit as st
import copy
import json
import os
from config.constants import STRATEGIES_FILE
from strategies.group_set_manager import load_group_sets
from strategies.strategy_validator import validate_strategy
from strategies.strategy_manager import migrate_within_last
from indicators.calculate_indicators import migrate_indicator_settings


TESTING_SELECTION_KEYS = ('wfo_selections', 'sopt_selections', 'test_selections')


PATTERN_COLUMN_SELECTION_KEYS = ('perf_selections', 'gs_selections')


def init_pattern_selections(state):
    """Initialise the Performance / Grid Search user-added pattern rows.

    Both tabs always show the fixed All Patterns / All Bullish / All Bearish /
    Global columns, so the default is an empty list (no user rows)."""
    for key in PATTERN_COLUMN_SELECTION_KEYS:
        if key not in state:
            state[key] = []


def init_gs_shortlist(state):
    """Initialise the Grid Search Shortlist.

    The key deliberately has no leading underscore: the sidebar clears every
    `_gs_` key when the aggregation timeframe changes, and the shortlist must
    survive that (it is only cleared when the browser session ends)."""
    if 'gs_shortlist' not in state:
        state['gs_shortlist'] = []


def init_testing_selections(state):
    """Initialise the Strategy Testing tab's per-section selection lists.

    A legacy shared 'testing_selections' list (from a running session) seeds
    all three sections and is then removed."""
    legacy = state.get('testing_selections')
    if legacy is not None:
        for key in TESTING_SELECTION_KEYS:
            if key not in state:
                state[key] = copy.deepcopy(legacy)
        del state['testing_selections']
    for key in TESTING_SELECTION_KEYS:
        if key not in state:
            state[key] = [{
                "mode": "All Patterns",
                "pattern_type": "Bullish",
                "primary": None,
                "secondary": None,
            }]


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
                    migrate_within_last(strategy)
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

    # Performance / Grid Search tab selections (user-added pattern columns;
    # the fixed All Patterns / All Bullish / All Bearish / Global columns
    # are always shown, so the default is no user rows)
    init_pattern_selections(st.session_state)

    # Grid Search Shortlist (no `_gs_` prefix on purpose — see init_gs_shortlist)
    init_gs_shortlist(st.session_state)

    # Instrument + Monte Carlo sizing (sidebar "Instrument" section)
    from config.constants import (DEFAULT_INSTRUMENT, MC_DEFAULT_BALANCE,
                                  MC_DEFAULT_TRADES_PER_SIM)
    if 'instrument' not in st.session_state:
        st.session_state['instrument'] = DEFAULT_INSTRUMENT
    if 'mc_starting_balance' not in st.session_state:
        st.session_state['mc_starting_balance'] = MC_DEFAULT_BALANCE
    if 'mc_trades_per_sim' not in st.session_state:
        st.session_state['mc_trades_per_sim'] = MC_DEFAULT_TRADES_PER_SIM
    if 'mc_n_simulations' not in st.session_state:
        st.session_state['mc_n_simulations'] = 20000
    if 'mc_risk_pct' not in st.session_state:
        st.session_state['mc_risk_pct'] = 1.0

    # Monte Carlo results cache
    if 'mc_results' not in st.session_state:
        st.session_state['mc_results'] = None

    # Grid Search tab
    if 'saved_group_sets' not in st.session_state:
        st.session_state['saved_group_sets'] = load_group_sets()

    # Strategy Testing tab selections (one list per section)
    init_testing_selections(st.session_state)

    # Correlation Analysis tab selections
    if 'corr_selections' not in st.session_state:
        st.session_state['corr_selections'] = [{
            "mode": "All Patterns",
            "pattern_type": "Bullish",
            "primary": None,
            "secondary": None,
        }]
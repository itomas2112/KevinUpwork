"""
Strategy management utilities - save, load, delete
"""
import streamlit as st
import json
import pandas as pd
from config.constants import STRATEGIES_FILE


def save_strategy_to_session(strategy_name):
    """Collect and save strategy data from session state"""

    # Use custom name if provided, otherwise generate default name
    if strategy_name and strategy_name.strip():
        final_strategy_name = strategy_name.strip()
    else:
        final_strategy_name = f"Strategy_{len(st.session_state['saved_strategies']) + 1}"

    # Get selected patterns
    selected_patterns = st.session_state.get('strategy_patterns', [])

    # Collect all strategy data
    strategy_data = {
        "strategy_name": final_strategy_name,
        "direction": st.session_state['strategy_direction'],
        "patterns": selected_patterns,
        "created_at": pd.Timestamp.now().strftime("%Y-%m-%d %H:%M:%S"),
        "entry": {
            "trigger": {
                "group": st.session_state.get('entry_trigger_group1'),
                "element1": st.session_state.get('entry_trigger_element1'),
                "event": st.session_state.get('entry_trigger_event'),
                "compare_type": st.session_state.get('entry_trigger_compare_type', 'Indicator'),
                "element2": st.session_state.get('entry_trigger_element2') if st.session_state.get(
                    'entry_trigger_compare_type', 'Indicator') == "Indicator" else None,
                "value": st.session_state.get('entry_trigger_value') if st.session_state.get(
                    'entry_trigger_compare_type', 'Indicator') == "Fixed Value" else None
            },
            "position_size": st.session_state.get('entry_position_size'),
            "conditions_count": st.session_state['entry_conditions_count'],
            "conditions": []
        },
        "initial_stop": st.session_state.get('initial_stop', {}),
        "exit_groups": []
    }

    # Capture indicator settings from strategy builder's own widgets (sb_ prefix)
    pfx = "sb_"
    strategy_data["indicator_settings"] = {
        "rsi_window": st.session_state.get(f"{pfx}rsi_window", 14),
        "bb_upper_period": st.session_state.get(f"{pfx}bb_upper_period", 20),
        "bb_upper_stdev": st.session_state.get(f"{pfx}bb_upper_stdev", 2.0),
        "bb_mid_period": st.session_state.get(f"{pfx}bb_mid_period", 20),
        "bb_lower_period": st.session_state.get(f"{pfx}bb_lower_period", 20),
        "bb_lower_stdev": st.session_state.get(f"{pfx}bb_lower_stdev", 2.0),
        "kc_upper_ema": st.session_state.get(f"{pfx}kc_upper_ema", 20),
        "kc_upper_mult": st.session_state.get(f"{pfx}kc_upper_mult", 2.0),
        "kc_mid_ema": st.session_state.get(f"{pfx}kc_mid_ema", 20),
        "kc_lower_ema": st.session_state.get(f"{pfx}kc_lower_ema", 20),
        "kc_lower_mult": st.session_state.get(f"{pfx}kc_lower_mult", 2.0),
        "kc_atr_period": st.session_state.get(f"{pfx}kc_atr_period", 10),
        "stoch_k_period": st.session_state.get(f"{pfx}stoch_k_period", 14),
        "stoch_k_smooth": st.session_state.get(f"{pfx}stoch_k_smooth", 3),
        "stoch_d_smooth": st.session_state.get(f"{pfx}stoch_d_smooth", 3),
        "adx_period": st.session_state.get(f"{pfx}adx_period", 14),
        "atr_period": st.session_state.get(f"{pfx}atr_period", 14),
        "macd_fast": st.session_state.get(f"{pfx}macd_fast", 12),
        "macd_slow": st.session_state.get(f"{pfx}macd_slow", 26),
        "macd_signal": st.session_state.get(f"{pfx}macd_signal", 9),
        "supertrend_period": st.session_state.get(f"{pfx}supertrend_period", 10),
        "supertrend_multiplier": st.session_state.get(f"{pfx}supertrend_multiplier", 3.0),
        "ema_periods": list(st.session_state.get(f"{pfx}ema_periods", [])),
        "dc_upper_period": st.session_state.get(f"{pfx}dc_upper_period", 20),
        "dc_mid_period": st.session_state.get(f"{pfx}dc_mid_period", 20),
        "dc_lower_period": st.session_state.get(f"{pfx}dc_lower_period", 20),
        "dc_offset": st.session_state.get(f"{pfx}dc_offset", 0),
        "psar_af_start": st.session_state.get(f"{pfx}psar_af_start", 0.02),
        "psar_af_increment": st.session_state.get(f"{pfx}psar_af_increment", 0.02),
        "psar_af_max": st.session_state.get(f"{pfx}psar_af_max", 0.20),
    }

    # Collect entry conditions
    for i in range(st.session_state['entry_conditions_count']):
        compare_type = st.session_state.get(f'entry_cond_{i}_compare_type', 'Indicator')

        condition = {
            "group": st.session_state.get(f'entry_cond_{i}_group1'),
            "element1": st.session_state.get(f'entry_cond_{i}_element1'),
            "operator": st.session_state.get(f'entry_cond_{i}_operator'),
            "compare_type": compare_type,
            "element2": st.session_state.get(f'entry_cond_{i}_element2') if compare_type == "Indicator" else None,
            "value": st.session_state.get(f'entry_cond_{i}_value') if compare_type == "Fixed Value" else None
        }
        strategy_data["entry"]["conditions"].append(condition)

    # Collect exit groups
    for group_idx, exit_group in enumerate(st.session_state.get('exit_groups', [])):
        group_data = {
            "group_id": exit_group.get('group_id', group_idx + 1),
            "allocation_pct": st.session_state.get(f'exit_group_{group_idx}_alloc', exit_group.get('allocation_pct', 100.0)),
            "targets": [],
            "stops": []
        }

        # Collect targets for this group
        for target_idx in range(len(exit_group.get('targets', []))):
            target = collect_exit_config(group_idx, 'Target', target_idx)
            if target:
                group_data["targets"].append(target)

        # Collect stops for this group
        for stop_idx in range(len(exit_group.get('stops', []))):
            stop = collect_exit_config(group_idx, 'Stop', stop_idx)
            if stop:
                group_data["stops"].append(stop)

        strategy_data["exit_groups"].append(group_data)

    # Save to session state
    st.session_state['saved_strategies'].append(strategy_data)

    # Save to file for persistence
    save_strategies_to_file()

    return len(st.session_state['saved_strategies'])


def collect_exit_config(group_idx, exit_type, exit_idx):
    """
    Collect a single exit (Target or Stop) configuration from session state.

    Reads the Streamlit widget keys created by render_exit_config() and
    render_exit_condition() in strategy_builder_tab.py.

    Parameters:
        group_idx : int   – index of the exit group
        exit_type : str   – 'Target' or 'Stop'
        exit_idx  : int   – index within the targets / stops list

    Returns:
        dict with 'type', 'trigger', and 'conditions' keys, or None if
        the trigger element is missing.
    """
    prefix = f"{exit_type}_{group_idx}_{exit_idx}"

    # ---- Trigger ----
    compare_type = st.session_state.get(f'{prefix}_trigger_compare_type', 'Indicator')

    trigger = {
        'group': st.session_state.get(f'{prefix}_trigger_group1'),
        'element1': st.session_state.get(f'{prefix}_trigger_element1'),
        'event': st.session_state.get(f'{prefix}_trigger_event'),
        'compare_type': compare_type,
        'element2': (
            st.session_state.get(f'{prefix}_trigger_element2')
            if compare_type == 'Indicator' else None
        ),
        'value': (
            st.session_state.get(f'{prefix}_trigger_value')
            if compare_type == 'Fixed Value' else None
        ),
    }

    # If element1 was never set the widget wasn't rendered – skip
    if trigger['element1'] is None:
        return None

    # ---- Conditions ----
    conditions_count = st.session_state.get(f'{prefix}_conditions_count', 0)
    conditions = []

    for cond_idx in range(conditions_count):
        cond_prefix = f"{prefix}_cond_{cond_idx}"
        cond_compare_type = st.session_state.get(f'{cond_prefix}_compare_type', 'Indicator')

        condition = {
            'group': st.session_state.get(f'{cond_prefix}_group'),
            'element1': st.session_state.get(f'{cond_prefix}_element1'),
            'operator': st.session_state.get(f'{cond_prefix}_operator'),
            'compare_type': cond_compare_type,
            'element2': (
                st.session_state.get(f'{cond_prefix}_element2')
                if cond_compare_type == 'Indicator' else None
            ),
            'value': (
                st.session_state.get(f'{cond_prefix}_value')
                if cond_compare_type == 'Fixed Value' else None
            ),
        }
        conditions.append(condition)

    return {
        'type': exit_type,
        'trigger': trigger,
        'conditions': conditions,
    }


def save_strategies_to_file():
    """Save all strategies to JSON file"""
    with open(STRATEGIES_FILE, 'w') as f:
        json.dump(st.session_state['saved_strategies'], f, indent=4)


def delete_strategy(idx):
    """Delete a strategy by index"""
    st.session_state['saved_strategies'].pop(idx)
    save_strategies_to_file()

    # Reset selected strategy if it was deleted
    if 'selected_custom_strategy_idx' in st.session_state:
        if st.session_state['selected_custom_strategy_idx'] > len(st.session_state['saved_strategies']):
            st.session_state['selected_custom_strategy_idx'] = 0


def delete_all_strategies():
    """Delete all strategies"""
    st.session_state['saved_strategies'] = []
    save_strategies_to_file()
    st.session_state['selected_custom_strategy_idx'] = 0
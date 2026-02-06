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

    # Collect all strategy data
    strategy_data = {
        "strategy_name": final_strategy_name,
        "direction": st.session_state['strategy_direction'],
        "created_at": pd.Timestamp.now().strftime("%Y-%m-%d %H:%M:%S"),
        "entry": {
            "trigger": {
                "group": st.session_state.get('entry_trigger_group1'),
                "element1": st.session_state.get('entry_trigger_element1'),
                "event": st.session_state.get('entry_trigger_event'),
                "element2": st.session_state.get('entry_trigger_element2') if st.session_state.get(
                    'entry_trigger_event') != "At Level" else None,
                "amplitude": st.session_state.get('entry_trigger_amplitude') if st.session_state.get(
                    'entry_trigger_event') == "At Level" else None
            },
            "position_size": st.session_state.get('entry_position_size'),
            "conditions_count": st.session_state['entry_conditions_count'],
            "conditions": []
        },
        "exit": {
            "trigger": {
                "group": st.session_state.get('exit_trigger_group1'),
                "element1": st.session_state.get('exit_trigger_element1'),
                "event": st.session_state.get('exit_trigger_event'),
                "element2": st.session_state.get('exit_trigger_element2') if st.session_state.get(
                    'exit_trigger_event') != "At Level" else None,
                "amplitude": st.session_state.get('exit_trigger_amplitude') if st.session_state.get(
                    'exit_trigger_event') == "At Level" else None
            },
            "position_size": st.session_state.get('exit_position_size'),
            "conditions_count": st.session_state['exit_conditions_count'],
            "conditions": []
        }
    }

    # Collect entry conditions - UPDATED TO USE 'operator' instead of 'event'
    for i in range(st.session_state['entry_conditions_count']):
        condition = {
            "group": st.session_state.get(f'entry_cond_{i}_group1'),
            "element1": st.session_state.get(f'entry_cond_{i}_element1'),
            "operator": st.session_state.get(f'entry_cond_{i}_operator'),  # Changed from 'event'
            "element2": st.session_state.get(f'entry_cond_{i}_element2'),
            "amplitude": None  # Conditions don't use amplitude
        }
        strategy_data["entry"]["conditions"].append(condition)

    # Collect exit conditions - UPDATED TO USE 'operator' instead of 'event'
    for i in range(st.session_state['exit_conditions_count']):
        condition = {
            "group": st.session_state.get(f'exit_cond_{i}_group1'),
            "element1": st.session_state.get(f'exit_cond_{i}_element1'),
            "operator": st.session_state.get(f'exit_cond_{i}_operator'),  # Changed from 'event'
            "element2": st.session_state.get(f'exit_cond_{i}_element2'),
            "amplitude": None  # Conditions don't use amplitude
        }
        strategy_data["exit"]["conditions"].append(condition)

    # Save to session state
    st.session_state['saved_strategies'].append(strategy_data)

    # Save to file for persistence
    save_strategies_to_file()

    return len(st.session_state['saved_strategies'])


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
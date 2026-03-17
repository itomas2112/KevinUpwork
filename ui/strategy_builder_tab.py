"""
Strategy Builder tab (Tab 2) UI and logic
"""
import streamlit as st
import json
from datetime import datetime
from config.constants import (
    PRICE_AND_INDICATORS,
    RSI_GROUP,
    CMB_GROUP,
    STOCH_GROUP,
    ADX_GROUP,
    MACD_GROUP,
    ATR_VOLUME_GROUP,
    EVENT_TYPES,
    CONDITION_OPERATORS,
    CONDITION_COMPARE_TYPES,
    EXIT_TYPES,
    STOP_EVENT_TYPES,
    GROUP_NAMES,
    GROUP_MAP,
    R_PROFIT_LOSS_ELEMENTS,
    ATR_TARGET_ELEMENTS,
    get_group_elements,
)
from strategies.strategy_manager import save_strategy_to_session, delete_strategy, delete_all_strategies, save_strategies_to_file


def _ema_count():
    """Get the current number of EMA overlays configured in strategy builder."""
    return len(st.session_state.get('sb_ema_periods', []))


def render_strategy_builder_tab():
    """Render the strategy builder tab content"""

    # Apply any pending edit BEFORE widgets are created
    _apply_pending_edit()

    col_left, col_center, col_right = st.columns([1, 3, 1])

    with col_center:
        st.header("Strategy Builder")

        if not st.session_state['strategy_started']:
            render_create_button()
        elif st.session_state['strategy_direction'] is None:
            render_direction_selection()
        else:
            render_strategy_form()

        st.divider()
        render_strategy_management()


def _apply_pending_edit():
    """
    Called at the top of render_strategy_builder_tab(), BEFORE any widgets.
    Reads the pending strategy and populates all session state keys.
    """
    if '_pending_edit_strategy' not in st.session_state:
        return

    strategy = st.session_state.pop('_pending_edit_strategy')
    strategy_idx = st.session_state.pop('_pending_edit_strategy_idx')

    # Mark that we're editing an existing strategy
    st.session_state['editing_strategy'] = True
    st.session_state['editing_strategy_idx'] = strategy_idx

    # Load basic info
    st.session_state['strategy_started'] = True
    st.session_state['strategy_direction'] = strategy.get('direction', 'Long')
    st.session_state['strategy_name_input'] = strategy.get('strategy_name', '')
    st.session_state['strategy_patterns'] = strategy.get('patterns', [])

    # Load max positions
    max_pos = strategy.get('max_positions', 1)
    if max_pos is None:
        st.session_state['max_positions_unlimited'] = True
        st.session_state['max_positions_count'] = 1
    else:
        st.session_state['max_positions_unlimited'] = False
        st.session_state['max_positions_count'] = max_pos

    # Load entry config
    entry = strategy.get('entry', {})
    entry_trigger = entry.get('trigger', {})

    st.session_state['entry_trigger_group1'] = entry_trigger.get('group', 'Price & Indicators')
    st.session_state['entry_trigger_element1'] = entry_trigger.get('element1')
    st.session_state['entry_trigger_event'] = entry_trigger.get('event')
    st.session_state['entry_trigger_compare_type'] = entry_trigger.get('compare_type', 'Indicator')

    if entry_trigger.get('compare_type') == 'Indicator':
        st.session_state['entry_trigger_element2'] = entry_trigger.get('element2')
    else:
        st.session_state['entry_trigger_value'] = entry_trigger.get('value', 50.0)

    st.session_state['entry_position_size'] = entry.get('position_size', 1.0)

    # Load entry conditions
    entry_conditions = entry.get('conditions', [])
    st.session_state['entry_conditions_count'] = len(entry_conditions)

    for i, cond in enumerate(entry_conditions):
        st.session_state[f'entry_cond_{i}_group1'] = cond.get('group', 'Price & Indicators')
        st.session_state[f'entry_cond_{i}_element1'] = cond.get('element1')
        st.session_state[f'entry_cond_{i}_operator'] = cond.get('operator')
        st.session_state[f'entry_cond_{i}_compare_type'] = cond.get('compare_type', 'Indicator')

        if cond.get('compare_type') == 'Indicator':
            st.session_state[f'entry_cond_{i}_element2'] = cond.get('element2')
        else:
            st.session_state[f'entry_cond_{i}_value'] = cond.get('value', 50.0)

    # Load initial stop
    st.session_state['initial_stop'] = strategy.get('initial_stop', None)

    if st.session_state['initial_stop']:
        initial = st.session_state['initial_stop']
        st.session_state['initial_stop_type'] = initial.get('stop_type', 'Indicator')
        st.session_state['initial_stop_event'] = initial.get('event', 'Cross Below')
        if initial.get('stop_type') == 'ATR':
            st.session_state['initial_stop_atr_period'] = initial.get('atr_period', 14)
            st.session_state['initial_stop_atr_multiplier'] = initial.get('atr_multiplier', 1.5)
        else:
            st.session_state['initial_stop_element2'] = initial.get('element2')

    # Load indicator settings into strategy builder's own keys (sb_ prefix)
    ind_settings = strategy.get('indicator_settings', {})
    if ind_settings:
        from indicators.calculate_indicators import migrate_indicator_settings
        ind_settings = migrate_indicator_settings(ind_settings)
        pfx = "sb_"
        setting_keys = [
            'rsi_window', 'bb_upper_period', 'bb_upper_stdev', 'bb_mid_period',
            'bb_lower_period', 'bb_lower_stdev', 'kc_upper_ema', 'kc_upper_mult',
            'kc_mid_ema', 'kc_lower_ema', 'kc_lower_mult', 'kc_atr_period',
            'stoch_k_period', 'stoch_k_smooth', 'stoch_d_smooth', 'adx_period',
            'atr_period', 'macd_fast', 'macd_slow', 'macd_signal',
            'supertrend_period', 'supertrend_multiplier',
            'dc_upper_period', 'dc_mid_period', 'dc_lower_period', 'dc_offset',
            'psar_af_start', 'psar_af_increment', 'psar_af_max',
        ]
        for key in setting_keys:
            if key in ind_settings:
                st.session_state[f'{pfx}{key}'] = ind_settings[key]
        if 'ema_periods' in ind_settings:
            st.session_state[f'{pfx}ema_periods'] = list(ind_settings['ema_periods'])

    # Load exit groups
    saved_groups = strategy.get('exit_groups', [])
    st.session_state['exit_groups'] = []

    for group_idx, group in enumerate(saved_groups):
        # Backward compat: if old format has position_size but no allocation_pct, split equally
        alloc_pct = group.get('allocation_pct', 100.0 / max(len(saved_groups), 1))

        group_data = {
            'group_id': group.get('group_id', group_idx + 1),
            'allocation_pct': alloc_pct,
            'targets': group.get('targets', []),
            'stops': group.get('stops', []),
        }
        st.session_state['exit_groups'].append(group_data)

        st.session_state[f'exit_group_{group_idx}_alloc'] = alloc_pct

        for target_idx, target in enumerate(group.get('targets', [])):
            _load_exit_widget_keys(group_idx, 'Target', target_idx, target)

        for stop_idx, stop in enumerate(group.get('stops', [])):
            _load_exit_widget_keys(group_idx, 'Stop', stop_idx, stop)


def render_create_button():
    """Render the create new strategy button"""
    if st.button("➕ Create New Strategy", type="primary", use_container_width=True):
        st.session_state['_deselect_strategy'] = True
        st.session_state['strategy_started'] = True
        st.rerun()


def render_direction_selection():
    """Render Long/Short selection"""
    st.subheader("Step 1: Choose Strategy Direction")

    col1, col2 = st.columns(2)

    with col1:
        if st.button("📈 Long", type="primary", use_container_width=True):
            st.session_state['strategy_direction'] = 'Long'
            st.rerun()

    with col2:
        if st.button("📉 Short", type="primary", use_container_width=True):
            st.session_state['strategy_direction'] = 'Short'
            st.rerun()


def render_strategy_form():
    """Render the main strategy creation form"""
    st.success(f"Strategy Direction: **{st.session_state['strategy_direction']}**")

    # Strategy name input
    strategy_name_input = st.text_input(
        "Strategy Name",
        value=st.session_state.get('strategy_name_input', ''),
        placeholder="Enter a name for your strategy...",
        key="strategy_name_field"
    )

    # Pattern selection for strategy
    st.markdown("#### Apply to Patterns")
    st.caption("Select which pattern combinations this strategy should apply to")

    # Get available patterns from PRIMARY_SECONDARY_MAP
    from data.helpers import PRIMARY_SECONDARY_MAP

    # Create pattern combination options
    pattern_options = []
    for primary, secondaries in PRIMARY_SECONDARY_MAP.items():
        for secondary in secondaries:
            pattern_options.append(f"{primary} → {secondary}")

    # Multi-select for patterns
    selected_patterns = st.multiselect(
        "Pattern Combinations",
        options=pattern_options,
        default=st.session_state.get('strategy_patterns', []),
        key="strategy_patterns_select",
        help="This strategy will only appear when these pattern combinations are selected"
    )

    # Store in session state
    st.session_state['strategy_patterns'] = selected_patterns

    # Max Positions
    st.markdown("#### Max Positions")
    st.caption("How many simultaneous positions this strategy can hold")

    mp_col1, mp_col2 = st.columns([1, 1])
    with mp_col1:
        unlimited = st.checkbox(
            "Unlimited",
            value=st.session_state.get('max_positions_unlimited', False),
            key="max_positions_unlimited_cb",
        )
        st.session_state['max_positions_unlimited'] = unlimited
    with mp_col2:
        max_pos = st.number_input(
            "Max Positions",
            min_value=1,
            value=int(st.session_state.get('max_positions_count', 1)),
            step=1,
            key="max_positions_count_input",
            disabled=unlimited,
            label_visibility="collapsed",
        )
        st.session_state['max_positions_count'] = max_pos

    # Reset / Cancel buttons
    if st.session_state.get('editing_strategy'):
        col_reset, col_cancel = st.columns(2)
        with col_reset:
            if st.button("🔄 Reset Strategy", type="secondary", use_container_width=True):
                reset_strategy_builder()
        with col_cancel:
            if st.button("✖ Cancel Edit", type="secondary", use_container_width=True):
                reset_strategy_builder()
    else:
        if st.button("🔄 Reset Strategy", type="secondary"):
            reset_strategy_builder()

    st.divider()

    # Indicator Settings (strategy-specific, independent from charting sidebar)
    render_strategy_indicator_settings()
    st.divider()

    # Entry box (includes Static Stop)
    render_entry_box()
    st.divider()

    # Exit Groups
    render_exit_groups()
    st.divider()

    # Validation and Save button
    render_save_button(strategy_name_input)


def render_strategy_indicator_settings():
    """Render indicator settings that are saved with the strategy (independent from sidebar)."""
    st.subheader("Indicator Settings")
    st.caption("These settings are saved with the strategy and used during backtesting. They are independent from the charting sidebar settings.")

    pfx = "sb_"  # strategy builder prefix to avoid collision with sidebar widget keys

    with st.expander("RSI", expanded=False):
        st.session_state[f'{pfx}rsi_window'] = st.number_input(
            "RSI Period", 5, 50,
            value=int(st.session_state.get(f'{pfx}rsi_window', 14)),
            step=1, key=f"{pfx}rsi_w"
        )

    with st.expander("Bollinger Bands", expanded=False):
        st.caption("**Upper Band**")
        st.session_state[f'{pfx}bb_upper_period'] = st.number_input(
            "Upper Period", 5, 100,
            value=int(st.session_state.get(f'{pfx}bb_upper_period', 20)),
            step=1, key=f"{pfx}bb_up_p"
        )
        st.session_state[f'{pfx}bb_upper_stdev'] = st.number_input(
            "Upper StdDev", 0.5, 5.0,
            value=float(st.session_state.get(f'{pfx}bb_upper_stdev', 2.0)),
            step=0.01, format="%.2f", key=f"{pfx}bb_up_s"
        )
        st.caption("**Middle Band**")
        st.session_state[f'{pfx}bb_mid_period'] = st.number_input(
            "Middle Period", 5, 100,
            value=int(st.session_state.get(f'{pfx}bb_mid_period', 20)),
            step=1, key=f"{pfx}bb_mid_p"
        )
        st.caption("**Lower Band**")
        st.session_state[f'{pfx}bb_lower_period'] = st.number_input(
            "Lower Period", 5, 100,
            value=int(st.session_state.get(f'{pfx}bb_lower_period', 20)),
            step=1, key=f"{pfx}bb_lo_p"
        )
        st.session_state[f'{pfx}bb_lower_stdev'] = st.number_input(
            "Lower StdDev", 0.5, 5.0,
            value=float(st.session_state.get(f'{pfx}bb_lower_stdev', 2.0)),
            step=0.01, format="%.2f", key=f"{pfx}bb_lo_s"
        )

    with st.expander("Keltner Channel", expanded=False):
        st.session_state[f'{pfx}kc_atr_period'] = st.number_input(
            "ATR Period", 5, 100,
            value=int(st.session_state.get(f'{pfx}kc_atr_period', 10)),
            step=1, key=f"{pfx}kc_atr"
        )
        st.caption("**Upper Band**")
        st.session_state[f'{pfx}kc_upper_ema'] = st.number_input(
            "Upper EMA Period", 5, 100,
            value=int(st.session_state.get(f'{pfx}kc_upper_ema', 20)),
            step=1, key=f"{pfx}kc_up_ema"
        )
        st.session_state[f'{pfx}kc_upper_mult'] = st.number_input(
            "Upper ATR Mult", 0.5, 5.0,
            value=float(st.session_state.get(f'{pfx}kc_upper_mult', 2.0)),
            step=0.01, format="%.2f", key=f"{pfx}kc_up_m"
        )
        st.caption("**Middle Band**")
        st.session_state[f'{pfx}kc_mid_ema'] = st.number_input(
            "Middle EMA Period", 5, 100,
            value=int(st.session_state.get(f'{pfx}kc_mid_ema', 20)),
            step=1, key=f"{pfx}kc_mid_e"
        )
        st.caption("**Lower Band**")
        st.session_state[f'{pfx}kc_lower_ema'] = st.number_input(
            "Lower EMA Period", 5, 100,
            value=int(st.session_state.get(f'{pfx}kc_lower_ema', 20)),
            step=1, key=f"{pfx}kc_lo_ema"
        )
        st.session_state[f'{pfx}kc_lower_mult'] = st.number_input(
            "Lower ATR Mult", 0.5, 5.0,
            value=float(st.session_state.get(f'{pfx}kc_lower_mult', 2.0)),
            step=0.01, format="%.2f", key=f"{pfx}kc_lo_m"
        )

    with st.expander("Stochastic", expanded=False):
        st.session_state[f'{pfx}stoch_k_period'] = st.number_input(
            "%K Period", 1, 100,
            value=int(st.session_state.get(f'{pfx}stoch_k_period', 14)),
            step=1, key=f"{pfx}stoch_kp"
        )
        st.session_state[f'{pfx}stoch_k_smooth'] = st.number_input(
            "%K Smoothing", 1, 50,
            value=int(st.session_state.get(f'{pfx}stoch_k_smooth', 3)),
            step=1, key=f"{pfx}stoch_ks"
        )
        st.session_state[f'{pfx}stoch_d_smooth'] = st.number_input(
            "%D Smoothing", 1, 50,
            value=int(st.session_state.get(f'{pfx}stoch_d_smooth', 3)),
            step=1, key=f"{pfx}stoch_ds"
        )

    with st.expander("ADX", expanded=False):
        st.session_state[f'{pfx}adx_period'] = st.number_input(
            "ADX Period", 5, 100,
            value=int(st.session_state.get(f'{pfx}adx_period', 14)),
            step=1, key=f"{pfx}adx_p"
        )

    with st.expander("ATR", expanded=False):
        st.session_state[f'{pfx}atr_period'] = st.number_input(
            "ATR Period", 5, 100,
            value=int(st.session_state.get(f'{pfx}atr_period', 14)),
            step=1, key=f"{pfx}atr_p"
        )

    with st.expander("MACD", expanded=False):
        st.session_state[f'{pfx}macd_fast'] = st.number_input(
            "Fast Period", 2, 100,
            value=int(st.session_state.get(f'{pfx}macd_fast', 12)),
            step=1, key=f"{pfx}macd_f"
        )
        st.session_state[f'{pfx}macd_slow'] = st.number_input(
            "Slow Period", 2, 200,
            value=int(st.session_state.get(f'{pfx}macd_slow', 26)),
            step=1, key=f"{pfx}macd_sl"
        )
        st.session_state[f'{pfx}macd_signal'] = st.number_input(
            "Signal Period", 2, 100,
            value=int(st.session_state.get(f'{pfx}macd_signal', 9)),
            step=1, key=f"{pfx}macd_sg"
        )

    with st.expander("Supertrend", expanded=False):
        st.session_state[f'{pfx}supertrend_period'] = st.number_input(
            "Period", 1, 100,
            value=int(st.session_state.get(f'{pfx}supertrend_period', 10)),
            step=1, key=f"{pfx}st_p"
        )
        st.session_state[f'{pfx}supertrend_multiplier'] = st.number_input(
            "Multiplier", 0.5, 10.0,
            value=float(st.session_state.get(f'{pfx}supertrend_multiplier', 3.0)),
            step=0.01, format="%.2f", key=f"{pfx}st_m"
        )

    with st.expander("EMA Overlay", expanded=False):
        ema_state_key = f'{pfx}ema_periods'
        if ema_state_key not in st.session_state:
            st.session_state[ema_state_key] = []

        emas_to_remove = []
        for idx, ema_val in enumerate(st.session_state[ema_state_key]):
            line_col, remove_col = st.columns([3, 1])
            with line_col:
                new_val = st.number_input(
                    f"EMA {idx + 1} Period", 2, 500, int(ema_val), step=1,
                    key=f"{pfx}ema_p_{idx}",
                )
                st.session_state[ema_state_key][idx] = new_val
            with remove_col:
                st.markdown("<br>", unsafe_allow_html=True)
                if st.button("X", key=f"{pfx}ema_rm_{idx}"):
                    emas_to_remove.append(idx)

        if emas_to_remove:
            for idx in sorted(emas_to_remove, reverse=True):
                st.session_state[ema_state_key].pop(idx)
            st.rerun()

        if st.button("+ Add EMA", key=f"{pfx}ema_add"):
            st.session_state[ema_state_key].append(20)
            st.rerun()

    with st.expander("Donchian Channel", expanded=False):
        st.caption("**Upper Band**")
        st.session_state[f'{pfx}dc_upper_period'] = st.number_input(
            "Upper Period", 5, 200,
            value=int(st.session_state.get(f'{pfx}dc_upper_period', 20)),
            step=1, key=f"{pfx}dc_up_p"
        )
        st.caption("**Middle Band**")
        st.session_state[f'{pfx}dc_mid_period'] = st.number_input(
            "Middle Period", 5, 200,
            value=int(st.session_state.get(f'{pfx}dc_mid_period', 20)),
            step=1, key=f"{pfx}dc_mid_p"
        )
        st.caption("**Lower Band**")
        st.session_state[f'{pfx}dc_lower_period'] = st.number_input(
            "Lower Period", 5, 200,
            value=int(st.session_state.get(f'{pfx}dc_lower_period', 20)),
            step=1, key=f"{pfx}dc_lo_p"
        )
        st.divider()
        st.session_state[f'{pfx}dc_offset'] = st.number_input(
            "Offset / Shift", -50, 50,
            value=int(st.session_state.get(f'{pfx}dc_offset', 0)),
            step=1, key=f"{pfx}dc_off"
        )

    with st.expander("Parabolic SAR", expanded=False):
        st.session_state[f'{pfx}psar_af_start'] = st.number_input(
            "AF Start", 0.001, 0.5,
            value=float(st.session_state.get(f'{pfx}psar_af_start', 0.02)),
            step=0.01, format="%.3f", key=f"{pfx}psar_afs"
        )
        st.session_state[f'{pfx}psar_af_increment'] = st.number_input(
            "AF Increment", 0.001, 0.5,
            value=float(st.session_state.get(f'{pfx}psar_af_increment', 0.02)),
            step=0.01, format="%.3f", key=f"{pfx}psar_afi"
        )
        st.session_state[f'{pfx}psar_af_max'] = st.number_input(
            "AF Maximum", 0.01, 1.0,
            value=float(st.session_state.get(f'{pfx}psar_af_max', 0.20)),
            step=0.01, format="%.2f", key=f"{pfx}psar_afm"
        )


def _restore_entry_keys_if_needed():
    """Re-populate entry widget keys from editing state if they were cleaned up by a rerun."""
    if 'editing_strategy' not in st.session_state or not st.session_state['editing_strategy']:
        return
    idx = st.session_state.get('editing_strategy_idx')
    if idx is None or idx >= len(st.session_state.get('saved_strategies', [])):
        return
    if 'entry_trigger_group1' in st.session_state:
        return  # Keys still exist, no need to restore

    strategy = st.session_state['saved_strategies'][idx]
    entry = strategy.get('entry', {})
    entry_trigger = entry.get('trigger', {})
    st.session_state['entry_trigger_group1'] = entry_trigger.get('group', 'Price & Indicators')
    st.session_state['entry_trigger_element1'] = entry_trigger.get('element1')
    st.session_state['entry_trigger_event'] = entry_trigger.get('event')
    st.session_state['entry_trigger_compare_type'] = entry_trigger.get('compare_type', 'Indicator')
    if entry_trigger.get('compare_type') == 'Indicator':
        st.session_state['entry_trigger_element2'] = entry_trigger.get('element2')
    else:
        st.session_state['entry_trigger_value'] = entry_trigger.get('value', 50.0)
    st.session_state['entry_position_size'] = entry.get('position_size', 1.0)
    st.session_state['entry_conditions_count'] = len(entry.get('conditions', []))
    for i, cond in enumerate(entry.get('conditions', [])):
        st.session_state[f'entry_cond_{i}_group1'] = cond.get('group', 'Price & Indicators')
        st.session_state[f'entry_cond_{i}_element1'] = cond.get('element1')
        st.session_state[f'entry_cond_{i}_operator'] = cond.get('operator')
        st.session_state[f'entry_cond_{i}_compare_type'] = cond.get('compare_type', 'Indicator')
        if cond.get('compare_type') == 'Indicator':
            st.session_state[f'entry_cond_{i}_element2'] = cond.get('element2')
        else:
            st.session_state[f'entry_cond_{i}_value'] = cond.get('value', 50.0)

    # Restore initial stop
    st.session_state['initial_stop'] = strategy.get('initial_stop', None)
    if st.session_state['initial_stop']:
        initial = st.session_state['initial_stop']
        st.session_state['initial_stop_type'] = initial.get('stop_type', 'Indicator')
        st.session_state['initial_stop_event'] = initial.get('event', 'Cross Below')
        if initial.get('stop_type') == 'ATR':
            st.session_state['initial_stop_atr_period'] = initial.get('atr_period', 14)
            st.session_state['initial_stop_atr_multiplier'] = initial.get('atr_multiplier', 1.5)
        else:
            st.session_state['initial_stop_element2'] = initial.get('element2')


def render_entry_box():
    """Render entry strategy configuration box"""
    _restore_entry_keys_if_needed()

    st.subheader("Entry Strategy")

    with st.container(border=True):
        # TRIGGER (Required) - Event between indicators/price
        st.markdown("#### Trigger (Required)")
        st.caption("Define an event between compatible elements")

        col1, col2, col3 = st.columns([2, 1, 2])

        with col1:
            # Group selection for element 1
            entry_trigger_group1 = st.selectbox(
                "Select Group",
                GROUP_NAMES,
                key="entry_trigger_group1"
            )

            available_elements1 = get_group_elements(entry_trigger_group1, _ema_count())

            entry_trigger_element1 = st.selectbox(
                "Element 1",
                available_elements1,
                key="entry_trigger_element1"
            )

        with col2:
            entry_trigger_event = st.selectbox(
                "Event",
                EVENT_TYPES,
                key="entry_trigger_event"
            )

        with col3:
            # Choose between indicator or fixed value
            entry_trigger_compare_type = st.radio(
                "Compare to",
                CONDITION_COMPARE_TYPES,
                key="entry_trigger_compare_type",
                horizontal=True
            )

            if entry_trigger_compare_type == "Indicator":
                # Element 2 must be from same group as Element 1
                compatible_elements = get_compatible_elements(entry_trigger_element1)

                entry_trigger_element2 = st.selectbox(
                    "Element 2",
                    [e for e in compatible_elements if e != entry_trigger_element1],
                    key="entry_trigger_element2"
                )
                st.caption(f"Example: {entry_trigger_element1} {entry_trigger_event} {entry_trigger_element2}")
            else:  # Fixed Value
                entry_trigger_value = st.number_input(
                    "Value/Level",
                    value=50.0,
                    key="entry_trigger_value",
                    help="e.g., RSI crosses above 50, Price crosses below 4000"
                )
                st.caption(f"Example: {entry_trigger_element1} {entry_trigger_event} {entry_trigger_value}")

        st.divider()

        # POSITION SIZE (Required) - R multiples
        st.markdown("#### Position Size (Required)")
        st.caption("Specify the position size in R multiples (risk units)")

        entry_position_size = st.number_input(
            "Position Size (R)",
            min_value=0.1,
            value=1.0,
            step=0.01,
            format="%.2f",
            key="entry_position_size"
        )

        st.divider()

        # STATIC STOP (Required) - Defines 1R
        st.markdown("#### Static Stop (Required)")
        st.caption("Defines 1R — distance between entry price and stop level at entry bar")

        initial_stop_type = st.radio(
            "Stop Type",
            ["Indicator", "ATR"],
            key="initial_stop_type",
            horizontal=True
        )

        sc1, sc2, sc3 = st.columns([2, 1, 2])

        with sc1:
            st.info("Element 1: **Price** (fixed)")

        with sc2:
            initial_stop_event = st.selectbox(
                "Event",
                STOP_EVENT_TYPES,
                key="initial_stop_event"
            )

        with sc3:
            if initial_stop_type == "Indicator":
                st.radio(
                    "Compare to",
                    ["Indicator"],
                    key="initial_stop_compare_type",
                    horizontal=True
                )

                initial_stop_element2 = st.selectbox(
                    "Indicator",
                    get_group_elements("Price & Indicators", _ema_count())[1:],
                    key="initial_stop_element2"
                )
                st.caption(f"Price {initial_stop_event} {initial_stop_element2}")
            else:  # ATR
                initial_stop_atr_period = st.number_input(
                    "ATR Period",
                    min_value=1,
                    value=14,
                    step=1,
                    key="initial_stop_atr_period"
                )
                initial_stop_atr_mult = st.number_input(
                    "ATR Multiplier",
                    min_value=0.01,
                    value=1.50,
                    step=0.01,
                    format="%.2f",
                    key="initial_stop_atr_multiplier"
                )
                st.caption(f"Stop = Entry ± ATR({initial_stop_atr_period}) × {initial_stop_atr_mult}")

        # Store initial stop in session state
        if initial_stop_type == "Indicator":
            st.session_state['initial_stop'] = {
                'element1': 'Price',
                'stop_type': 'Indicator',
                'event': initial_stop_event,
                'compare_type': 'Indicator',
                'element2': initial_stop_element2
            }
        else:  # ATR
            st.session_state['initial_stop'] = {
                'element1': 'Price',
                'stop_type': 'ATR',
                'event': initial_stop_event,
                'atr_period': initial_stop_atr_period,
                'atr_multiplier': initial_stop_atr_mult,
            }

        st.divider()

        # CONDITIONS (Optional, 0-10)
        st.markdown("#### Conditions")
        st.caption(
            "All conditions must be met for the trigger to activate. If any condition fails, entry will not occur.")

        # Add condition button
        if st.button("➕ Add Condition", key="add_entry_condition"):
            if st.session_state['entry_conditions_count'] < 10:
                st.session_state['entry_conditions_count'] += 1

        # Display conditions
        if st.session_state['entry_conditions_count'] > 0:
            st.markdown(f"**Active Conditions: {st.session_state['entry_conditions_count']}**")

            for i in range(st.session_state['entry_conditions_count']):
                with st.expander(f"Condition {i + 1}", expanded=True):
                    # Remove button inside the condition box
                    if st.button("Remove", key=f"remove_entry_cond_{i}"):
                        _remove_entry_condition(i)
                        st.rerun()

                    col1, col2, col3 = st.columns([2, 1, 2])

                    with col1:
                        # Group selection for condition element 1
                        cond_group1 = st.selectbox(
                            "Select Group",
                            GROUP_NAMES,
                            key=f"entry_cond_{i}_group1"
                        )

                        cond_available_elements1 = get_group_elements(cond_group1, _ema_count())

                        cond_element1 = st.selectbox(
                            "Element 1",
                            cond_available_elements1,
                            key=f"entry_cond_{i}_element1"
                        )

                    with col2:
                        cond_operator = st.selectbox(
                            "Operator",
                            CONDITION_OPERATORS,
                            key=f"entry_cond_{i}_operator"
                        )

                    with col3:
                        # Choose between indicator or fixed value
                        cond_compare_type = st.radio(
                            "Compare to",
                            CONDITION_COMPARE_TYPES,
                            key=f"entry_cond_{i}_compare_type",
                            horizontal=True
                        )

                        if cond_compare_type == "Indicator":
                            # Element 2 must be from same group as Element 1
                            cond_compatible_elements = get_compatible_elements(cond_element1)

                            cond_element2 = st.selectbox(
                                "Element 2",
                                [e for e in cond_compatible_elements if e != cond_element1],
                                key=f"entry_cond_{i}_element2"
                            )
                            st.caption(f"{cond_element1} {cond_operator} {cond_element2}")
                        else:  # Fixed Value
                            cond_value = st.number_input(
                                "Value",
                                value=50.0,
                                key=f"entry_cond_{i}_value"
                            )
                            st.caption(f"{cond_element1} {cond_operator} {cond_value}")
        else:
            st.info("No conditions added. Trigger will activate without additional requirements.")


def render_exit_box():
    """Render exit strategy configuration box"""
    st.subheader("Exit Strategy")

    with st.container(border=True):
        # TRIGGER (Required) - Event between indicators/price
        st.markdown("#### Trigger (Required)")
        st.caption("Define an event between compatible elements")

        col1, col2, col3 = st.columns([2, 1, 2])

        with col1:
            # Group selection for element 1
            exit_trigger_group1 = st.selectbox(
                "Select Group",
                GROUP_NAMES,
                key="exit_trigger_group1"
            )

            available_elements1 = get_group_elements(exit_trigger_group1, _ema_count())

            exit_trigger_element1 = st.selectbox(
                "Element 1",
                available_elements1,
                key="exit_trigger_element1"
            )

        with col2:
            exit_trigger_event = st.selectbox(
                "Event",
                EVENT_TYPES,
                key="exit_trigger_event"
            )

        with col3:
            # Choose between indicator or fixed value
            exit_trigger_compare_type = st.radio(
                "Compare to",
                CONDITION_COMPARE_TYPES,
                key="exit_trigger_compare_type",
                horizontal=True
            )

            if exit_trigger_compare_type == "Indicator":
                # Element 2 must be from same group as Element 1
                compatible_elements = get_compatible_elements(exit_trigger_element1)

                exit_trigger_element2 = st.selectbox(
                    "Element 2",
                    [e for e in compatible_elements if e != exit_trigger_element1],
                    key="exit_trigger_element2"
                )
                st.caption(f"Example: {exit_trigger_element1} {exit_trigger_event} {exit_trigger_element2}")
            else:  # Fixed Value
                exit_trigger_value = st.number_input(
                    "Value/Level",
                    value=50.0,
                    key="exit_trigger_value",
                    help="e.g., RSI crosses above 50, Price crosses below 4000"
                )
                st.caption(f"Example: {exit_trigger_element1} {exit_trigger_event} {exit_trigger_value}")

        st.divider()

        # POSITION SIZE (Required) - Units of position
        st.markdown("#### Position Size (Required)")
        st.caption("Specify the units/size of the position to exit")

        exit_position_size = st.number_input(
            "Position Size (units)",
            min_value=0.0,
            value=1.0,
            step=0.01,
            format="%.2f",
            key="exit_position_size"
        )

        st.divider()

        # CONDITIONS (Optional, 0-10)
        st.markdown("#### Conditions")
        st.caption(
            "All conditions must be met for the trigger to activate. If any condition fails, exit will not occur.")

        # Add condition button
        if st.button("➕ Add Condition", key="add_exit_condition"):
            if st.session_state['exit_conditions_count'] < 10:
                st.session_state['exit_conditions_count'] += 1
                st.rerun()

        # Display conditions
        if st.session_state['exit_conditions_count'] > 0:
            st.markdown(f"**Active Conditions: {st.session_state['exit_conditions_count']}**")

            for i in range(st.session_state['exit_conditions_count']):
                with st.expander(f"Condition {i + 1}", expanded=True):
                    # Remove button inside the condition box
                    if st.button("Remove", key=f"remove_exit_cond_{i}"):
                        _remove_exit_condition(i)
                        st.rerun()

                    col1, col2, col3 = st.columns([2, 1, 2])

                    with col1:
                        # Group selection for condition element 1
                        cond_group1 = st.selectbox(
                            "Select Group",
                            GROUP_NAMES,
                            key=f"exit_cond_{i}_group1"
                        )

                        cond_available_elements1 = get_group_elements(cond_group1, _ema_count())

                        cond_element1 = st.selectbox(
                            "Element 1",
                            cond_available_elements1,
                            key=f"exit_cond_{i}_element1"
                        )

                    with col2:
                        cond_operator = st.selectbox(
                            "Operator",
                            CONDITION_OPERATORS,
                            key=f"exit_cond_{i}_operator"
                        )

                    with col3:
                        # Choose between indicator or fixed value
                        cond_compare_type = st.radio(
                            "Compare to",
                            CONDITION_COMPARE_TYPES,
                            key=f"exit_cond_{i}_compare_type",
                            horizontal=True
                        )

                        if cond_compare_type == "Indicator":
                            # Element 2 must be from same group as Element 1
                            cond_compatible_elements = get_compatible_elements(cond_element1)

                            cond_element2 = st.selectbox(
                                "Element 2",
                                [e for e in cond_compatible_elements if e != cond_element1],
                                key=f"exit_cond_{i}_element2"
                            )
                            st.caption(f"{cond_element1} {cond_operator} {cond_element2}")
                        else:  # Fixed Value
                            cond_value = st.number_input(
                                "Value",
                                value=50.0,
                                key=f"exit_cond_{i}_value"
                            )
                            st.caption(f"{cond_element1} {cond_operator} {cond_value}")
        else:
            st.info("No conditions added. Trigger will activate without additional requirements.")


def render_save_button(strategy_name_input: str):
    """Render save/update button with validation"""

    # Check if we're editing
    is_editing = st.session_state.get('editing_strategy', False)
    button_label = "Update Strategy" if is_editing else "Save Strategy"

    # Validation
    is_valid = validate_exit_groups()

    if not is_valid:
        exit_groups = st.session_state.get('exit_groups', [])
        has_empty_group = any(
            len(g.get('targets', [])) == 0 and len(g.get('stops', [])) == 0
            for g in exit_groups
        )

        if not exit_groups:
            st.error("⚠️ You must add at least one exit group!")
        elif has_empty_group:
            st.error("⚠️ Every exit group must have at least one target or stop!")
        else:
            st.error("⚠️ Total allocation must equal 100%!")

    col1, col2 = st.columns([3, 1])
    with col2:
        if st.button(button_label, type="primary", use_container_width=True, disabled=not is_valid):
            if is_editing:
                # Update existing strategy
                editing_idx = st.session_state.get('editing_strategy_idx')
                # Delete the old version
                st.session_state['saved_strategies'].pop(editing_idx)
                # Save the updated version
                count = save_strategy_to_session(strategy_name_input)

                # Clear editing flags
                st.session_state['editing_strategy'] = False
                st.session_state['editing_strategy_idx'] = None

                st.success(f"✅ Strategy updated!")
            else:
                # Create new strategy
                count = save_strategy_to_session(strategy_name_input)
                st.success(f"✅ Strategy saved! Total: {count}")

            # Reset the strategy builder
            st.session_state['strategy_started'] = False
            st.session_state['strategy_direction'] = None
            st.session_state['entry_conditions_count'] = 0
            st.session_state['exit_groups'] = []
            st.session_state['initial_stop'] = None
            st.session_state['strategy_name_input'] = ""
            st.session_state.pop('max_positions_unlimited', None)
            st.session_state.pop('max_positions_count', None)

            st.rerun()


# =============================================================================
# Export/Import Functions
# =============================================================================

def render_export_import_section():
    """Render the export/import section for strategies"""
    st.subheader("Export / Import Strategies")
    st.caption("Save your strategies to a file or load them back after app restart")

    col1, col2 = st.columns(2)

    with col1:
        st.markdown("#### ⬇️ Export Strategies")

        if st.session_state.get('saved_strategies'):
            # Prepare export data
            export_data = {
                'export_date': datetime.now().isoformat(),
                'version': '1.0',
                'strategies': st.session_state['saved_strategies']
            }

            # Convert to JSON string
            json_string = json.dumps(export_data, indent=2)

            # Generate filename with timestamp
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"strategies_backup_{timestamp}.json"

            st.download_button(
                label="Download All Strategies",
                data=json_string,
                file_name=filename,
                mime="application/json",
                type="primary",
                use_container_width=True
            )
            st.caption(f"Will export {len(st.session_state['saved_strategies'])} strategy(ies)")
        else:
            st.info("No strategies to export. Create and save some strategies first.")

    with col2:
        st.markdown("#### ⬆️ Import Strategies")

        uploaded_file = st.file_uploader(
            "Upload strategies JSON file",
            type=['json'],
            key="strategy_import_uploader",
            help="Select a previously exported strategies file"
        )

        if uploaded_file is not None:
            try:
                # Read and parse the JSON file
                content = uploaded_file.read().decode('utf-8')
                import_data = json.loads(content)

                # Support both new format (with metadata) and old format (plain list)
                if isinstance(import_data, list):
                    # Old format: plain list of strategies (e.g., from saved_strategies.json)
                    strategies_to_import = import_data
                    export_date = None
                elif 'strategies' in import_data:
                    # New format: with metadata wrapper
                    strategies_to_import = import_data['strategies']
                    export_date = import_data.get('export_date')
                else:
                    st.error("❌ Invalid file format: expected a list of strategies or an object with 'strategies' key")
                    return

                if not isinstance(strategies_to_import, list):
                    st.error("❌ Invalid file format: strategies must be a list")
                    return

                # Show import preview
                st.success(f"✅ Found {len(strategies_to_import)} strategy(ies) in file")

                if export_date:
                    export_date_str = export_date[:19].replace('T', ' ')
                    st.caption(f"Exported on: {export_date_str}")

                # Import mode selection
                import_mode = st.radio(
                    "Import mode",
                    ["Merge (add to existing)", "Replace (clear existing first)"],
                    key="import_mode_radio",
                    horizontal=True
                )

                # Show preview of strategies to import
                with st.expander("Preview strategies to import", expanded=False):
                    for idx, strategy in enumerate(strategies_to_import):
                        name = strategy.get('strategy_name', f'Strategy_{idx+1}')
                        direction = strategy.get('direction', 'N/A')
                        patterns = strategy.get('patterns', [])
                        pattern_info = f"{len(patterns)} patterns" if patterns else "All patterns"
                        st.markdown(f"- **{name}** ({direction}) - {pattern_info}")

                # Import button
                if st.button("Import Strategies", type="primary", use_container_width=True):
                    if import_mode == "Replace (clear existing first)":
                        st.session_state['saved_strategies'] = []

                    # Add imported strategies
                    existing_names = {s.get('strategy_name') for s in st.session_state.get('saved_strategies', [])}
                    imported_count = 0
                    skipped_count = 0

                    for strategy in strategies_to_import:
                        strategy_name = strategy.get('strategy_name', '')

                        # Check for duplicates in merge mode
                        if import_mode == "Merge (add to existing)" and strategy_name in existing_names:
                            # Rename to avoid conflict
                            new_name = f"{strategy_name}_imported"
                            counter = 1
                            while new_name in existing_names:
                                new_name = f"{strategy_name}_imported_{counter}"
                                counter += 1
                            strategy['strategy_name'] = new_name
                            existing_names.add(new_name)

                        st.session_state['saved_strategies'].append(strategy)
                        imported_count += 1

                    # Save to file for persistence
                    save_strategies_to_file()

                    st.success(f"✅ Successfully imported {imported_count} strategy(ies)!")
                    st.rerun()

            except json.JSONDecodeError as e:
                st.error(f"❌ Invalid JSON file: {str(e)}")
            except Exception as e:
                st.error(f"❌ Error importing file: {str(e)}")


def render_strategy_management():
    """Render strategy management section"""
    st.subheader("Strategy Management")

    if st.session_state['saved_strategies']:
        st.caption(f"Total strategies saved: {len(st.session_state['saved_strategies'])}")

        # Create a table view of strategies
        for idx, strategy in enumerate(st.session_state['saved_strategies']):
            with st.container(border=True):
                col1, col2, col3, col4, col5 = st.columns([3, 2, 2, 1, 1])

                with col1:
                    st.markdown(f"**{strategy.get('strategy_name', f'Strategy_{idx + 1}')}**")

                with col2:
                    direction_emoji = "📈" if strategy.get('direction') == 'Long' else "📉"
                    st.markdown(f"{direction_emoji} {strategy.get('direction', 'N/A')}")

                with col3:
                    patterns = strategy.get('patterns', [])
                    if patterns:
                        st.caption(f"Patterns: {len(patterns)}")
                    else:
                        st.caption("Patterns: All")

                with col4:
                    if st.button("✏️", key=f"edit_strategy_{idx}", help="Edit this strategy"):
                        # Load strategy into builder for editing
                        load_strategy_for_editing(strategy, idx)
                        st.rerun()

                with col5:
                    if st.button("🗑️", key=f"delete_strategy_{idx}", help="Delete this strategy"):
                        delete_strategy(idx)
                        st.success(f"Strategy deleted!")
                        st.rerun()

                # Expandable details view
                with st.expander("View Strategy Details", expanded=False):

                    # Show patterns
                    st.markdown("### Applied to Patterns")
                    patterns = strategy.get('patterns', [])
                    if patterns:
                        for pattern in patterns:
                            st.markdown(f"- {pattern}")
                    else:
                        st.info("This strategy applies to all patterns")

                    st.divider()

                    # Entry Strategy Section
                    st.markdown("### Entry Strategy")

                    with st.container(border=True):
                        entry = strategy.get('entry', {})
                        trigger = entry.get('trigger', {})

                        # Trigger
                        st.markdown("#### Trigger")
                        trigger_element1 = trigger.get('element1', 'N/A')
                        trigger_event = trigger.get('event', 'N/A')
                        trigger_compare_type = trigger.get('compare_type', 'Indicator')

                        if trigger_compare_type == "Fixed Value":
                            trigger_value = trigger.get('value', 'N/A')
                            st.info(f"**{trigger_element1}** {trigger_event} **{trigger_value}**")
                        else:
                            trigger_element2 = trigger.get('element2', 'N/A')
                            st.info(f"**{trigger_element1}** {trigger_event} **{trigger_element2}**")

                        # Position Size
                        st.markdown("#### Position Size")
                        position_size = entry.get('position_size', 'N/A')
                        st.info(f"**{position_size}R**")

                        # Conditions
                        st.markdown("#### Conditions")
                        conditions_count = len(entry.get('conditions', []))

                        if conditions_count > 0:
                            st.markdown(f"**{conditions_count} condition(s) must be met:**")
                            for i, cond in enumerate(entry.get('conditions', []), 1):
                                cond_element1 = cond.get('element1', 'N/A')
                                cond_operator = cond.get('operator', 'N/A')
                                cond_compare_type = cond.get('compare_type', 'Indicator')

                                if cond_compare_type == "Fixed Value":
                                    cond_value = cond.get('value', 'N/A')
                                    st.markdown(f"{i}. {cond_element1} **{cond_operator}** {cond_value}")
                                else:
                                    cond_element2 = cond.get('element2', 'N/A')
                                    st.markdown(f"{i}. {cond_element1} **{cond_operator}** {cond_element2}")
                        else:
                            st.markdown("*No conditions - trigger activates immediately*")

                    st.divider()

                    # Exit Strategy Section
                    st.markdown("### Exit Strategy")

                    # Static Stop
                    initial_stop = strategy.get('initial_stop')
                    if initial_stop and initial_stop.get('element1'):
                        with st.container(border=True):
                            st.markdown("#### 🛑 Static Stop (defines 1R, closes all remaining position)")
                            if initial_stop.get('stop_type') == 'ATR':
                                atr_p = initial_stop.get('atr_period', 14)
                                atr_m = initial_stop.get('atr_multiplier', 1.5)
                                st.info(f"**ATR Stop** — Entry ± ATR({atr_p}) × {atr_m}")
                            else:
                                stop_el1 = initial_stop.get('element1', 'N/A')
                                stop_event = initial_stop.get('event', 'N/A')
                                stop_el2 = initial_stop.get('element2', 'N/A')
                                st.info(f"**{stop_el1}** {stop_event} **{stop_el2}**")

                    # Exit Groups
                    exit_groups = strategy.get('exit_groups', [])

                    if exit_groups:
                        for g_idx, group in enumerate(exit_groups):
                            with st.container(border=True):
                                g_alloc = group.get('allocation_pct', group.get('position_size', 'N/A'))
                                st.markdown(f"#### Exit Group {g_idx + 1}  —  {g_alloc}%")

                                # Targets
                                targets = group.get('targets', [])
                                if targets:
                                    for t_idx, target in enumerate(targets):
                                        t_trigger = target.get('trigger', {})
                                        t_el1 = t_trigger.get('element1', 'N/A')
                                        t_event = t_trigger.get('event', 'N/A')
                                        t_ctype = t_trigger.get('compare_type', 'Indicator')

                                        if t_el1 == "ATR Target":
                                            atr_p = t_trigger.get('atr_period', 14)
                                            atr_m = t_trigger.get('atr_multiplier', 2.0)
                                            t_el2 = f"ATR({atr_p}) × {atr_m}"
                                        elif t_ctype == "Fixed Value":
                                            t_el2 = t_trigger.get('value', 'N/A')
                                            # Add R suffix for R Profit/R Loss
                                            if t_el1 in ("R Profit", "R Loss"):
                                                t_el2 = f"{t_el2}R"
                                        else:
                                            t_el2 = t_trigger.get('element2', 'N/A')

                                        st.markdown(f"**Target {t_idx + 1}:** {t_el1} {t_event} {t_el2}")

                                        # Target conditions
                                        for c_idx, cond in enumerate(target.get('conditions', []), 1):
                                            c_el1 = cond.get('element1', 'N/A')
                                            c_op = cond.get('operator', 'N/A')
                                            c_ctype = cond.get('compare_type', 'Indicator')
                                            if c_ctype == "Fixed Value":
                                                c_el2 = cond.get('value', 'N/A')
                                            else:
                                                c_el2 = cond.get('element2', 'N/A')
                                            st.caption(f"   Condition {c_idx}: {c_el1} {c_op} {c_el2}")

                                # Dynamic Stops
                                stops = group.get('stops', [])
                                if stops:
                                    for s_idx, stop in enumerate(stops):
                                        s_trigger = stop.get('trigger', {})
                                        s_el1 = s_trigger.get('element1', 'N/A')
                                        s_event = s_trigger.get('event', 'N/A')
                                        s_ctype = s_trigger.get('compare_type', 'Indicator')

                                        if s_el1 == "ATR Target":
                                            atr_p = s_trigger.get('atr_period', 14)
                                            atr_m = s_trigger.get('atr_multiplier', 2.0)
                                            s_el2 = f"ATR({atr_p}) × {atr_m}"
                                        elif s_ctype == "Fixed Value":
                                            s_el2 = s_trigger.get('value', 'N/A')
                                            if s_el1 in ("R Profit", "R Loss"):
                                                s_el2 = f"{s_el2}R"
                                        else:
                                            s_el2 = s_trigger.get('element2', 'N/A')

                                        st.markdown(f"**Dynamic Stop {s_idx + 1}:** {s_el1} {s_event} {s_el2}")

                                        # Stop conditions
                                        for c_idx, cond in enumerate(stop.get('conditions', []), 1):
                                            c_el1 = cond.get('element1', 'N/A')
                                            c_op = cond.get('operator', 'N/A')
                                            c_ctype = cond.get('compare_type', 'Indicator')
                                            if c_ctype == "Fixed Value":
                                                c_el2 = cond.get('value', 'N/A')
                                            else:
                                                c_el2 = cond.get('element2', 'N/A')
                                            st.caption(f"   Condition {c_idx}: {c_el1} {c_op} {c_el2}")

                                if not targets and not stops:
                                    st.info("No targets or stops configured in this group")
                    else:
                        st.info("No exit groups configured")

                    # Advanced: Show JSON for debugging
                    with st.expander("🔧 Advanced: View Raw JSON", expanded=False):
                        st.json(strategy)

        # Bulk delete option
        st.divider()
        col1, col2, col3 = st.columns([2, 1, 2])
        with col2:
            if st.button("🗑️ Delete All Strategies", type="secondary", use_container_width=True):
                if st.session_state.get('confirm_delete_all', False):
                    delete_all_strategies()
                    st.session_state['confirm_delete_all'] = False
                    st.success("All strategies deleted!")
                    st.rerun()
                else:
                    st.session_state['confirm_delete_all'] = True
                    st.warning("⚠️ Click again to confirm deletion of ALL strategies")
                    st.rerun()
    else:
        st.info("No strategies saved yet. Create and save a strategy to see it here.")

    # Export/Import section
    st.divider()
    render_export_import_section()




def _remove_entry_condition(idx):
    """Remove entry condition at given index, shifting subsequent keys down."""
    count = st.session_state['entry_conditions_count']
    keys_per_cond = ['group1', 'element1', 'operator', 'compare_type', 'element2', 'value']
    for i in range(idx, count - 1):
        for k in keys_per_cond:
            src = f"entry_cond_{i + 1}_{k}"
            dst = f"entry_cond_{i}_{k}"
            if src in st.session_state:
                st.session_state[dst] = st.session_state[src]
    # Clean up last slot
    for k in keys_per_cond:
        st.session_state.pop(f"entry_cond_{count - 1}_{k}", None)
    st.session_state['entry_conditions_count'] = count - 1


def _remove_exit_condition(idx):
    """Remove exit condition at given index, shifting subsequent keys down."""
    count = st.session_state['exit_conditions_count']
    keys_per_cond = ['group1', 'element1', 'operator', 'compare_type', 'element2', 'value']
    for i in range(idx, count - 1):
        for k in keys_per_cond:
            src = f"exit_cond_{i + 1}_{k}"
            dst = f"exit_cond_{i}_{k}"
            if src in st.session_state:
                st.session_state[dst] = st.session_state[src]
    # Clean up last slot
    for k in keys_per_cond:
        st.session_state.pop(f"exit_cond_{count - 1}_{k}", None)
    st.session_state['exit_conditions_count'] = count - 1


def get_compatible_elements(selected_element):
    """Get compatible elements based on selection"""
    for group_name, group_list in GROUP_MAP.items():
        if selected_element in group_list:
            if group_name == "Price & Indicators":
                # Include dynamic EMAs
                return get_group_elements(group_name, _ema_count())
            return group_list
    # Fallback: check dynamic EMA names
    if selected_element and selected_element.startswith("EMA "):
        return get_group_elements("Price & Indicators", _ema_count())
    return PRICE_AND_INDICATORS


def load_strategy_for_editing(strategy, strategy_idx):
    """
    """
    st.session_state['_deselect_strategy'] = True
    st.session_state['_pending_edit_strategy'] = strategy
    st.session_state['_pending_edit_strategy_idx'] = strategy_idx


def validate_exit_groups():
    """Validate that total allocation equals 100% and every group has at least one target or stop"""
    exit_groups = st.session_state.get('exit_groups', [])

    # Must have at least one exit group
    if not exit_groups:
        return False

    total_alloc = 0
    for exit_group in exit_groups:
        total_alloc += exit_group.get('allocation_pct', 0)

        # Every group must have at least one target or one stop
        has_target = len(exit_group.get('targets', [])) > 0
        has_stop = len(exit_group.get('stops', [])) > 0
        if not has_target and not has_stop:
            return False

    return abs(total_alloc - 100.0) < 0.01  # Small tolerance for float comparison



def add_exit_group():
    """Add a new exit group"""
    if 'exit_groups' not in st.session_state:
        st.session_state['exit_groups'] = []

    new_group = {
        'group_id': len(st.session_state['exit_groups']) + 1,
        'allocation_pct': 100.0,
        'targets': [],
        'stops': []
    }

    st.session_state['exit_groups'].append(new_group)

def remove_exit_group(group_idx):
    """Remove an exit group"""
    if 'exit_groups' in st.session_state and 0 <= group_idx < len(st.session_state['exit_groups']):
        total = len(st.session_state['exit_groups'])
        # Clear stale widget keys for this and all subsequent groups
        # (indices shift down after pop, so old keys become misaligned)
        for idx in range(group_idx, total):
            _clear_exit_group_widget_keys(idx)
        st.session_state['exit_groups'].pop(group_idx)


def add_exit_to_group(group_idx, exit_type):
    """Add a target or stop to a specific exit group"""
    if 'exit_groups' not in st.session_state:
        return

    if 0 <= group_idx < len(st.session_state['exit_groups']):
        exit_config = {
            'type': exit_type,
            'trigger': {},
            'conditions': []
        }

        if exit_type == 'Target':
            st.session_state['exit_groups'][group_idx]['targets'].append(exit_config)
        else:  # Stop
            st.session_state['exit_groups'][group_idx]['stops'].append(exit_config)


def _clear_exit_group_widget_keys(group_idx):
    """Clear all widget keys for an exit group so they get re-populated from data."""
    prefixes = (f"Target_{group_idx}_", f"Stop_{group_idx}_", f"exit_group_{group_idx}_")
    for k in list(st.session_state.keys()):
        if any(k.startswith(p) for p in prefixes):
            del st.session_state[k]


def remove_exit_from_group(group_idx, exit_type, exit_idx):
    """Remove a specific exit from a group"""
    if 'exit_groups' not in st.session_state:
        return

    if 0 <= group_idx < len(st.session_state['exit_groups']):
        list_key = 'targets' if exit_type == 'Target' else 'stops'
        items = st.session_state['exit_groups'][group_idx].get(list_key, [])
        if 0 <= exit_idx < len(items):
            total = len(items)
            # Clear stale widget keys from removed index onward
            # (indices shift down after pop, so old keys become misaligned)
            for idx in range(exit_idx, total):
                prefix = f"{exit_type}_{group_idx}_{idx}"
                for k in list(st.session_state.keys()):
                    if k.startswith(prefix):
                        del st.session_state[k]
            items.pop(exit_idx)


def render_exit_groups():
    """Render all exit groups with targets and dynamic stops"""
    st.subheader("Exit Strategy Groups")
    st.caption(
        "Each group handles a percentage of your position. Targets and Dynamic Stops within a group are OCO (One-Cancels-Other)")

    # Initialize exit groups if not exists
    if 'exit_groups' not in st.session_state:
        st.session_state['exit_groups'] = []

    # Calculate total allocation
    total_alloc = sum(group.get('allocation_pct', 0) for group in st.session_state.get('exit_groups', []))

    # Validation message
    if len(st.session_state.get('exit_groups', [])) > 0:
        if abs(total_alloc - 100.0) < 0.01:
            st.success(f"✓ Total Allocation: {total_alloc:.0f}%")
        else:
            st.error(f"⚠️ Total Allocation ({total_alloc:.1f}%) must equal 100% - Strategy invalid!")

    # Render each exit group
    for group_idx, exit_group in enumerate(st.session_state.get('exit_groups', [])):
        with st.container(border=True):
            # Group header
            col1, col2, col3 = st.columns([3, 2, 1])

            with col1:
                st.markdown(f"### Exit Group {group_idx + 1}")

            with col2:
                group_alloc = st.number_input(
                    "Allocation (%)",
                    min_value=0.0,
                    max_value=100.0,
                    value=exit_group.get('allocation_pct', 100.0),
                    step=1.0,
                    key=f"exit_group_{group_idx}_alloc"
                )
                st.session_state['exit_groups'][group_idx]['allocation_pct'] = group_alloc

            with col3:
                if st.button("🗑️", key=f"remove_exit_group_{group_idx}", help="Remove this exit group"):
                    remove_exit_group(group_idx)
                    st.rerun()

            st.divider()

            # Targets section
            st.markdown("#### Targets")
            targets = exit_group.get('targets', [])

            if len(targets) == 0:
                st.info("No targets added yet")
            else:
                for target_idx, target in enumerate(targets):
                    render_exit_config(group_idx, 'Target', target_idx, target)

            if st.button(f"➕ Add Target", key=f"add_target_group_{group_idx}"):
                add_exit_to_group(group_idx, 'Target')
                st.rerun()

            st.divider()

            # Dynamic Stops section
            st.markdown("#### Dynamic Stops")
            st.caption("Static Stop is defined in the Entry box and applies to all groups")

            stops = exit_group.get('stops', [])

            # Show static stop reference (always present)
            if st.session_state.get('initial_stop'):
                initial = st.session_state['initial_stop']
                if initial.get('stop_type') == 'ATR':
                    atr_p = initial.get('atr_period', 14)
                    atr_m = initial.get('atr_multiplier', 1.5)
                    st.info(f"🔒 Static Stop: ATR({atr_p}) × {atr_m} (auto-included)")
                else:
                    st.info(
                        f"🔒 Static Stop: {initial['element1']} {initial['event']} {initial.get('element2', 'N/A')} (auto-included)")

            if len(stops) == 0:
                st.info("No dynamic stops added")
            else:
                for stop_idx, stop in enumerate(stops):
                    render_exit_config(group_idx, 'Stop', stop_idx, stop)

            if st.button(f"➕ Add Dynamic Stop", key=f"add_stop_group_{group_idx}"):
                add_exit_to_group(group_idx, 'Stop')
                st.rerun()

    # Add new exit group button
    st.divider()
    if st.button("➕ Add Exit Group", type="primary"):
        add_exit_group()
        st.rerun()


def render_exit_config(group_idx, exit_type, exit_idx, exit_config):
    """Render a single exit (target or stop) configuration"""
    icon = "" if exit_type == "Target" else ""

    prefix = f"{exit_type}_{group_idx}_{exit_idx}"

    # Re-populate widget keys from exit_config if they were cleaned up by a rerun
    # (happens when st.rerun() fires in entry/indicator sections before exit widgets render)
    if f"{prefix}_trigger_group1" not in st.session_state:
        _load_exit_widget_keys(group_idx, exit_type, exit_idx, exit_config)

    with st.expander(f"{icon} {exit_type} {exit_idx + 1}", expanded=True):
        col_delete = st.columns([10, 1])

        with col_delete[1]:
            if st.button("×", key=f"remove_{exit_type}_{group_idx}_{exit_idx}",
                         help=f"Remove this {exit_type.lower()}"):
                remove_exit_from_group(group_idx, exit_type, exit_idx)
                st.rerun()

        # Trigger
        st.markdown("**Trigger**")
        exit_group_options = ["R Profit / R Loss", "ATR Target"] + GROUP_NAMES

        col1, col2, col3 = st.columns([2, 1, 2])

        with col1:
            # Default to "Price & Indicators" (index 1) for new exits
            group_key = f"{prefix}_trigger_group1"
            if group_key not in st.session_state:
                st.session_state[group_key] = "Price & Indicators"

            trigger_group = st.selectbox(
                "Select Group",
                exit_group_options,
                key=group_key
            )

            if trigger_group == "R Profit / R Loss":
                available_elements = R_PROFIT_LOSS_ELEMENTS
            elif trigger_group == "ATR Target":
                available_elements = ATR_TARGET_ELEMENTS
            else:
                available_elements = get_group_elements(trigger_group, _ema_count())

            trigger_element1 = st.selectbox(
                "Element 1",
                available_elements,
                key=f"{prefix}_trigger_element1"
            )

        is_r_element = trigger_element1 in R_PROFIT_LOSS_ELEMENTS
        is_atr_target = trigger_element1 in ATR_TARGET_ELEMENTS

        with col2:
            if is_atr_target:
                # ATR Target: event is always Cross Above for Long / Cross Below for Short
                # but let user choose (they may want Cross Below for a stop-like ATR Target)
                event_options = STOP_EVENT_TYPES if exit_type == "Stop" else EVENT_TYPES
                trigger_event = st.selectbox(
                    "Event",
                    event_options,
                    key=f"{prefix}_trigger_event"
                )
            else:
                event_options = STOP_EVENT_TYPES if exit_type == "Stop" else EVENT_TYPES
                trigger_event = st.selectbox(
                    "Event",
                    event_options,
                    key=f"{prefix}_trigger_event"
                )

        with col3:
            if is_atr_target:
                # ATR Target: show ATR period + multiplier inputs
                if st.session_state.get(f"{prefix}_trigger_compare_type") != "ATR":
                    st.session_state[f"{prefix}_trigger_compare_type"] = "ATR"
                atr_period = st.number_input(
                    "ATR Period",
                    min_value=1,
                    value=int(st.session_state.get(f"{prefix}_atr_period", 14)),
                    step=1,
                    key=f"{prefix}_atr_period"
                )
                atr_mult = st.number_input(
                    "ATR Multiplier",
                    min_value=0.01,
                    value=float(st.session_state.get(f"{prefix}_atr_multiplier", 2.0)),
                    step=0.01,
                    format="%.2f",
                    key=f"{prefix}_atr_multiplier"
                )
                st.caption(f"Target = Entry ± ATR({atr_period}) × {atr_mult}")
            elif is_r_element:
                # R Profit/R Loss: always Fixed Value, no indicator comparison
                # Force compare type to Fixed Value if switching from indicator mode
                if st.session_state.get(f"{prefix}_trigger_compare_type") != "Fixed Value":
                    st.session_state[f"{prefix}_trigger_compare_type"] = "Fixed Value"
                st.radio(
                    "Compare to",
                    ["Fixed Value"],
                    key=f"{prefix}_trigger_compare_type",
                    horizontal=True
                )
                r_label = "R" if trigger_element1 == "R Profit" else "R"
                trigger_value = st.number_input(
                    f"Value ({r_label})",
                    value=0.5,
                    min_value=0.0,
                    step=0.01,
                    format="%.2f",
                    key=f"{prefix}_trigger_value",
                    help=f"e.g., 0.7 means 0.7R {'profit' if trigger_element1 == 'R Profit' else 'loss'}"
                )
                st.caption(f"{trigger_element1} {trigger_event} {trigger_value}R")
            else:
                trigger_compare_type = st.radio(
                    "Compare to",
                    CONDITION_COMPARE_TYPES,
                    key=f"{prefix}_trigger_compare_type",
                    horizontal=True
                )

                if trigger_compare_type == "Indicator":
                    compatible_elements = get_compatible_elements(trigger_element1)
                    trigger_element2 = st.selectbox(
                        "Element 2",
                        [e for e in compatible_elements if e != trigger_element1],
                        key=f"{prefix}_trigger_element2"
                    )
                else:
                    trigger_value = st.number_input(
                        "Value/Level",
                        value=50.0,
                        key=f"{prefix}_trigger_value"
                    )

        # Conditions (optional)
        st.markdown("**Conditions (Optional)**")
        conditions_key = f"{prefix}_conditions_count"

        if conditions_key not in st.session_state:
            st.session_state[conditions_key] = 0

        col1, col2 = st.columns([1, 1])
        with col1:
            if st.button("➕ Add Condition", key=f"add_cond_{exit_type}_{group_idx}_{exit_idx}"):
                if st.session_state[conditions_key] < 5:  # Limit to 5 conditions per exit
                    st.session_state[conditions_key] += 1
        with col2:
            if st.button("➖ Remove", key=f"rem_cond_{exit_type}_{group_idx}_{exit_idx}"):
                if st.session_state[conditions_key] > 0:
                    st.session_state[conditions_key] -= 1

        # Render conditions
        for cond_idx in range(st.session_state.get(conditions_key, 0)):
            render_exit_condition(group_idx, exit_type, exit_idx, cond_idx)

def _load_exit_widget_keys(group_idx, exit_type, exit_idx, exit_config):
    """Helper: populate session state widget keys for a single target or stop"""
    prefix = f"{exit_type}_{group_idx}_{exit_idx}"

    trigger = exit_config.get('trigger', {})
    element1 = trigger.get('element1')

    # R Profit/R Loss use special group name
    if element1 in R_PROFIT_LOSS_ELEMENTS:
        st.session_state[f'{prefix}_trigger_group1'] = 'R Profit / R Loss'
    elif element1 == 'ATR Target':
        st.session_state[f'{prefix}_trigger_group1'] = 'ATR Target'
    else:
        st.session_state[f'{prefix}_trigger_group1'] = trigger.get('group', 'Price & Indicators')

    st.session_state[f'{prefix}_trigger_element1'] = element1
    st.session_state[f'{prefix}_trigger_event'] = trigger.get('event')
    st.session_state[f'{prefix}_trigger_compare_type'] = trigger.get('compare_type', 'Indicator')

    if element1 == 'ATR Target':
        st.session_state[f'{prefix}_trigger_compare_type'] = 'ATR'
        st.session_state[f'{prefix}_atr_period'] = trigger.get('atr_period', 14)
        st.session_state[f'{prefix}_atr_multiplier'] = trigger.get('atr_multiplier', 2.0)
    elif trigger.get('compare_type') == 'Indicator':
        st.session_state[f'{prefix}_trigger_element2'] = trigger.get('element2')
    else:
        st.session_state[f'{prefix}_trigger_value'] = trigger.get('value', 50.0)

    # Load conditions
    conditions = exit_config.get('conditions', [])
    st.session_state[f'{prefix}_conditions_count'] = len(conditions)

    for cond_idx, cond in enumerate(conditions):
        cond_prefix = f"{prefix}_cond_{cond_idx}"
        st.session_state[f'{cond_prefix}_group'] = cond.get('group', 'Price & Indicators')
        st.session_state[f'{cond_prefix}_element1'] = cond.get('element1')
        st.session_state[f'{cond_prefix}_operator'] = cond.get('operator')
        st.session_state[f'{cond_prefix}_compare_type'] = cond.get('compare_type', 'Indicator')

        if cond.get('compare_type') == 'Indicator':
            st.session_state[f'{cond_prefix}_element2'] = cond.get('element2')
        else:
            st.session_state[f'{cond_prefix}_value'] = cond.get('value', 50.0)


def render_exit_condition(group_idx, exit_type, exit_idx, cond_idx):
    """Render a condition for an exit"""
    col1, col2, col3 = st.columns([2, 1, 2])

    with col1:
        cond_group = st.selectbox(
            "Group",
            GROUP_NAMES,
            key=f"{exit_type}_{group_idx}_{exit_idx}_cond_{cond_idx}_group"
        )

        cond_available = get_group_elements(cond_group, _ema_count())

        cond_element1 = st.selectbox(
            "Element 1",
            cond_available,
            key=f"{exit_type}_{group_idx}_{exit_idx}_cond_{cond_idx}_element1"
        )

    with col2:
        cond_operator = st.selectbox(
            "Operator",
            CONDITION_OPERATORS,
            key=f"{exit_type}_{group_idx}_{exit_idx}_cond_{cond_idx}_operator"
        )

    with col3:
        cond_compare_type = st.radio(
            "Compare to",
            CONDITION_COMPARE_TYPES,
            key=f"{exit_type}_{group_idx}_{exit_idx}_cond_{cond_idx}_compare_type",
            horizontal=True
        )

        if cond_compare_type == "Indicator":
            cond_compatible = get_compatible_elements(cond_element1)
            cond_element2 = st.selectbox(
                "Element 2",
                [e for e in cond_compatible if e != cond_element1],
                key=f"{exit_type}_{group_idx}_{exit_idx}_cond_{cond_idx}_element2"
            )
        else:
            cond_value = st.number_input(
                "Value",
                value=50.0,
                key=f"{exit_type}_{group_idx}_{exit_idx}_cond_{cond_idx}_value"
            )


def reset_strategy_builder():
    """Reset all strategy builder state"""
    st.session_state['strategy_started'] = False
    st.session_state['strategy_direction'] = None
    st.session_state['entry_conditions_count'] = 0
    st.session_state['exit_groups'] = []
    st.session_state['initial_stop'] = None
    st.session_state.pop('initial_stop_type', None)
    st.session_state.pop('initial_stop_atr_period', None)
    st.session_state.pop('initial_stop_atr_multiplier', None)
    st.session_state['strategy_name_input'] = ""
    st.session_state['editing_strategy'] = False
    st.session_state['editing_strategy_idx'] = None
    st.session_state.pop('max_positions_unlimited', None)
    st.session_state.pop('max_positions_count', None)
    # Reset strategy builder indicator settings to defaults
    pfx = "sb_"
    for key in ['rsi_window', 'bb_upper_period', 'bb_upper_stdev', 'bb_mid_period',
                'bb_lower_period', 'bb_lower_stdev', 'kc_upper_ema', 'kc_upper_mult',
                'kc_mid_ema', 'kc_lower_ema', 'kc_lower_mult', 'kc_atr_period',
                'stoch_k_period', 'stoch_k_smooth', 'stoch_d_smooth', 'adx_period',
                'atr_period', 'macd_fast', 'macd_slow', 'macd_signal',
                'supertrend_period', 'supertrend_multiplier',
                'dc_upper_period', 'dc_mid_period', 'dc_lower_period', 'dc_offset',
                'psar_af_start', 'psar_af_increment', 'psar_af_max']:
        st.session_state.pop(f'{pfx}{key}', None)
    st.session_state.pop(f'{pfx}ema_periods', None)
    st.rerun()
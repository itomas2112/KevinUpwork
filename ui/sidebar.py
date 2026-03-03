"""
Sidebar UI components
"""
import streamlit as st
from data.helpers import (
    PRIMARY_SECONDARY_MAP, PRIMARY_LIST, ALL_UNIQUE_SECONDARIES,
    expand_selection,
)

CHARTING_MODES = [
    "Specified Primary",
    "Specified Secondary",
    "Secondary Across Primaries",
]


def render_sidebar():
    """Render the complete sidebar with all controls"""

    # Analysis Mode Toggle
    analysis_mode = st.sidebar.radio(
        "Analysis Mode",
        options=["15m", "1H"],
        index=0,
        key="analysis_mode",
        horizontal=True,
    )

    # Pattern Parameters — multi-row selection
    with st.sidebar.expander("Pattern Parameters", expanded=True):
        selections = st.session_state['charting_selections']

        rows_to_remove = []
        for idx, sel in enumerate(selections):
            # Mode
            current_mode = sel.get("mode", "Specified Secondary")
            mode = st.selectbox(
                "Mode",
                CHARTING_MODES,
                index=CHARTING_MODES.index(current_mode) if current_mode in CHARTING_MODES else 1,
                key=f"chart_sel_{idx}_mode",
            )
            selections[idx]["mode"] = mode

            # Pattern Type
            ptype_options = ["Bullish", "Bearish"]
            current_ptype = sel.get("pattern_type", "Bullish")
            ptype = st.selectbox(
                "Pattern Type",
                ptype_options,
                index=ptype_options.index(current_ptype) if current_ptype in ptype_options else 0,
                key=f"chart_sel_{idx}_pattern_type",
            )
            selections[idx]["pattern_type"] = ptype

            # Primary (visible for Specified Primary and Specified Secondary)
            show_primary = mode in ("Specified Primary", "Specified Secondary")
            show_secondary = mode in ("Specified Secondary", "Secondary Across Primaries")

            if show_primary:
                current_primary = sel.get("primary")
                primary_idx = PRIMARY_LIST.index(current_primary) if current_primary in PRIMARY_LIST else 0
                primary = st.selectbox(
                    "Primary",
                    PRIMARY_LIST,
                    index=primary_idx,
                    key=f"chart_sel_{idx}_primary",
                )
                selections[idx]["primary"] = primary

            # Secondary
            if show_secondary:
                if mode == "Specified Secondary":
                    chosen_primary = selections[idx].get("primary") or PRIMARY_LIST[0]
                    sec_options = PRIMARY_SECONDARY_MAP.get(chosen_primary, [])
                else:
                    sec_options = ALL_UNIQUE_SECONDARIES

                if sec_options:
                    current_sec = sel.get("secondary")
                    sec_idx = sec_options.index(current_sec) if current_sec in sec_options else 0
                    secondary = st.selectbox(
                        "Secondary",
                        sec_options,
                        index=sec_idx,
                        key=f"chart_sel_{idx}_secondary",
                    )
                    selections[idx]["secondary"] = secondary

            # Delete button
            if st.button("Remove", key=f"chart_sel_{idx}_remove", type="primary"):
                rows_to_remove.append(idx)

            if idx < len(selections) - 1:
                st.markdown("---")

        # Remove rows
        if rows_to_remove:
            for i in sorted(rows_to_remove, reverse=True):
                st.session_state['charting_selections'].pop(i)
            if not st.session_state['charting_selections']:
                st.session_state['charting_selections'] = [{
                    "mode": "Specified Secondary",
                    "pattern_type": "Bullish",
                    "primary": None,
                    "secondary": None,
                }]
            st.rerun()

        # Add selection button
        if st.button("+ Add Selection", key="chart_add_selection"):
            st.session_state['charting_selections'].append({
                "mode": "Specified Secondary",
                "pattern_type": "Bullish",
                "primary": None,
                "secondary": None,
            })
            st.rerun()

    # Indicators
    with st.sidebar.expander("Indicators"):
        show_ichimoku = st.checkbox("Show Ichimoku Cloud", value=False)
        show_bb = st.checkbox("Show Bollinger Bands", value=False)
        show_kc = st.checkbox("Show Keltner Channel", value=False)

    # Chart Tools
    draw_mode = st.sidebar.checkbox("Draw Boxes on Charts", value=False, key="draw_mode")

    show_tenkan_kijun = False

    # Custom Strategies (single selection) - FILTERED BY ALL SELECTED PATTERNS
    charting_selections = st.session_state['charting_selections']

    # Build set of all pattern strings from expanded selections
    all_pattern_strings = set()
    for sel in charting_selections:
        combos = expand_selection(sel)
        for _ptype, primary, secondary in combos:
            all_pattern_strings.add(f"{primary} → {secondary}")

    if st.session_state['saved_strategies'] and all_pattern_strings:
        # Filter strategies that apply to any selected pattern
        filtered_strategies = [
            (idx, strategy)
            for idx, strategy in enumerate(st.session_state['saved_strategies'])
            if any(
                pat in all_pattern_strings
                for pat in strategy.get('patterns', [])
            )
        ]

        if filtered_strategies:
            with st.sidebar.expander("Custom Strategies", expanded=True):
                strategy_options = ["None"] + [
                    strategy.get('strategy_name', f'Strategy_{idx + 1}')
                    for idx, strategy in filtered_strategies
                ]

                # Map display indices to actual strategy indices
                strategy_index_map = {i: idx for i, (idx, _) in enumerate(filtered_strategies, 1)}

                if 'selected_custom_strategy_idx' not in st.session_state:
                    st.session_state['selected_custom_strategy_idx'] = 0

                selected_option = st.radio(
                    "Select one strategy:",
                    options=range(len(strategy_options)),
                    format_func=lambda x: strategy_options[x],
                    index=st.session_state['selected_custom_strategy_idx'],
                    key="custom_strategy_radio"
                )

                # Check if selection changed and update
                if selected_option != st.session_state['selected_custom_strategy_idx']:
                    st.session_state['selected_custom_strategy_idx'] = selected_option
                    # Store the actual strategy index
                    if selected_option > 0:
                        st.session_state['selected_custom_strategy_actual_idx'] = strategy_index_map[selected_option]
                    else:
                        st.session_state['selected_custom_strategy_actual_idx'] = None
                    st.rerun()

    # Indicator Parameters
    show_1h = analysis_mode == "1H"
    params_1h = render_timeframe_parameters("1H") if show_1h else None
    params_15m = render_timeframe_parameters("15m") if not show_1h else None

    return {
        'analysis_mode': analysis_mode,
        'charting_selections': charting_selections,
        'show_ichimoku': show_ichimoku,
        'show_bb': show_bb,
        'show_kc': show_kc,
        'show_tenkan_kijun': show_tenkan_kijun,
        'draw_mode': draw_mode,
        'params_1h': params_1h,
        'params_15m': params_15m
    }

def render_timeframe_parameters(timeframe):
    """Render indicator parameters for a specific timeframe"""
    st.sidebar.header("Indicator Settings")

    key_prefix = timeframe.lower().replace('h', '_h').replace('m', '_m')

    params = {}

    with st.sidebar.expander("RSI"):
        params['rsi_window'] = st.slider(
            f"RSI Period", 5, 50, 14,
            key=f"rsi_{key_prefix}"
        )
        params['rsi_upper_1'] = st.number_input(
            "Upper Line 1", 0.0, 100.0, 70.0, step=1.0,
            key=f"rsi_u1_{key_prefix}"
        )
        params['rsi_upper_2'] = st.number_input(
            "Upper Line 2", 0.0, 100.0, 67.0, step=1.0,
            key=f"rsi_u2_{key_prefix}"
        )
        params['rsi_lower_1'] = st.number_input(
            "Lower Line 1", 0.0, 100.0, 33.0, step=1.0,
            key=f"rsi_l1_{key_prefix}"
        )
        params['rsi_lower_2'] = st.number_input(
            "Lower Line 2", 0.0, 100.0, 30.0, step=1.0,
            key=f"rsi_l2_{key_prefix}"
        )

    with st.sidebar.expander("CMB"):
        # Dynamic CMB lines: default 0, user can add/remove
        cmb_state_key = f"cmb_lines_{key_prefix}"
        if cmb_state_key not in st.session_state:
            st.session_state[cmb_state_key] = []

        # Render existing lines with remove buttons
        lines_to_remove = []
        for idx, line_val in enumerate(st.session_state[cmb_state_key]):
            line_col, remove_col = st.columns([3, 1])
            with line_col:
                new_val = st.number_input(
                    f"Line {idx + 1}", 0.0, 200.0, float(line_val), step=1.0,
                    key=f"cmb_line_{key_prefix}_{idx}"
                )
                st.session_state[cmb_state_key][idx] = new_val
            with remove_col:
                st.markdown("<br>", unsafe_allow_html=True)
                if st.button("X", key=f"cmb_rm_{key_prefix}_{idx}"):
                    lines_to_remove.append(idx)

        # Remove lines (in reverse to preserve indices)
        if lines_to_remove:
            for idx in sorted(lines_to_remove, reverse=True):
                st.session_state[cmb_state_key].pop(idx)
            st.rerun()

        # Add line button
        if st.button("+ Add Line", key=f"cmb_add_{key_prefix}"):
            st.session_state[cmb_state_key].append(50.0)
            st.rerun()

        params['cmb_lines'] = list(st.session_state[cmb_state_key])

    with st.sidebar.expander("Stochastic"):
        params['stoch_k_period'] = st.number_input(
            "%K Period", 1, 100, 14, step=1,
            key=f"stoch_kp_{key_prefix}"
        )
        params['stoch_k_smooth'] = st.number_input(
            "%K Smoothing", 1, 50, 3, step=1,
            key=f"stoch_ks_{key_prefix}"
        )
        params['stoch_d_smooth'] = st.number_input(
            "%D Smoothing", 1, 50, 3, step=1,
            key=f"stoch_ds_{key_prefix}"
        )

    with st.sidebar.expander("Ichimoku"):
        params['ichi_show_tenkan'] = st.checkbox(
            "Show Tenkan", value=True, key=f"ichi_tenkan_{key_prefix}"
        )
        params['ichi_show_kijun'] = st.checkbox(
            "Show Kijun", value=True, key=f"ichi_kijun_{key_prefix}"
        )
        params['ichi_show_senkou_a'] = st.checkbox(
            "Show Senkou A", value=True, key=f"ichi_senkou_a_{key_prefix}"
        )
        params['ichi_show_senkou_b'] = st.checkbox(
            "Show Senkou B", value=True, key=f"ichi_senkou_b_{key_prefix}"
        )
        params['ichi_show_chikou'] = st.checkbox(
            "Show Chikou", value=True, key=f"ichi_chikou_{key_prefix}"
        )

    with st.sidebar.expander("Bollinger Bands"):
        params['bb_show_upper'] = st.checkbox(
            "Show Upper Band", value=True, key=f"bb_upper_{key_prefix}"
        )
        params['bb_show_middle'] = st.checkbox(
            "Show Middle Band", value=True, key=f"bb_mid_{key_prefix}"
        )
        params['bb_show_lower'] = st.checkbox(
            "Show Lower Band", value=True, key=f"bb_lower_{key_prefix}"
        )
        params['bb_period'] = st.number_input(
            f"BB Period", 5, 100, 20, step=1,
            key=f"bb_p_{key_prefix}"
        )
        params['bb_stdev'] = st.number_input(
            f"BB StdDev", 0.5, 5.0, 2.0, step=0.1,
            key=f"bb_s_{key_prefix}"
        )

    with st.sidebar.expander("Keltner Channel"):
        params['kc_show_upper'] = st.checkbox(
            "Show Upper Band", value=True, key=f"kc_upper_{key_prefix}"
        )
        params['kc_show_middle'] = st.checkbox(
            "Show Middle Band", value=True, key=f"kc_mid_{key_prefix}"
        )
        params['kc_show_lower'] = st.checkbox(
            "Show Lower Band", value=True, key=f"kc_lower_{key_prefix}"
        )
        params['kc_ema_period'] = st.number_input(
            f"KC EMA Period", 5, 100, 20, step=1,
            key=f"kc_ema_{key_prefix}"
        )
        params['kc_atr_period'] = st.number_input(
            f"KC ATR Period", 5, 100, 10, step=1,
            key=f"kc_atr_{key_prefix}"
        )
        params['kc_atr_mult'] = st.number_input(
            f"KC ATR Mult", 0.5, 5.0, 2.0, step=0.1,
            key=f"kc_mult_{key_prefix}"
        )

    return params

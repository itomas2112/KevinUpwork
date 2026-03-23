"""
Sidebar UI components
"""
import streamlit as st
from datetime import date
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

    # Historical Data Aggregation
    base_tf = st.session_state.get("base_timeframe", "15m")
    with st.sidebar.expander("Historical Data Aggregation", expanded=False):
        ALL_AGG = ["15m", "1H", "4H", "1D"]
        base_idx = ALL_AGG.index(base_tf)
        agg_options = ALL_AGG[base_idx:]
        current_agg = st.session_state.get("_agg_timeframe", base_tf)
        if current_agg not in agg_options:
            current_agg = agg_options[0]
            st.session_state["_agg_timeframe"] = current_agg
        agg_tf = st.selectbox("Timeframe", agg_options,
                               index=agg_options.index(current_agg) if current_agg in agg_options else 0,
                               key="agg_tf_select")
        if st.button("Apply", key="agg_apply"):
            if "df_raw" in st.session_state:
                from data.loader import resample_ohlc
                st.session_state["df_ohlc"] = resample_ohlc(st.session_state["df_raw"], agg_tf, base_timeframe=base_tf)
                st.session_state["_agg_timeframe"] = agg_tf
                # Clear indicator caches so they recompute on new timeframe
                for k in ["df_features", "_indicator_params", "_indicator_params_data_fp"]:
                    st.session_state.pop(k, None)
                # Clear backtest caches
                for k in list(st.session_state.keys()):
                    if k.startswith("_bt_cache_") or k.startswith("_perf_") or k.startswith("_gs_") or k.startswith("_cv_") or k.startswith("_test_"):
                        st.session_state.pop(k, None)
                st.success(f"Data aggregated to **{agg_tf}**")
                st.rerun()
            else:
                st.warning("Upload OHLC data first.")
        if st.session_state.get("_agg_timeframe", base_tf) != base_tf:
            st.caption(f"Currently using: **{st.session_state['_agg_timeframe']}**")

    # Training Set — required before any calculations
    with st.sidebar.expander("Training Set", expanded=False):
        date_col1, date_col2 = st.columns(2)
        with date_col1:
            global_start_date = st.date_input(
                "Start Date",
                value=st.session_state.get("global_start_date"),
                key="global_start_date",
            )
        with date_col2:
            global_end_date = st.date_input(
                "End Date",
                value=st.session_state.get("global_end_date"),
                key="global_end_date",
            )
        if st.button("Apply Training Set", key="apply_date_range", type="primary"):
            st.session_state['date_range_applied'] = True
            st.rerun()
        if st.session_state.get('date_range_applied'):
            st.success(f"{global_start_date} → {global_end_date}")
        else:
            st.warning("Set dates and click Apply to proceed.")

    # Test Set — used only by Strategy Testing tab
    with st.sidebar.expander("Test Set", expanded=False):
        test_col1, test_col2 = st.columns(2)
        with test_col1:
            test_start_date = st.date_input(
                "Start Date",
                value=st.session_state.get("test_start_date"),
                key="test_start_date",
            )
        with test_col2:
            test_end_date = st.date_input(
                "End Date",
                value=st.session_state.get("test_end_date"),
                key="test_end_date",
            )
        if st.button("Apply Test Set", key="apply_test_set", type="primary"):
            st.session_state['test_set_applied'] = True
            st.rerun()
        if st.session_state.get('test_set_applied'):
            st.success(f"{test_start_date} → {test_end_date}")
        else:
            st.info("Optional: set dates for Strategy Testing tab.")

    # Pattern Parameters — multi-row selection
    with st.sidebar.expander("Pattern Parameters", expanded=False):
        selections = st.session_state['charting_selections']

        # Generation counter to guarantee fresh widget keys after deletions
        gen = st.session_state.get("_chart_sel_gen", 0)

        rows_to_remove = []
        for idx, sel in enumerate(selections):
            kp = f"chart_sel_g{gen}_{idx}"

            # Mode
            current_mode = sel.get("mode", "Specified Secondary")
            mode = st.selectbox(
                "Mode",
                CHARTING_MODES,
                index=CHARTING_MODES.index(current_mode) if current_mode in CHARTING_MODES else 1,
                key=f"{kp}_mode",
            )
            selections[idx]["mode"] = mode

            # Pattern Type
            ptype_options = ["Bullish", "Bearish"]
            current_ptype = sel.get("pattern_type", "Bullish")
            ptype = st.selectbox(
                "Pattern Type",
                ptype_options,
                index=ptype_options.index(current_ptype) if current_ptype in ptype_options else 0,
                key=f"{kp}_pattern_type",
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
                    key=f"{kp}_primary",
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
                        key=f"{kp}_secondary",
                    )
                    selections[idx]["secondary"] = secondary

            # Delete button
            if st.button("Remove", key=f"{kp}_remove", type="primary"):
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
            # Bump generation so all widget keys are fresh on next render
            st.session_state["_chart_sel_gen"] = gen + 1
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

    # Price Overlays
    with st.sidebar.expander("Price Overlays"):
        show_ichimoku = st.checkbox("Show Ichimoku Cloud",
            value=st.session_state.get("overlay_ichimoku", False), key="overlay_ichimoku")
        show_bb = st.checkbox("Show Bollinger Bands",
            value=st.session_state.get("overlay_bb", False), key="overlay_bb")
        show_kc = st.checkbox("Show Keltner Channel",
            value=st.session_state.get("overlay_kc", False), key="overlay_kc")
        show_supertrend = st.checkbox("Show Supertrend",
            value=st.session_state.get("overlay_supertrend", False), key="overlay_supertrend")
        show_ema = st.checkbox("Show EMA Overlay",
            value=st.session_state.get("overlay_ema", False), key="overlay_ema")
        show_donchian = st.checkbox("Show Donchian Channel",
            value=st.session_state.get("overlay_donchian", False), key="overlay_donchian")
        show_psar = st.checkbox("Show Parabolic SAR",
            value=st.session_state.get("overlay_psar", False), key="overlay_psar")

    # Oscillator Panel Toggles
    with st.sidebar.expander("Oscillator Panels", expanded=False):
        show_rsi = st.checkbox("Show RSI Panel",
            value=st.session_state.get("show_rsi_panel", True), key="show_rsi_panel")
        show_cmb = st.checkbox("Show CMB Panel",
            value=st.session_state.get("show_cmb_panel", True), key="show_cmb_panel")
        show_stoch = st.checkbox("Show Stochastic Panel",
            value=st.session_state.get("show_stoch_panel", True), key="show_stoch_panel")
        show_adx = st.checkbox("Show ADX Panel",
            value=st.session_state.get("show_adx_panel", False), key="show_adx_panel")
        show_atr = st.checkbox("Show ATR Panel",
            value=st.session_state.get("show_atr_panel", False), key="show_atr_panel")
        show_macd = st.checkbox("Show MACD Panel",
            value=st.session_state.get("show_macd_panel", False), key="show_macd_panel")
        show_obv = st.checkbox("Show OBV Panel",
            value=st.session_state.get("show_obv_panel", False), key="show_obv_panel")
        show_accdist = st.checkbox("Show Acc/Dist Panel",
            value=st.session_state.get("show_accdist_panel", False), key="show_accdist_panel")

    # Chart Tools
    draw_mode = st.sidebar.checkbox("Shade Sections on Price Chart", value=False, key="draw_mode")
    chart_height = st.sidebar.slider("Chart Height", min_value=600, max_value=2000, value=920, step=20, key="chart_height")

    show_tenkan_kijun = False

    # Custom Strategies (single selection) - FILTERED BY ALL SELECTED PATTERNS
    charting_selections = st.session_state['charting_selections']

    # Build set of all pattern strings from expanded selections
    all_pattern_strings = set()
    for sel in charting_selections:
        combos = expand_selection(sel)
        for _ptype, primary, secondary in combos:
            all_pattern_strings.add(f"{primary} → {secondary}")

    # Handle deselect request from Strategy Builder tab
    if st.session_state.pop('_deselect_strategy', False):
        st.session_state['selected_custom_strategy_idx'] = 0
        st.session_state['selected_custom_strategy_actual_idx'] = None
        if 'custom_strategy_radio' in st.session_state:
            st.session_state['custom_strategy_radio'] = 0

    if st.session_state['saved_strategies']:
        # Filter strategies that apply to any selected pattern
        # If no patterns are expanded yet (no DRM uploaded), show all strategies
        if all_pattern_strings:
            filtered_strategies = [
                (idx, strategy)
                for idx, strategy in enumerate(st.session_state['saved_strategies'])
                if not strategy.get('patterns') or any(
                    pat in all_pattern_strings
                    for pat in strategy.get('patterns', [])
                )
            ]
        else:
            filtered_strategies = list(enumerate(st.session_state['saved_strategies']))

        if filtered_strategies:
            with st.sidebar.expander("Custom Strategies", expanded=False):
                strategy_options = ["None"] + [
                    strategy.get('strategy_name', f'Strategy_{idx + 1}')
                    for idx, strategy in filtered_strategies
                ]

                # Map display indices to actual strategy indices
                strategy_index_map = {i: idx for i, (idx, _) in enumerate(filtered_strategies, 1)}

                if 'selected_custom_strategy_idx' not in st.session_state:
                    st.session_state['selected_custom_strategy_idx'] = 0

                # Clamp index to valid range (filtered list may have shrunk)
                current_idx = st.session_state['selected_custom_strategy_idx']
                if current_idx >= len(strategy_options):
                    current_idx = 0
                    st.session_state['selected_custom_strategy_idx'] = 0
                    st.session_state['selected_custom_strategy_actual_idx'] = None

                selected_option = st.radio(
                    "Select one strategy:",
                    options=range(len(strategy_options)),
                    format_func=lambda x: strategy_options[x],
                    index=current_idx,
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
        else:
            # No strategies match current patterns — reset selection to avoid being stuck
            if st.session_state.get('selected_custom_strategy_idx', 0) > 0:
                st.session_state['selected_custom_strategy_idx'] = 0
                st.session_state['selected_custom_strategy_actual_idx'] = None
                if 'custom_strategy_radio' in st.session_state:
                    st.session_state['custom_strategy_radio'] = 0

    # Detect if a strategy is actively selected (for disabling sidebar indicator settings)
    strategy_active = False
    if st.session_state.get('selected_custom_strategy_idx', 0) > 0:
        actual_idx = st.session_state.get('selected_custom_strategy_actual_idx')
        if actual_idx is not None and actual_idx < len(st.session_state['saved_strategies']):
            saved_settings = st.session_state['saved_strategies'][actual_idx].get('indicator_settings')
            if saved_settings:
                strategy_active = True

    # Indicator Parameters
    # In strategy mode, sidebar settings are disabled — strategy's saved settings are used
    tf_label = st.session_state.get("_agg_timeframe", "15m")
    params = render_timeframe_parameters(tf_label, disabled=strategy_active)

    return {
        'global_start_date': global_start_date,
        'global_end_date': global_end_date,
        'date_range_applied': st.session_state.get('date_range_applied', False),
        'charting_selections': charting_selections,
        'show_ichimoku': show_ichimoku,
        'show_bb': show_bb,
        'show_kc': show_kc,
        'show_supertrend': show_supertrend,
        'show_ema': show_ema,
        'show_donchian': show_donchian,
        'show_psar': show_psar,
        'show_rsi': show_rsi,
        'show_cmb': show_cmb,
        'show_stoch': show_stoch,
        'show_adx': show_adx,
        'show_atr': show_atr,
        'show_macd': show_macd,
        'show_obv': show_obv,
        'show_accdist': show_accdist,
        'show_tenkan_kijun': show_tenkan_kijun,
        'draw_mode': draw_mode,
        'chart_height': chart_height,
        'params': params,
        'test_start_date': test_start_date,
        'test_end_date': test_end_date,
        'test_set_applied': st.session_state.get('test_set_applied', False),
    }

def render_timeframe_parameters(timeframe, disabled=False):
    """Render indicator parameters for a specific timeframe"""
    if disabled:
        st.sidebar.header("Indicator Settings (Strategy Mode)")
        st.sidebar.caption("*Using indicator settings saved with the strategy. These controls are locked. Switch to viewing mode (deselect strategy) to change.*")
    else:
        st.sidebar.header("Indicator Settings (Viewing Mode)")

    key_prefix = timeframe.lower().replace('h', '_h').replace('m', '_m')

    params = {}

    with st.sidebar.expander("RSI"):
        params['rsi_window'] = st.slider(
            f"RSI Period", 5, 50, 14,
            key=f"rsi_{key_prefix}",
            disabled=disabled,
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

        # Generation counter to guarantee fresh widget keys after deletions
        cmb_gen_key = f"_cmb_gen_{key_prefix}"
        cmb_gen = st.session_state.get(cmb_gen_key, 0)

        # Render existing lines with remove buttons
        lines_to_remove = []
        for idx, line_val in enumerate(st.session_state[cmb_state_key]):
            line_col, remove_col = st.columns([3, 1])
            with line_col:
                new_val = st.number_input(
                    f"Line {idx + 1}", -200.0, 200.0, float(line_val), step=1.0,
                    key=f"cmb_line_{key_prefix}_g{cmb_gen}_{idx}"
                )
                st.session_state[cmb_state_key][idx] = new_val
            with remove_col:
                st.markdown("<br>", unsafe_allow_html=True)
                if st.button("X", key=f"cmb_rm_{key_prefix}_g{cmb_gen}_{idx}"):
                    lines_to_remove.append(idx)

        # Remove lines (in reverse to preserve indices)
        if lines_to_remove:
            for idx in sorted(lines_to_remove, reverse=True):
                st.session_state[cmb_state_key].pop(idx)
            st.session_state[cmb_gen_key] = cmb_gen + 1
            st.rerun()

        # Add line button
        if st.button("+ Add Line", key=f"cmb_add_{key_prefix}"):
            st.session_state[cmb_state_key].append(50.0)
            st.rerun()

        params['cmb_lines'] = list(st.session_state[cmb_state_key])

    with st.sidebar.expander("Stochastic"):
        params['stoch_k_period'] = st.number_input(
            "%K Period", 1, 100, 14, step=1,
            key=f"stoch_kp_{key_prefix}",
            disabled=disabled,
        )
        params['stoch_k_smooth'] = st.number_input(
            "%K Smoothing", 1, 50, 3, step=1,
            key=f"stoch_ks_{key_prefix}",
            disabled=disabled,
        )
        params['stoch_d_smooth'] = st.number_input(
            "%D Smoothing", 1, 50, 3, step=1,
            key=f"stoch_ds_{key_prefix}",
            disabled=disabled,
        )

    with st.sidebar.expander("ADX"):
        params['adx_period'] = st.number_input(
            "ADX Period", 5, 100, 14, step=1,
            key=f"adx_p_{key_prefix}",
            disabled=disabled,
        )

    with st.sidebar.expander("ATR"):
        params['atr_period'] = st.number_input(
            "ATR Period", 5, 100, 14, step=1,
            key=f"atr_p_{key_prefix}",
            disabled=disabled,
        )

    with st.sidebar.expander("MACD"):
        params['macd_fast'] = st.number_input(
            "Fast Period", 2, 100, 12, step=1,
            key=f"macd_fast_{key_prefix}",
            disabled=disabled,
        )
        params['macd_slow'] = st.number_input(
            "Slow Period", 2, 200, 26, step=1,
            key=f"macd_slow_{key_prefix}",
            disabled=disabled,
        )
        params['macd_signal'] = st.number_input(
            "Signal Period", 2, 100, 9, step=1,
            key=f"macd_sig_{key_prefix}",
            disabled=disabled,
        )

    with st.sidebar.expander("Supertrend"):
        params['supertrend_period'] = st.number_input(
            "Period", 1, 100, 10, step=1,
            key=f"st_p_{key_prefix}",
            disabled=disabled,
        )
        params['supertrend_multiplier'] = st.number_input(
            "Multiplier", 0.5, 10.0, 3.0, step=0.01, format="%.2f",
            key=f"st_m_{key_prefix}",
            disabled=disabled,
        )

    with st.sidebar.expander("EMA Overlay"):
        # Dynamic EMA periods: add/remove like CMB lines
        ema_state_key = f"ema_periods_{key_prefix}"
        if ema_state_key not in st.session_state:
            st.session_state[ema_state_key] = []

        # Generation counter to guarantee fresh widget keys after deletions
        ema_gen_key = f"_ema_gen_{key_prefix}"
        ema_gen = st.session_state.get(ema_gen_key, 0)

        emas_to_remove = []
        for idx, ema_val in enumerate(st.session_state[ema_state_key]):
            line_col, remove_col = st.columns([3, 1])
            with line_col:
                new_val = st.number_input(
                    f"EMA {idx + 1} Period", 2, 500, int(ema_val), step=1,
                    key=f"ema_p_{key_prefix}_g{ema_gen}_{idx}",
                    disabled=disabled,
                )
                st.session_state[ema_state_key][idx] = new_val
            with remove_col:
                st.markdown("<br>", unsafe_allow_html=True)
                if st.button("X", key=f"ema_rm_{key_prefix}_g{ema_gen}_{idx}"):
                    emas_to_remove.append(idx)

        if emas_to_remove:
            for idx in sorted(emas_to_remove, reverse=True):
                st.session_state[ema_state_key].pop(idx)
            st.session_state[ema_gen_key] = ema_gen + 1
            st.rerun()

        if st.button("+ Add EMA", key=f"ema_add_{key_prefix}"):
            st.session_state[ema_state_key].append(20)
            st.rerun()

        params['ema_periods'] = list(st.session_state[ema_state_key])

    with st.sidebar.expander("Ichimoku"):
        params['ichi_show_tenkan'] = st.checkbox(
            "Show Tenkan",
            value=st.session_state.get(f"ichi_tenkan_{key_prefix}", True),
            key=f"ichi_tenkan_{key_prefix}"
        )
        params['ichi_show_kijun'] = st.checkbox(
            "Show Kijun",
            value=st.session_state.get(f"ichi_kijun_{key_prefix}", True),
            key=f"ichi_kijun_{key_prefix}"
        )
        params['ichi_show_senkou_a'] = st.checkbox(
            "Show Senkou A",
            value=st.session_state.get(f"ichi_senkou_a_{key_prefix}", True),
            key=f"ichi_senkou_a_{key_prefix}"
        )
        params['ichi_show_senkou_b'] = st.checkbox(
            "Show Senkou B",
            value=st.session_state.get(f"ichi_senkou_b_{key_prefix}", True),
            key=f"ichi_senkou_b_{key_prefix}"
        )
        params['ichi_show_chikou'] = st.checkbox(
            "Show Chikou",
            value=st.session_state.get(f"ichi_chikou_{key_prefix}", True),
            key=f"ichi_chikou_{key_prefix}"
        )
        st.divider()
        st.caption("Strategy decision lines (un-displaced)")
        params['ichi_show_senkou_a_current'] = st.checkbox(
            "Show Senkou A (current)",
            value=st.session_state.get(f"ichi_senkou_a_current_{key_prefix}", False),
            key=f"ichi_senkou_a_current_{key_prefix}"
        )
        params['ichi_show_senkou_b_current'] = st.checkbox(
            "Show Senkou B (current)",
            value=st.session_state.get(f"ichi_senkou_b_current_{key_prefix}", False),
            key=f"ichi_senkou_b_current_{key_prefix}"
        )
        params['ichi_show_chikou_decision'] = st.checkbox(
            "Show Chikou (decision)",
            value=st.session_state.get(f"ichi_chikou_decision_{key_prefix}", False),
            key=f"ichi_chikou_decision_{key_prefix}"
        )

    with st.sidebar.expander("Bollinger Bands"):
        params['bb_show_upper'] = st.checkbox(
            "Show Upper Band",
            value=st.session_state.get(f"bb_upper_{key_prefix}", True),
            key=f"bb_upper_{key_prefix}"
        )
        params['bb_show_middle'] = st.checkbox(
            "Show Middle Band",
            value=st.session_state.get(f"bb_mid_{key_prefix}", True),
            key=f"bb_mid_{key_prefix}"
        )
        params['bb_show_lower'] = st.checkbox(
            "Show Lower Band",
            value=st.session_state.get(f"bb_lower_{key_prefix}", True),
            key=f"bb_lower_{key_prefix}"
        )
        st.divider()
        st.caption("**Upper Band**")
        params['bb_upper_period'] = st.number_input(
            "Upper Period", 5, 100, 20, step=1,
            key=f"bb_up_p_{key_prefix}",
            disabled=disabled,
        )
        params['bb_upper_stdev'] = st.number_input(
            "Upper StdDev", 0.5, 5.0, 2.0, step=0.01, format="%.2f",
            key=f"bb_up_s_{key_prefix}",
            disabled=disabled,
        )
        st.caption("**Middle Band**")
        params['bb_mid_period'] = st.number_input(
            "Middle Period", 5, 100, 20, step=1,
            key=f"bb_mid_p_{key_prefix}",
            disabled=disabled,
        )
        st.caption("**Lower Band**")
        params['bb_lower_period'] = st.number_input(
            "Lower Period", 5, 100, 20, step=1,
            key=f"bb_lo_p_{key_prefix}",
            disabled=disabled,
        )
        params['bb_lower_stdev'] = st.number_input(
            "Lower StdDev", 0.5, 5.0, 2.0, step=0.01, format="%.2f",
            key=f"bb_lo_s_{key_prefix}",
            disabled=disabled,
        )

    with st.sidebar.expander("Keltner Channel"):
        params['kc_show_upper'] = st.checkbox(
            "Show Upper Band",
            value=st.session_state.get(f"kc_upper_{key_prefix}", True),
            key=f"kc_upper_{key_prefix}"
        )
        params['kc_show_middle'] = st.checkbox(
            "Show Middle Band",
            value=st.session_state.get(f"kc_mid_{key_prefix}", True),
            key=f"kc_mid_{key_prefix}"
        )
        params['kc_show_lower'] = st.checkbox(
            "Show Lower Band",
            value=st.session_state.get(f"kc_lower_{key_prefix}", True),
            key=f"kc_lower_{key_prefix}"
        )
        st.divider()
        params['kc_atr_period'] = st.number_input(
            "ATR Period", 5, 100, 10, step=1,
            key=f"kc_atr_{key_prefix}",
            disabled=disabled,
        )
        st.caption("**Upper Band**")
        params['kc_upper_ema'] = st.number_input(
            "Upper EMA Period", 5, 100, 20, step=1,
            key=f"kc_up_ema_{key_prefix}",
            disabled=disabled,
        )
        params['kc_upper_mult'] = st.number_input(
            "Upper ATR Mult", 0.5, 5.0, 2.0, step=0.01, format="%.2f",
            key=f"kc_up_mult_{key_prefix}",
            disabled=disabled,
        )
        st.caption("**Middle Band**")
        params['kc_mid_ema'] = st.number_input(
            "Middle EMA Period", 5, 100, 20, step=1,
            key=f"kc_mid_ema_{key_prefix}",
            disabled=disabled,
        )
        st.caption("**Lower Band**")
        params['kc_lower_ema'] = st.number_input(
            "Lower EMA Period", 5, 100, 20, step=1,
            key=f"kc_lo_ema_{key_prefix}",
            disabled=disabled,
        )
        params['kc_lower_mult'] = st.number_input(
            "Lower ATR Mult", 0.5, 5.0, 2.0, step=0.01, format="%.2f",
            key=f"kc_lo_mult_{key_prefix}",
            disabled=disabled,
        )

    with st.sidebar.expander("Donchian Channel"):
        params['dc_show_upper'] = st.checkbox(
            "Show Upper Band",
            value=st.session_state.get(f"dc_upper_{key_prefix}", True),
            key=f"dc_upper_{key_prefix}"
        )
        params['dc_show_middle'] = st.checkbox(
            "Show Middle Band",
            value=st.session_state.get(f"dc_mid_{key_prefix}", True),
            key=f"dc_mid_{key_prefix}"
        )
        params['dc_show_lower'] = st.checkbox(
            "Show Lower Band",
            value=st.session_state.get(f"dc_lower_{key_prefix}", True),
            key=f"dc_lower_{key_prefix}"
        )
        st.divider()
        st.caption("**Upper Band**")
        params['dc_upper_period'] = st.number_input(
            "Upper Period", 5, 200, 20, step=1,
            key=f"dc_up_p_{key_prefix}",
            disabled=disabled,
        )
        st.caption("**Middle Band**")
        params['dc_mid_period'] = st.number_input(
            "Middle Period", 5, 200, 20, step=1,
            key=f"dc_mid_p_{key_prefix}",
            disabled=disabled,
        )
        st.caption("**Lower Band**")
        params['dc_lower_period'] = st.number_input(
            "Lower Period", 5, 200, 20, step=1,
            key=f"dc_lo_p_{key_prefix}",
            disabled=disabled,
        )
        st.divider()
        params['dc_offset'] = st.number_input(
            "Offset / Shift", -50, 50, 0, step=1,
            key=f"dc_off_{key_prefix}",
            disabled=disabled,
        )

    with st.sidebar.expander("Parabolic SAR"):
        params['psar_af_start'] = st.number_input(
            "AF Start", 0.001, 0.5, 0.02, step=0.01, format="%.3f",
            key=f"psar_afs_{key_prefix}",
            disabled=disabled,
        )
        params['psar_af_increment'] = st.number_input(
            "AF Increment", 0.001, 0.5, 0.02, step=0.01, format="%.3f",
            key=f"psar_afi_{key_prefix}",
            disabled=disabled,
        )
        params['psar_af_max'] = st.number_input(
            "AF Maximum", 0.01, 1.0, 0.20, step=0.01, format="%.2f",
            key=f"psar_afm_{key_prefix}",
            disabled=disabled,
        )

    return params

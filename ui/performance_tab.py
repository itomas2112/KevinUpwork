"""
Performance tab (Tab 3) - Aggregated backtest metrics across multiple Elliott Wave patterns.
Independent pattern selection UI with 6 modes, "+ Add Selection" rows, and global metrics.
"""
import streamlit as st
import pandas as pd

from data.loader import parse_drm_periods
from data.helpers import PRIMARY_SECONDARY_MAP, PRIMARY_LIST, ALL_UNIQUE_SECONDARIES, expand_selection, selection_label
from indicators.calculate_indicators import calculate_indicators, slice_for_graph, migrate_indicator_settings
from strategies.first_strategy import execute_custom_strategy
from ui.charting_tab import _aggregate_stats

SELECTION_MODES = [
    "All Patterns",
    "All Bullish",
    "All Bearish",
    "Specified Primary",
    "Specified Secondary",
    "Secondary Across Primaries",
]

def render_performance_tab(sidebar_config):
    """Render the Performance tab content."""

    # --------------------------------------------------
    # Strategy selector (all saved strategies, unfiltered)
    # --------------------------------------------------
    saved = st.session_state.get('saved_strategies', [])
    if not saved:
        st.info("No saved strategies. Create one in the Strategy Builder tab.")
        return

    strategy_names = [s.get('strategy_name', f'Strategy_{i+1}') for i, s in enumerate(saved)]
    sel_col, _ = st.columns([1, 3])
    with sel_col:
        selected_idx = st.selectbox(
            "Select Strategy",
            options=range(len(strategy_names)),
            format_func=lambda x: strategy_names[x],
            key="perf_strategy_select",
        )
    selected_strategy = saved[selected_idx]

    # --------------------------------------------------
    # Prerequisites
    # --------------------------------------------------
    show_1h = sidebar_config['analysis_mode'] == "1H"
    df_key = 'df_1h' if show_1h else 'df_15m'
    params_key = 'params_1h' if show_1h else 'params_15m'
    tf_label = "1H" if show_1h else "15m"

    if df_key not in st.session_state:
        st.info(f"Please upload {tf_label} OHLC data in the Charting tab first.")
        return

    drm_bullish = st.session_state.get('drm_bullish')
    drm_bearish = st.session_state.get('drm_bearish')
    if drm_bullish is None and drm_bearish is None:
        st.info("Please upload a DRM file in the Charting tab first.")
        return

    # --------------------------------------------------
    # Selection UI — dynamic rows with add/remove
    # --------------------------------------------------
    st.subheader("Pattern Selections")

    selections = st.session_state['perf_selections']

    # Column headers
    h_mode, h_ptype, h_primary, h_secondary, h_del = st.columns([2, 2, 2, 2, 0.5])
    h_mode.markdown("**Mode**")
    h_ptype.markdown("**Pattern Type**")
    h_primary.markdown("**Primary**")
    h_secondary.markdown("**Secondary**")

    rows_to_remove = []
    for idx, sel in enumerate(selections):
        # Fixed layout: Mode | Pattern Type | Primary | Secondary | Delete
        # All labels collapsed so everything aligns on one line.
        c_mode, c_ptype, c_primary, c_secondary, c_delete = st.columns([2, 2, 2, 2, 0.5])

        with c_mode:
            current_mode = sel.get("mode", "All Patterns")
            mode = st.selectbox(
                "Mode",
                SELECTION_MODES,
                index=SELECTION_MODES.index(current_mode) if current_mode in SELECTION_MODES else 0,
                key=f"perf_sel_{idx}_mode",
                label_visibility="collapsed",
            )
            selections[idx]["mode"] = mode

        # Use the live widget value for visibility
        show_ptype = mode in ("Specified Primary", "Specified Secondary", "Secondary Across Primaries")
        show_primary = mode in ("Specified Primary", "Specified Secondary")
        show_secondary = mode in ("Specified Secondary", "Secondary Across Primaries")

        with c_ptype:
            if show_ptype:
                ptype_options = ["Bullish", "Bearish"]
                current_ptype = sel.get("pattern_type", "Bullish")
                ptype = st.selectbox(
                    "Pattern Type",
                    ptype_options,
                    index=ptype_options.index(current_ptype) if current_ptype in ptype_options else 0,
                    key=f"perf_sel_{idx}_pattern_type",
                    label_visibility="collapsed",
                )
                selections[idx]["pattern_type"] = ptype

        with c_primary:
            if show_primary:
                current_primary = sel.get("primary")
                primary_idx = PRIMARY_LIST.index(current_primary) if current_primary in PRIMARY_LIST else 0
                primary = st.selectbox(
                    "Primary",
                    PRIMARY_LIST,
                    index=primary_idx,
                    key=f"perf_sel_{idx}_primary",
                    label_visibility="collapsed",
                )
                selections[idx]["primary"] = primary

        with c_secondary:
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
                        key=f"perf_sel_{idx}_secondary",
                        label_visibility="collapsed",
                    )
                    selections[idx]["secondary"] = secondary

        with c_delete:
            if st.button("X", key=f"perf_sel_{idx}_remove", type="primary"):
                rows_to_remove.append(idx)

    # Remove rows (reverse order to preserve indices)
    if rows_to_remove:
        for idx in sorted(rows_to_remove, reverse=True):
            st.session_state['perf_selections'].pop(idx)
        # Ensure at least one selection remains
        if not st.session_state['perf_selections']:
            st.session_state['perf_selections'] = [{
                "mode": "All Patterns",
                "pattern_type": "Bullish",
                "primary": None,
                "secondary": None,
            }]
        st.rerun()

    # Add selection button
    if st.button("+ Add Selection", key="perf_add_selection"):
        st.session_state['perf_selections'].append({
            "mode": "All Patterns",
            "pattern_type": "Bullish",
            "primary": None,
            "secondary": None,
        })
        st.rerun()

    st.markdown("---")

    # --------------------------------------------------
    # Calculate button (prevents auto-calculate on every widget change)
    # --------------------------------------------------
    if st.button("Calculate", key="perf_calculate_btn", type="primary"):
        display_only_keys = {'rsi_upper_1', 'rsi_upper_2', 'rsi_lower_1', 'rsi_lower_2', 'cmb_lines',
                             'ichi_show_tenkan', 'ichi_show_kijun', 'ichi_show_senkou_a',
                             'ichi_show_senkou_b', 'ichi_show_chikou',
                             'ichi_show_senkou_a_current', 'ichi_show_senkou_b_current',
                             'ichi_show_chikou_decision',
                             'bb_show_upper', 'bb_show_middle', 'bb_show_lower',
                             'kc_show_upper', 'kc_show_middle', 'kc_show_lower'}
        indicator_params = {k: v for k, v in sidebar_config[params_key].items() if k not in display_only_keys}
        strategy_settings = selected_strategy.get('indicator_settings')
        if strategy_settings:
            strategy_settings = migrate_indicator_settings(strategy_settings)
            indicator_params.update(strategy_settings)
        df_full = calculate_indicators(df=st.session_state[df_key], **indicator_params)

        # --------------------------------------------------
        # Run backtest for each selection
        # --------------------------------------------------
        results = {}  # selection_label -> aggregated dict
        global_stats = []  # deduplicated stats for global metrics
        global_seen_combos = set()  # track (pattern_type, primary, secondary) already counted globally

        # Cache backtest results per combo to avoid re-running identical combos
        combo_cache = {}  # (pattern_type, primary, secondary) -> list of stats

        # Count total combos for progress bar
        total_combos = 0
        expanded_per_selection = []
        for sel in selections:
            combos = expand_selection(sel)
            expanded_per_selection.append(combos)
            total_combos += len(combos)

        progress_bar = st.progress(0)
        combo_counter = 0

        for sel, combos in zip(selections, expanded_per_selection):
            label = selection_label(sel)
            # Ensure unique column labels when duplicate selections exist
            if label in results:
                n = 2
                while f"{label} ({n})" in results:
                    n += 1
                label = f"{label} ({n})"

            if not combos:
                results[label] = _empty_agg()
                continue

            selection_stats = []
            for pattern_type, primary, secondary in combos:
                combo_key = (pattern_type, primary, secondary)

                # Use cached results if this combo was already computed
                if combo_key in combo_cache:
                    selection_stats.extend(combo_cache[combo_key])
                    if combo_key not in global_seen_combos:
                        global_seen_combos.add(combo_key)
                        global_stats.extend(combo_cache[combo_key])
                    combo_counter += 1
                    if total_combos > 0:
                        progress_bar.progress(combo_counter / total_combos)
                    continue

                drm_df = drm_bullish if pattern_type == 'Bullish' else drm_bearish
                if drm_df is None:
                    combo_cache[combo_key] = []
                    combo_counter += 1
                    if total_combos > 0:
                        progress_bar.progress(combo_counter / total_combos)
                    continue

                periods = parse_drm_periods(drm_df, pattern_type, primary, secondary)

                combo_stats = []
                for start_dt, end_dt in periods:
                    df_slice, period_start, period_end = slice_for_graph(
                        df=df_full, start_date=start_dt, end_date=end_dt,
                        show_ichimoku=sidebar_config['show_ichimoku'],
                        show_bb=sidebar_config['show_bb'],
                        show_kc=sidebar_config['show_kc'],
                    )
                    if df_slice.empty:
                        continue

                    _, stats = execute_custom_strategy(df_slice, selected_strategy, period_start, period_end)
                    combo_stats.append(stats)

                combo_cache[combo_key] = combo_stats
                selection_stats.extend(combo_stats)

                # Only add to global stats once per unique combo
                if combo_key not in global_seen_combos:
                    global_seen_combos.add(combo_key)
                    global_stats.extend(combo_stats)

                combo_counter += 1
                if total_combos > 0:
                    progress_bar.progress(combo_counter / total_combos)

            if selection_stats:
                results[label] = _aggregate_stats(selection_stats)
            else:
                results[label] = _empty_agg()

        progress_bar.empty()

        # Store results in session state so they persist across reruns
        st.session_state['perf_cached_results'] = results
        st.session_state['perf_cached_global_stats'] = global_stats
        st.session_state['perf_cached_strategy_name'] = selected_strategy.get('strategy_name', 'Custom')

    # --------------------------------------------------
    # Display results (from session state cache)
    # --------------------------------------------------
    cached_results = st.session_state.get('perf_cached_results')
    cached_global_stats = st.session_state.get('perf_cached_global_stats')
    cached_strategy_name = st.session_state.get('perf_cached_strategy_name', '')

    if cached_results is not None:
        # Global Performance Metrics
        st.subheader("Global Performance Metrics")
        st.caption(f"Strategy: **{cached_strategy_name}**")
        if cached_global_stats:
            global_agg = _aggregate_stats(cached_global_stats)
        else:
            global_agg = _empty_agg()

        global_table = _build_metrics_table({"Global": global_agg})
        st.table(global_table)

        # Per-selection results table
        if cached_results:
            st.subheader("Performance by Selection")
            perf_table = _build_metrics_table(cached_results)
            st.table(perf_table)
    else:
        st.info("Configure your pattern selections above, then click **Calculate** to run the backtest.")


# ------------------------------------------------------------------
# Helpers
# ------------------------------------------------------------------

def _build_metrics_table(results_dict):
    """Build a DataFrame with metric rows and one column per result."""
    metric_names = [
        "Number of Trades",
        "Win %",
        "Lose %",
        "Avg Profit",
        "Avg Loss",
        "Total P&L",
        "Expected Value",
        "Target Exit %",
        "Stop Exit %",
    ]

    table_data = {}
    for label, agg in results_dict.items():
        table_data[label] = [
            f"{agg['num_trades']}",
            f"{agg['win_pct']:.0f}%",
            f"{agg['lose_pct']:.0f}%",
            f"{agg['avg_win_pnl']:.2f}R",
            f"{agg['avg_lose_pnl']:.2f}R",
            f"{agg['total_pnl']:.2f}R",
            f"{agg['expected_value']:.2f}R",
            f"{agg['target_exit_pct']:.0f}%",
            f"{agg['stop_exit_pct']:.0f}%",
        ]

    return pd.DataFrame(table_data, index=metric_names)


def _empty_agg():
    """Return an empty aggregation dict (no trades)."""
    return {
        'num_trades': 0,
        'win_pct': 0.0,
        'lose_pct': 0.0,
        'avg_win_pnl': 0.0,
        'avg_lose_pnl': 0.0,
        'total_pnl': 0.0,
        'expected_value': 0.0,
        'target_exit_pct': 0.0,
        'stop_exit_pct': 0.0,
    }

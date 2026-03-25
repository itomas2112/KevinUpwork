"""
Strategy Testing tab (Tab 4) - Cross Validation and Test Set validation.
Own pattern selection UI, own strategy selector, manual calculate buttons.
Strategy pattern filtering applied: combos are intersected with the strategy's
saved patterns so a strategy only runs on the patterns it was designed for.
"""
import json
import streamlit as st
import pandas as pd

from data.loader import parse_drm_periods
from data.helpers import (
    PRIMARY_SECONDARY_MAP, PRIMARY_LIST, ALL_UNIQUE_SECONDARIES,
    expand_selection, selection_label,
)
from indicators.calculate_indicators import slice_for_graph, migrate_indicator_settings, strategy_indicator_flags
from strategies.first_strategy import execute_custom_strategy
from ui.charting_tab import _aggregate_stats, _get_or_calculate

SELECTION_MODES = [
    "All Patterns",
    "All Bullish",
    "All Bearish",
    "Specified Primary",
    "Specified Secondary",
    "Secondary Across Primaries",
]

# Default indicator params (matching calculate_indicators defaults)
_DEFAULT_INDICATOR_PARAMS = {
    'rsi_window': 14,
    'bb_upper_period': 20, 'bb_upper_stdev': 2.0, 'bb_mid_period': 20,
    'bb_lower_period': 20, 'bb_lower_stdev': 2.0,
    'kc_upper_ema': 20, 'kc_mid_ema': 20, 'kc_lower_ema': 20,
    'kc_atr_period': 10, 'kc_upper_mult': 2.0, 'kc_lower_mult': 2.0,
    'stoch_k_period': 14, 'stoch_k_smooth': 3, 'stoch_d_smooth': 3,
    'adx_period': 14, 'atr_period': 14,
    'macd_fast': 12, 'macd_slow': 26, 'macd_signal': 9,
    'supertrend_period': 10, 'supertrend_multiplier': 3.0,
    'ema_periods': [],
    'dc_upper_period': 20, 'dc_mid_period': 20, 'dc_lower_period': 20, 'dc_offset': 0,
    'psar_af_start': 0.02, 'psar_af_increment': 0.02, 'psar_af_max': 0.20,
}


def render_strategy_testing_tab(sidebar_config):
    """Render the Strategy Testing tab content."""

    # --------------------------------------------------
    # Strategy selector (inside tab, all saved strategies, unfiltered)
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
            key="testing_strategy_select",
        )
    selected_strategy = saved[selected_idx]
    strategy_name = selected_strategy.get('strategy_name', 'Custom')

    # Show strategy's defined patterns for reference
    strategy_patterns = selected_strategy.get('patterns', [])
    if strategy_patterns:
        st.caption(f"Strategy patterns: {', '.join(strategy_patterns)}")
    else:
        st.caption("Strategy patterns: **All** (no pattern filter defined)")

    # --------------------------------------------------
    # Prerequisites — need OHLC data and DRM uploaded
    # --------------------------------------------------
    df_key = 'df_ohlc'
    tf_label = st.session_state.get("_agg_timeframe", "15m")

    if df_key not in st.session_state:
        st.info("Please upload OHLC data in the Charting tab first.")
        return

    drm_bullish = st.session_state.get('drm_bullish')
    drm_bearish = st.session_state.get('drm_bearish')
    if drm_bullish is None and drm_bearish is None:
        st.info("Please upload a DRM file in the Charting tab first.")
        return

    # --------------------------------------------------
    # Build indicator params
    # --------------------------------------------------
    indicator_params = dict(_DEFAULT_INDICATOR_PARAMS)
    strategy_settings = selected_strategy.get('indicator_settings')
    if strategy_settings:
        strategy_settings = migrate_indicator_settings(strategy_settings)
        indicator_params.update(strategy_settings)

    # --------------------------------------------------
    # Pattern Selection UI
    # --------------------------------------------------
    st.subheader("Pattern Selections")

    selections = st.session_state['testing_selections']

    # Column headers
    h_mode, h_ptype, h_primary, h_secondary, h_del = st.columns([2, 2, 2, 2, 0.5])
    h_mode.markdown("**Mode**")
    h_ptype.markdown("**Pattern Type**")
    h_primary.markdown("**Primary**")
    h_secondary.markdown("**Secondary**")

    # Generation counter to guarantee fresh widget keys after deletions
    gen = st.session_state.get("_testing_sel_gen", 0)

    rows_to_remove = []
    for idx, sel in enumerate(selections):
        c_mode, c_ptype, c_primary, c_secondary, c_delete = st.columns([2, 2, 2, 2, 0.5])

        kp = f"testing_sel_g{gen}_{idx}"

        with c_mode:
            current_mode = sel.get("mode", "All Patterns")
            mode = st.selectbox(
                "Mode",
                SELECTION_MODES,
                index=SELECTION_MODES.index(current_mode) if current_mode in SELECTION_MODES else 0,
                key=f"{kp}_mode",
                label_visibility="collapsed",
            )
            selections[idx]["mode"] = mode

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
                    key=f"{kp}_pattern_type",
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
                    key=f"{kp}_primary",
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
                        key=f"{kp}_secondary",
                        label_visibility="collapsed",
                    )
                    selections[idx]["secondary"] = secondary

        with c_delete:
            if st.button("X", key=f"{kp}_remove", type="primary"):
                rows_to_remove.append(idx)

    # Remove rows
    if rows_to_remove:
        for idx in sorted(rows_to_remove, reverse=True):
            st.session_state['testing_selections'].pop(idx)
        if not st.session_state['testing_selections']:
            st.session_state['testing_selections'] = [{
                "mode": "All Patterns",
                "pattern_type": "Bullish",
                "primary": None,
                "secondary": None,
            }]
        # Bump generation so all widget keys are fresh on next render
        st.session_state["_testing_sel_gen"] = gen + 1
        st.rerun()

    # Add selection button
    if st.button("+ Add Selection", key="testing_add_selection"):
        st.session_state['testing_selections'].append({
            "mode": "All Patterns",
            "pattern_type": "Bullish",
            "primary": None,
            "secondary": None,
        })
        st.rerun()

    # --------------------------------------------------
    # Expand selections and apply strategy pattern filter
    # --------------------------------------------------
    all_combos = _expand_and_filter(selections, strategy_patterns)

    if not all_combos:
        st.warning("No pattern combos match the strategy's defined patterns. "
                    "The strategy will have no trades with these selections.")

    st.markdown("---")

    # ==================================================
    # Section A: Cross Validation (Training Set)
    # ==================================================
    st.subheader("Cross Validation (Training Set)")

    if not sidebar_config.get('date_range_applied', False):
        st.info("Please set a Training Set date range in the sidebar and click **Apply Training Set** to proceed.")
    else:
        train_start = sidebar_config.get('global_start_date')
        train_end = sidebar_config.get('global_end_date')
        st.caption(f"Training Set: **{train_start}** → **{train_end}**")

        cv_k_col, cv_btn_col, _ = st.columns([1, 1, 2])
        with cv_k_col:
            cv_k = st.number_input("K Folds", min_value=2, max_value=50, value=10, step=1, key="cv_k_folds")
        with cv_btn_col:
            st.markdown("<br>", unsafe_allow_html=True)
            cv_calculate = st.button("Calculate", key="cv_calculate", type="primary")

        # Invalidate cache if strategy changed
        cv_cache = st.session_state.get('_cv_cached_results')
        strategy_fingerprint = json.dumps(selected_strategy, sort_keys=True, default=str)
        if cv_cache and cv_cache.get('_strategy_fingerprint') != strategy_fingerprint:
            st.session_state.pop('_cv_cached_results', None)
            cv_cache = None

        if cv_calculate:
            _run_cross_validation(
                df_key, indicator_params, selected_strategy, strategy_name,
                train_start, train_end, all_combos, drm_bullish, drm_bearish, cv_k
            )
        elif cv_cache:
            _display_cv_results(cv_cache)
        else:
            st.info("Set K folds and click **Calculate** to run cross validation on the Training Set.")

    st.markdown("---")

    # ==================================================
    # Section B: Test Set (one-shot)
    # ==================================================
    st.subheader("Test Set")

    if not sidebar_config.get('test_set_applied', False):
        st.info("Please set a Test Set date range in the sidebar and click **Apply Test Set** to proceed.")
    else:
        test_start = sidebar_config.get('test_start_date')
        test_end = sidebar_config.get('test_end_date')
        st.caption(f"Test Set: **{test_start}** → **{test_end}**")

        test_calculate = st.button("Calculate", key="test_calculate", type="primary")

        # Invalidate cache if strategy changed
        test_cache = st.session_state.get('_test_cached_results')
        if test_cache and test_cache.get('_strategy_fingerprint') != strategy_fingerprint:
            st.session_state.pop('_test_cached_results', None)
            test_cache = None

        if test_calculate:
            _run_test_set(
                df_key, indicator_params, selected_strategy, strategy_name,
                test_start, test_end, all_combos, drm_bullish, drm_bearish,
            )
        elif test_cache:
            _display_test_set_results(test_cache)
        else:
            st.info("Click **Calculate** to run the strategy on the Test Set.")


# ------------------------------------------------------------------
# Pattern expansion + strategy pattern filtering
# ------------------------------------------------------------------

def _expand_and_filter(selections, strategy_patterns):
    """
    Expand pattern selections into combos, then filter by the strategy's
    saved patterns. If the strategy has no patterns defined, all combos pass.

    Strategy patterns are stored as "Primary → Secondary" strings.
    A combo (pattern_type, primary, secondary) matches if
    "primary → secondary" is in the strategy's pattern list.
    """
    # Expand all selections into unique combos
    seen = set()
    all_combos = []
    for sel in selections:
        for combo in expand_selection(sel):
            if combo not in seen:
                seen.add(combo)
                all_combos.append(combo)

    # If strategy has no pattern filter, return all combos
    if not strategy_patterns:
        return all_combos

    # Build set of allowed "primary → secondary" strings from strategy
    allowed = set(strategy_patterns)

    # Filter: only keep combos whose "primary → secondary" is in allowed set
    filtered = [
        (ptype, primary, secondary)
        for ptype, primary, secondary in all_combos
        if f"{primary} \u2192 {secondary}" in allowed
    ]

    return filtered


# ------------------------------------------------------------------
# Cross Validation
# ------------------------------------------------------------------

def _run_cross_validation(df_key, indicator_params, selected_strategy, strategy_name,
                          date_start, date_end, all_combos, drm_bullish, drm_bearish, K=10,
                          cache_key='_cv_cached_results', features_key='_cv_features', params_key='_cv_params',
                          download_key='cv_download', file_prefix='cv'):
    """Run K-fold cross validation on a date range."""

    # Calculate indicators on full date range (cached)
    df_full = _get_or_calculate(
        df_key, features_key, params_key, indicator_params,
        global_start_date=date_start, global_end_date=date_end
    )

    if df_full.empty:
        st.warning("No data available for the selected date range.")
        return

    # Split date range into K equal folds by date
    start_ts = pd.Timestamp(date_start)
    end_ts = pd.Timestamp(date_end) + pd.Timedelta(days=1)
    total_duration = end_ts - start_ts
    fold_duration = total_duration / K

    folds = []
    for f in range(K):
        fold_start = start_ts + fold_duration * f
        fold_end = start_ts + fold_duration * (f + 1)
        folds.append((fold_start, fold_end))

    # Run strategy per fold
    fold_results = {}
    all_fold_stats = []

    progress_bar = st.progress(0)

    for fold_idx, (fold_start, fold_end) in enumerate(folds):
        fold_label = f"Fold {fold_idx + 1}"
        fold_stats = _run_on_date_range(
            df_full, selected_strategy, all_combos,
            drm_bullish, drm_bearish, fold_start, fold_end
        )

        if fold_stats:
            fold_results[fold_label] = _aggregate_stats(fold_stats)
            all_fold_stats.extend(fold_stats)
        else:
            fold_results[fold_label] = _empty_agg()

        progress_bar.progress((fold_idx + 1) / K)

    progress_bar.empty()

    # Overall aggregation
    if all_fold_stats:
        overall_agg = _aggregate_stats(all_fold_stats)
    else:
        overall_agg = _empty_agg()

    cache = {
        'strategy_name': strategy_name,
        '_strategy_fingerprint': json.dumps(selected_strategy, sort_keys=True, default=str),
        'fold_results': fold_results,
        'overall_agg': overall_agg,
    }
    st.session_state[cache_key] = cache
    _display_cv_results(cache, download_key=download_key, file_prefix=file_prefix)


def _display_cv_results(cache, download_key="cv_download", file_prefix="cv"):
    """Display cross validation results."""
    strategy_name = cache['strategy_name']
    fold_results = cache['fold_results']
    overall_agg = cache['overall_agg']

    k = len(fold_results)
    st.caption(f"Strategy: **{strategy_name}** | {k}-Fold Cross Validation")

    # Build table: Overall + per fold
    results_dict = {"Overall": overall_agg}
    results_dict.update(fold_results)

    table = _build_metrics_table(results_dict)
    st.table(table)

    # Copy to clipboard
    _copy_to_clipboard(table.to_csv(sep='\t', header=False, index=False), key=download_key)


# ------------------------------------------------------------------
# Test Set (one-shot)
# ------------------------------------------------------------------

def _run_test_set(df_key, indicator_params, selected_strategy, strategy_name,
                  date_start, date_end, all_combos, drm_bullish, drm_bearish):
    """Run strategy once on the full test set date range (no folds)."""

    df_full = _get_or_calculate(
        df_key, '_test_features', '_test_params', indicator_params,
        global_start_date=date_start, global_end_date=date_end
    )

    if df_full.empty:
        st.warning("No data available for the selected date range.")
        return

    progress_bar = st.progress(0)

    all_stats = _run_on_date_range(
        df_full, selected_strategy, all_combos,
        drm_bullish, drm_bearish,
        pd.Timestamp(date_start),
        pd.Timestamp(date_end) + pd.Timedelta(days=1),
        progress_callback=lambda p: progress_bar.progress(p)
    )

    progress_bar.empty()

    if all_stats:
        agg = _aggregate_stats(all_stats)
    else:
        agg = _empty_agg()

    cache = {
        'strategy_name': strategy_name,
        '_strategy_fingerprint': json.dumps(selected_strategy, sort_keys=True, default=str),
        'agg': agg,
    }
    st.session_state['_test_cached_results'] = cache
    _display_test_set_results(cache)


def _display_test_set_results(cache):
    """Display test set results (single result, no folds)."""
    strategy_name = cache['strategy_name']
    agg = cache['agg']

    st.caption(f"Strategy: **{strategy_name}**")

    table = _build_metrics_table({"Result": agg})
    st.table(table)

    _copy_to_clipboard(table.to_csv(sep='\t', header=False, index=False), key="test_download")


# ------------------------------------------------------------------
# Shared helpers
# ------------------------------------------------------------------

def _copy_to_clipboard(text: str, key: str = "copy_btn"):
    """Render a 'Copy to Clipboard' button using HTML/JS."""
    import streamlit.components.v1 as components
    escaped = text.replace('\\', '\\\\').replace('`', '\\`').replace('$', '\\$')
    components.html(f"""
    <button id="btn_{key}" onclick="
        navigator.clipboard.writeText(`{escaped}`).then(function() {{
            document.getElementById('btn_{key}').innerText = 'Copied!';
            setTimeout(function() {{ document.getElementById('btn_{key}').innerText = 'Copy to Clipboard'; }}, 2000);
        }})
    " style="
        background-color: #FF4B4B; color: white; border: none; padding: 8px 16px;
        border-radius: 6px; cursor: pointer; font-size: 14px; font-weight: 500;
    ">Copy to Clipboard</button>
    """, height=50)

def _run_on_date_range(df_full, selected_strategy, all_combos,
                       drm_bullish, drm_bearish, range_start, range_end,
                       progress_callback=None):
    """
    Run strategy on all DRM periods whose start date falls within [range_start, range_end).
    Returns list of stats dicts for aggregation.
    """
    all_stats = []
    total = len(all_combos)
    ind_flags = strategy_indicator_flags(selected_strategy)

    for idx, (pattern_type, primary, secondary) in enumerate(all_combos):
        drm_df = drm_bullish if pattern_type == 'Bullish' else drm_bearish
        if drm_df is None:
            if progress_callback and total > 0:
                progress_callback((idx + 1) / total)
            continue

        periods = parse_drm_periods(drm_df, pattern_type, primary, secondary)

        for start_dt, end_dt in periods:
            # Only include periods whose start falls within the date range
            if start_dt < range_start or start_dt >= range_end:
                continue

            df_slice, period_start, period_end = slice_for_graph(
                df=df_full, start_date=start_dt, end_date=end_dt,
                **ind_flags,
            )
            if df_slice.empty:
                continue

            _, stats = execute_custom_strategy(df_slice, selected_strategy, period_start, period_end)
            all_stats.append(stats)

        if progress_callback and total > 0:
            progress_callback((idx + 1) / total)

    return all_stats


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
        "Static %",
        "Dynamic %",
        "Sharpe Ratio",
        "MAR Ratio",
        "SQN",
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
            f"{agg['static_exit_pct']:.0f}%",
            f"{agg['dynamic_exit_pct']:.0f}%",
            f"{agg['sharpe_ratio']:.2f}",
            f"{agg['mar_ratio']:.2f}",
            f"{agg['sqn']:.2f}",
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
        'static_exit_pct': 0.0,
        'dynamic_exit_pct': 0.0,
        'sharpe_ratio': 0.0,
        'max_drawdown': 0.0,
        'mar_ratio': 0.0,
        'sqn': 0.0,
    }

"""
Strategy Testing tab (Tab 4) - Walk Forward Optimization and Test Set validation.
Own pattern selection UI, own strategy selector, manual calculate buttons.
Strategy pattern filtering applied: combos are intersected with the strategy's
saved patterns so a strategy only runs on the patterns it was designed for.
"""
import json
import copy
import math
import multiprocessing
from collections import OrderedDict
import streamlit as st
import pandas as pd

from data.loader import parse_drm_periods
from data.helpers import (
    PRIMARY_SECONDARY_MAP, PRIMARY_LIST, ALL_UNIQUE_SECONDARIES,
    expand_selection, selection_label,
)
from indicators.calculate_indicators import (
    slice_for_graph, migrate_indicator_settings, strategy_indicator_flags,
    calculate_indicators,
)
from strategies.first_strategy import execute_custom_strategy
from ui.charting_tab import _aggregate_stats, _get_or_calculate
from strategies.wfo_engine import (
    split_folds, detect_used_groups, generate_param_grid, count_grid_size,
    aggregate_stats_dicts,
)
from ui.performance_tab import (
    _build_metrics_table as _perf_build_metrics_table,
    _empty_agg as _perf_empty_agg,
)
from strategies.wfo_worker import init_wfo_worker, run_wfo_batch
from strategies.first_strategy_numpy import execute_custom_strategy_numpy
from config.constants import WFO_DEFAULT_RANGES as _WFO_RANGES

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
    'ema_periods': [10, 20, 50, 200],
    'dc_upper_period': 20, 'dc_mid_period': 20, 'dc_lower_period': 20, 'dc_offset': 0,
    'psar_af_start': 0.02, 'psar_af_increment': 0.02, 'psar_af_max': 0.20,
    'willr_period': 14,
    'cci_period': 20,
    'roc_period': 12, 'roc_signal_period': 9,
    'lr_period': 50, 'lr_multiplier': 2.0,
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
    # Section A: Walk Forward Optimization (Training Set)
    # ==================================================
    st.subheader("Walk Forward Optimization (Training Set)")

    if not sidebar_config.get('date_range_applied', False):
        st.info("Please set a Training Set date range in the sidebar and click **Apply Training Set** to proceed.")
    else:
        train_start = sidebar_config.get('global_start_date')
        train_end = sidebar_config.get('global_end_date')
        st.caption(f"Training Set: **{train_start}** → **{train_end}**")

        # Detect which indicator groups the strategy uses
        used_groups = detect_used_groups(selected_strategy)

        # ---- Fold configuration ----
        fc1, fc2, fc3 = st.columns([1, 1, 2])
        with fc1:
            wfo_folds = st.number_input("Number of Folds", 2, 20, 5, step=1, key="wfo_folds")
        with fc2:
            wfo_ratio = st.number_input("Train % per Fold", 50, 95, 80, step=5, key="wfo_ratio")

        folds = split_folds(train_start, train_end, wfo_folds, wfo_ratio / 100.0)
        try:
            fold_dur = pd.Timestamp(train_end) + pd.Timedelta(days=1) - pd.Timestamp(train_start)
            fold_days = fold_dur.total_seconds() / 86400 / wfo_folds
            train_days = int(fold_days * wfo_ratio / 100)
            test_days = int(fold_days * (100 - wfo_ratio) / 100)
        except Exception:
            fold_days = train_days = test_days = 0
        st.caption(f"Fold size: **{int(fold_days)}** days | "
                   f"Train: **{train_days}** days | "
                   f"Test: **{test_days}** days")

        # ---- Parameter ranges (only used groups) ----
        if not used_groups:
            st.info("Strategy does not reference any optimisable indicators.")
        else:
            # Wrapped in @st.fragment so editing param inputs only reruns this
            # block instead of the entire tab — keeps focus, prevents jumps.
            _render_wfo_param_section(used_groups, indicator_params, wfo_folds,
                                      selected_strategy)

            wfo_ranges = st.session_state.get('_wfo_current_ranges', {})
            grid_size = st.session_state.get('_wfo_current_grid_size', 0)
            grid_too_large = st.session_state.get('_wfo_grid_too_large', False)

            # ---- Filter & sort ----
            _render_wfo_filters()

            # ---- Run button ----
            strategy_fingerprint = json.dumps(selected_strategy, sort_keys=True, default=str)
            wfo_calculate = st.button("Run Walk Forward Optimization", key="wfo_run",
                                      type="primary", disabled=grid_too_large)

            # Invalidate cache if strategy changed
            wfo_cache = st.session_state.get('_wfo_cached_results')
            if wfo_cache and wfo_cache.get('_strategy_fingerprint') != strategy_fingerprint:
                st.session_state.pop('_wfo_cached_results', None)
                wfo_cache = None

            if wfo_calculate:
                _run_wfo(
                    df_key, indicator_params, selected_strategy, strategy_name,
                    train_start, train_end, all_combos, drm_bullish, drm_bearish,
                    wfo_folds, wfo_ratio / 100.0, used_groups, wfo_ranges, grid_size,
                )
            elif wfo_cache:
                _display_wfo_results(wfo_cache)

    st.markdown("---")

    # ==================================================
    # Section B: Simple Optimization (80/20 split of training set)
    # ==================================================
    st.subheader("Simple Optimization")
    st.caption("Splits the training set 80% in-sample / 20% out-of-sample. "
               "Runs every parameter combo on the in-sample period; shows the "
               "top 10 by your selected sort metric, with both in-sample and "
               "out-of-sample stats per candidate.")

    if not sidebar_config.get('date_range_applied', False):
        st.info("Set a Training Set date range in the sidebar and click "
                "**Apply Training Set** to use Simple Optimization.")
    else:
        sopt_train_start = sidebar_config.get('global_start_date')
        sopt_train_end = sidebar_config.get('global_end_date')
        # Compute the 80/20 split
        try:
            train_ts = pd.Timestamp(sopt_train_start)
            test_ts = pd.Timestamp(sopt_train_end) + pd.Timedelta(days=1)
            duration = test_ts - train_ts
            split_ts = train_ts + duration * 0.8
            sopt_in_end = split_ts
            sopt_oos_start = split_ts
            sopt_in_start = train_ts
            sopt_oos_end = test_ts
            st.caption(
                f"In-sample (80%): **{sopt_in_start.date()}** → **{(sopt_in_end - pd.Timedelta(seconds=1)).date()}** | "
                f"Out-of-sample (20%): **{sopt_oos_start.date()}** → **{(sopt_oos_end - pd.Timedelta(seconds=1)).date()}**"
            )
        except Exception as e:
            st.error(f"Could not compute 80/20 split: {e}")
            sopt_in_start = sopt_in_end = sopt_oos_start = sopt_oos_end = None

        sopt_used_groups = detect_used_groups(selected_strategy)

        if not sopt_used_groups:
            st.info("Strategy does not reference any optimisable indicators or values.")
        elif sopt_in_start is not None:
            # Param ranges (fragment, independent prefix from WFO)
            _render_wfo_param_section(sopt_used_groups, indicator_params, 1,
                                      selected_strategy, key_prefix="sopt")

            sopt_ranges = st.session_state.get('_sopt_current_ranges', {})
            sopt_grid_size = st.session_state.get('_sopt_current_grid_size', 0)
            sopt_grid_too_large = st.session_state.get('_sopt_grid_too_large', False)

            # Filter & sort UI (same metric set as WFO/Grid Search)
            _render_wfo_filters(key_prefix="sopt")

            sopt_calc = st.button("Run Simple Optimization", key="sopt_run",
                                   type="primary", disabled=sopt_grid_too_large)

            sopt_cache = st.session_state.get('_sopt_cached_results')
            if sopt_cache and sopt_cache.get('_strategy_fingerprint') != strategy_fingerprint:
                st.session_state.pop('_sopt_cached_results', None)
                sopt_cache = None

            if sopt_calc:
                _run_simple_opt(
                    df_key, indicator_params, selected_strategy, strategy_name,
                    sopt_in_start, sopt_in_end, sopt_oos_start, sopt_oos_end,
                    all_combos, drm_bullish, drm_bearish,
                    sopt_used_groups, sopt_ranges, sopt_grid_size,
                )
            elif sopt_cache:
                _display_simple_opt_results(sopt_cache)

    st.markdown("---")

    # ==================================================
    # Section C: Test Set (one-shot)
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
# Walk Forward Optimization
# ------------------------------------------------------------------

# Mirror Grid Search's sort/filter metrics so WFO and Grid Search stay in
# sync. WFO does not compute correlation against reference strategies, so
# `abs_correlation` is filtered out.
from ui.grid_search_tab import (
    SORT_METRICS as _GS_SORT_METRICS,
    GS_MC_N_SIMULATIONS, filter_props, passes_thresholds,
    enrich_aggs_with_mc,
)
SORT_METRICS = [m for m in _GS_SORT_METRICS if m[0] != "abs_correlation"]
_DEFAULT_SORT_KEY = "mc_avg_profit"


@st.fragment
def _render_wfo_param_section(used_groups, base_settings, n_runs_per_combo, strategy,
                              key_prefix="wfo"):
    """Render param ranges + grid-size estimate as a Streamlit fragment.

    Editing inputs inside a fragment only reruns this function — the rest of
    the tab stays put. Latest values are persisted to session state under
    `_{key_prefix}_current_ranges` etc. so the parent (which doesn't rerun)
    can read them.

    n_runs_per_combo: number of executions per combo (folds for WFO,
    1 for Simple Optimization). Used only for time estimate.
    """
    wfo_ranges = _render_wfo_param_ranges(used_groups, base_settings, strategy,
                                          key_prefix=key_prefix)

    grid_size = count_grid_size(used_groups, wfo_ranges, base_settings)
    est_seconds = grid_size * n_runs_per_combo * 0.008  # ~8ms per combo per run
    grid_too_large = False
    if grid_size <= 1:
        st.caption(f"Combinations: **{grid_size:,}** — Widen min/max ranges above to create parameter variations.")
    elif grid_size > 100000:
        st.error(f"Grid too large ({grid_size:,} combos). Increase step sizes or narrow min/max ranges to stay under 100,000.")
        grid_too_large = True
    elif grid_size > 10000:
        st.warning(f"Combinations: **{grid_size:,}** | Est. time: **~{est_seconds/60:.0f} min** — consider reducing.")
    else:
        st.caption(f"Combinations: **{grid_size:,}** | Est. time: **~{max(1, int(est_seconds))} sec**")

    st.session_state[f'_{key_prefix}_current_ranges'] = wfo_ranges
    st.session_state[f'_{key_prefix}_current_grid_size'] = grid_size
    st.session_state[f'_{key_prefix}_grid_too_large'] = grid_too_large


def _render_wfo_param_ranges(used_groups, base_settings, strategy=None, key_prefix="wfo"):
    """Render parameter range editors for each used indicator group.

    Defaults: min = max = strategy's saved value (no variation).
    User widens only the params they want to optimise.
    Shows per-group combo count so user sees the impact.

    ATR slots (initial ATR stop + ATR Target triggers) and value slots
    (every fixed numeric value in the strategy) are discovered from
    *strategy* and rendered as dedicated sections.

    Returns the user-configured ranges dict.
    """
    from strategies.wfo_engine import (
        _range_values, detect_atr_slots, detect_value_slots,
        ATR_SLOT_DEFAULT_RANGES,
    )

    st.markdown("**Parameter Ranges** *(defaults = strategy's saved values — widen to optimise)*")
    ranges = {}

    # Build a lookup of slot metadata by slot_id
    atr_slots_by_id = {}
    value_slots_by_id = {}
    if strategy is not None:
        for slot in detect_atr_slots(strategy):
            atr_slots_by_id[slot["slot_id"]] = slot
        for slot in detect_value_slots(strategy):
            value_slots_by_id[slot["slot_id"]] = slot

    for group in sorted(used_groups):
        # --- ATR Slot pseudo-group ---
        if group.startswith("atr_slot:"):
            slot_id = group[len("atr_slot:"):]
            slot = atr_slots_by_id.get(slot_id)
            if slot is None:
                continue
            ranges[group] = _render_atr_slot_range(slot, key_prefix=key_prefix)
            continue

        # --- Value Slot pseudo-group (any fixed numeric value) ---
        if group.startswith("value_slot:"):
            slot_id = group[len("value_slot:"):]
            slot = value_slots_by_id.get(slot_id)
            if slot is None:
                continue
            ranges[group] = _render_value_slot_range(slot, key_prefix=key_prefix)
            continue

        templates = _WFO_RANGES.get(group, {})
        if not templates:
            continue

        # Count combos for this group to show in the header
        group_ranges_tmp = {}
        ranges[group] = {}

        if group == "ema":
            ema_count = len(base_settings.get("ema_periods", []))
            ema_ranges = []
            group_combos = 1
            for idx in range(ema_count):
                saved_val = base_settings.get("ema_periods", [])[idx] if idx < len(base_settings.get("ema_periods", [])) else 20
                # Get step from template
                tmpl = templates.get("_ema_ranges", [])
                tmpl_step = tmpl[idx][2] if idx < len(tmpl) else 1
                lo = st.session_state.get(f"{key_prefix}_ema_{idx}_lo", int(saved_val))
                hi = st.session_state.get(f"{key_prefix}_ema_{idx}_hi", int(saved_val))
                step = st.session_state.get(f"{key_prefix}_ema_{idx}_step", int(tmpl_step))
                n_vals = len(_range_values((lo, hi, step)))
                group_combos *= n_vals
                ema_ranges.append((lo, hi, step))
            ranges[group]["_ema_ranges"] = ema_ranges

            with st.expander(f"EMA  ({group_combos:,} combos)", expanded=False):
                for idx in range(ema_count):
                    saved_val = base_settings.get("ema_periods", [])[idx]
                    tmpl = templates.get("_ema_ranges", [])
                    tmpl_step = tmpl[idx][2] if idx < len(tmpl) else 1
                    c1, c2, c3 = st.columns(3)
                    with c1:
                        lo = st.number_input(f"EMA {idx+1} Min", 1, 500, int(saved_val),
                                             key=f"{key_prefix}_ema_{idx}_lo")
                    with c2:
                        hi = st.number_input(f"EMA {idx+1} Max", 1, 500, int(saved_val),
                                             key=f"{key_prefix}_ema_{idx}_hi")
                    with c3:
                        step = st.number_input(f"EMA {idx+1} Step", 1, 100, int(tmpl_step),
                                               key=f"{key_prefix}_ema_{idx}_step")
                    ema_ranges[idx] = (lo, hi, step)
                ranges[group]["_ema_ranges"] = ema_ranges
        else:
            # First pass: read current widget values to compute combo count
            group_combos = 1
            param_specs = []
            for param, spec in templates.items():
                if param.startswith("_"):
                    continue
                saved_val = base_settings.get(param)
                is_discrete = isinstance(spec, list)
                if is_discrete:
                    # Discrete: default to just the saved value
                    cur_vals = st.session_state.get(f"{key_prefix}_{group}_{param}_disc")
                    if cur_vals is not None:
                        try:
                            n_vals = len([float(v.strip()) for v in cur_vals.split(",") if v.strip()])
                        except ValueError:
                            n_vals = 1
                    else:
                        n_vals = 1
                else:
                    lo_key = f"{key_prefix}_{group}_{param}_lo"
                    hi_key = f"{key_prefix}_{group}_{param}_hi"
                    step_key = f"{key_prefix}_{group}_{param}_step"
                    _, _, tmpl_step = spec
                    lo = st.session_state.get(lo_key, saved_val if saved_val is not None else spec[0])
                    hi = st.session_state.get(hi_key, saved_val if saved_val is not None else spec[0])
                    step = st.session_state.get(step_key, tmpl_step)
                    n_vals = len(_range_values((lo, hi, step)))
                group_combos *= n_vals
                param_specs.append((param, spec))

            with st.expander(f"{group.upper().replace('_', ' ')}  ({group_combos:,} combos)", expanded=False):
                for param, spec in param_specs:
                    saved_val = base_settings.get(param)
                    is_discrete = isinstance(spec, list)

                    if is_discrete:
                        default_str = str(saved_val) if saved_val is not None else ", ".join(str(v) for v in spec)
                        st.caption(f"**{param}**: discrete values")
                        vals_str = st.text_input(
                            f"{param} values", value=default_str,
                            key=f"{key_prefix}_{group}_{param}_disc",
                        )
                        try:
                            ranges[group][param] = [float(v.strip()) for v in vals_str.split(",") if v.strip()]
                        except ValueError:
                            ranges[group][param] = [float(saved_val)] if saved_val is not None else list(spec)
                    else:
                        lo_tmpl, hi_tmpl, step_tmpl = spec
                        is_float = isinstance(lo_tmpl, float)
                        # Default: min = max = saved value (no variation)
                        default_val = saved_val if saved_val is not None else lo_tmpl

                        c1, c2, c3 = st.columns(3)
                        with c1:
                            if is_float:
                                lo = st.number_input(f"{param} Min", 0.001, 999.0, float(default_val),
                                                     step=0.01, format="%.3f",
                                                     key=f"{key_prefix}_{group}_{param}_lo")
                            else:
                                lo = st.number_input(f"{param} Min", 1, 999, int(default_val),
                                                     key=f"{key_prefix}_{group}_{param}_lo")
                        with c2:
                            if is_float:
                                hi = st.number_input(f"{param} Max", 0.001, 999.0, float(default_val),
                                                     step=0.01, format="%.3f",
                                                     key=f"{key_prefix}_{group}_{param}_hi")
                            else:
                                hi = st.number_input(f"{param} Max", 1, 999, int(default_val),
                                                     key=f"{key_prefix}_{group}_{param}_hi")
                        with c3:
                            if is_float:
                                step = st.number_input(f"{param} Step", 0.001, 100.0, float(step_tmpl),
                                                       step=0.01, format="%.3f",
                                                       key=f"{key_prefix}_{group}_{param}_step")
                            else:
                                step = st.number_input(f"{param} Step", 1, 100, int(step_tmpl),
                                                       key=f"{key_prefix}_{group}_{param}_step")
                        ranges[group][param] = (lo, hi, step)
    return ranges


def _render_atr_slot_range(slot, key_prefix="wfo"):
    """Render range editors for one ATR slot (period + multiplier).

    Defaults: min = max = strategy's saved value (no variation).
    Step defaults come from ATR_SLOT_DEFAULT_RANGES.
    Returns a dict shaped like {atr_period: (lo, hi, step), atr_multiplier: (...)}.
    """
    from strategies.wfo_engine import _range_values, ATR_SLOT_DEFAULT_RANGES

    slot_id = slot["slot_id"]
    saved_period = int(slot["atr_period"])
    saved_mult = float(slot["atr_multiplier"])
    period_step_tmpl = ATR_SLOT_DEFAULT_RANGES["atr_period"][2]
    mult_step_tmpl = ATR_SLOT_DEFAULT_RANGES["atr_multiplier"][2]

    # Pre-read widget values so the expander header can show a live combo count
    p_lo = int(st.session_state.get(f"{key_prefix}_atr_{slot_id}_p_lo", saved_period))
    p_hi = int(st.session_state.get(f"{key_prefix}_atr_{slot_id}_p_hi", saved_period))
    p_step = int(st.session_state.get(f"{key_prefix}_atr_{slot_id}_p_step", int(period_step_tmpl)))
    m_lo = float(st.session_state.get(f"{key_prefix}_atr_{slot_id}_m_lo", saved_mult))
    m_hi = float(st.session_state.get(f"{key_prefix}_atr_{slot_id}_m_hi", saved_mult))
    m_step = float(st.session_state.get(f"{key_prefix}_atr_{slot_id}_m_step", float(mult_step_tmpl)))

    combos = max(1, len(_range_values((p_lo, p_hi, p_step)))) * \
             max(1, len(_range_values((m_lo, m_hi, m_step))))

    header = f"{slot['label']}  ({combos:,} combos)"
    with st.expander(header, expanded=False):
        st.caption(f"Saved: period={saved_period}, multiplier={saved_mult:.2f}")

        c1, c2, c3 = st.columns(3)
        with c1:
            p_lo = st.number_input("Period Min", 1, 500, saved_period, step=1,
                                   key=f"{key_prefix}_atr_{slot_id}_p_lo")
        with c2:
            p_hi = st.number_input("Period Max", 1, 500, saved_period, step=1,
                                   key=f"{key_prefix}_atr_{slot_id}_p_hi")
        with c3:
            p_step = st.number_input("Period Step", 1, 100, int(period_step_tmpl),
                                     step=1, key=f"{key_prefix}_atr_{slot_id}_p_step")

        c4, c5, c6 = st.columns(3)
        with c4:
            m_lo = st.number_input("Multiplier Min", 0.1, 100.0, saved_mult,
                                   step=0.1, format="%.2f",
                                   key=f"{key_prefix}_atr_{slot_id}_m_lo")
        with c5:
            m_hi = st.number_input("Multiplier Max", 0.1, 100.0, saved_mult,
                                   step=0.1, format="%.2f",
                                   key=f"{key_prefix}_atr_{slot_id}_m_hi")
        with c6:
            m_step = st.number_input("Multiplier Step", 0.01, 100.0,
                                     float(mult_step_tmpl), step=0.1, format="%.2f",
                                     key=f"{key_prefix}_atr_{slot_id}_m_step")

    return {
        "atr_period": (int(p_lo), int(p_hi), int(p_step)),
        "atr_multiplier": (float(m_lo), float(m_hi), float(m_step)),
    }


def _render_value_slot_range(slot, key_prefix="wfo"):
    """Render Min/Max/Step for one fixed-value slot.

    Default: Min=Max=current value (no variation). Step is auto-sized to the
    saved value. Returns {"value": (lo, hi, step)} consumed by generate_param_grid.
    """
    from strategies.wfo_engine import _range_values, _step_for

    slot_id = slot["slot_id"]
    saved = float(slot["value"])
    step_default = _step_for(saved)

    # Pre-read widget values so the expander header can show a live combo count
    lo = float(st.session_state.get(f"{key_prefix}_val_{slot_id}_lo", saved))
    hi = float(st.session_state.get(f"{key_prefix}_val_{slot_id}_hi", saved))
    step = float(st.session_state.get(f"{key_prefix}_val_{slot_id}_step", step_default))
    combos = max(1, len(_range_values((lo, hi, step))))

    header = f"{slot['label']}  ({combos:,} combos)"
    with st.expander(header, expanded=False):
        st.caption(f"Saved value: {saved:g}")

        c1, c2, c3 = st.columns(3)
        with c1:
            lo = st.number_input("Min", value=saved, step=step_default,
                                 format="%.4f", key=f"{key_prefix}_val_{slot_id}_lo")
        with c2:
            hi = st.number_input("Max", value=saved, step=step_default,
                                 format="%.4f", key=f"{key_prefix}_val_{slot_id}_hi")
        with c3:
            step = st.number_input("Step", min_value=1e-6, value=step_default,
                                   step=step_default, format="%.4f",
                                   key=f"{key_prefix}_val_{slot_id}_step")

    return {"value": (float(lo), float(hi), float(step))}


def _render_wfo_filters(key_prefix="wfo"):
    """Render filter threshold and sort metric controls.

    Uses the same metric set, filter bounds, and sort defaults as Grid Search.
    Default sort metric is MC Avg Profit @ 5% DD.
    """
    st.markdown("**Filters & Sort**")
    # Filters — 4 columns per row, mirroring grid search filter layout
    rows = [SORT_METRICS[i:i+4] for i in range(0, len(SORT_METRICS), 4)]
    for row in rows:
        cols = st.columns(len(row))
        for col, (key, label) in zip(cols, row):
            with col:
                lbl, default, step, fmt = filter_props(key, label)
                st.number_input(lbl, value=default, step=step,
                                key=f"{key_prefix}_filter_{key}", format=fmt)
    # Sort — default to MC Avg Profit
    sc1, sc2 = st.columns(2)
    with sc1:
        sort_labels = [m[1] for m in SORT_METRICS]
        default_idx = next(
            (i for i, (k, _) in enumerate(SORT_METRICS) if k == _DEFAULT_SORT_KEY), 0)
        st.selectbox("Sort by", sort_labels, index=default_idx,
                     key=f"{key_prefix}_sort_metric")
    with sc2:
        st.radio("Order", ["Highest to Lowest", "Lowest to Highest"],
                 horizontal=True, key=f"{key_prefix}_sort_order")


def _run_wfo(df_key, indicator_params, selected_strategy, strategy_name,
             train_start, train_end, all_combos, drm_bullish, drm_bearish,
             n_folds, train_ratio, used_groups, param_ranges, grid_size):
    """Run the full Walk Forward Optimization.

    Performance: indicators are calculated ONCE with base params.  Workers
    receive the pre-computed DataFrame and only recalculate the changed
    indicator groups via recalculate_groups() (10-50x faster than full recalc).
    """

    df_ohlc = st.session_state[df_key]
    if df_ohlc.empty:
        st.warning("No OHLC data available.")
        return

    folds = split_folds(train_start, train_end, n_folds, train_ratio)

    # Pre-parse ALL DRM periods once
    all_periods = []
    for ptype, primary, secondary in all_combos:
        drm_df = drm_bullish if ptype == "Bullish" else drm_bearish
        if drm_df is None:
            continue
        periods = parse_drm_periods(drm_df, ptype, primary, secondary)
        for start_dt, end_dt in periods:
            all_periods.append((start_dt, end_dt))

    if not all_periods:
        st.warning("No DRM periods found for the selected patterns.")
        return

    # Generate parameter grid
    param_grid = generate_param_grid(used_groups, param_ranges, indicator_params)

    # Collect filter thresholds for every metric in SORT_METRICS, using the
    # widget value when set (defaults are chosen so unset filters don't exclude).
    thresholds = {}
    for key, label in SORT_METRICS:
        _, default, _, _ = filter_props(key, label)
        thresholds[key] = st.session_state.get(f"wfo_filter_{key}", default)

    # Sort settings — default to MC Avg Profit
    sort_labels = [m[1] for m in SORT_METRICS]
    default_sort_label = next(
        (label for k, label in SORT_METRICS if k == _DEFAULT_SORT_KEY), sort_labels[0])
    sort_label = st.session_state.get("wfo_sort_metric", default_sort_label)
    sort_key = next((k for k, l in SORT_METRICS if l == sort_label), _DEFAULT_SORT_KEY)
    sort_desc = st.session_state.get("wfo_sort_order", "Highest to Lowest") == "Highest to Lowest"

    # MC enrichment uses the same balance as Grid Search
    mc_balance = st.session_state.get('mc_starting_balance', 10000.0)

    # Pre-compute indicators ONCE with the base strategy params
    progress = st.progress(0, text="Calculating base indicators...")
    df_featured = calculate_indicators(df_ohlc, **indicator_params)

    # Multiprocessing setup
    n_workers = max(1, multiprocessing.cpu_count() - 1)
    batch_size = max(1, math.ceil(len(param_grid) / (n_workers * 4)))

    total_steps = n_folds + 1  # optimisation steps + final evaluation
    strategy_fingerprint = json.dumps(selected_strategy, sort_keys=True, default=str)

    winning_params = []  # one per OOS step

    for oos_idx in range(n_folds):
        progress.progress(oos_idx / total_steps,
                          text=f"OOS {oos_idx+1}/{n_folds}: optimising on cumulative training data...")

        # Cumulative training date ranges (fold 1 train … fold k train)
        cum_train_ranges = [
            (folds[i]["train_start"], folds[i]["train_end"])
            for i in range(oos_idx + 1)
        ]

        # Build batches — no longer send df_ohlc per batch (it's shared via init)
        batches = []
        for b_start in range(0, len(param_grid), batch_size):
            b_end = min(b_start + batch_size, len(param_grid))
            indices = list(range(b_start, b_end))
            combos = param_grid[b_start:b_end]
            batches.append((indices, combos, cum_train_ranges))

        all_results = []
        try:
            with multiprocessing.Pool(
                n_workers,
                initializer=init_wfo_worker,
                initargs=(copy.deepcopy(selected_strategy), all_periods,
                          dict(indicator_params), df_featured),
            ) as pool:
                for batch_result in pool.imap_unordered(run_wfo_batch, batches):
                    all_results.extend(batch_result)
        except Exception as e:
            st.error(f"Multiprocessing error: {e}")
            # Fallback to single-process
            init_wfo_worker(copy.deepcopy(selected_strategy), all_periods,
                            dict(indicator_params), df_featured)
            for batch in batches:
                all_results.extend(run_wfo_batch(batch))

        # Filter out failed candidates (agg is None)
        valid = [(idx, params, agg) for idx, params, agg in all_results if agg is not None]

        # Enrich every surviving candidate with MC Avg Profit so users can
        # filter / sort by it (matches Grid Search behaviour). Mutates aggs in-place.
        if valid:
            enrich_aggs_with_mc(
                [agg for _, _, agg in valid],
                mc_balance, GS_MC_N_SIMULATIONS, target_dd=5.0,
                progress_label=f"OOS {oos_idx+1}/{n_folds}: Computing MC stats")

        # Apply filters using grid-search threshold semantics (Min/Max per metric)
        filtered = [item for item in valid if passes_thresholds(item[2], thresholds)]

        # Sort
        filtered.sort(key=lambda x: x[2].get(sort_key, 0), reverse=sort_desc)

        if filtered:
            winning_params.append(filtered[0][1])  # best param combo
        else:
            # Fallback: use base strategy params
            winning_params.append(dict(indicator_params))

    # ---- Final evaluation: test each winning param set on ALL test portions ----
    progress.progress(n_folds / total_steps, text="Evaluating winning parameters on all test sets...")

    all_test_ranges = [
        (folds[i]["test_start"], folds[i]["test_end"])
        for i in range(n_folds)
    ]

    final_results = []
    from strategies.wfo_engine import (
        _extract_stats as _wfo_extract,
        split_atr_overlay, apply_atr_overlay,
        split_value_overlay, apply_value_overlay,
    )
    from indicators.calculate_indicators import recalculate_groups, changed_groups as _cg
    for params in winning_params:
        # Split overlays out of the combo dict before indicator recalc
        rest, atr_overlay = split_atr_overlay(params)
        ind_params, value_overlay = split_value_overlay(rest)

        # Incremental recalc for final evaluation too (indicator-only view)
        groups_changed = _cg(indicator_params, ind_params)
        if groups_changed:
            df_feat = df_featured.copy()
            recalculate_groups(df_feat, groups_changed, **ind_params)
        else:
            df_feat = df_featured

        # Apply both overlays to the strategy for this fold's winning combo
        strategy_for_fold = apply_value_overlay(selected_strategy, value_overlay)
        strategy_for_fold = apply_atr_overlay(strategy_for_fold, atr_overlay)

        full_index = df_feat.index
        test_stats = []
        for start_dt, end_dt in all_periods:
            in_test = any(start_dt >= r[0] and start_dt < r[1] for r in all_test_ranges)
            if not in_test:
                continue
            try:
                start_pos = full_index.searchsorted(start_dt, side="left")
                end_pos = full_index.searchsorted(end_dt, side="right") - 1
                if end_pos < start_pos:
                    continue
                ext_start = max(0, start_pos - 50)
                df_slice = df_feat.iloc[ext_start: end_pos + 1]
                if df_slice.empty:
                    continue
                ps = df_slice.index[df_slice.index.searchsorted(start_dt, side="left")]
                pe_pos = df_slice.index.searchsorted(end_dt, side="right") - 1
                pe = df_slice.index[min(pe_pos, len(df_slice.index) - 1)]
                _, stats = execute_custom_strategy_numpy(
                    df_slice.copy(), copy.deepcopy(strategy_for_fold), ps, pe
                )
                if stats is not None:
                    test_stats.append(_wfo_extract(stats))
            except Exception:
                continue

        if test_stats:
            final_results.append(aggregate_stats_dicts(test_stats))
        else:
            final_results.append(_empty_agg())

    progress.empty()

    # Cache results
    from strategies.wfo_engine import detect_atr_slots, detect_value_slots
    cache = {
        "strategy_name": strategy_name,
        "_strategy_fingerprint": strategy_fingerprint,
        "n_folds": n_folds,
        "winning_params": winning_params,
        "final_results": final_results,
        "base_settings": indicator_params,
        "used_groups": sorted(used_groups),
        "atr_slots": detect_atr_slots(selected_strategy),
        "value_slots": detect_value_slots(selected_strategy),
    }
    st.session_state["_wfo_cached_results"] = cache
    _display_wfo_results(cache)


def _display_wfo_results(cache):
    """Display WFO results: winning parameters + OOS performance."""
    strategy_name = cache["strategy_name"]
    n_folds = cache["n_folds"]
    winning_params = cache["winning_params"]
    final_results = cache["final_results"]
    base_settings = cache["base_settings"]
    used_groups = cache["used_groups"]

    st.caption(f"Strategy: **{strategy_name}** | {n_folds}-Fold Walk Forward Optimization")

    # ---- Parameters table ----
    st.markdown("**Winning Parameters per OOS Step**")
    from strategies.wfo_engine import ATR_OVERLAY_PREFIX, VALUE_OVERLAY_PREFIX
    atr_slots = cache.get("atr_slots", [])
    value_slots = cache.get("value_slots", [])
    # Collect only the params that were optimised (differ from base)
    param_keys = set()
    for group in used_groups:
        if group.startswith("atr_slot:") or group.startswith("value_slot:"):
            # ATR / value slots get their own dedicated rows below
            continue
        defaults = _WFO_RANGES.get(group, {})
        if group == "ema":
            ema_count = len(base_settings.get("ema_periods", []))
            for i in range(ema_count):
                param_keys.add(f"EMA {i+1}")
        else:
            for param in defaults:
                if not param.startswith("_"):
                    param_keys.add(param)

    param_rows = {}
    for pk in sorted(param_keys):
        row = []
        for params in winning_params:
            if pk.startswith("EMA "):
                idx = int(pk.split(" ")[1]) - 1
                ema_list = params.get("ema_periods", [])
                row.append(str(ema_list[idx]) if idx < len(ema_list) else "—")
            else:
                val = params.get(pk)
                if val is not None:
                    row.append(f"{val:.3f}" if isinstance(val, float) else str(val))
                else:
                    row.append("—")
        param_rows[pk] = row

    # ATR slot rows
    for slot in atr_slots:
        slot_id = slot["slot_id"]
        period_label = f"{slot['label']} — period"
        mult_label = f"{slot['label']} — multiplier"
        period_row = []
        mult_row = []
        for params in winning_params:
            p = params.get(f"{ATR_OVERLAY_PREFIX}{slot_id}__atr_period")
            m = params.get(f"{ATR_OVERLAY_PREFIX}{slot_id}__atr_multiplier")
            period_row.append(str(int(p)) if p is not None else str(slot["atr_period"]))
            mult_row.append(f"{float(m):.2f}" if m is not None else f"{slot['atr_multiplier']:.2f}")
        param_rows[period_label] = period_row
        param_rows[mult_label] = mult_row

    # Value slot rows
    for slot in value_slots:
        slot_id = slot["slot_id"]
        row = []
        for params in winning_params:
            v = params.get(f"{VALUE_OVERLAY_PREFIX}{slot_id}")
            row.append(f"{float(v):.4g}" if v is not None else f"{slot['value']:.4g}")
        param_rows[slot["label"]] = row

    col_names = [f"OOS {i+1}" for i in range(n_folds)]
    params_table = pd.DataFrame(param_rows, index=col_names).T
    st.table(params_table)

    # ---- Performance table ----
    st.markdown("**Aggregated OOS Performance (All Test Sets Combined)**")
    results_dict = {}
    for i, agg in enumerate(final_results):
        results_dict[f"OOS {i+1} Params"] = agg

    perf_table = _build_metrics_table(results_dict)
    st.table(perf_table)
    _copy_to_clipboard(perf_table.to_csv(sep='\t', header=False, index=False), key="wfo_copy")


# ------------------------------------------------------------------
# Simple Optimization (single 80/20 split)
# ------------------------------------------------------------------

SIMPLE_OPT_TOP_N = 10


def _run_simple_opt(df_key, indicator_params, selected_strategy, strategy_name,
                    in_start, in_end, oos_start, oos_end,
                    all_combos, drm_bullish, drm_bearish,
                    used_groups, param_ranges, grid_size):
    """Single 80/20 optimisation: every combo runs on in-sample, top N by
    sort metric runs additionally on out-of-sample. Caches both result sets.
    """
    df_ohlc = st.session_state[df_key]
    if df_ohlc.empty:
        st.warning("No OHLC data available.")
        return

    # Pre-parse all DRM periods once (same shape WFO uses)
    all_periods = []
    for ptype, primary, secondary in all_combos:
        drm_df = drm_bullish if ptype == "Bullish" else drm_bearish
        if drm_df is None:
            continue
        for s_dt, e_dt in parse_drm_periods(drm_df, ptype, primary, secondary):
            all_periods.append((s_dt, e_dt))

    if not all_periods:
        st.warning("No DRM periods found for the selected patterns.")
        return

    # Generate parameter grid
    param_grid = generate_param_grid(used_groups, param_ranges, indicator_params)

    # Read filter / sort settings from the sopt-prefixed widget keys
    thresholds = {}
    for key, label in SORT_METRICS:
        _, default, _, _ = filter_props(key, label)
        thresholds[key] = st.session_state.get(f"sopt_filter_{key}", default)
    sort_labels = [m[1] for m in SORT_METRICS]
    default_sort_label = next(
        (label for k, label in SORT_METRICS if k == _DEFAULT_SORT_KEY), sort_labels[0])
    sort_label = st.session_state.get("sopt_sort_metric", default_sort_label)
    sort_key = next((k for k, l in SORT_METRICS if l == sort_label), _DEFAULT_SORT_KEY)
    sort_desc = st.session_state.get("sopt_sort_order", "Highest to Lowest") == "Highest to Lowest"

    mc_balance = st.session_state.get('mc_starting_balance', 10000.0)

    progress = st.progress(0, text="Calculating base indicators...")
    df_featured = calculate_indicators(df_ohlc, **indicator_params)

    n_workers = max(1, multiprocessing.cpu_count() - 1)
    batch_size = max(1, math.ceil(len(param_grid) / (n_workers * 4)))
    strategy_fingerprint = json.dumps(selected_strategy, sort_keys=True, default=str)

    # ---- In-sample run ----
    in_ranges = [(in_start, in_end)]
    batches = []
    for b_start in range(0, len(param_grid), batch_size):
        b_end = min(b_start + batch_size, len(param_grid))
        batches.append((list(range(b_start, b_end)), param_grid[b_start:b_end], in_ranges))

    progress.progress(0.05, text=f"Running {len(param_grid):,} combos on in-sample...")
    in_results = []
    try:
        with multiprocessing.Pool(
            n_workers,
            initializer=init_wfo_worker,
            initargs=(copy.deepcopy(selected_strategy), all_periods,
                      dict(indicator_params), df_featured),
        ) as pool:
            for batch_result in pool.imap_unordered(run_wfo_batch, batches):
                in_results.extend(batch_result)
    except Exception as e:
        st.error(f"Multiprocessing error: {e}")
        init_wfo_worker(copy.deepcopy(selected_strategy), all_periods,
                        dict(indicator_params), df_featured)
        for batch in batches:
            in_results.extend(run_wfo_batch(batch))

    valid = [(idx, params, agg) for idx, params, agg in in_results if agg is not None]
    if not valid:
        progress.empty()
        st.warning("No combos produced any trades on the in-sample period.")
        return

    progress.progress(0.5, text=f"Computing MC stats for {len(valid):,} candidates...")
    enrich_aggs_with_mc(
        [agg for _, _, agg in valid],
        mc_balance, GS_MC_N_SIMULATIONS, target_dd=5.0,
        progress_label="Simple Opt: in-sample MC")

    # Filter + sort + take top N
    filtered = [item for item in valid if passes_thresholds(item[2], thresholds)]
    filtered.sort(key=lambda x: x[2].get(sort_key, 0) or 0, reverse=sort_desc)
    top_n = filtered[:SIMPLE_OPT_TOP_N]

    # ---- Out-of-sample evaluation for the top N ----
    progress.progress(0.75, text=f"Evaluating top {len(top_n)} on out-of-sample...")
    oos_ranges = [(oos_start, oos_end)]
    if top_n:
        oos_top_indices = [idx for idx, _, _ in top_n]
        oos_top_combos = [params for _, params, _ in top_n]
        oos_batch = (oos_top_indices, oos_top_combos, oos_ranges)
        try:
            with multiprocessing.Pool(
                n_workers,
                initializer=init_wfo_worker,
                initargs=(copy.deepcopy(selected_strategy), all_periods,
                          dict(indicator_params), df_featured),
            ) as pool:
                oos_results_raw = list(pool.imap_unordered(run_wfo_batch, [oos_batch]))
            oos_flat = []
            for batch in oos_results_raw:
                oos_flat.extend(batch)
        except Exception:
            init_wfo_worker(copy.deepcopy(selected_strategy), all_periods,
                            dict(indicator_params), df_featured)
            oos_flat = run_wfo_batch(oos_batch)

        # Index OOS aggs by original grid index for stable matching
        oos_by_idx = {idx: agg for idx, _, agg in oos_flat if agg is not None}

        if oos_by_idx:
            enrich_aggs_with_mc(
                list(oos_by_idx.values()),
                mc_balance, GS_MC_N_SIMULATIONS, target_dd=5.0,
                progress_label="Simple Opt: OOS MC")
    else:
        oos_by_idx = {}

    progress.empty()

    # Build final cache
    from strategies.wfo_engine import (
        detect_atr_slots, detect_value_slots,
        ATR_OVERLAY_PREFIX, VALUE_OVERLAY_PREFIX,
    )
    cache = {
        "strategy_name": strategy_name,
        "_strategy_fingerprint": strategy_fingerprint,
        "in_start": in_start, "in_end": in_end,
        "oos_start": oos_start, "oos_end": oos_end,
        "total_combos": len(param_grid),
        "filtered_count": len(filtered),
        "top_n": top_n,                  # list of (grid_idx, params, in_sample_agg)
        "oos_by_idx": oos_by_idx,        # {grid_idx: oos_agg}
        "sort_label": sort_label,
        "base_settings": indicator_params,
        "used_groups": sorted(used_groups),
        "atr_slots": detect_atr_slots(selected_strategy),
        "value_slots": detect_value_slots(selected_strategy),
    }
    st.session_state["_sopt_cached_results"] = cache
    _display_simple_opt_results(cache)


def _display_simple_opt_results(cache):
    """Display Simple Optimization results: top N candidates with their winning
    parameters and both in-sample + out-of-sample metric tables."""
    from strategies.wfo_engine import ATR_OVERLAY_PREFIX, VALUE_OVERLAY_PREFIX

    strategy_name = cache["strategy_name"]
    top_n = cache.get("top_n", [])
    oos_by_idx = cache.get("oos_by_idx", {})
    base_settings = cache.get("base_settings", {})
    used_groups = cache.get("used_groups", [])
    atr_slots = cache.get("atr_slots", [])
    value_slots = cache.get("value_slots", [])

    st.caption(
        f"Strategy: **{strategy_name}** | Sorted by **{cache.get('sort_label', '?')}** | "
        f"{cache.get('filtered_count', 0):,} of {cache.get('total_combos', 0):,} combos passed filters"
    )

    if not top_n:
        st.warning("No candidates passed the filters.")
        return

    n_shown = len(top_n)
    st.markdown(f"**Top {n_shown} candidates** (in-sample → out-of-sample)")

    # Build results-dict for _build_metrics_table: 2 columns per candidate
    # (IS / OOS), labelled with the candidate rank.
    results_dict = OrderedDict()
    for rank, (grid_idx, params, in_agg) in enumerate(top_n, start=1):
        results_dict[f"#{rank} IS"] = in_agg
        oos_agg = oos_by_idx.get(grid_idx)
        if oos_agg is None:
            oos_agg = _empty_agg()
        results_dict[f"#{rank} OOS"] = oos_agg

    perf_table = _build_metrics_table(results_dict)
    st.table(perf_table)
    _copy_to_clipboard(perf_table.to_csv(sep='\t', header=False, index=False),
                       key="sopt_perf_copy")

    # ---- Winning parameters per candidate ----
    st.markdown(f"**Winning parameters per candidate**")
    param_keys = set()
    for group in used_groups:
        if group.startswith("atr_slot:") or group.startswith("value_slot:"):
            continue
        defaults = _WFO_RANGES.get(group, {})
        if group == "ema":
            ema_count = len(base_settings.get("ema_periods", []))
            for i in range(ema_count):
                param_keys.add(f"EMA {i + 1}")
        else:
            for param in defaults:
                if not param.startswith("_"):
                    param_keys.add(param)

    rows_by_param = {}
    col_names = [f"#{r}" for r in range(1, n_shown + 1)]

    for pk in sorted(param_keys):
        row = []
        for _, params, _ in top_n:
            if pk.startswith("EMA "):
                idx = int(pk.split(" ")[1]) - 1
                ema_list = params.get("ema_periods", [])
                row.append(str(ema_list[idx]) if idx < len(ema_list) else "—")
            else:
                v = params.get(pk)
                if v is not None:
                    row.append(f"{v:.3f}" if isinstance(v, float) else str(v))
                else:
                    row.append("—")
        rows_by_param[pk] = row

    for slot in atr_slots:
        slot_id = slot["slot_id"]
        period_row = []
        mult_row = []
        for _, params, _ in top_n:
            p = params.get(f"{ATR_OVERLAY_PREFIX}{slot_id}__atr_period")
            m = params.get(f"{ATR_OVERLAY_PREFIX}{slot_id}__atr_multiplier")
            period_row.append(str(int(p)) if p is not None else str(slot["atr_period"]))
            mult_row.append(f"{float(m):.2f}" if m is not None else f"{slot['atr_multiplier']:.2f}")
        rows_by_param[f"{slot['label']} — period"] = period_row
        rows_by_param[f"{slot['label']} — multiplier"] = mult_row

    for slot in value_slots:
        slot_id = slot["slot_id"]
        row = []
        for _, params, _ in top_n:
            v = params.get(f"{VALUE_OVERLAY_PREFIX}{slot_id}")
            row.append(f"{float(v):.4g}" if v is not None else f"{slot['value']:.4g}")
        rows_by_param[slot["label"]] = row

    if rows_by_param:
        params_table = pd.DataFrame(rows_by_param, index=col_names).T
        st.table(params_table)
        _copy_to_clipboard(params_table.to_csv(sep='\t', header=False, index=False),
                           key="sopt_params_copy")


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


# Share the metrics table + empty-agg definitions with Grid Search and the
# Performance tab so all three display identical metric sets.
_build_metrics_table = _perf_build_metrics_table
_empty_agg = _perf_empty_agg

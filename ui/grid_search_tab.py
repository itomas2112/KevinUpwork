"""
Grid Search tab — load a saved strategy, swap components with pre-saved group sets,
and batch-run backtests across all candidates.
"""
import streamlit as st
import streamlit.components.v1 as _components
import pandas as pd
import copy
import math
import os
import multiprocessing
from collections import OrderedDict

from data.loader import parse_drm_periods
from data.helpers import (PRIMARY_SECONDARY_MAP, PRIMARY_LIST,
                          ALL_UNIQUE_SECONDARIES, expand_selection, selection_label)
from indicators.calculate_indicators import slice_for_graph, migrate_indicator_settings
from strategies.group_set_manager import (
    load_group_sets, save_group_set, update_group_set, delete_group_set,
    export_group_set, import_group_set,
    save_group_sets_to_file, get_mode,
    MODE_RUNTIME, MODE_PER_CANDIDATE,
    extract_groups_from_set, extract_value_eligible_groups, apply_view,
    candidate_groups,
    candidate_wfo_groups, group_variant_combos, offset_label,
    INDICATOR_PRIMARY_PARAM,
)
from ui.charting_tab import _get_or_calculate
from ui.performance_tab import (_DEFAULT_INDICATOR_PARAMS, SELECTION_MODES,
                                 _build_metrics_table, _copy_to_clipboard, _empty_agg)
from ui.grid_search_helpers import (
    format_candidate_label, format_run_label, generate_run_configs,
    collect_gs_indicator_settings,
)
from config.constants import (GROUP_NAMES, EVENT_TYPES, STOP_EVENT_TYPES,
                               CONDITION_OPERATORS, get_group_elements,
                               R_PROFIT_LOSS_ELEMENTS, ATR_TARGET_ELEMENTS)

# Search component types (what part of the strategy to swap)
SEARCH_COMPONENTS = [
    ("trigger", "Trigger"),
    ("condition", "Condition"),
    ("static_stop", "Static Stop"),
    ("dynamic_stop", "Dynamic Stop"),
    ("target", "Target"),
]

SORT_METRICS = [
    ("expected_value", "Expected Value (R)"),
    ("num_trades", "Number of Trades"),
    ("win_pct", "Win %"),
    ("lose_pct", "Lose %"),
    ("mc_avg_profit", "MC Avg Profit @ 5% DD"),
    ("avg_win_pnl", "Avg Profit (R)"),
    ("avg_lose_pnl", "Avg Loss (R)"),
    ("total_pnl", "Total P&L (R)"),
    ("target_exit_pct", "Target Exit %"),
    ("static_exit_pct", "Static %"),
    ("dynamic_exit_pct", "Dynamic %"),
    ("rr_ratio", "Avg RR Ratio"),
    ("abs_correlation", "|Correlation|"),
    ("avg_holding_period", "Avg Holding (periods)"),
]

PAGE_SIZE = 50

# Grid Search uses a lower MC simulation count than the Monte Carlo tab —
# keeps the per-candidate enrichment fast across hundreds/thousands of candidates.
GS_MC_N_SIMULATIONS = 1000

# Metrics where the threshold acts as a maximum (value must be <= threshold).
# All other metrics in SORT_METRICS use minimum semantics (value must be >= threshold).
MAX_FILTER_METRICS = {"abs_correlation", "avg_holding_period"}


def filter_props(metric_key, metric_label):
    """Return (label, default, step, fmt) for a filter input.

    Defaults are chosen so that the out-of-the-box value disables the filter.
    Min metrics: very low default. Max metrics: very high default.
    """
    if metric_key == "num_trades":
        return f"Min {metric_label}", 0.0, 1.0, "%.0f"
    if metric_key == "avg_lose_pnl":
        return f"Max {metric_label}", -999.0, 0.01, "%.2f"
    if metric_key == "mc_avg_profit":
        return f"Min {metric_label}", 0.0, 1000.0, "%.0f"
    if metric_key == "abs_correlation":
        return f"Max {metric_label}", 100.0, 1.0, "%.0f"
    if metric_key == "avg_holding_period":
        # Default high so unset filter admits everything (holding periods are
        # typically < 1,000 bars). Lower the value to cap long-holding candidates.
        return f"Max {metric_label}", 100000.0, 1.0, "%.1f"
    return f"Min {metric_label}", -999.0, 0.01, "%.2f"


def passes_thresholds(agg, thresholds):
    """Return True if `agg` passes every (key -> threshold) bound in `thresholds`.

    Honours MAX_FILTER_METRICS for metrics where the threshold is an upper bound.
    Missing values (None) skip that filter rather than excluding the candidate.
    """
    for metric_key, threshold_val in thresholds.items():
        val = agg.get(metric_key)
        if val is None:
            continue
        if metric_key in MAX_FILTER_METRICS:
            if val > threshold_val:
                return False
        else:
            if val < threshold_val:
                return False
    return True


def enrich_aggs_with_mc(agg_dicts, balance, n_sims, target_dd=5.0,
                        progress_label="Computing Monte Carlo stats"):
    """Enrich a flat list of agg dicts with `mc_avg_profit` (in-place).

    Uses each agg's own `num_trades` as trades/sim. Spawns a multiprocessing
    pool; falls back to serial execution if pool setup fails. Aggs with
    num_trades == 0 are filled with 0.0 without entering the pool.
    """
    from strategies.mc_enrichment_worker import init_worker as mc_init, enrich_one
    from strategies.monte_carlo_core import compute_mc_avg_profit_at_target_dd

    tasks = []  # (idx, win_pct, rr_ratio, num_trades)
    for idx, agg in enumerate(agg_dicts):
        n_trades = agg.get('num_trades', 0)
        if n_trades == 0:
            agg['mc_avg_profit'] = 0.0
            continue
        tasks.append((idx, agg.get('win_pct', 0),
                      agg.get('rr_ratio', 0), n_trades))

    if not tasks:
        return

    n_workers = min(len(tasks), max(1, os.cpu_count() or 1))
    progress = st.progress(0, text=f"{progress_label} ({len(tasks)} aggs, {n_workers} workers)...")

    try:
        with multiprocessing.Pool(
            processes=n_workers,
            initializer=mc_init,
            initargs=(balance, n_sims, target_dd),
        ) as pool:
            completed = 0
            for idx, value in pool.imap_unordered(enrich_one, tasks, chunksize=4):
                agg_dicts[idx]['mc_avg_profit'] = value
                completed += 1
                progress.progress(completed / len(tasks),
                                  text=f"{progress_label} {completed}/{len(tasks)}")
    except Exception as e:
        st.warning(f"MC parallel enrichment failed ({e}); falling back to serial.")
        for idx, win_pct, rr_ratio, num_trades in tasks:
            agg_dicts[idx]['mc_avg_profit'] = compute_mc_avg_profit_at_target_dd(
                win_pct, rr_ratio, balance, target_dd=target_dd,
                trades_per_sim=num_trades, n_sims=n_sims)

    progress.empty()


# ======================================================================
# Main entry point
# ======================================================================

def render_grid_search_tab(sidebar_config):
    """Render the Grid Search tab."""
    st.subheader("Grid Search")

    # ── Data checks ─────────────────────────────────────
    df_key = "df_ohlc"
    if df_key not in st.session_state:
        st.info("Please upload OHLC data in the Charting tab.")
        return

    drm_bullish = st.session_state.get("drm_bullish")
    drm_bearish = st.session_state.get("drm_bearish")
    if drm_bullish is None and drm_bearish is None:
        st.info("Please upload a DRM file in the Charting tab.")
        return

    if not sidebar_config.get("date_range_applied", False):
        st.info("Please apply a Training Set date range in the sidebar.")
        return

    # ── Section A: Strategy Loader ──────────────────────
    saved = st.session_state.get("saved_strategies", [])
    if not saved:
        st.info("No saved strategies. Create one in the Strategy Builder tab.")
        return

    strategy_names = [s.get("strategy_name", f"Strategy_{i+1}") for i, s in enumerate(saved)]
    sel_col, info_col = st.columns([1, 3])
    with sel_col:
        selected_idx = st.selectbox(
            "Load Strategy", range(len(strategy_names)),
            format_func=lambda x: strategy_names[x],
            key="gs_strategy_select")
    selected_strategy = saved[selected_idx]

    with info_col:
        st.markdown(f"**Direction:** {selected_strategy.get('direction', '?')}  "
                    f"| **Max Positions:** {selected_strategy.get('max_positions', 1) or 'Unlimited'}")

    st.markdown("---")

    # ── Section B: Group Set Management ─────────────────
    with st.expander("Group Set Management", expanded=False):
        _render_group_set_management()

    st.markdown("---")

    # ── Section C: Search Configuration ─────────────────
    st.markdown("**Search Configuration**")

    all_group_sets = st.session_state.get("saved_group_sets", [])
    if not all_group_sets:
        st.warning("No group sets saved. Create one in **Group Set Management** above.")
        return

    sc1, sc2 = st.columns(2)
    with sc1:
        search_type_labels = [label for _, label in SEARCH_COMPONENTS]
        search_type_idx = st.selectbox(
            "Component to Search",
            range(len(SEARCH_COMPONENTS)),
            format_func=lambda x: search_type_labels[x],
            key="gs_search_type_idx")
        search_group = SEARCH_COMPONENTS[search_type_idx][0]

    with sc2:
        gs_names = [gs["name"] for gs in all_group_sets]
        gs_sel = st.selectbox(
            "Group Set to Use",
            range(len(gs_names)),
            format_func=lambda x: gs_names[x],
            key="gs_search_set_sel")
        search_set = all_group_sets[gs_sel]

    search_set_mode = get_mode(search_set)

    # Filter view: group toggles + per-group value ranges (mutates search_set in place if user clicks Save)
    effective_candidates, view_state = _render_set_view_filter(search_set, gs_sel)

    # Event / Operator multi-select — only for MODE_RUNTIME sets.
    # MODE_PER_CANDIDATE sets carry their event on each candidate.
    if search_set_mode == MODE_PER_CANDIDATE:
        st.caption("This group set has an event embedded per candidate — "
                   "no global event selection needed.")
        selected_events = None  # signals per-candidate mode downstream
    else:
        if search_group == "condition":
            selected_events = st.multiselect(
                "Operators", CONDITION_OPERATORS,
                default=CONDITION_OPERATORS,
                key="gs_events")
        elif search_group == "trigger":
            selected_events = st.multiselect(
                "Events", EVENT_TYPES,
                default=["Cross Above", "Cross Below"],
                key="gs_events")
        else:
            selected_events = st.multiselect(
                "Events", STOP_EVENT_TYPES,
                default=["Cross Above", "Cross Below"],
                key="gs_events")

        if not selected_events:
            st.warning("Select at least one event/operator.")
            return

    # Cross-combination: condition group set for target/dynamic
    condition_candidates = None
    condition_event = None
    if search_group in ("target", "dynamic_stop"):
        cond_options_names = ["No cross-combination"] + gs_names
        cond_sel = st.selectbox(
            "Condition Group Set (cross-combine)",
            range(len(cond_options_names)),
            format_func=lambda x: cond_options_names[x],
            key="gs_cross_cond_sel")
        if cond_sel > 0:
            condition_candidates = all_group_sets[cond_sel - 1]["candidates"]
            condition_event = st.selectbox(
                "Condition Operator",
                CONDITION_OPERATORS,
                key="gs_cond_event")

    # Show run count (after filter)
    n_total = len(search_set.get("candidates", []))
    n_search = len(effective_candidates)
    if n_search != n_total:
        st.caption(f"Filter view: **{n_search}** of {n_total} candidates after toggles/ranges.")
    n_events = 1 if search_set_mode == MODE_PER_CANDIDATE else len(selected_events)
    n_cond = len(condition_candidates) if condition_candidates else 0
    if search_group in ("target", "dynamic_stop") and n_cond > 0:
        total_runs = n_search * n_events * (n_cond + 1)
        if search_set_mode == MODE_PER_CANDIDATE:
            st.info(f"**{n_search}** candidates (per-candidate events) x **{n_cond + 1}** (standalone + {n_cond} conditions) = **{total_runs}** total runs")
        else:
            st.info(f"**{n_search}** candidates x **{n_events}** events x **{n_cond + 1}** (standalone + {n_cond} conditions) = **{total_runs}** total runs")
    else:
        total_runs = n_search * n_events
        if search_set_mode == MODE_PER_CANDIDATE:
            st.info(f"**{n_search}** candidates (per-candidate events) = **{total_runs}** total runs")
        else:
            st.info(f"**{n_search}** candidates x **{n_events}** events = **{total_runs}** total runs")

    st.markdown("---")

    # ── Section D: Pattern Selection ────────────────────
    st.markdown("**Pattern Selection**")
    _render_pattern_selection()

    st.markdown("---")

    # ── Section E: Indicator Settings ───────────────────
    st.caption(
        "Indicator settings come from the selected strategy and the group "
        "set's *Indicator Ranges* (edit them in **Group Set Management** "
        "above). The old global Indicator Settings panel has been retired."
    )

    st.markdown("---")

    # ── Section F: Correlation ─────────────────────────
    _render_reference_strategy_selection(saved, strategy_names, selected_idx)

    st.markdown("---")

    # ── Section G: Filters & Sort ───────────────────────
    st.markdown("**Performance Filters**")
    # Build threshold inputs for every metric in SORT_METRICS
    thresholds = {}

    # Row 1 — first 5 metrics (includes MC Avg Profit @ 5% DD)
    cols_r1 = st.columns(5)
    for ci, (metric_key, metric_label) in enumerate(SORT_METRICS[:5]):
        with cols_r1[ci]:
            lbl, default, step, fmt = filter_props(metric_key, metric_label)
            thresholds[metric_key] = st.number_input(
                lbl, value=default, step=step,
                format=fmt, key=f"gs_thresh_{metric_key}")
    # Row 2 — next 5 metrics
    cols_r2 = st.columns(5)
    for ci, (metric_key, metric_label) in enumerate(SORT_METRICS[5:10]):
        with cols_r2[ci]:
            lbl, default, step, fmt = filter_props(metric_key, metric_label)
            thresholds[metric_key] = st.number_input(
                lbl, value=default, step=step,
                format=fmt, key=f"gs_thresh_{metric_key}")
    # Row 3 — remaining metrics
    remaining = SORT_METRICS[10:]
    if remaining:
        cols_r3 = st.columns(max(len(remaining), 1))
        for ci, (metric_key, metric_label) in enumerate(remaining):
            with cols_r3[ci]:
                lbl, default, step, fmt = filter_props(metric_key, metric_label)
                thresholds[metric_key] = st.number_input(
                    lbl, value=default, step=step,
                    format=fmt, key=f"gs_thresh_{metric_key}")

    sort_col1, sort_col2 = st.columns(2)
    with sort_col1:
        sort_metric_labels = [label for _, label in SORT_METRICS]
        default_sort_key = "mc_avg_profit"
        default_sort_idx = next(
            (i for i, (k, _) in enumerate(SORT_METRICS) if k == default_sort_key), 0)
        sort_idx = st.selectbox("Sort By", range(len(SORT_METRICS)),
                                index=default_sort_idx,
                                format_func=lambda x: sort_metric_labels[x],
                                key="gs_sort_metric")
        sort_key = SORT_METRICS[sort_idx][0]
    with sort_col2:
        sort_order = st.radio("Order", ["Highest to Lowest", "Lowest to Highest"],
                              horizontal=True, key="gs_sort_order")
        sort_descending = sort_order == "Highest to Lowest"

    st.markdown("---")

    # ── Section G: Calculate + Results ──────────────────
    # Hidden debug toggle — set gs_use_original_engine in session state to True to use original engine
    use_original_engine = st.session_state.get("gs_use_original_engine", False)
    calculate_clicked = st.button("Calculate", key="gs_calculate", type="primary")
    st.caption("To stop a running calculation, click the **Stop** button (top-right corner) or refresh the page.")

    # Cache invalidation
    cached = st.session_state.get("_gs_cached_results")
    if cached:
        fp = _build_cache_fingerprint(selected_strategy, search_group,
                                       search_set, selected_events, condition_candidates,
                                       view_state)
        if cached.get("fingerprint") != fp:
            st.session_state.pop("_gs_cached_results", None)
            cached = None

    if calculate_clicked:
        results = _run_grid_search(
            selected_strategy, search_group, search_set, selected_events,
            condition_candidates, condition_event, sidebar_config,
            effective_candidates=effective_candidates,
            use_original_engine=use_original_engine)
        fp = _build_cache_fingerprint(selected_strategy, search_group,
                                       search_set, selected_events, condition_candidates,
                                       view_state)
        sel_labels = [selection_label(s) for s in st.session_state.get("gs_selections", [])]
        st.session_state["_gs_cached_results"] = {
            "fingerprint": fp,
            "results": results,
            "strategy_name": selected_strategy.get("strategy_name", "Custom"),
            "selection_labels": sel_labels,
        }
        cached = st.session_state["_gs_cached_results"]

    if cached and cached.get("results"):
        _display_results(cached["results"], cached["strategy_name"],
                         thresholds, sort_key, sort_descending)
    elif not calculate_clicked:
        st.info("Configure search and click **Calculate** to run.")


# ======================================================================
# Set "view" filter — toggle indicator groups on/off + per-group value ranges
# (operates on the SELECTED set in the Search Configuration section)
# ======================================================================

def _render_set_view_filter(search_set, set_idx):
    """Render the indicator-toggle / value-range filter beneath the selected
    group set. Returns (filtered_candidates, view_state_dict).

    view_state_dict captures the current (potentially unsaved) widget state
    so the cache fingerprint can react to user changes.
    """
    candidates = search_set.get("candidates", [])
    set_name = search_set.get("name", "")

    if not candidates:
        return [], {}

    available_groups = extract_groups_from_set(candidates)
    value_eligible = extract_value_eligible_groups(candidates)

    # Saved view (defaults: all groups active, no value filters)
    saved_active = search_set.get("active_groups")
    if saved_active is None:
        saved_active = list(available_groups)
    else:
        # Prune any saved groups that no longer exist in the set's candidates
        saved_active = [g for g in saved_active if g in available_groups]
    saved_filters = dict(search_set.get("value_filters") or {})

    # Per-set widget keys so different sets don't clobber each other
    safe_id = f"{set_idx}_{set_name}"
    active_key = f"gs_view_active__{safe_id}"
    reset_flag_key = f"gs_view_reset_flag__{safe_id}"

    # Reset flag: if Reset was clicked last run, force widget defaults back
    # to "everything on / no ranges" before instantiating the widgets.
    if st.session_state.pop(reset_flag_key, False):
        st.session_state.pop(active_key, None)
        for g in value_eligible:
            for f in ("low", "high", "step"):
                st.session_state.pop(f"gs_view_{f}__{safe_id}__{g}", None)
        saved_active = list(available_groups)
        saved_filters = {}

    with st.expander(f"Filter view ({len(candidates)} candidates available)",
                     expanded=False):
        st.caption("Untick indicators to drop their candidates for this run. "
                   "For oscillators, set a value range to keep only matching "
                   "fixed-value candidates. Click **Save Filter to Set** to "
                   "persist; **Reset** to drop the saved view.")

        # Group toggle multiselect (defaults to saved selection)
        if active_key not in st.session_state:
            st.session_state[active_key] = saved_active
        active_selected = st.multiselect(
            "Active indicator groups",
            options=available_groups,
            key=active_key,
        )

        # Per-group value range inputs (only for value-eligible groups
        # that are currently active)
        value_filters = {}
        if value_eligible:
            st.markdown("**Value ranges (Fixed Value candidates)**")
            for g in value_eligible:
                if g not in active_selected:
                    continue
                saved_vf = saved_filters.get(g, {})
                # Sensible default range based on existing values in this group
                vals_in_group = [c["value"] for c in candidates
                                 if c.get("compare_type") == "Fixed Value"
                                 and c.get("value") is not None
                                 and g in candidate_groups(c)]
                if vals_in_group:
                    auto_low = float(min(vals_in_group))
                    auto_high = float(max(vals_in_group))
                else:
                    auto_low, auto_high = 0.0, 100.0

                low_key = f"gs_view_low__{safe_id}__{g}"
                high_key = f"gs_view_high__{safe_id}__{g}"
                step_key = f"gs_view_step__{safe_id}__{g}"
                if low_key not in st.session_state:
                    st.session_state[low_key] = float(saved_vf.get("low", auto_low))
                if high_key not in st.session_state:
                    st.session_state[high_key] = float(saved_vf.get("high", auto_high))
                if step_key not in st.session_state:
                    st.session_state[step_key] = float(saved_vf.get("step", 0))

                col_lbl, col_lo, col_hi, col_st, col_off = st.columns([2, 1.2, 1.2, 1.2, 1])
                with col_lbl:
                    st.markdown(f"`{g}`")
                with col_lo:
                    low = st.number_input("Low", key=low_key, step=1.0, format="%.4f")
                with col_hi:
                    high = st.number_input("High", key=high_key, step=1.0, format="%.4f")
                with col_st:
                    step = st.number_input("Step (0 = any)", key=step_key,
                                            min_value=0.0, step=0.5, format="%.4f")
                with col_off:
                    st.markdown("&nbsp;", unsafe_allow_html=True)
                    enabled = st.checkbox("Apply", value=g in saved_filters,
                                          key=f"gs_view_apply__{safe_id}__{g}")
                if enabled:
                    value_filters[g] = {"low": low, "high": high, "step": step}

        # Save / Reset buttons — persist or clear the view on the saved set
        bcol1, bcol2 = st.columns(2)
        with bcol1:
            if st.button("Save Filter to Set", key=f"gs_view_save__{safe_id}"):
                search_set["active_groups"] = list(active_selected)
                search_set["value_filters"] = value_filters
                update_group_set(set_idx, search_set)
                st.success("Filter view saved to the group set.")
                st.rerun()
        with bcol2:
            if st.button("Reset (clear saved view)", key=f"gs_view_reset__{safe_id}"):
                search_set.pop("active_groups", None)
                search_set.pop("value_filters", None)
                update_group_set(set_idx, search_set)
                st.session_state[reset_flag_key] = True
                st.success("Filter view cleared.")
                st.rerun()

    filtered = apply_view(candidates,
                          active_groups=set(active_selected) if active_selected is not None else None,
                          value_filters=value_filters)
    view_state = {
        "active": sorted(active_selected) if active_selected else [],
        "filters": value_filters,
    }
    return filtered, view_state


# ======================================================================
# Group Set Management UI
# ======================================================================

def _render_group_set_management():
    """Render the universal group set create/edit/delete/import/export UI."""
    all_sets = st.session_state.get("saved_group_sets", [])

    if all_sets:
        gs_names = [gs["name"] for gs in all_sets]
        sel = st.selectbox("Saved Group Sets", range(len(gs_names)),
                           format_func=lambda x, n=gs_names: n[x],
                           key="gs_mgmt_sel")
        selected_gs = all_sets[sel]
        sel_mode = get_mode(selected_gs)

        # Show candidates
        if selected_gs.get("candidates"):
            cand_labels = [format_candidate_label(c) for c in selected_gs["candidates"]]
            mode_lbl = ("event chosen at search time"
                        if sel_mode == MODE_RUNTIME
                        else "event embedded per candidate")
            st.caption(f"{len(cand_labels)} candidates — *{mode_lbl}*")
            with st.expander("View candidates", expanded=False):
                for i, lbl in enumerate(cand_labels):
                    cand = selected_gs["candidates"][i]
                    if sel_mode == MODE_PER_CANDIDATE and cand.get("event"):
                        st.text(f"{i+1}. {lbl}  [{cand['event']}]")
                    else:
                        st.text(f"{i+1}. {lbl}")

        # Action buttons
        bc1, bc2, bc3 = st.columns(3)
        with bc1:
            if st.button("Delete", key="gs_mgmt_del", type="secondary"):
                delete_group_set(sel)
                st.rerun()
        with bc2:
            json_data = export_group_set(selected_gs)
            st.download_button("Export JSON", json_data,
                               file_name=f"{selected_gs['name']}.json",
                               mime="application/json",
                               key="gs_mgmt_export")
        with bc3:
            if st.button("Edit", key="gs_mgmt_edit_btn"):
                st.session_state["gs_editing"] = sel
                st.rerun()

        # Edit mode
        if st.session_state.get("gs_editing") is not None:
            edit_idx = st.session_state["gs_editing"]
            if edit_idx < len(all_sets):
                edit_gs = all_sets[edit_idx]
                edit_mode = get_mode(edit_gs)
                st.markdown("---")
                mode_lbl = ("Cross-product (event chosen at search time)"
                            if edit_mode == MODE_RUNTIME
                            else "Per-candidate event (event chosen per row)")
                st.markdown(f"**Editing: {edit_gs['name']}**  \n*Logic: {mode_lbl}*")
                edited_candidates = _render_candidate_editor(
                    list(edit_gs.get("candidates", [])), "gs_edit", mode=edit_mode)
                edited_ranges = _render_indicator_ranges_editor(
                    edited_candidates, edit_gs.get("indicator_ranges") or {}, "gs_edit")
                ec1, ec2 = st.columns(2)
                with ec1:
                    if st.button("Save Changes", key="gs_edit_save", type="primary"):
                        edit_gs["candidates"] = _strip_uids(edited_candidates)
                        edit_gs["mode"] = edit_mode
                        edit_gs["indicator_ranges"] = edited_ranges
                        dupes = update_group_set(edit_idx, edit_gs)
                        if dupes:
                            st.warning(f"{dupes} duplicate candidate(s) removed.")
                        st.session_state.pop("gs_editing", None)
                        st.session_state.pop("gs_edit_candidates", None)
                        st.rerun()
                with ec2:
                    if st.button("Cancel", key="gs_edit_cancel"):
                        st.session_state.pop("gs_editing", None)
                        st.session_state.pop("gs_edit_candidates", None)
                        st.rerun()
    else:
        st.caption("No group sets saved yet.")

    st.markdown("---")

    # Import
    uploaded = st.file_uploader("Import Group Set",
                                type=["json"], key="gs_import")
    if uploaded:
        import_id = f"{uploaded.name}_{uploaded.size}"
        last_import_key = "_gs_last_import"
        if st.session_state.get(last_import_key) != import_id:
            try:
                data = import_group_set(uploaded.read().decode("utf-8"))
                dupes_removed = data.pop("_duplicates_removed", 0)
                save_group_set(data)
                st.session_state[last_import_key] = import_id
                msg = f"Imported **{data['name']}** with {len(data['candidates'])} candidates."
                if dupes_removed:
                    msg += f" ({dupes_removed} duplicate(s) removed.)"
                st.success(msg)
                st.rerun()
            except (ValueError, Exception) as e:
                st.error(f"Import failed: {e}")
        else:
            st.info("File already imported. Upload a different file or remove and re-upload.")

    # Create new
    with st.expander("Create New Group Set", expanded=False):
        new_name = st.text_input("Name", key="gs_new_name")

        mode_options = [
            (MODE_RUNTIME,
             "Cross-product — pick event(s) at search time"),
            (MODE_PER_CANDIDATE,
             "Per-candidate — choose an event for each row"),
        ]
        mode_idx = st.radio(
            "Logic",
            options=range(len(mode_options)),
            format_func=lambda i: mode_options[i][1],
            key="gs_new_mode_idx",
            horizontal=False,
        )
        new_mode = mode_options[mode_idx][0]

        # If user toggles mode, drop previously-built candidates so the editor
        # rebuilds with the right shape (event field present or absent).
        last_mode_key = "_gs_new_last_mode"
        if st.session_state.get(last_mode_key) != new_mode:
            st.session_state.pop("gs_new_candidates", None)
            st.session_state[last_mode_key] = new_mode

        new_candidates = _render_candidate_editor([], "gs_new", mode=new_mode)
        new_ranges = _render_indicator_ranges_editor(new_candidates, {}, "gs_new")
        if st.button("Save New Set", key="gs_new_save", type="primary"):
            if not new_name.strip():
                st.error("Please provide a name.")
            elif not new_candidates:
                st.error("Add at least one candidate.")
            else:
                new_gs = {
                    "name": new_name.strip(),
                    "mode": new_mode,
                    "candidates": _strip_uids(new_candidates),
                    "indicator_ranges": new_ranges,
                }
                dupes = save_group_set(new_gs)
                st.session_state.pop("gs_new_candidates", None)
                st.session_state.pop(last_mode_key, None)
                msg = f"Created **{new_name}** with {len(new_gs['candidates'])} candidates."
                if dupes:
                    msg += f" ({dupes} duplicate(s) removed.)"
                st.success(msg)
                st.rerun()


# ======================================================================
# Indicator Ranges editor (per Group Set)
# ======================================================================

# Friendly display names for the rangeable indicator groups.
_GROUP_KEY_DISPLAY = {
    "rsi": "RSI", "stoch": "Stochastic", "adx": "ADX", "atr": "ATR",
    "macd": "MACD", "supertrend": "Supertrend", "bb": "Bollinger Bands",
    "kc": "Keltner Channel", "donchian": "Donchian Channel", "psar": "Parabolic SAR",
    "willr": "Williams %R", "roc": "Rate of Change", "cci": "CCI",
    "lr": "Linear Regression",
}

# Parameters whose UI step + default values should be float, not int.
_FLOAT_PARAMS = {"psar_af_max", "supertrend_multiplier", "lr_multiplier",
                 "bb_upper_stdev", "bb_lower_stdev",
                 "kc_upper_mult", "kc_lower_mult"}


def _ranged_groups_in_candidates(candidates):
    """Return the sorted list of WFO group keys touched by these candidates
    AND that we know how to range (have a primary param)."""
    keys = set()
    for c in candidates:
        keys.update(candidate_wfo_groups(c))
    return sorted(k for k in keys if k in INDICATOR_PRIMARY_PARAM)


def _render_indicator_ranges_editor(candidates, current_ranges, prefix):
    """Render (min, max, step) inputs per rangeable indicator group present
    in the candidates. Returns a dict in the same shape as a Group Set's
    `indicator_ranges` field. Groups left at their default ("not ranged")
    are omitted from the returned dict."""
    eligible = _ranged_groups_in_candidates(candidates)
    if not eligible:
        return {}

    out = {}
    with st.expander(f"Indicator Ranges ({len(eligible)} group(s))",
                     expanded=False):
        st.caption(
            "For each indicator referenced by this set, optionally set a "
            "min / max / step range. During Grid Search, candidates that "
            "touch the indicator are duplicated for each value, labeled by "
            "its offset from the range midpoint: `(0)` for the middle, "
            "`(+1)` / `(-1)` per step. Untick *Apply* to skip ranging "
            "that indicator."
        )

        for grp in eligible:
            primary = INDICATOR_PRIMARY_PARAM[grp]
            display = _GROUP_KEY_DISPLAY.get(grp, grp.upper())
            saved = (current_ranges or {}).get(grp, {}).get(primary)

            apply_key = f"{prefix}_irng_apply__{grp}"
            lo_key = f"{prefix}_irng_lo__{grp}"
            hi_key = f"{prefix}_irng_hi__{grp}"
            st_key = f"{prefix}_irng_step__{grp}"

            is_float = primary in _FLOAT_PARAMS
            cast = float if is_float else int
            step_w = 0.1 if is_float else 1.0
            fmt = "%.4f" if is_float else "%d"

            if saved is not None:
                default_lo, default_hi, default_step = saved
            else:
                # Sensible defaults based on the param's type
                if is_float:
                    default_lo, default_hi, default_step = 1.0, 3.0, 0.5
                else:
                    default_lo, default_hi, default_step = 5, 50, 5

            if apply_key not in st.session_state:
                st.session_state[apply_key] = saved is not None
            if lo_key not in st.session_state:
                st.session_state[lo_key] = cast(default_lo)
            if hi_key not in st.session_state:
                st.session_state[hi_key] = cast(default_hi)
            if st_key not in st.session_state:
                st.session_state[st_key] = cast(default_step)

            c_lbl, c_apply, c_lo, c_hi, c_st = st.columns([2, 1, 1.2, 1.2, 1.2])
            with c_lbl:
                st.markdown(f"**{display}** &nbsp; `{primary}`",
                            unsafe_allow_html=True)
            with c_apply:
                st.markdown("&nbsp;", unsafe_allow_html=True)
                applied = st.checkbox("Apply", key=apply_key)
            with c_lo:
                if is_float:
                    lo = st.number_input("Min", key=lo_key, step=step_w, format=fmt)
                else:
                    lo = st.number_input("Min", key=lo_key, step=int(step_w), format=fmt)
            with c_hi:
                if is_float:
                    hi = st.number_input("Max", key=hi_key, step=step_w, format=fmt)
                else:
                    hi = st.number_input("Max", key=hi_key, step=int(step_w), format=fmt)
            with c_st:
                if is_float:
                    sp = st.number_input("Step", key=st_key, step=step_w, format=fmt,
                                          min_value=step_w / 100)
                else:
                    sp = st.number_input("Step", key=st_key, step=int(step_w), format=fmt,
                                          min_value=1)

            if applied:
                # Quick preview of the variant count and offsets
                from strategies.group_set_manager import (
                    enumerate_variant_values, compute_offsets,
                )
                vals = enumerate_variant_values(lo, hi, sp)
                offsets = compute_offsets([cast(v) for v in vals])
                if offsets:
                    preview = ", ".join(
                        f"{offset_label(o)}={v}" for o, v in offsets
                    )
                    st.caption(f"&nbsp;&nbsp;{len(offsets)} variants: {preview}",
                               unsafe_allow_html=True)
                out[grp] = {primary: [cast(lo), cast(hi), cast(sp)]}

    return out


# ======================================================================
# Candidate Editor
# ======================================================================

def _clear_candidate_widget_keys_by_uid(prefix, uid):
    """Clear all widget keys for a candidate identified by unique ID."""
    suffixes = [
        "_grp", "_e1", "_ev", "_cmp", "_e2", "_op", "_val",
        "_stype", "_atr_p", "_atr_m", "_cmp_d", "_cmp_at", "_rm",
        "_event",
    ]
    for sfx in suffixes:
        st.session_state.pop(f"{prefix}_{uid}{sfx}", None)


# Union of all events any search component might use — used by the
# per-candidate Event picker (the candidate doesn't know yet which
# component slot it'll be assigned to at search time).
_ALL_EVENT_CHOICES = list(dict.fromkeys(
    list(EVENT_TYPES) + list(STOP_EVENT_TYPES) + list(CONDITION_OPERATORS)
))


_candidate_uid_counter_key = "_gs_candidate_uid_counter"


def _strip_uids(candidates):
    """Return a deep copy of candidates with internal _uid fields removed."""
    cleaned = copy.deepcopy(candidates)
    for c in cleaned:
        c.pop('_uid', None)
    return cleaned


def _next_candidate_uid():
    """Generate a unique ID for a candidate (monotonically increasing integer)."""
    uid = st.session_state.get(_candidate_uid_counter_key, 0)
    st.session_state[_candidate_uid_counter_key] = uid + 1
    return uid


def _ensure_candidate_uids(candidates):
    """Ensure every candidate dict has a '_uid' field."""
    for cand in candidates:
        if '_uid' not in cand:
            cand['_uid'] = _next_candidate_uid()


CANDIDATE_PAGE_SIZE = 20


def _render_candidate_editor(initial_candidates, prefix, mode=MODE_RUNTIME):
    """Render an editable list of candidates with pagination. Returns list of candidate dicts.

    `mode` controls whether each row also shows an Event picker (MODE_PER_CANDIDATE).
    """
    # Use session state to track candidates for this editor
    state_key = f"{prefix}_candidates"
    if state_key not in st.session_state:
        st.session_state[state_key] = list(initial_candidates) if initial_candidates else []

    candidates = st.session_state[state_key]
    _ensure_candidate_uids(candidates)
    ema_count = len(st.session_state.get("gs_ema_periods", []))

    total = len(candidates)
    page_key = f"{prefix}_page"
    total_pages = max(1, math.ceil(total / CANDIDATE_PAGE_SIZE))

    # Clamp page to valid range
    current_page = st.session_state.get(page_key, 0)
    if current_page >= total_pages:
        current_page = max(0, total_pages - 1)
        st.session_state[page_key] = current_page

    start_idx = current_page * CANDIDATE_PAGE_SIZE
    end_idx = min(start_idx + CANDIDATE_PAGE_SIZE, total)

    # Pagination controls (top)
    if total > CANDIDATE_PAGE_SIZE:
        st.caption(f"Showing candidates {start_idx + 1}–{end_idx} of {total}")
        p1, p2, p3 = st.columns([1, 1, 1])
        with p1:
            if st.button("◀ Previous", key=f"{prefix}_prev", disabled=current_page == 0):
                st.session_state[page_key] = current_page - 1
                st.rerun()
        with p2:
            st.markdown(f"Page **{current_page + 1}** / {total_pages}")
        with p3:
            if st.button("Next ▶", key=f"{prefix}_next", disabled=current_page >= total_pages - 1):
                st.session_state[page_key] = current_page + 1
                st.rerun()

    # Render only the current page of candidates
    to_remove = []
    for i in range(start_idx, end_idx):
        cand = candidates[i]
        uid = cand['_uid']
        st.markdown(f"**Candidate {i+1}**")
        updated = _render_single_candidate(cand, f"{prefix}_{uid}", ema_count, mode=mode)
        # Preserve UID
        updated['_uid'] = uid
        candidates[i] = updated

        if st.button("Remove", key=f"{prefix}_{uid}_rm"):
            to_remove.append(i)

    if to_remove:
        for idx in sorted(to_remove, reverse=True):
            removed = candidates.pop(idx)
            _clear_candidate_widget_keys_by_uid(prefix, removed['_uid'])
        st.session_state[state_key] = candidates
        # If we deleted all items on the last page, go back one page
        new_total_pages = max(1, math.ceil(len(candidates) / CANDIDATE_PAGE_SIZE))
        if st.session_state.get(page_key, 0) >= new_total_pages:
            st.session_state[page_key] = max(0, new_total_pages - 1)
        st.rerun()

    if st.button("+ Add Candidate", key=f"{prefix}_add"):
        new_cand = _default_candidate(mode=mode)
        new_cand['_uid'] = _next_candidate_uid()
        candidates.append(new_cand)
        st.session_state[state_key] = candidates
        # Jump to the last page where the new candidate was added
        st.session_state[page_key] = math.ceil(len(candidates) / CANDIDATE_PAGE_SIZE) - 1
        st.rerun()

    return candidates


def _default_candidate(mode=MODE_RUNTIME):
    """Return a default universal candidate dict."""
    cand = {
        "group": "Price & Indicators",
        "element1": "Price",
        "compare_type": "Indicator",
        "element2": "Tenkan",
        "value": None,
    }
    if mode == MODE_PER_CANDIDATE:
        cand["event"] = "Cross Above"
    return cand


def _render_single_candidate(cand, prefix, ema_count, mode=MODE_RUNTIME):
    """Render widgets for a single universal candidate and return updated dict.

    In MODE_RUNTIME the row defines only (group, element1, element2/value);
    the event is chosen at search time. In MODE_PER_CANDIDATE the row also
    has its own event picker.
    """
    # Check if this is an ATR stop candidate
    is_atr_stop = cand.get("stop_type") == "ATR"

    # ATR Stop toggle
    atr_stop_checked = st.checkbox("ATR Stop", value=is_atr_stop, key=f"{prefix}_atr_stop")

    if atr_stop_checked:
        if mode == MODE_PER_CANDIDATE:
            evt_default = cand.get("event", "Cross Below")
            if evt_default not in _ALL_EVENT_CHOICES:
                evt_default = _ALL_EVENT_CHOICES[0]
            ev_col, ac1_col, ac2_col = st.columns([1.5, 1, 1])
            with ev_col:
                event = st.selectbox("Event", _ALL_EVENT_CHOICES,
                                     index=_ALL_EVENT_CHOICES.index(evt_default),
                                     key=f"{prefix}_event")
        else:
            event = None
            ac1_col, ac2_col = st.columns(2)
        with ac1_col:
            atr_period = st.number_input("ATR Period", 1, 200,
                                         value=int(cand.get("atr_period", 14)),
                                         step=1, key=f"{prefix}_atr_p")
        with ac2_col:
            atr_mult = st.number_input("ATR Multiplier", 0.1, 20.0,
                                        value=float(cand.get("atr_multiplier", 2.0)),
                                        step=0.1, format="%.1f", key=f"{prefix}_atr_m")
        result = {
            "stop_type": "ATR",
            "atr_period": atr_period,
            "atr_multiplier": atr_mult,
        }
        if mode == MODE_PER_CANDIDATE:
            result["event"] = event
        return result

    # Standard candidate: (optional Event in MODE_PER_CANDIDATE), Group, Element 1, Compare, Element 2/Value
    if mode == MODE_PER_CANDIDATE:
        evt_default = cand.get("event", "Cross Above")
        if evt_default not in _ALL_EVENT_CHOICES:
            evt_default = _ALL_EVENT_CHOICES[0]
        c_ev, c_grp, c_e1, c_cmp, c_e2 = st.columns([1.5, 2, 2, 1.5, 2])
        with c_ev:
            event = st.selectbox("Event", _ALL_EVENT_CHOICES,
                                 index=_ALL_EVENT_CHOICES.index(evt_default),
                                 key=f"{prefix}_event")
    else:
        event = None
        c_grp, c_e1, c_cmp, c_e2 = st.columns([2, 2, 1.5, 2])

    with c_grp:
        group_idx = GROUP_NAMES.index(cand.get("group", GROUP_NAMES[0])) if cand.get("group") in GROUP_NAMES else 0
        group = st.selectbox("Group", GROUP_NAMES, index=group_idx, key=f"{prefix}_grp")

    with c_e1:
        elements = get_group_elements(group, ema_count)
        # Include special elements (R Profit/Loss, ATR Target) — universal
        extra = list(R_PROFIT_LOSS_ELEMENTS) + list(ATR_TARGET_ELEMENTS)
        all_e1 = elements + extra
        e1_val = cand.get("element1", all_e1[0])
        if e1_val not in all_e1:
            e1_val = all_e1[0]
        element1 = st.selectbox("Element 1", all_e1, index=all_e1.index(e1_val), key=f"{prefix}_e1")

    is_r = element1 in R_PROFIT_LOSS_ELEMENTS
    is_atr_target = element1 in ATR_TARGET_ELEMENTS

    with c_cmp:
        if is_r:
            compare = "Fixed Value"
            st.radio("Compare", ["Fixed Value"], key=f"{prefix}_cmp_d", disabled=True)
        elif is_atr_target:
            compare = "Indicator"
            st.radio("Compare", ["Indicator"], key=f"{prefix}_cmp_at", disabled=True)
        else:
            cmp_val = cand.get("compare_type", "Indicator")
            cmp_opts = ["Indicator", "Fixed Value"]
            compare = st.radio("Compare", cmp_opts,
                               index=cmp_opts.index(cmp_val) if cmp_val in cmp_opts else 0,
                               key=f"{prefix}_cmp", horizontal=True)

    with c_e2:
        if is_atr_target:
            atr_period = st.number_input("ATR Period", 1, 200,
                                         value=int(cand.get("atr_period", 14)),
                                         step=1, key=f"{prefix}_atr_p")
            atr_mult = st.number_input("ATR Multiplier", 0.1, 20.0,
                                        value=float(cand.get("atr_multiplier", 2.0)),
                                        step=0.1, format="%.1f", key=f"{prefix}_atr_m")
            atr_target_result = {
                "element1": element1,
                "atr_period": atr_period,
                "atr_multiplier": atr_mult,
            }
            if mode == MODE_PER_CANDIDATE:
                atr_target_result["event"] = event
            return atr_target_result

        elif compare == "Indicator":
            all_elements = []
            for g in GROUP_NAMES:
                all_elements.extend(get_group_elements(g, ema_count))
            e2_val = cand.get("element2", all_elements[0] if all_elements else "Price")
            if e2_val not in all_elements:
                e2_val = all_elements[0]
            element2 = st.selectbox("Element 2", all_elements,
                                    index=all_elements.index(e2_val), key=f"{prefix}_e2")
            value = None
        else:
            element2 = None
            value = st.number_input("Value", value=float(cand.get("value") or 0.0),
                                    step=0.01, format="%.4f", key=f"{prefix}_val")

    result = {
        "group": group,
        "element1": element1,
        "compare_type": compare,
        "element2": element2,
        "value": value,
    }
    if mode == MODE_PER_CANDIDATE:
        result["event"] = event
    return result


# ======================================================================
# Reference strategy selection UI (for Correlation)
# ======================================================================

def _render_reference_strategy_selection(saved_strategies, strategy_names, current_strategy_idx):
    """Render UI for selecting reference strategies used in correlation."""
    st.markdown("**Reference Strategies (for Correlation)**")

    ref_indices = st.session_state.get("gs_ref_strategies", [])

    if not ref_indices:
        st.caption(
            "Add reference strategies to compute correlation \u2014 measures how "
            "diversified each candidate is vs your existing strategies."
        )

    available = [(i, name) for i, name in enumerate(strategy_names)
                 if i != current_strategy_idx]

    if not available:
        st.info("Save at least one additional strategy to use as a correlation reference.")
        return

    to_remove = []
    for row_idx, ref_idx in enumerate(ref_indices):
        col_sel, col_rm = st.columns([4, 1])
        with col_sel:
            avail_indices = [a[0] for a in available]
            current_pos = avail_indices.index(ref_idx) if ref_idx in avail_indices else 0
            new_ref = st.selectbox(
                f"Reference {row_idx + 1}",
                avail_indices,
                index=current_pos,
                format_func=lambda x: strategy_names[x],
                key=f"gs_ref_sel_{row_idx}",
            )
            ref_indices[row_idx] = new_ref
        with col_rm:
            st.markdown("")
            if st.button("X", key=f"gs_ref_rm_{row_idx}"):
                to_remove.append(row_idx)

    if to_remove:
        for idx in sorted(to_remove, reverse=True):
            ref_indices.pop(idx)
        st.session_state["gs_ref_strategies"] = ref_indices
        st.rerun()

    if st.button("+ Add Reference Strategy", key="gs_ref_add"):
        used = set(ref_indices)
        default = next((i for i, _ in available if i not in used), available[0][0])
        ref_indices.append(default)
        st.session_state["gs_ref_strategies"] = ref_indices
        st.rerun()

    st.session_state["gs_ref_strategies"] = ref_indices


# ======================================================================
# Correlation helpers (DRM period directional signals)
# ======================================================================

def _compute_reference_directions(ref_indices, combo_slices, global_combo_keys):
    """Run reference strategies and return combined directional signal array.

    For each DRM period (slice):
      1. Run each reference strategy, get raw total P&L
      2. Sum raw P&Ls across all references into one composite P&L
      3. Convert composite to +1 (profit), -1 (loss), 0 (flat/no trades)

    Returns numpy array of length = total slices, or None if no references.
    """
    import numpy as np
    from strategies.first_strategy_numpy import execute_custom_strategy_numpy

    saved = st.session_state.get("saved_strategies", [])
    if not ref_indices or not saved:
        return None

    # Count total slices across all combos
    total_slices = sum(len(combo_slices.get(ck, [])) for ck in global_combo_keys)
    if total_slices == 0:
        return None

    # Accumulate raw P&L per slice across all reference strategies
    composite_pnl = np.zeros(total_slices, dtype=np.float64)

    for ref_idx in ref_indices:
        if ref_idx >= len(saved):
            continue
        strategy = saved[ref_idx]
        slice_idx = 0
        for ck in global_combo_keys:
            for df_slice, ps, pe in combo_slices.get(ck, []):
                try:
                    _, stats_df = execute_custom_strategy_numpy(
                        df_slice.copy(), strategy, ps, pe)
                    if stats_df is not None:
                        win_pnl = float(stats_df.loc['Winning trades P&L (R)', 'value'])
                        lose_pnl = float(stats_df.loc['Losing trades P&L (R)', 'value'])
                        composite_pnl[slice_idx] += win_pnl + lose_pnl
                except Exception:
                    pass  # slice produces no result → 0 contribution
                slice_idx += 1

    # Discretize composite P&L to directional signal
    return np.sign(composite_pnl)


def _compute_candidate_correlation(combo_results, global_combo_keys, ref_directions):
    """Compute correlation between candidate's directional signals and reference.

    Returns correlation as percentage (-100 to +100), or 0.0 if not computable.
    """
    import numpy as np

    if ref_directions is None:
        return None

    # Build candidate direction array from per-slice P&L
    candidate_dirs = []
    for ck in global_combo_keys:
        for sd in combo_results.get(ck, []):
            total_pnl = sd.get('win_pnl', 0) + sd.get('lose_pnl', 0)
            if total_pnl > 0:
                candidate_dirs.append(1.0)
            elif total_pnl < 0:
                candidate_dirs.append(-1.0)
            else:
                candidate_dirs.append(0.0)

    candidate_arr = np.array(candidate_dirs, dtype=np.float64)

    if len(candidate_arr) != len(ref_directions) or len(candidate_arr) < 3:
        return 0.0

    if np.std(candidate_arr) == 0 or np.std(ref_directions) == 0:
        return 0.0

    corr = np.corrcoef(candidate_arr, ref_directions)[0, 1]
    if np.isnan(corr):
        return 0.0
    return round(float(corr) * 100, 1)  # percentage


# ======================================================================
# Pattern selection UI (same as Performance tab)
# ======================================================================

def _render_pattern_selection():
    """Pattern selection UI identical to Performance tab."""
    selections = st.session_state.get("gs_selections", [])

    # Generation counter to guarantee fresh widget keys after deletions
    gen = st.session_state.get("_gs_sel_gen", 0)

    for idx, sel in enumerate(selections):
        c_mode, c_pt, c_prim, c_sec, c_rm = st.columns([2, 1.5, 2, 2, 0.5])

        kp = f"gs_sel_g{gen}_{idx}"

        with c_mode:
            mode = st.selectbox("Mode", SELECTION_MODES, key=f"{kp}_mode",
                                index=SELECTION_MODES.index(sel.get("mode", SELECTION_MODES[0])))
        sel["mode"] = mode

        with c_pt:
            need_pt = mode not in ("All Patterns",)
            if need_pt:
                pt = st.selectbox("Type", ["Bullish", "Bearish"], key=f"{kp}_pt",
                                  index=["Bullish", "Bearish"].index(sel.get("pattern_type", "Bullish")))
            else:
                pt = sel.get("pattern_type", "Bullish")
                st.selectbox("Type", ["—"], key=f"{kp}_pt_d", disabled=True)
            sel["pattern_type"] = pt

        with c_prim:
            need_prim = mode in ("Specified Primary", "Specified Secondary")
            if need_prim:
                prim_opts = PRIMARY_LIST
                prim_val = sel.get("primary") or prim_opts[0]
                if prim_val not in prim_opts:
                    prim_val = prim_opts[0]
                prim = st.selectbox("Primary", prim_opts, key=f"{kp}_prim",
                                    index=prim_opts.index(prim_val))
            else:
                prim = sel.get("primary")
                st.selectbox("Primary", ["—"], key=f"{kp}_prim_d", disabled=True)
            sel["primary"] = prim

        with c_sec:
            need_sec = mode in ("Specified Secondary", "Secondary Across Primaries")
            if need_sec:
                if mode == "Specified Secondary" and prim:
                    sec_opts = PRIMARY_SECONDARY_MAP.get(prim, ALL_UNIQUE_SECONDARIES)
                else:
                    sec_opts = ALL_UNIQUE_SECONDARIES
                sec_val = sel.get("secondary") or (sec_opts[0] if sec_opts else None)
                if sec_val not in sec_opts:
                    sec_val = sec_opts[0] if sec_opts else None
                sec = st.selectbox("Secondary", sec_opts, key=f"{kp}_sec",
                                   index=sec_opts.index(sec_val) if sec_val in sec_opts else 0)
            else:
                sec = sel.get("secondary")
                st.selectbox("Secondary", ["—"], key=f"{kp}_sec_d", disabled=True)
            sel["secondary"] = sec

        with c_rm:
            st.markdown("<br>", unsafe_allow_html=True)
            if len(selections) > 1 and st.button("X", key=f"{kp}_rm"):
                selections.pop(idx)
                st.session_state["gs_selections"] = selections
                # Bump generation so all widget keys are fresh on next render
                st.session_state["_gs_sel_gen"] = gen + 1
                st.rerun()

    if st.button("+ Add Selection", key="gs_add_sel"):
        selections.append({
            "mode": "All Patterns",
            "pattern_type": "Bullish",
            "primary": None,
            "secondary": None,
        })
        st.session_state["gs_selections"] = selections
        st.rerun()


# ======================================================================
# Indicator settings UI (gs_ prefixed)
# ======================================================================

def _render_indicator_settings():
    """Render indicator parameter widgets with gs_ prefix."""
    pfx = "gs_"

    with st.expander("RSI", expanded=False):
        st.session_state[f'{pfx}rsi_window'] = st.number_input(
            "RSI Period", 5, 50,
            value=int(st.session_state.get(f'{pfx}rsi_window', 14)),
            step=1, key=f"{pfx}rsi_w")

    with st.expander("Bollinger Bands", expanded=False):
        st.caption("**Upper Band**")
        st.session_state[f'{pfx}bb_upper_period'] = st.number_input(
            "Upper Period", 5, 100, value=int(st.session_state.get(f'{pfx}bb_upper_period', 20)),
            step=1, key=f"{pfx}bb_up_p")
        st.session_state[f'{pfx}bb_upper_stdev'] = st.number_input(
            "Upper StdDev", 0.5, 5.0, value=float(st.session_state.get(f'{pfx}bb_upper_stdev', 2.0)),
            step=0.01, format="%.2f", key=f"{pfx}bb_up_s")
        st.caption("**Middle Band**")
        st.session_state[f'{pfx}bb_mid_period'] = st.number_input(
            "Middle Period", 5, 100, value=int(st.session_state.get(f'{pfx}bb_mid_period', 20)),
            step=1, key=f"{pfx}bb_mid_p")
        st.caption("**Lower Band**")
        st.session_state[f'{pfx}bb_lower_period'] = st.number_input(
            "Lower Period", 5, 100, value=int(st.session_state.get(f'{pfx}bb_lower_period', 20)),
            step=1, key=f"{pfx}bb_lo_p")
        st.session_state[f'{pfx}bb_lower_stdev'] = st.number_input(
            "Lower StdDev", 0.5, 5.0, value=float(st.session_state.get(f'{pfx}bb_lower_stdev', 2.0)),
            step=0.01, format="%.2f", key=f"{pfx}bb_lo_s")

    with st.expander("Keltner Channel", expanded=False):
        st.session_state[f'{pfx}kc_atr_period'] = st.number_input(
            "ATR Period", 5, 100, value=int(st.session_state.get(f'{pfx}kc_atr_period', 10)),
            step=1, key=f"{pfx}kc_atr")
        st.caption("**Upper Band**")
        st.session_state[f'{pfx}kc_upper_ema'] = st.number_input(
            "Upper EMA Period", 5, 100, value=int(st.session_state.get(f'{pfx}kc_upper_ema', 20)),
            step=1, key=f"{pfx}kc_up_ema")
        st.session_state[f'{pfx}kc_upper_mult'] = st.number_input(
            "Upper ATR Mult", 0.5, 5.0, value=float(st.session_state.get(f'{pfx}kc_upper_mult', 2.0)),
            step=0.01, format="%.2f", key=f"{pfx}kc_up_m")
        st.caption("**Middle Band**")
        st.session_state[f'{pfx}kc_mid_ema'] = st.number_input(
            "Middle EMA Period", 5, 100, value=int(st.session_state.get(f'{pfx}kc_mid_ema', 20)),
            step=1, key=f"{pfx}kc_mid_e")
        st.caption("**Lower Band**")
        st.session_state[f'{pfx}kc_lower_ema'] = st.number_input(
            "Lower EMA Period", 5, 100, value=int(st.session_state.get(f'{pfx}kc_lower_ema', 20)),
            step=1, key=f"{pfx}kc_lo_ema")
        st.session_state[f'{pfx}kc_lower_mult'] = st.number_input(
            "Lower ATR Mult", 0.5, 5.0, value=float(st.session_state.get(f'{pfx}kc_lower_mult', 2.0)),
            step=0.01, format="%.2f", key=f"{pfx}kc_lo_m")

    with st.expander("Stochastic", expanded=False):
        st.session_state[f'{pfx}stoch_k_period'] = st.number_input(
            "%K Period", 1, 100, value=int(st.session_state.get(f'{pfx}stoch_k_period', 14)),
            step=1, key=f"{pfx}stoch_kp")
        st.session_state[f'{pfx}stoch_k_smooth'] = st.number_input(
            "%K Smoothing", 1, 50, value=int(st.session_state.get(f'{pfx}stoch_k_smooth', 3)),
            step=1, key=f"{pfx}stoch_ks")
        st.session_state[f'{pfx}stoch_d_smooth'] = st.number_input(
            "%D Smoothing", 1, 50, value=int(st.session_state.get(f'{pfx}stoch_d_smooth', 3)),
            step=1, key=f"{pfx}stoch_ds")

    with st.expander("ADX", expanded=False):
        st.session_state[f'{pfx}adx_period'] = st.number_input(
            "ADX Period", 5, 100, value=int(st.session_state.get(f'{pfx}adx_period', 14)),
            step=1, key=f"{pfx}adx_p")

    with st.expander("ATR", expanded=False):
        st.session_state[f'{pfx}atr_period'] = st.number_input(
            "ATR Period", 5, 100, value=int(st.session_state.get(f'{pfx}atr_period', 14)),
            step=1, key=f"{pfx}atr_p")

    with st.expander("MACD", expanded=False):
        st.session_state[f'{pfx}macd_fast'] = st.number_input(
            "Fast Period", 2, 100, value=int(st.session_state.get(f'{pfx}macd_fast', 12)),
            step=1, key=f"{pfx}macd_f")
        st.session_state[f'{pfx}macd_slow'] = st.number_input(
            "Slow Period", 2, 200, value=int(st.session_state.get(f'{pfx}macd_slow', 26)),
            step=1, key=f"{pfx}macd_sl")
        st.session_state[f'{pfx}macd_signal'] = st.number_input(
            "Signal Period", 2, 100, value=int(st.session_state.get(f'{pfx}macd_signal', 9)),
            step=1, key=f"{pfx}macd_sg")

    with st.expander("Supertrend", expanded=False):
        st.session_state[f'{pfx}supertrend_period'] = st.number_input(
            "Period", 1, 100, value=int(st.session_state.get(f'{pfx}supertrend_period', 10)),
            step=1, key=f"{pfx}st_p")
        st.session_state[f'{pfx}supertrend_multiplier'] = st.number_input(
            "Multiplier", 0.5, 10.0, value=float(st.session_state.get(f'{pfx}supertrend_multiplier', 3.0)),
            step=0.01, format="%.2f", key=f"{pfx}st_m")

    with st.expander("EMA Overlay", expanded=False):
        ema_key = f'{pfx}ema_periods'
        if ema_key not in st.session_state:
            from config.constants import DEFAULT_EMA_PERIODS
            st.session_state[ema_key] = list(DEFAULT_EMA_PERIODS)
        gs_ema_gen_key = f"_gs_ema_gen_{pfx}"
        gs_ema_gen = st.session_state.get(gs_ema_gen_key, 0)

        emas_to_remove = []
        for idx, ema_val in enumerate(st.session_state[ema_key]):
            lc, rc = st.columns([3, 1])
            with lc:
                new_val = st.number_input(f"EMA {idx+1} Period", 2, 500, int(ema_val),
                                          step=1, key=f"{pfx}ema_p_g{gs_ema_gen}_{idx}")
                st.session_state[ema_key][idx] = new_val
            with rc:
                st.markdown("<br>", unsafe_allow_html=True)
                if st.button("X", key=f"{pfx}ema_rm_g{gs_ema_gen}_{idx}"):
                    emas_to_remove.append(idx)
        if emas_to_remove:
            for i in sorted(emas_to_remove, reverse=True):
                st.session_state[ema_key].pop(i)
            st.session_state[gs_ema_gen_key] = gs_ema_gen + 1
            st.rerun()
        if st.button("+ Add EMA", key=f"{pfx}ema_add"):
            st.session_state[ema_key].append(20)
            st.rerun()

    with st.expander("Donchian Channel", expanded=False):
        st.caption("**Upper Band**")
        st.session_state[f'{pfx}dc_upper_period'] = st.number_input(
            "Upper Period", 5, 200, value=int(st.session_state.get(f'{pfx}dc_upper_period', 20)),
            step=1, key=f"{pfx}dc_up_p")
        st.caption("**Middle Band**")
        st.session_state[f'{pfx}dc_mid_period'] = st.number_input(
            "Middle Period", 5, 200, value=int(st.session_state.get(f'{pfx}dc_mid_period', 20)),
            step=1, key=f"{pfx}dc_mid_p")
        st.caption("**Lower Band**")
        st.session_state[f'{pfx}dc_lower_period'] = st.number_input(
            "Lower Period", 5, 200, value=int(st.session_state.get(f'{pfx}dc_lower_period', 20)),
            step=1, key=f"{pfx}dc_lo_p")
        st.divider()
        st.session_state[f'{pfx}dc_offset'] = st.number_input(
            "Offset / Shift", -50, 50, value=int(st.session_state.get(f'{pfx}dc_offset', 0)),
            step=1, key=f"{pfx}dc_off")

    with st.expander("Parabolic SAR", expanded=False):
        st.session_state[f'{pfx}psar_af_start'] = st.number_input(
            "AF Start", 0.001, 0.5, value=float(st.session_state.get(f'{pfx}psar_af_start', 0.02)),
            step=0.01, format="%.3f", key=f"{pfx}psar_afs")
        st.session_state[f'{pfx}psar_af_increment'] = st.number_input(
            "AF Increment", 0.001, 0.5, value=float(st.session_state.get(f'{pfx}psar_af_increment', 0.02)),
            step=0.01, format="%.3f", key=f"{pfx}psar_afi")
        st.session_state[f'{pfx}psar_af_max'] = st.number_input(
            "AF Max", 0.01, 1.0, value=float(st.session_state.get(f'{pfx}psar_af_max', 0.20)),
            step=0.01, format="%.2f", key=f"{pfx}psar_afm")

    with st.expander("Williams %R", expanded=False):
        st.session_state[f'{pfx}willr_period'] = st.number_input(
            "Period", 1, 100, value=int(st.session_state.get(f'{pfx}willr_period', 14)),
            step=1, key=f"{pfx}willr_p")

    with st.expander("ROC", expanded=False):
        st.session_state[f'{pfx}roc_period'] = st.number_input(
            "ROC Period", 1, 100, value=int(st.session_state.get(f'{pfx}roc_period', 12)),
            step=1, key=f"{pfx}roc_p")
        st.session_state[f'{pfx}roc_signal_period'] = st.number_input(
            "Signal Period (EMA)", 1, 100, value=int(st.session_state.get(f'{pfx}roc_signal_period', 9)),
            step=1, key=f"{pfx}roc_sig")

    with st.expander("CCI", expanded=False):
        st.session_state[f'{pfx}cci_period'] = st.number_input(
            "CCI Period", 1, 200, value=int(st.session_state.get(f'{pfx}cci_period', 20)),
            step=1, key=f"{pfx}cci_p")

    with st.expander("Linear Regression Channel", expanded=False):
        st.session_state[f'{pfx}lr_period'] = st.number_input(
            "Period", 2, 500, value=int(st.session_state.get(f'{pfx}lr_period', 50)),
            step=1, key=f"{pfx}lr_p")
        st.session_state[f'{pfx}lr_multiplier'] = st.number_input(
            "Channel Multiplier", 0.1, 10.0, value=float(st.session_state.get(f'{pfx}lr_multiplier', 2.0)),
            step=0.1, format="%.1f", key=f"{pfx}lr_m")


# ======================================================================
# Dict-based aggregation (for multiprocessing — workers return dicts, not DataFrames)
# ======================================================================

def _aggregate_stats_dicts(all_stats_dicts):
    """Same logic as _aggregate_stats but works with lightweight dicts from workers."""
    import numpy as np

    all_trade_pnls = []
    all_holding_periods = []
    total_win_pnl = 0.0
    total_lose_pnl = 0.0
    total_static_alloc = 0.0
    total_dynamic_alloc = 0.0
    total_target_alloc = 0.0
    total_eod_alloc = 0.0

    for sd in all_stats_dicts:
        total_win_pnl += sd['win_pnl']
        total_lose_pnl += sd['lose_pnl']
        total_static_alloc += sd['total_static_alloc']
        total_dynamic_alloc += sd['total_dynamic_alloc']
        total_target_alloc += sd['total_target_alloc']
        total_eod_alloc += sd.get('total_eod_alloc', 0.0)
        all_trade_pnls.extend(sd['trade_pnls_r'])
        all_holding_periods.extend(sd.get('trade_holding_periods', []))

    total_trades = len(all_trade_pnls)
    total_wins = sum(pnl > 0 for pnl in all_trade_pnls)
    total_flat = sum(pnl == 0 for pnl in all_trade_pnls)
    total_losses = total_trades - total_wins - total_flat
    total_pnl = total_win_pnl + total_lose_pnl

    if total_trades > 0:
        meaningful = total_wins + total_losses
        win_pct = (total_wins / meaningful * 100) if meaningful > 0 else 0.0
        lose_pct = (total_losses / meaningful * 100) if meaningful > 0 else 0.0
        target_exit_pct = total_target_alloc / total_trades
        static_exit_pct = total_static_alloc / total_trades
        dynamic_exit_pct = total_dynamic_alloc / total_trades
        eod_exit_pct = total_eod_alloc / total_trades

        avg_win_pnl = total_win_pnl / total_wins if total_wins > 0 else 0.0
        avg_lose_pnl = total_lose_pnl / total_losses if total_losses > 0 else 0.0
        expected_value = (win_pct / 100 * avg_win_pnl) + (lose_pct / 100 * avg_lose_pnl)
        rr_ratio = abs(avg_win_pnl / avg_lose_pnl) if avg_lose_pnl != 0 else 0.0

        if all_trade_pnls:
            cumulative = np.cumsum(all_trade_pnls)
            peak = np.maximum.accumulate(cumulative)
            drawdowns = cumulative - peak
            max_drawdown = abs(drawdowns.min())
        else:
            max_drawdown = 0.0

        if len(all_trade_pnls) >= 2:
            pnl_std = np.std(all_trade_pnls, ddof=1)
            sqn = (np.mean(all_trade_pnls) / pnl_std * np.sqrt(len(all_trade_pnls))) if pnl_std > 0 else 0.0
        else:
            sqn = 0.0
    else:
        win_pct = lose_pct = total_pnl = avg_win_pnl = avg_lose_pnl = 0.0
        expected_value = target_exit_pct = static_exit_pct = dynamic_exit_pct = eod_exit_pct = 0.0
        rr_ratio = max_drawdown = sqn = 0.0

    avg_holding_period = (sum(all_holding_periods) / len(all_holding_periods)) if all_holding_periods else 0.0

    return {
        'num_trades': total_trades,
        'win_pct': win_pct,
        'lose_pct': lose_pct,
        'avg_win_pnl': avg_win_pnl,
        'avg_lose_pnl': avg_lose_pnl,
        'total_pnl': total_pnl,
        'expected_value': expected_value,
        'target_exit_pct': target_exit_pct,
        'static_exit_pct': static_exit_pct,
        'dynamic_exit_pct': dynamic_exit_pct,
        'eod_exit_pct': eod_exit_pct,
        'rr_ratio': rr_ratio,
        'max_drawdown': max_drawdown,
        'sqn': sqn,
        'avg_holding_period': avg_holding_period,
    }


# ======================================================================
# MC enrichment — parallel version
# ======================================================================

def _enrich_mc_parallel(results, balance, n_sims, target_dd=5.0):
    """Enrich Grid Search results (list of (label, global_agg, sel_results))
    with `mc_avg_profit` for every agg dict (global + each per-selection).

    Thin wrapper around `enrich_aggs_with_mc` that flattens the nested shape.
    """
    flat_aggs = []
    for _label, global_agg, sel_results in results:
        flat_aggs.append(global_agg)
        flat_aggs.extend(sel_results.values())
    enrich_aggs_with_mc(flat_aggs, balance, n_sims, target_dd=target_dd)


# ======================================================================
# Grid Search execution
# ======================================================================

def _run_grid_search(selected_strategy, search_group, search_set, selected_events,
                     condition_candidates, condition_event, sidebar_config,
                     effective_candidates=None, use_original_engine=False):
    """Run backtests for all candidate runs.

    Returns list of (label, global_agg, selection_results) where
    selection_results is an OrderedDict {selection_label: agg_dict}.
    """

    # Build base indicator params: start from defaults, overlay the selected
    # strategy's saved settings. The retired global gs_* widgets are no longer
    # consulted — ranges come from the group set's indicator_ranges field.
    indicator_params = dict(_DEFAULT_INDICATOR_PARAMS)
    strategy_settings = selected_strategy.get("indicator_settings")
    if strategy_settings:
        strategy_settings = migrate_indicator_settings(strategy_settings)
        indicator_params.update(strategy_settings)

    # Calculate base indicators once
    g_start = sidebar_config.get("global_start_date")
    g_end = sidebar_config.get("global_end_date")

    df_full = _get_or_calculate(
        "df_ohlc", "_gs_features", "_gs_params",
        indicator_params, global_start_date=g_start, global_end_date=g_end)

    if df_full.empty:
        st.warning("No data available for the selected date range.")
        return []

    # Expand pattern selections per selection group
    selections = st.session_state.get("gs_selections", [])
    drm_bullish = st.session_state.get("drm_bullish")
    drm_bearish = st.session_state.get("drm_bearish")

    # Build period slices per unique combo, and track which combos belong to each selection
    from collections import OrderedDict
    combo_slices = OrderedDict()  # combo_key -> [(df_slice, ps, pe), ...]
    selection_combo_map = OrderedDict()  # sel_label -> [combo_key, ...]
    global_combo_keys = []  # all unique combo keys (deduplicated, ordered)

    for sel in selections:
        label = selection_label(sel)
        # Ensure unique labels
        if label in selection_combo_map:
            n = 2
            while f"{label} ({n})" in selection_combo_map:
                n += 1
            label = f"{label} ({n})"

        combos = expand_selection(sel)
        sel_combo_keys = []
        for pattern_type, primary, secondary in combos:
            combo_key = (pattern_type, primary, secondary)
            sel_combo_keys.append(combo_key)

            if combo_key not in combo_slices:
                # First time seeing this combo — compute its period slices
                drm_df = drm_bullish if pattern_type == "Bullish" else drm_bearish
                slices = []
                if drm_df is not None:
                    periods = parse_drm_periods(drm_df, pattern_type, primary, secondary)
                    for start_dt, end_dt in periods:
                        df_slice, ps, pe = slice_for_graph(
                            df=df_full, start_date=start_dt, end_date=end_dt,
                            show_ichimoku=False,
                            show_bb=False,
                            show_kc=False,
                            show_donchian=False,
                            show_psar=False)
                        if not df_slice.empty:
                            slices.append((df_slice, ps, pe))
                combo_slices[combo_key] = slices
                global_combo_keys.append(combo_key)

        selection_combo_map[label] = sel_combo_keys

    if not any(combo_slices[ck] for ck in global_combo_keys):
        st.warning("No valid DRM periods found for selected patterns.")
        return []

    # Generate all run configs
    base = copy.deepcopy(selected_strategy)
    base["indicator_settings"] = dict(indicator_params)

    search_candidates = (effective_candidates if effective_candidates is not None
                         else search_set.get("candidates", []))
    events_per_candidate = get_mode(search_set) == MODE_PER_CANDIDATE

    # Variant expansion: read per-set indicator ranges and enumerate offsets.
    variant_groups = group_variant_combos(search_set, indicator_params)

    run_configs = generate_run_configs(
        base, search_group, search_candidates, selected_events,
        condition_candidates=condition_candidates, condition_event=condition_event,
        events_per_candidate=events_per_candidate,
        variant_groups=variant_groups)

    if not run_configs:
        st.warning("No run configurations generated.")
        return []

    # Compute the per-variant DataFrames + slice them per pattern combo.
    # Always include the default variant (None) so non-ranged candidates work.
    variant_combo_slices = _build_variant_combo_slices(
        df_full, indicator_params, run_configs, variant_groups,
        combo_slices, global_combo_keys)

    # Compute reference directions for correlation (against default DataFrame)
    ref_indices = st.session_state.get("gs_ref_strategies", [])
    ref_directions = None
    if ref_indices:
        ref_directions = _compute_reference_directions(
            ref_indices, combo_slices, global_combo_keys)

    if use_original_engine:
        results = _run_grid_search_original_engine(
            run_configs, variant_combo_slices, global_combo_keys,
            selection_combo_map, ref_directions)
    else:
        results = _run_grid_search_multiprocessing(
            run_configs, variant_combo_slices, global_combo_keys,
            selection_combo_map, ref_directions)

    # Enrich every agg dict with MC Avg Profit @ 5% avg max DD
    # Binary-searches for the risk % that yields exactly 5% avg max DD.
    # trades/sim = each candidate's own num_trades (not a fixed constant),
    # so low-trade-count candidates are simulated with matching path length.
    # Parallelised across workers — enrichment used to dominate runtime.
    balance = st.session_state.get('mc_starting_balance', 10000.0)
    _enrich_mc_parallel(results, balance, GS_MC_N_SIMULATIONS, target_dd=5.0)

    # Diagnostic: if all candidates produced zero trades, tell the user
    if results and all(r[1].get('num_trades', 0) == 0 for r in results):
        st.info(
            f"All {len(results)} candidates ran successfully but produced 0 trades. "
            f"Check entry/exit conditions, DRM periods, or date range."
        )

    return results


def _build_variant_combo_slices(df_full, base_indicator_params, run_configs,
                                 variant_groups, default_combo_slices,
                                 global_combo_keys):
    """For every distinct variant_id used by run_configs, recompute the
    indicator group(s) for that variant on a copy of df_full, then slice it
    per pattern combo using the same combo_slices structure.

    Returns: {variant_id: {combo_key: [(df_slice, ps, pe), ...]}}
    The default variant (None) always points at default_combo_slices."""
    from indicators.calculate_indicators import recalculate_groups
    from collections import OrderedDict

    out = {None: default_combo_slices}

    if not variant_groups:
        return out

    # Build (group, offset) -> params lookup once.
    offset_params = {}
    for grp, variants in variant_groups.items():
        for offset, params in variants:
            offset_params[(grp, offset)] = params

    # Collect distinct non-default variant ids actually used
    needed = sorted({vid for _label, _strat, vid in run_configs if vid is not None})

    # Build pattern slice structure (which periods belong to which combo).
    # Reuse the period boundaries from default_combo_slices so we can re-slice
    # the per-variant DataFrame consistently.
    combo_periods = OrderedDict()
    for ck in global_combo_keys:
        combo_periods[ck] = [(ps, pe) for _df, ps, pe in default_combo_slices.get(ck, [])]

    for vid in needed:
        # vid is a tuple of ((group, offset), ...) — recover params overrides.
        df_v = df_full.copy()
        for (group, offset) in vid:
            params = offset_params.get((group, offset), {})
            merged = dict(base_indicator_params)
            merged.update(params)
            recalculate_groups(df_v, [group], **merged)

        # Slice the variant DataFrame using the same period bounds per combo.
        variant_slices = OrderedDict()
        for ck, periods in combo_periods.items():
            slices = []
            for ps, pe in periods:
                # df_full is already date-bounded; periods are already trimmed to
                # match each combo's DRM windows. Use the same boundaries.
                df_slice = df_v.loc[ps:pe]
                if not df_slice.empty:
                    slices.append((df_slice, ps, pe))
            variant_slices[ck] = slices
        out[vid] = variant_slices

    return out


def _run_grid_search_original_engine(run_configs, variant_combo_slices, global_combo_keys,
                                      selection_combo_map, ref_directions=None):
    """Run grid search using the ORIGINAL (non-numpy) engine, single-process.
    Used for debugging to compare results with the numpy engine.
    """
    from strategies.first_strategy import execute_custom_strategy

    def _extract_stats(stats_df):
        return {
            'win_pnl': float(stats_df.loc['Winning trades P&L (R)', 'value']),
            'lose_pnl': float(stats_df.loc['Losing trades P&L (R)', 'value']),
            'trade_pnls_r': list(stats_df.attrs.get('trade_pnls_r', [])),
            'trade_holding_periods': list(stats_df.attrs.get('trade_holding_periods', [])),
            'total_static_alloc': float(stats_df.attrs.get('total_static_alloc', 0.0)),
            'total_dynamic_alloc': float(stats_df.attrs.get('total_dynamic_alloc', 0.0)),
            'total_target_alloc': float(stats_df.attrs.get('total_target_alloc', 0.0)),
            'total_eod_alloc': float(stats_df.attrs.get('total_eod_alloc', 0.0)),
        }

    n_candidates = len(run_configs)
    progress = st.progress(0, text=f"Running grid search (original engine, single-process)...")

    results = []
    for idx, (label, strategy, variant_id) in enumerate(run_configs):
        slice_store = variant_combo_slices.get(variant_id) or variant_combo_slices.get(None) or {}
        combo_results = {}
        for combo_key in global_combo_keys:
            slices = slice_store.get(combo_key, [])
            stats_list = []
            for df_slice, ps, pe in slices:
                try:
                    _, stats_df = execute_custom_strategy(df_slice.copy(), strategy, ps, pe)
                    if stats_df is not None:
                        stats_list.append(_extract_stats(stats_df))
                except Exception:
                    pass
            combo_results[combo_key] = stats_list

        # Global stats
        global_stats_dicts = []
        for ck in global_combo_keys:
            global_stats_dicts.extend(combo_results.get(ck, []))
        global_agg = _aggregate_stats_dicts(global_stats_dicts) if global_stats_dicts else _empty_agg()

        # Correlation with reference strategies
        if ref_directions is not None:
            corr_value = _compute_candidate_correlation(
                combo_results, global_combo_keys, ref_directions)
            global_agg['correlation'] = corr_value
            global_agg['abs_correlation'] = abs(corr_value) if corr_value is not None else None
        else:
            global_agg['correlation'] = None
            global_agg['abs_correlation'] = None

        # Per-selection stats
        sel_results = OrderedDict()
        for sel_label, sel_combo_keys in selection_combo_map.items():
            sel_stats_dicts = []
            for ck in sel_combo_keys:
                sel_stats_dicts.extend(combo_results.get(ck, []))
            sel_results[sel_label] = _aggregate_stats_dicts(sel_stats_dicts) if sel_stats_dicts else _empty_agg()

        results.append((label, global_agg, sel_results))
        progress.progress((idx + 1) / n_candidates,
                          text=f"Completed {idx + 1}/{n_candidates} candidates")

    progress.empty()
    return results


def _run_grid_search_multiprocessing(run_configs, variant_combo_slices, global_combo_keys,
                                      selection_combo_map, ref_directions=None):
    """Run grid search using multiprocessing Pool.
    Each worker process handles one candidate across all combo/period slices.
    """
    from strategies.grid_search_worker import init_worker, run_candidate

    n_candidates = len(run_configs)
    n_workers = min(n_candidates, max(1, os.cpu_count() or 1))

    progress = st.progress(0, text=f"Running grid search with {n_workers} processes...")
    n_variants = len([k for k in variant_combo_slices.keys() if k is not None])
    if n_variants:
        st.caption(f"Using {n_workers} CPU cores for {n_candidates} runs across {n_variants + 1} indicator variant(s)")
    else:
        st.caption(f"Using {n_workers} CPU cores for {n_candidates} candidates")

    # Convert structures to plain dicts (ensure picklable)
    variant_slices_dict = {k: dict(v) for k, v in variant_combo_slices.items()}
    combo_keys_list = list(global_combo_keys)

    # Build task args: (idx, label, strategy, variant_id) per run
    tasks = [(idx, label, strategy, variant_id)
             for idx, (label, strategy, variant_id) in enumerate(run_configs)]

    # Run with Pool — shared data passed via initializer (pickled once per worker)
    results = []
    try:
        with multiprocessing.Pool(
            processes=n_workers,
            initializer=init_worker,
            initargs=(variant_slices_dict, combo_keys_list)
        ) as pool:
            completed = 0
            candidate_results = {}

            for idx, label, combo_results in pool.imap_unordered(run_candidate, tasks):
                candidate_results[idx] = (label, combo_results)
                completed += 1
                progress.progress(completed / n_candidates,
                                  text=f"Completed {completed}/{n_candidates} candidates")

    except Exception as e:
        st.error(f"Multiprocessing error: {e}")
        progress.empty()
        return []

    # Diagnostic: warn if workers produced fewer results than expected
    if len(candidate_results) == 0:
        st.error(
            f"Grid search produced no candidate results. "
            f"Expected {len(run_configs)} candidates, got 0 from workers. "
            f"This usually means every worker crashed — check the terminal "
            f"where Streamlit is running for error messages."
        )
        progress.empty()
        return []
    elif len(candidate_results) < len(run_configs):
        st.warning(
            f"Partial failure: only {len(candidate_results)} of {len(run_configs)} "
            f"candidates returned results. Check the terminal for worker errors."
        )

    # Assemble results in original order
    for idx in range(len(run_configs)):
        if idx not in candidate_results:
            continue
        label, combo_results = candidate_results[idx]

        # Global stats: all unique combos
        global_stats_dicts = []
        for ck in global_combo_keys:
            global_stats_dicts.extend(combo_results.get(ck, []))
        global_agg = _aggregate_stats_dicts(global_stats_dicts) if global_stats_dicts else _empty_agg()

        # Correlation with reference strategies
        if ref_directions is not None:
            corr_value = _compute_candidate_correlation(
                combo_results, global_combo_keys, ref_directions)
            global_agg['correlation'] = corr_value
            global_agg['abs_correlation'] = abs(corr_value) if corr_value is not None else None
        else:
            global_agg['correlation'] = None
            global_agg['abs_correlation'] = None

        # Per-selection stats
        sel_results = OrderedDict()
        for sel_label, sel_combo_keys in selection_combo_map.items():
            sel_stats_dicts = []
            for ck in sel_combo_keys:
                sel_stats_dicts.extend(combo_results.get(ck, []))
            sel_results[sel_label] = _aggregate_stats_dicts(sel_stats_dicts) if sel_stats_dicts else _empty_agg()

        results.append((label, global_agg, sel_results))

    progress.empty()
    return results


# ======================================================================
# Results display with filtering, sorting, pagination
# ======================================================================

def _display_results(results, strategy_name, thresholds, sort_key, sort_descending):
    """Display filtered, sorted, paginated results with Global + per-selection breakdown."""

    # Filter based on Global metrics — all thresholds applied dynamically
    filtered = [(label, global_agg, sel_results)
                for label, global_agg, sel_results in results
                if passes_thresholds(global_agg, thresholds)]

    st.subheader(f"Results — {strategy_name}")
    st.caption(f"{len(filtered)} of {len(results)} candidates pass filters")

    if not filtered:
        st.warning("No candidates pass the threshold filters.")
        return

    # Sort based on Global metrics
    # Sort — None values go to bottom regardless of direction
    def _sort_val(x):
        v = x[1].get(sort_key)
        if v is None:
            return float('inf') if not sort_descending else float('-inf')
        return v
    filtered.sort(key=_sort_val, reverse=sort_descending)

    # Build Global results table (rows = candidates, columns = metrics)
    rows = []
    for label, global_agg, sel_results in filtered:
        rows.append({
            "Candidate": label,
            "Trades": global_agg["num_trades"],
            "Win%": f"{global_agg['win_pct']:.0f}%",
            "Lose%": f"{global_agg['lose_pct']:.0f}%",
            "Avg Profit": f"{global_agg['avg_win_pnl']:.2f}R",
            "Avg Loss": f"{global_agg['avg_lose_pnl']:.2f}R",
            "Total P&L": f"{global_agg['total_pnl']:.2f}R",
            "EV": f"{global_agg['expected_value']:.2f}R",
            "Target%": f"{global_agg['target_exit_pct']:.0f}%",
            "Static%": f"{global_agg['static_exit_pct']:.0f}%",
            "Dynamic%": f"{global_agg['dynamic_exit_pct']:.0f}%",
            "RR": f"{global_agg.get('rr_ratio', 0):.2f}",
            "MC": f"${global_agg.get('mc_avg_profit', 0):,.0f}",
            "Corr": f"{global_agg['correlation']:.0f}%" if global_agg.get('correlation') is not None else "\u2014",
            "Hold": f"{global_agg.get('avg_holding_period', 0):.1f}",
        })

    df_results = pd.DataFrame(rows)

    # MC context caption
    _mc_bal = st.session_state.get('mc_starting_balance', 10000.0)
    st.caption(f"MC Avg Profit @ 5% DD based on ${_mc_bal:,.0f} starting balance, each candidate's own trade count as trades/sim, {GS_MC_N_SIMULATIONS:,} simulations (risk % auto-adjusted to 5% avg max DD)")

    # Copy to clipboard (all filtered results, TSV)
    tsv_data = df_results.to_csv(sep='\t', index=False, header=False)
    _copy_to_clipboard(tsv_data, key="gs_copy_results")

    st.caption(f"{len(filtered)} results (Global Performance)")
    st.dataframe(df_results, use_container_width=True, hide_index=True)

    # Expandable detail view for each candidate (paginated)
    st.markdown("---")
    st.subheader("Detailed Performance")

    detail_page_size = PAGE_SIZE
    total_detail_pages = max(1, math.ceil(len(filtered) / detail_page_size))
    detail_page = st.session_state.get("_gs_detail_page", 0)
    if detail_page >= total_detail_pages:
        detail_page = max(0, total_detail_pages - 1)
        st.session_state["_gs_detail_page"] = detail_page

    detail_start = detail_page * detail_page_size
    detail_end = min(detail_start + detail_page_size, len(filtered))

    if len(filtered) > detail_page_size:
        st.caption(f"Showing {detail_start + 1}–{detail_end} of {len(filtered)}")
        dp1, dp2, dp3 = st.columns([1, 1, 1])
        with dp1:
            if st.button("◀ Previous", key="gs_detail_prev", disabled=detail_page == 0):
                st.session_state["_gs_detail_page"] = detail_page - 1
                st.rerun()
        with dp2:
            st.markdown(f"Page **{detail_page + 1}** / {total_detail_pages}")
        with dp3:
            if st.button("Next ▶", key="gs_detail_next", disabled=detail_page >= total_detail_pages - 1):
                st.session_state["_gs_detail_page"] = detail_page + 1
                st.rerun()

    for idx in range(detail_start, detail_end):
        label, global_agg, sel_results = filtered[idx]
        with st.expander(f"**{label}**", expanded=False):
            # Build table: Global on the left + per-selection columns (if multiple)
            table_data = {"Global": global_agg}
            if len(sel_results) > 1:
                table_data.update(sel_results)
            table = _build_metrics_table(table_data)
            st.table(table)
            _copy_to_clipboard(
                table.to_csv(sep='\t', header=False, index=False),
                key=f"gs_sel_detail_{idx}")


# ======================================================================
# Cache helpers
# ======================================================================

def _build_cache_fingerprint(strategy, search_group, search_set, selected_events,
                             condition_candidates, view_state=None):
    """Build a hashable fingerprint for cache invalidation."""
    import json
    parts = [
        strategy.get("strategy_name", ""),
        search_group,
        search_set.get("name", ""),
        get_mode(search_set),
        json.dumps(search_set.get("candidates", []), sort_keys=True),
        json.dumps(search_set.get("indicator_ranges") or {}, sort_keys=True),
        json.dumps(sorted(selected_events)) if selected_events else "",
        json.dumps(condition_candidates, sort_keys=True) if condition_candidates else "",
        json.dumps(view_state, sort_keys=True) if view_state else "",
    ]
    return "|".join(parts)

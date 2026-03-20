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
    export_group_set, import_group_set, get_group_sets_by_type,
    save_group_sets_to_file,
)
from ui.charting_tab import _get_or_calculate
from ui.performance_tab import (_DEFAULT_INDICATOR_PARAMS, SELECTION_MODES,
                                 _build_metrics_table, _copy_to_clipboard, _empty_agg)
from ui.grid_search_helpers import (
    format_candidate_label, generate_run_configs,
    collect_gs_indicator_settings,
)
from config.constants import (GROUP_NAMES, EVENT_TYPES, STOP_EVENT_TYPES,
                               CONDITION_OPERATORS, get_group_elements,
                               R_PROFIT_LOSS_ELEMENTS, ATR_TARGET_ELEMENTS)

# Group types for the group set manager
GROUP_TYPES = [
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
    ("avg_win_pnl", "Avg Profit (R)"),
    ("avg_lose_pnl", "Avg Loss (R)"),
    ("total_pnl", "Total P&L (R)"),
    ("target_exit_pct", "Target Exit %"),
    ("static_exit_pct", "Static %"),
    ("dynamic_exit_pct", "Dynamic %"),
    ("sharpe_ratio", "Sharpe Ratio"),
    ("mar_ratio", "MAR Ratio"),
    ("sqn", "SQN"),
]

PAGE_SIZE = 50


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

    sc1, sc2 = st.columns(2)
    with sc1:
        search_type_labels = [label for _, label in GROUP_TYPES]
        search_type_idx = st.selectbox(
            "Component to Search",
            range(len(GROUP_TYPES)),
            format_func=lambda x: search_type_labels[x],
            key="gs_search_type_idx")
        search_group = GROUP_TYPES[search_type_idx][0]

    with sc2:
        available = get_group_sets_by_type(search_group)
        if not available:
            st.warning(f"No group sets saved for **{GROUP_TYPES[search_type_idx][1]}**. Create one above.")
            return
        gs_options = [(i, gs["name"]) for i, gs in available]
        gs_sel = st.selectbox(
            "Group Set to Use",
            range(len(gs_options)),
            format_func=lambda x: gs_options[x][1],
            key="gs_search_set_sel")
        search_set_global_idx = gs_options[gs_sel][0]
        search_set = st.session_state["saved_group_sets"][search_set_global_idx]

    # Cross-combination: condition group set for target/dynamic
    condition_candidates = None
    if search_group in ("target", "dynamic_stop"):
        cond_available = get_group_sets_by_type("condition")
        if cond_available:
            cond_options = [("none", "No cross-combination")] + \
                           [(i, gs["name"]) for i, gs in cond_available]
            cond_sel = st.selectbox(
                "Condition Group Set (cross-combine)",
                range(len(cond_options)),
                format_func=lambda x: cond_options[x][1],
                key="gs_cross_cond_sel")
            if cond_sel > 0:
                cond_global_idx = cond_options[cond_sel][0]
                condition_candidates = st.session_state["saved_group_sets"][cond_global_idx]["candidates"]

    # Show run count
    n_search = len(search_set.get("candidates", []))
    n_cond = len(condition_candidates) if condition_candidates else 0
    if search_group in ("target", "dynamic_stop") and n_cond > 0:
        total_runs = n_search * (n_cond + 1)
        st.info(f"**{n_search}** candidates x **{n_cond + 1}** (standalone + {n_cond} conditions) = **{total_runs}** total runs")
    else:
        st.info(f"**{n_search}** candidates to test")

    st.markdown("---")

    # ── Section D: Pattern Selection ────────────────────
    st.markdown("**Pattern Selection**")
    _render_pattern_selection()

    st.markdown("---")

    # ── Section E: Indicator Settings ───────────────────
    with st.expander("Indicator Settings", expanded=False):
        _render_indicator_settings()

    st.markdown("---")

    # ── Section F: Filters & Sort ───────────────────────
    st.markdown("**Performance Filters**")
    fc1, fc2, fc3, fc4 = st.columns(4)
    with fc1:
        min_trades = st.number_input("Min Trades", value=0, step=1, key="gs_thresh_min_trades")
    with fc2:
        min_ev = st.number_input("Min EV (R)", value=-999.0, step=0.01,
                                 format="%.2f", key="gs_thresh_min_ev")
    with fc3:
        min_sharpe = st.number_input("Min Sharpe", value=-999.0, step=0.01,
                                     format="%.2f", key="gs_thresh_min_sharpe")
    with fc4:
        min_sqn = st.number_input("Min SQN", value=-999.0, step=0.01,
                                  format="%.2f", key="gs_thresh_min_sqn")

    thresholds = {
        "min_trades": min_trades,
        "min_ev": min_ev,
        "min_sharpe": min_sharpe,
        "min_sqn": min_sqn,
    }

    sort_col1, sort_col2 = st.columns(2)
    with sort_col1:
        sort_metric_labels = [label for _, label in SORT_METRICS]
        sort_idx = st.selectbox("Sort By", range(len(SORT_METRICS)),
                                format_func=lambda x: sort_metric_labels[x],
                                key="gs_sort_metric")
        sort_key = SORT_METRICS[sort_idx][0]
    with sort_col2:
        sort_order = st.radio("Order", ["Highest to Lowest", "Lowest to Highest"],
                              horizontal=True, key="gs_sort_order")
        sort_descending = sort_order == "Highest to Lowest"

    st.markdown("---")

    # ── Section G: Calculate + Results ──────────────────
    calculate_clicked = st.button("Calculate", key="gs_calculate", type="primary")
    st.caption("To stop a running calculation, click the **Stop** button (top-right corner) or refresh the page.")

    # Cache invalidation
    cached = st.session_state.get("_gs_cached_results")
    if cached:
        fp = _build_cache_fingerprint(selected_strategy, search_group,
                                       search_set, condition_candidates)
        if cached.get("fingerprint") != fp:
            st.session_state.pop("_gs_cached_results", None)
            cached = None

    if calculate_clicked:
        results = _run_grid_search(
            selected_strategy, search_group, search_set, condition_candidates,
            sidebar_config)
        fp = _build_cache_fingerprint(selected_strategy, search_group,
                                       search_set, condition_candidates)
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
# Group Set Management UI
# ======================================================================

def _render_group_set_management():
    """Render the group set create/edit/delete/import/export UI."""
    tabs = st.tabs([label for _, label in GROUP_TYPES])

    for tab, (gs_type, gs_label) in zip(tabs, GROUP_TYPES):
        with tab:
            available = get_group_sets_by_type(gs_type)

            if available:
                gs_names = [gs["name"] for _, gs in available]
                sel = st.selectbox(f"Saved {gs_label} Sets", range(len(gs_names)),
                                   format_func=lambda x, n=gs_names: n[x],
                                   key=f"gs_mgmt_{gs_type}_sel")
                global_idx, selected_gs = available[sel]

                # Show candidates
                if selected_gs.get("candidates"):
                    cand_labels = []
                    for c in selected_gs["candidates"]:
                        cand_labels.append(format_candidate_label(c, gs_type))
                    st.caption(f"{len(cand_labels)} candidates")
                    with st.expander("View candidates", expanded=False):
                        for i, lbl in enumerate(cand_labels):
                            st.text(f"{i+1}. {lbl}")

                # Action buttons
                bc1, bc2, bc3 = st.columns(3)
                with bc1:
                    if st.button("Delete", key=f"gs_mgmt_{gs_type}_del", type="secondary"):
                        delete_group_set(global_idx)
                        st.rerun()
                with bc2:
                    json_data = export_group_set(selected_gs)
                    st.download_button("Export JSON", json_data,
                                       file_name=f"{selected_gs['name']}.json",
                                       mime="application/json",
                                       key=f"gs_mgmt_{gs_type}_export")
                with bc3:
                    if st.button("Edit", key=f"gs_mgmt_{gs_type}_edit_btn"):
                        st.session_state[f"gs_editing_{gs_type}"] = global_idx
                        st.rerun()

                # Edit mode
                if st.session_state.get(f"gs_editing_{gs_type}") is not None:
                    edit_idx = st.session_state[f"gs_editing_{gs_type}"]
                    edit_gs = st.session_state["saved_group_sets"][edit_idx]
                    st.markdown("---")
                    st.markdown(f"**Editing: {edit_gs['name']}**")
                    edited_candidates = _render_candidate_editor(
                        gs_type, list(edit_gs.get("candidates", [])),
                        f"gs_edit_{gs_type}")
                    ec1, ec2 = st.columns(2)
                    with ec1:
                        if st.button("Save Changes", key=f"gs_edit_{gs_type}_save", type="primary"):
                            edit_gs["candidates"] = copy.deepcopy(edited_candidates)
                            update_group_set(edit_idx, edit_gs)
                            st.session_state.pop(f"gs_editing_{gs_type}", None)
                            st.session_state.pop(f"gs_edit_{gs_type}_candidates", None)
                            st.rerun()
                    with ec2:
                        if st.button("Cancel", key=f"gs_edit_{gs_type}_cancel"):
                            st.session_state.pop(f"gs_editing_{gs_type}", None)
                            st.session_state.pop(f"gs_edit_{gs_type}_candidates", None)
                            st.rerun()
            else:
                st.caption(f"No {gs_label} group sets saved yet.")

            st.markdown("---")

            # Import
            uploaded = st.file_uploader(f"Import {gs_label} Group Set",
                                        type=["json"], key=f"gs_import_{gs_type}")
            if uploaded:
                # Prevent re-importing the same file on every rerun
                import_id = f"{uploaded.name}_{uploaded.size}"
                last_import_key = f"_gs_last_import_{gs_type}"
                if st.session_state.get(last_import_key) != import_id:
                    try:
                        data = import_group_set(uploaded.read().decode("utf-8"))
                        data["type"] = gs_type  # force correct type
                        save_group_set(data)
                        st.session_state[last_import_key] = import_id
                        st.success(f"Imported **{data['name']}** with {len(data['candidates'])} candidates.")
                        st.rerun()
                    except (ValueError, Exception) as e:
                        st.error(f"Import failed: {e}")
                else:
                    st.info("File already imported. Upload a different file or remove and re-upload.")

            # Create new
            with st.expander(f"Create New {gs_label} Set", expanded=False):
                new_name = st.text_input("Name", key=f"gs_new_{gs_type}_name")
                new_candidates = _render_candidate_editor(gs_type, [], f"gs_new_{gs_type}")
                if st.button("Save New Set", key=f"gs_new_{gs_type}_save", type="primary"):
                    if not new_name.strip():
                        st.error("Please provide a name.")
                    elif not new_candidates:
                        st.error("Add at least one candidate.")
                    else:
                        new_gs = {
                            "name": new_name.strip(),
                            "type": gs_type,
                            "candidates": copy.deepcopy(new_candidates),
                        }
                        save_group_set(new_gs)
                        # Clear the "Create New" editor state so it resets
                        st.session_state.pop(f"gs_new_{gs_type}_candidates", None)
                        _clear_candidate_widget_keys(f"gs_new_{gs_type}", 0, len(new_candidates))
                        st.success(f"Created **{new_name}** with {len(new_candidates)} candidates.")
                        st.rerun()


# ======================================================================
# Candidate Editor
# ======================================================================

def _clear_candidate_widget_keys(prefix, start_idx, end_idx):
    """Clear widget keys for candidate indices [start_idx, end_idx) to prevent
    stale values when candidates are removed and indices shift."""
    suffixes = [
        "_grp", "_e1", "_ev", "_cmp", "_e2", "_op", "_val",
        "_stype", "_atr_p", "_atr_m", "_cmp_d", "_cmp_at", "_rm",
    ]
    for i in range(start_idx, end_idx):
        for sfx in suffixes:
            st.session_state.pop(f"{prefix}_{i}{sfx}", None)


def _render_candidate_editor(group_type, initial_candidates, prefix):
    """Render an editable list of candidates. Returns list of candidate dicts."""
    # Use session state to track candidates for this editor
    state_key = f"{prefix}_candidates"
    if state_key not in st.session_state:
        st.session_state[state_key] = list(initial_candidates) if initial_candidates else []

    candidates = st.session_state[state_key]
    ema_count = len(st.session_state.get("gs_ema_periods", []))

    to_remove = []
    for i, cand in enumerate(candidates):
        st.markdown(f"**Candidate {i+1}**")
        updated = _render_single_candidate(group_type, cand, f"{prefix}_{i}", ema_count)
        candidates[i] = updated

        if st.button("Remove", key=f"{prefix}_{i}_rm"):
            to_remove.append(i)

    if to_remove:
        for idx in sorted(to_remove, reverse=True):
            candidates.pop(idx)
        # Clear stale widget keys so remaining candidates don't pick up
        # values from wrong indices after the list shifts
        _clear_candidate_widget_keys(prefix, min(to_remove), len(candidates) + len(to_remove))
        st.session_state[state_key] = candidates
        st.rerun()

    if st.button("+ Add Candidate", key=f"{prefix}_add"):
        candidates.append(_default_candidate(group_type))
        st.session_state[state_key] = candidates
        st.rerun()

    return candidates


def _default_candidate(group_type):
    """Return a default empty candidate dict for a given group type."""
    if group_type == "condition":
        return {
            "group": "Price & Indicators",
            "element1": "Price",
            "operator": "Above",
            "compare_type": "Indicator",
            "element2": "Tenkan",
            "value": None,
        }
    elif group_type == "static_stop":
        return {
            "stop_type": "ATR",
            "element1": "Price",
            "event": "Cross Below",
            "atr_period": 14,
            "atr_multiplier": 2.0,
        }
    else:
        return {
            "group": "Price & Indicators",
            "element1": "Price",
            "event": "Cross Above",
            "compare_type": "Indicator",
            "element2": "Tenkan",
            "value": None,
        }


def _render_single_candidate(group_type, cand, prefix, ema_count):
    """Render widgets for a single candidate and return updated dict."""

    if group_type == "static_stop":
        return _render_static_stop_edit(cand, prefix, ema_count)

    # Common layout for trigger, condition, dynamic_stop, target
    include_r = group_type in ("dynamic_stop", "target")
    include_atr_target = group_type == "target"
    is_condition = group_type == "condition"

    c_grp, c_e1, c_ev, c_cmp, c_e2 = st.columns([2, 2, 2, 1.5, 2])

    with c_grp:
        group_idx = GROUP_NAMES.index(cand.get("group", GROUP_NAMES[0])) if cand.get("group") in GROUP_NAMES else 0
        group = st.selectbox("Group", GROUP_NAMES, index=group_idx, key=f"{prefix}_grp")

    with c_e1:
        elements = get_group_elements(group, ema_count)
        extra = []
        if include_r:
            extra.extend(R_PROFIT_LOSS_ELEMENTS)
        if include_atr_target:
            extra.extend(ATR_TARGET_ELEMENTS)
        all_e1 = elements + extra
        e1_val = cand.get("element1", all_e1[0])
        if e1_val not in all_e1:
            e1_val = all_e1[0]
        element1 = st.selectbox("Element 1", all_e1, index=all_e1.index(e1_val), key=f"{prefix}_e1")

    is_r = element1 in R_PROFIT_LOSS_ELEMENTS
    is_atr_target = element1 in ATR_TARGET_ELEMENTS

    with c_ev:
        if is_condition:
            ops = CONDITION_OPERATORS
            op_val = cand.get("operator", ops[0])
            if op_val not in ops:
                op_val = ops[0]
            operator = st.selectbox("Operator", ops, index=ops.index(op_val), key=f"{prefix}_op")
        else:
            evts = STOP_EVENT_TYPES if group_type in ("dynamic_stop", "target") else EVENT_TYPES
            ev_val = cand.get("event", evts[0])
            if ev_val not in evts:
                ev_val = evts[0]
            event = st.selectbox("Event", evts, index=evts.index(ev_val), key=f"{prefix}_ev")

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
            result = {
                "group": group,
                "element1": element1,
                "event": event if not is_condition else None,
                "compare_type": "Indicator",
                "element2": None,
                "value": None,
                "atr_period": atr_period,
                "atr_multiplier": atr_mult,
            }
            if is_condition:
                result["operator"] = operator
            return result

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
    if is_condition:
        result["operator"] = operator
    else:
        result["event"] = event
    return result


def _render_static_stop_edit(cand, prefix, ema_count):
    """Render a static stop candidate editor."""
    c_type, c_detail = st.columns([1.5, 3])
    with c_type:
        st_val = cand.get("stop_type", "ATR")
        st_opts = ["ATR", "Indicator"]
        stop_type = st.selectbox("Stop Type", st_opts,
                                 index=st_opts.index(st_val) if st_val in st_opts else 0,
                                 key=f"{prefix}_stype")

    with c_detail:
        if stop_type == "ATR":
            ac1, ac2 = st.columns(2)
            with ac1:
                atr_period = st.number_input("ATR Period", 1, 200,
                                             value=int(cand.get("atr_period", 14)),
                                             step=1, key=f"{prefix}_atr_p")
            with ac2:
                atr_mult = st.number_input("ATR Multiplier", 0.1, 20.0,
                                           value=float(cand.get("atr_multiplier", 2.0)),
                                           step=0.1, format="%.1f", key=f"{prefix}_atr_m")
            return {
                "element1": "Price",
                "stop_type": "ATR",
                "event": cand.get("event", "Cross Below"),
                "atr_period": atr_period,
                "atr_multiplier": atr_mult,
            }
        else:
            all_elements = []
            for g in GROUP_NAMES:
                all_elements.extend(get_group_elements(g, ema_count))
            e2_val = cand.get("element2", all_elements[0] if all_elements else "Tenkan")
            if e2_val not in all_elements:
                e2_val = all_elements[0]
            element2 = st.selectbox("Stop Element", all_elements,
                                    index=all_elements.index(e2_val),
                                    key=f"{prefix}_e2")
            ev_val = cand.get("event", STOP_EVENT_TYPES[0])
            if ev_val not in STOP_EVENT_TYPES:
                ev_val = STOP_EVENT_TYPES[0]
            event = st.selectbox("Event", STOP_EVENT_TYPES,
                                 index=STOP_EVENT_TYPES.index(ev_val),
                                 key=f"{prefix}_ev")
            return {
                "stop_type": "Indicator",
                "group": "Price & Indicators",
                "element1": "Price",
                "element2": element2,
                "event": event,
                "compare_type": "Indicator",
            }


# ======================================================================
# Pattern selection UI (same as Performance tab)
# ======================================================================

def _render_pattern_selection():
    """Pattern selection UI identical to Performance tab."""
    selections = st.session_state.get("gs_selections", [])

    for idx, sel in enumerate(selections):
        c_mode, c_pt, c_prim, c_sec, c_rm = st.columns([2, 1.5, 2, 2, 0.5])

        with c_mode:
            mode = st.selectbox("Mode", SELECTION_MODES, key=f"gs_sel_{idx}_mode",
                                index=SELECTION_MODES.index(sel.get("mode", SELECTION_MODES[0])))
        sel["mode"] = mode

        with c_pt:
            need_pt = mode not in ("All Patterns",)
            if need_pt:
                pt = st.selectbox("Type", ["Bullish", "Bearish"], key=f"gs_sel_{idx}_pt",
                                  index=["Bullish", "Bearish"].index(sel.get("pattern_type", "Bullish")))
            else:
                pt = sel.get("pattern_type", "Bullish")
                st.selectbox("Type", ["—"], key=f"gs_sel_{idx}_pt_d", disabled=True)
            sel["pattern_type"] = pt

        with c_prim:
            need_prim = mode in ("Specified Primary", "Specified Secondary")
            if need_prim:
                prim_opts = PRIMARY_LIST
                prim_val = sel.get("primary") or prim_opts[0]
                if prim_val not in prim_opts:
                    prim_val = prim_opts[0]
                prim = st.selectbox("Primary", prim_opts, key=f"gs_sel_{idx}_prim",
                                    index=prim_opts.index(prim_val))
            else:
                prim = sel.get("primary")
                st.selectbox("Primary", ["—"], key=f"gs_sel_{idx}_prim_d", disabled=True)
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
                sec = st.selectbox("Secondary", sec_opts, key=f"gs_sel_{idx}_sec",
                                   index=sec_opts.index(sec_val) if sec_val in sec_opts else 0)
            else:
                sec = sel.get("secondary")
                st.selectbox("Secondary", ["—"], key=f"gs_sel_{idx}_sec_d", disabled=True)
            sel["secondary"] = sec

        with c_rm:
            st.markdown("<br>", unsafe_allow_html=True)
            if len(selections) > 1 and st.button("X", key=f"gs_sel_{idx}_rm"):
                selections.pop(idx)
                st.session_state["gs_selections"] = selections
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
            st.session_state[ema_key] = []
        emas_to_remove = []
        for idx, ema_val in enumerate(st.session_state[ema_key]):
            lc, rc = st.columns([3, 1])
            with lc:
                new_val = st.number_input(f"EMA {idx+1} Period", 2, 500, int(ema_val),
                                          step=1, key=f"{pfx}ema_p_{idx}")
                st.session_state[ema_key][idx] = new_val
            with rc:
                st.markdown("<br>", unsafe_allow_html=True)
                if st.button("X", key=f"{pfx}ema_rm_{idx}"):
                    emas_to_remove.append(idx)
        if emas_to_remove:
            for i in sorted(emas_to_remove, reverse=True):
                st.session_state[ema_key].pop(i)
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


# ======================================================================
# Dict-based aggregation (for multiprocessing — workers return dicts, not DataFrames)
# ======================================================================

def _aggregate_stats_dicts(all_stats_dicts):
    """Same logic as _aggregate_stats but works with lightweight dicts from workers."""
    import numpy as np

    all_trade_pnls = []
    total_win_pnl = 0.0
    total_lose_pnl = 0.0
    total_static_alloc = 0.0
    total_dynamic_alloc = 0.0
    total_target_alloc = 0.0

    for sd in all_stats_dicts:
        total_win_pnl += sd['win_pnl']
        total_lose_pnl += sd['lose_pnl']
        total_static_alloc += sd['total_static_alloc']
        total_dynamic_alloc += sd['total_dynamic_alloc']
        total_target_alloc += sd['total_target_alloc']
        all_trade_pnls.extend(sd['trade_pnls_r'])

    total_trades = len(all_trade_pnls)
    total_wins = sum(pnl > 0 for pnl in all_trade_pnls)
    total_losses = total_trades - total_wins
    total_pnl = total_win_pnl + total_lose_pnl

    if total_trades > 0:
        win_pct = (total_wins / total_trades) * 100
        lose_pct = (total_losses / total_trades) * 100
        target_exit_pct = total_target_alloc / total_trades
        static_exit_pct = total_static_alloc / total_trades
        dynamic_exit_pct = total_dynamic_alloc / total_trades

        avg_win_pnl = total_win_pnl / total_wins if total_wins > 0 else 0.0
        avg_lose_pnl = total_lose_pnl / total_losses if total_losses > 0 else 0.0
        expected_value = (win_pct / 100 * avg_win_pnl) + (lose_pct / 100 * avg_lose_pnl)

        if len(all_trade_pnls) >= 2:
            pnl_std = np.std(all_trade_pnls, ddof=1)
            sharpe_ratio = (np.mean(all_trade_pnls) / pnl_std) if pnl_std > 0 else 0.0
        else:
            pnl_std = 0.0
            sharpe_ratio = 0.0

        if all_trade_pnls:
            cumulative = np.cumsum(all_trade_pnls)
            peak = np.maximum.accumulate(cumulative)
            drawdowns = cumulative - peak
            max_drawdown = abs(drawdowns.min())
        else:
            max_drawdown = 0.0

        mar_ratio = (total_pnl / max_drawdown) if max_drawdown > 0 else 0.0

        if len(all_trade_pnls) >= 2 and pnl_std > 0:
            sqn = (np.mean(all_trade_pnls) / pnl_std * np.sqrt(len(all_trade_pnls)))
        else:
            sqn = 0.0
    else:
        win_pct = lose_pct = total_pnl = avg_win_pnl = avg_lose_pnl = 0.0
        expected_value = target_exit_pct = static_exit_pct = dynamic_exit_pct = 0.0
        sharpe_ratio = max_drawdown = mar_ratio = sqn = 0.0

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
        'sharpe_ratio': sharpe_ratio,
        'max_drawdown': max_drawdown,
        'mar_ratio': mar_ratio,
        'sqn': sqn,
    }


# ======================================================================
# Grid Search execution
# ======================================================================

def _run_grid_search(selected_strategy, search_group, search_set,
                     condition_candidates, sidebar_config):
    """Run backtests for all candidate runs.

    Returns list of (label, global_agg, selection_results) where
    selection_results is an OrderedDict {selection_label: agg_dict}.
    """

    # Build indicator params: start from defaults, overlay strategy settings, then gs_ overrides
    indicator_params = dict(_DEFAULT_INDICATOR_PARAMS)
    strategy_settings = selected_strategy.get("indicator_settings")
    if strategy_settings:
        strategy_settings = migrate_indicator_settings(strategy_settings)
        indicator_params.update(strategy_settings)
    # Override with gs_ UI settings
    gs_settings = collect_gs_indicator_settings(st.session_state)
    indicator_params.update(gs_settings)

    # Calculate indicators once
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
                            show_ichimoku=True, show_bb=True, show_kc=True,
                            show_donchian=True, show_psar=True)
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

    search_candidates = search_set.get("candidates", [])
    run_configs = generate_run_configs(
        base, search_group, search_candidates, condition_candidates)

    if not run_configs:
        st.warning("No run configurations generated.")
        return []

    return _run_grid_search_multiprocessing(
        run_configs, combo_slices, global_combo_keys,
        selection_combo_map)


def _run_grid_search_multiprocessing(run_configs, combo_slices, global_combo_keys,
                                      selection_combo_map):
    """Run grid search using multiprocessing Pool.
    Each worker process handles one candidate across all combo/period slices.
    """
    from strategies.grid_search_worker import init_worker, run_candidate

    n_candidates = len(run_configs)
    n_workers = min(n_candidates, max(1, os.cpu_count() or 1))

    progress = st.progress(0, text=f"Running grid search with {n_workers} processes...")
    st.caption(f"Using {n_workers} CPU cores for {n_candidates} candidates")

    # Convert combo_slices keys to plain tuples (ensure picklable)
    combo_slices_dict = dict(combo_slices)
    combo_keys_list = list(global_combo_keys)

    # Build task args: (idx, label, strategy) per candidate — idx for safe keying
    tasks = [(idx, label, strategy) for idx, (label, strategy) in enumerate(run_configs)]

    # Run with Pool — shared data passed via initializer (pickled once per worker)
    results = []
    try:
        with multiprocessing.Pool(
            processes=n_workers,
            initializer=init_worker,
            initargs=(combo_slices_dict, combo_keys_list)
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

    # Filter based on Global metrics
    filtered = []
    for label, global_agg, sel_results in results:
        if (global_agg["num_trades"] >= thresholds["min_trades"]
                and global_agg["expected_value"] >= thresholds["min_ev"]
                and global_agg["sharpe_ratio"] >= thresholds["min_sharpe"]
                and global_agg["sqn"] >= thresholds["min_sqn"]):
            filtered.append((label, global_agg, sel_results))

    st.subheader(f"Results — {strategy_name}")
    st.caption(f"{len(filtered)} of {len(results)} candidates pass filters")

    if not filtered:
        st.warning("No candidates pass the threshold filters.")
        return

    # Sort based on Global metrics
    filtered.sort(key=lambda x: x[1].get(sort_key, 0), reverse=sort_descending)

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
            "Sharpe": f"{global_agg['sharpe_ratio']:.2f}",
            "MAR": f"{global_agg['mar_ratio']:.2f}",
            "SQN": f"{global_agg['sqn']:.2f}",
        })

    df_results = pd.DataFrame(rows)

    # Copy to clipboard (all filtered results, TSV)
    tsv_data = df_results.to_csv(sep='\t', index=False)
    _copy_to_clipboard(tsv_data, key="gs_copy_results")

    st.caption(f"{len(filtered)} results (Global Performance)")
    st.dataframe(df_results, use_container_width=True, hide_index=True)

    # Expandable detail view for each candidate
    st.markdown("---")
    st.subheader("Detailed Performance")
    for idx in range(len(filtered)):
        label, global_agg, sel_results = filtered[idx]
        with st.expander(f"**{label}**", expanded=False):
            # Build table: Global on the left + per-selection columns (if multiple)
            table_data = {"Global": global_agg}
            if len(sel_results) > 1:
                table_data.update(sel_results)
            table = _build_metrics_table(table_data)
            st.table(table)
            _copy_to_clipboard(
                table.to_csv(sep='\t'),
                key=f"gs_sel_detail_{idx}")


# ======================================================================
# Cache helpers
# ======================================================================

def _build_cache_fingerprint(strategy, search_group, search_set, condition_candidates):
    """Build a hashable fingerprint for cache invalidation."""
    import json
    parts = [
        strategy.get("strategy_name", ""),
        search_group,
        search_set.get("name", ""),
        json.dumps(search_set.get("candidates", []), sort_keys=True),
        json.dumps(condition_candidates, sort_keys=True) if condition_candidates else "",
    ]
    return "|".join(parts)

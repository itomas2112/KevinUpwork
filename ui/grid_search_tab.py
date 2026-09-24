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
                          ALL_UNIQUE_SECONDARIES, selection_label,
                          FIXED_COLUMNS, FIXED_SELECTIONS, all_combos,
                          build_pattern_columns, unique_label)
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
    generate_candidates, validate_generator, count_candidates,
)
from ui.charting_tab import _get_or_calculate
from ui.performance_tab import (_DEFAULT_INDICATOR_PARAMS, SELECTION_MODES,
                                 _build_metrics_table, _copy_to_clipboard, _empty_agg,
                                 _new_user_selection, PATTERN_COLUMNS_CAPTION)
from ui.grid_search_helpers import (
    format_candidate_label, format_run_label, generate_run_configs,
    _ema_periods_for_strategy, result_strategy_for_variant,
    make_shortlist_entry, shortlist_has, shortlist_add, shortlist_remove,
    shortlist_strategy_for_save,
)
from config.constants import (GROUP_NAMES, EVENT_TYPES, STOP_EVENT_TYPES,
                               CONDITION_OPERATORS, get_group_elements)

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


def selection_labels_for(selections):
    """Unique, ordered labels for the user-added pattern-selection rows.

    Mirrors the user-column labels produced by build_pattern_columns: labels
    are de-duplicated against the fixed columns and each other
    ("X", "X (2)", ...)."""
    labels = []
    for sel in selections:
        labels.append(unique_label(selection_label(sel), set(FIXED_COLUMNS) | set(labels)))
    return labels


def scope_agg(all_patterns_agg, columns, scope):
    """Return the agg dict that filters/sort/table should use for `scope`.

    `columns` is the build_pattern_columns OrderedDict (All Patterns, All
    Bullish, All Bearish, Global, user rows). An unknown scope (e.g. stale
    cache) -> all_patterns_agg. Otherwise the column's agg, with correlation
    copied from All Patterns because correlation is only computed there.
    """
    agg = (columns or {}).get(scope, all_patterns_agg)
    if agg is all_patterns_agg:
        return all_patterns_agg
    agg = dict(agg)
    agg.setdefault("correlation", all_patterns_agg.get("correlation"))
    agg.setdefault("abs_correlation", all_patterns_agg.get("abs_correlation"))
    return agg


def filter_and_sort_results(results, thresholds, sort_key, sort_descending,
                            scope="All Patterns"):
    """Filter and sort grid search results on the metrics of `scope`.

    results: list of (label, all_patterns_agg, columns, strategy).
    Returns a list of (label, metric_agg, all_patterns_agg, columns, strategy)
    where metric_agg is scope_agg(all_patterns_agg, columns, scope). None sort
    values go to the bottom regardless of direction.
    """
    filtered = []
    for label, all_patterns_agg, columns, strategy in results:
        metric_agg = scope_agg(all_patterns_agg, columns, scope)
        if passes_thresholds(metric_agg, thresholds):
            filtered.append((label, metric_agg, all_patterns_agg, columns, strategy))

    def _sort_val(x):
        v = x[1].get(sort_key)
        if v is None:
            return float('inf') if not sort_descending else float('-inf')
        return v
    filtered.sort(key=_sort_val, reverse=sort_descending)
    return filtered


def mc_settings():
    """Current MC sizing settings from the sidebar Instrument section."""
    from config.constants import (DEFAULT_INSTRUMENT, MC_DEFAULT_BALANCE,
                                  MC_DEFAULT_TRADES_PER_SIM, INSTRUMENTS,
                                  instrument_point_value, instrument_margin)
    symbol = st.session_state.get('instrument', DEFAULT_INSTRUMENT)
    if symbol not in INSTRUMENTS:
        symbol = DEFAULT_INSTRUMENT
    return {
        'instrument': symbol,
        'point_value': instrument_point_value(symbol),
        'margin': instrument_margin(symbol),
        'balance': float(st.session_state.get('mc_starting_balance', MC_DEFAULT_BALANCE)),
        'trades_per_sim': int(st.session_state.get('mc_trades_per_sim', MC_DEFAULT_TRADES_PER_SIM)),
    }


def mc_fingerprint():
    """(instrument, balance, trades_per_sim) — MC results are stale when this changes."""
    cfg = mc_settings()
    return (cfg['instrument'], cfg['balance'], cfg['trades_per_sim'])


def mc_caption(n_sims=None):
    """Caption describing how 'MC Avg Profit @ 5% DD' is computed."""
    cfg = mc_settings()
    n_sims = GS_MC_N_SIMULATIONS if n_sims is None else n_sims
    return (f"MC Avg Profit @ 5% Avg Max DD: {cfg['instrument']}, ${cfg['balance']:,.0f} start, "
            f"{cfg['trades_per_sim']} trades/sim bootstrapped from each candidate's trades, "
            f"{n_sims:,} sims. Trades 1–30 risk 1% (0.75–1.25% after whole-contract rounding, "
            f"trade skipped if no size fits); every 200 trades thereafter the risk % is "
            f"re-assessed from the trades so far (30-sim MC of the next block) to target 5% "
            f"avg max DD. "
            f"Contracts capped by margin (balance ÷ ${cfg['margin']:,.0f}) and re-sized after "
            f"every trade. '(cap)' = most re-assessed blocks ran at full margin.")


def mc_caption_short():
    """One-line version of mc_caption for the Strategy Testing / Performance tabs."""
    cfg = mc_settings()
    return (f"MC Avg Profit @ 5% Avg Max DD ({cfg['instrument']}, ${cfg['balance']:,.0f}, "
            f"{cfg['trades_per_sim']} trades): trades 1–30 risk 1% (0.75–1.25%, else skipped), "
            f"then every 200 trades the risk % is re-assessed (30-sim MC of the next block) "
            f"to target 5% avg max DD; margin-capped. "
            f"'(cap)' = mostly full margin, '(skip x%)' = share of first-30 trades skipped.")


def format_mc_value(agg):
    """'$12,345' with ' (cap)' when most re-assessed blocks ran at full margin,
    and ' (skip x%)' when more than 10% of the 1%-phase trades were skipped
    because no whole-contract size fit the 0.75–1.25% band."""
    text = f"${agg.get('mc_avg_profit', 0) or 0:,.0f}"
    if agg.get('mc_margin_capped'):
        text += " (cap)"
    skipped = agg.get('mc_initial_skipped') or 0.0
    if skipped > 0.10:
        text += f" (skip {skipped:.0%})"
    return text


def _mc_tasks(agg_dicts):
    """Fill trade-less aggs with 0.0 / False; return (idx, pnls_r, r_dists) tasks."""
    tasks = []
    for idx, agg in enumerate(agg_dicts):
        pnls = list(agg.get('trade_pnls_r') or [])
        if not pnls:
            agg['mc_avg_profit'] = 0.0
            agg['mc_margin_capped'] = False
            agg['mc_initial_skipped'] = 0.0
            continue
        tasks.append((idx, pnls, list(agg.get('trade_r_distances') or [])))
    return tasks


def _enrich_mc_serial(agg_dicts, balance, n_sims, target_dd, point_value, margin,
                      trades_per_sim, tasks=None):
    """Serial MC enrichment (no pool, no streamlit). Mutates agg_dicts in place."""
    from strategies.monte_carlo_core import mc_phased_profit
    if tasks is None:
        tasks = _mc_tasks(agg_dicts)
    for idx, pnls, r_dists in tasks:
        res = mc_phased_profit(
            pnls, r_dists, balance, point_value, margin,
            trades_per_sim=trades_per_sim, n_sims=n_sims)
        agg_dicts[idx]['mc_avg_profit'] = res['avg_profit']
        agg_dicts[idx]['mc_margin_capped'] = res['margin_capped']
        agg_dicts[idx]['mc_initial_skipped'] = res['initial_skipped_fraction']


def enrich_aggs_with_mc(agg_dicts, balance, n_sims, target_dd=5.0,
                        point_value=None, margin=None, trades_per_sim=None,
                        progress_label="Computing Monte Carlo stats"):
    """Enrich a flat list of agg dicts with `mc_avg_profit`,
    `mc_margin_capped` and `mc_initial_skipped` (in-place).

    Bootstraps each agg's own `trade_pnls_r` / `trade_r_distances` into a
    dollar-based simulation of `trades_per_sim` trades, sized in whole
    contracts from each trade's stop distance and capped by margin, with
    phased risk (mc_phased_profit: 1% for trades 1–30, re-assessed every 200 trades thereafter).
    point_value / margin / trades_per_sim default to the sidebar Instrument
    settings. Spawns a multiprocessing pool; falls back to serial execution
    if pool setup fails. Aggs without trades get 0.0 / False without
    entering the pool.
    """
    from strategies.mc_enrichment_worker import init_worker as mc_init, enrich_one

    if point_value is None or margin is None or trades_per_sim is None:
        cfg = mc_settings()
        point_value = cfg['point_value'] if point_value is None else point_value
        margin = cfg['margin'] if margin is None else margin
        trades_per_sim = cfg['trades_per_sim'] if trades_per_sim is None else trades_per_sim

    tasks = _mc_tasks(agg_dicts)
    if not tasks:
        return

    n_workers = min(len(tasks), max(1, os.cpu_count() or 1))
    progress = st.progress(0, text=f"{progress_label} ({len(tasks)} aggs, {n_workers} workers)...")

    try:
        with multiprocessing.Pool(
            processes=n_workers,
            initializer=mc_init,
            initargs=(balance, n_sims, target_dd, point_value, margin, trades_per_sim),
        ) as pool:
            completed = 0
            for idx, value, capped, skipped in pool.imap_unordered(enrich_one, tasks,
                                                                   chunksize=4):
                agg_dicts[idx]['mc_avg_profit'] = value
                agg_dicts[idx]['mc_margin_capped'] = capped
                agg_dicts[idx]['mc_initial_skipped'] = skipped
                completed += 1
                progress.progress(completed / len(tasks),
                                  text=f"{progress_label} {completed}/{len(tasks)}")
    except Exception as e:
        st.warning(f"MC parallel enrichment failed ({e}); falling back to serial.")
        _enrich_mc_serial(agg_dicts, balance, n_sims, target_dd, point_value,
                          margin, trades_per_sim, tasks=tasks)

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
        _render_group_set_management(_ema_periods_for_strategy(selected_strategy))

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
    scope_options = FIXED_COLUMNS + selection_labels_for(st.session_state.get("gs_selections", []))
    if st.session_state.get("gs_filter_scope") not in scope_options:
        st.session_state["gs_filter_scope"] = "All Patterns"
    filter_scope = st.selectbox("Apply filters to", scope_options, key="gs_filter_scope")
    st.caption("Filters, sorting and the results table use this column's metrics.")
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

    # Cache invalidation (pattern rows are NOT part of the fingerprint: they
    # only change how the cached per-combo results are aggregated)
    cached = st.session_state.get("_gs_cached_results")
    if cached:
        fp = _build_cache_fingerprint(selected_strategy, search_group,
                                       search_set, selected_events, condition_candidates,
                                       view_state)
        if cached.get("fingerprint") != fp or "combo_results" not in cached:
            st.session_state.pop("_gs_cached_results", None)
            cached = None

    user_selections = st.session_state.get("gs_selections", [])
    selections_json = _selections_json(user_selections)

    if calculate_clicked:
        results, combo_results_list = _run_grid_search(
            selected_strategy, search_group, search_set, selected_events,
            condition_candidates, condition_event, sidebar_config,
            effective_candidates=effective_candidates,
            use_original_engine=use_original_engine)
        fp = _build_cache_fingerprint(selected_strategy, search_group,
                                       search_set, selected_events, condition_candidates,
                                       view_state)
        st.session_state["_gs_cached_results"] = {
            "fingerprint": fp,
            "mc_fingerprint": mc_fingerprint(),
            "results": results,
            "combo_results": combo_results_list,
            "selections_json": selections_json,
            "strategy_name": selected_strategy.get("strategy_name", "Custom"),
            "search_group": search_group,
            "group_set": search_set.get("name", ""),
        }
        cached = st.session_state["_gs_cached_results"]

    if cached and cached.get("results") and cached.get("selections_json") != selections_json:
        # User pattern rows changed — re-aggregate the pattern columns from
        # the cached per-combo stats (no re-run), MC-enrich only the new aggs.
        new_results, changed_aggs = reaggregate_results(
            cached["results"], cached["combo_results"], user_selections)
        if changed_aggs:
            mc_cfg = mc_settings()
            enrich_aggs_with_mc(changed_aggs, mc_cfg['balance'], GS_MC_N_SIMULATIONS,
                                target_dd=5.0, point_value=mc_cfg['point_value'],
                                margin=mc_cfg['margin'], trades_per_sim=mc_cfg['trades_per_sim'])
        cached["results"] = new_results
        cached["selections_json"] = selections_json
        st.caption("Pattern columns re-aggregated from the last run.")

    if cached and cached.get("results") and cached.get("mc_fingerprint") != mc_fingerprint():
        # Instrument / MC balance / trades-per-sim changed — trades are still
        # valid, only the MC column is stale. Re-enrich the cached aggs.
        mc_cfg = mc_settings()
        _enrich_mc_parallel(cached["results"], mc_cfg['balance'], GS_MC_N_SIMULATIONS,
                            target_dd=5.0, point_value=mc_cfg['point_value'],
                            margin=mc_cfg['margin'], trades_per_sim=mc_cfg['trades_per_sim'])
        cached["mc_fingerprint"] = mc_fingerprint()

    if cached and cached.get("results"):
        _display_results(cached["results"], cached["strategy_name"],
                         thresholds, sort_key, sort_descending, scope=filter_scope,
                         search_group=cached.get("search_group", search_group),
                         group_set=cached.get("group_set", search_set.get("name", "")))
    elif not calculate_clicked:
        st.info("Configure search and click **Calculate** to run.")

    _render_shortlist()


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

_GEN_EDIT_PREFIX = "gs_gen_edit"
_GEN_NEW_PREFIX = "gs_gen_new"
_GS_MGMT_MSG_KEY = "_gs_mgmt_msg"

# Above this many candidates the preview warns (search time grows linearly).
GENERATOR_WARN_CANDIDATES = 2000


def _clear_prefixed_keys(prefix):
    """Drop every session-state key under `prefix_` (generator form widgets)."""
    for k in [k for k in st.session_state.keys()
              if isinstance(k, str) and k.startswith(prefix + "_")]:
        del st.session_state[k]


def _render_legacy_set_summary(group_set):
    """Read-only view of a set built with the old line editor (no generator)."""
    cands = group_set.get("candidates", [])
    st.caption("Created with the old line editor — export it to keep a copy, "
               "or create a new set with the generator.")
    st.markdown(f"{len(cands)} candidates")
    for i, c in enumerate(cands[:20]):
        lbl = format_candidate_label(c)
        if get_mode(group_set) == MODE_PER_CANDIDATE and c.get("event"):
            lbl += f"  [{c['event']}]"
        st.text(f"{i + 1}. {lbl}")
    if len(cands) > 20:
        st.caption(f"… and {len(cands) - 20} more.")


def _render_group_set_management(ema_periods):
    """Render the universal group set create/edit/delete/import/export UI.

    `ema_periods` comes from the selected strategy and sizes the EMA elements
    offered in the generator form (and the EMA limit when regenerating).
    """
    all_sets = st.session_state.get("saved_group_sets", [])
    ema_count = len(ema_periods)

    msg = st.session_state.pop(_GS_MGMT_MSG_KEY, None)
    if msg:
        st.success(msg)

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
                st.session_state.pop("gs_editing", None)
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
                _clear_prefixed_keys(_GEN_EDIT_PREFIX)
                st.rerun()

        # Edit mode
        if st.session_state.get("gs_editing") is not None:
            edit_idx = st.session_state["gs_editing"]
            if edit_idx < len(all_sets):
                edit_gs = all_sets[edit_idx]
                st.markdown("---")
                st.markdown(f"**Editing: {edit_gs['name']}**")
                if edit_gs.get("generator") is None:
                    _render_legacy_set_summary(edit_gs)
                    if st.button("Close", key="gs_edit_cancel"):
                        st.session_state.pop("gs_editing", None)
                        st.rerun()
                else:
                    name_key = f"{_GEN_EDIT_PREFIX}_name"
                    if name_key not in st.session_state:
                        st.session_state[name_key] = edit_gs["name"]
                    edit_name = st.text_input("Name", key=name_key)
                    generator, errors, gen_cands = _render_generator_form(
                        edit_gs["generator"], _GEN_EDIT_PREFIX, ema_periods)
                    edited_ranges = _render_indicator_ranges_editor(
                        gen_cands, edit_gs.get("indicator_ranges") or {}, _GEN_EDIT_PREFIX)
                    ec1, ec2 = st.columns(2)
                    with ec1:
                        if st.button("Save Changes", key="gs_edit_save", type="primary"):
                            if errors:
                                st.error("Fix the problems above before saving.")
                            elif not edit_name.strip():
                                st.error("Please provide a name.")
                            else:
                                # Keep the saved filter view (active_groups / value_filters)
                                new_gs = dict(edit_gs)
                                new_gs.update({
                                    "name": edit_name.strip(),
                                    "mode": MODE_RUNTIME,
                                    "generator": generator,
                                    "indicator_ranges": edited_ranges,
                                })
                                try:
                                    update_group_set(edit_idx, new_gs, ema_count)
                                except ValueError as e:
                                    st.error(str(e))
                                else:
                                    st.session_state.pop("gs_editing", None)
                                    _clear_prefixed_keys(_GEN_EDIT_PREFIX)
                                    st.session_state[_GS_MGMT_MSG_KEY] = (
                                        f"Saved **{new_gs['name']}** with "
                                        f"{len(new_gs['candidates'])} candidates.")
                                    st.rerun()
                    with ec2:
                        if st.button("Cancel", key="gs_edit_cancel"):
                            st.session_state.pop("gs_editing", None)
                            _clear_prefixed_keys(_GEN_EDIT_PREFIX)
                            st.rerun()
    else:
        st.caption("No group sets saved yet.")

    st.markdown("---")

    # Import
    uploaded = st.file_uploader("Import Group Set",
                                type=["json"], key="gs_import")
    if uploaded:
        import_id = f"{uploaded.name}_{uploaded.size}"
        # No "_gs_" prefix on purpose: the sidebar's timeframe-change sweep
        # clears every "_gs_*" key, which would re-import the still-loaded
        # file on the next rerun (seen as a duplicated group set).
        last_import_key = "gs_last_import"
        if st.session_state.get(last_import_key) != import_id:
            try:
                data = import_group_set(uploaded.read().decode("utf-8"),
                                        ema_count=ema_count)
                dupes_removed = data.pop("_duplicates_removed", 0)
                regenerated = data.pop("_regenerated", None)
                save_group_set(data, ema_count)
                st.session_state[last_import_key] = import_id
                msg = f"Imported **{data['name']}** with {len(data['candidates'])} candidates."
                if regenerated is not None:
                    msg += (f" Regenerated {regenerated} candidates from the "
                            f"generator block (candidates in the file were ignored).")
                if dupes_removed:
                    msg += f" ({dupes_removed} duplicate(s) removed.)"
                st.session_state[_GS_MGMT_MSG_KEY] = msg
                st.rerun()
            except (ValueError, Exception) as e:
                st.error(f"Import failed: {e}")
        else:
            st.info("File already imported. Upload a different file or remove and re-upload.")

    # Create new
    with st.expander("Create New Group Set", expanded=False):
        new_name = st.text_input("Name", key="gs_new_name")
        generator, errors, gen_cands = _render_generator_form(
            None, _GEN_NEW_PREFIX, ema_periods)
        new_ranges = _render_indicator_ranges_editor(gen_cands, {}, _GEN_NEW_PREFIX)
        if st.button("Save New Set", key="gs_new_save", type="primary"):
            if not new_name.strip():
                st.error("Please provide a name.")
            elif errors:
                st.error("Fix the problems above before saving.")
            else:
                new_gs = {
                    "name": new_name.strip(),
                    "mode": MODE_RUNTIME,
                    "generator": generator,
                    "indicator_ranges": new_ranges,
                }
                try:
                    save_group_set(new_gs, ema_count)
                except ValueError as e:
                    st.error(str(e))
                else:
                    _clear_prefixed_keys(_GEN_NEW_PREFIX)
                    st.session_state.pop("gs_new_name", None)
                    st.session_state[_GS_MGMT_MSG_KEY] = (
                        f"Created **{new_gs['name']}** with "
                        f"{len(new_gs['candidates'])} candidates.")
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
# Group Set generator form
# ======================================================================

def _fmt_ema_element(name, ema_periods):
    """Display "EMA 1 (10)" while the stored value stays "EMA 1"."""
    if isinstance(name, str) and name.startswith("EMA "):
        try:
            idx = int(name.split(" ")[1]) - 1
        except (IndexError, ValueError):
            return name
        if 0 <= idx < len(ema_periods):
            return f"{name} ({ema_periods[idx]})"
    return name


def _render_range_inputs(label, prefix, saved, defaults, with_period=False):
    """One "Apply + [period] + min / max / step" row. Returns the spec dict
    ({min, max, step[, period]}) when applied, else None."""
    apply_key = f"{prefix}_apply"
    keys = {f: f"{prefix}_{f}" for f in ("min", "max", "step")}
    if apply_key not in st.session_state:
        st.session_state[apply_key] = bool(saved)
    for f, dflt in zip(("min", "max", "step"), defaults):
        if keys[f] not in st.session_state:
            st.session_state[keys[f]] = float((saved or {}).get(f, dflt))
    period_key = f"{prefix}_period"
    if with_period and period_key not in st.session_state:
        st.session_state[period_key] = int((saved or {}).get("period", 14))

    if with_period:
        c_lbl, c_apply, c_p, c_lo, c_hi, c_st = st.columns([2, 1, 1, 1.2, 1.2, 1.2])
    else:
        c_lbl, c_apply, c_lo, c_hi, c_st = st.columns([2, 1, 1.2, 1.2, 1.2])
    with c_lbl:
        st.markdown(f"**{label}**")
    with c_apply:
        st.markdown("&nbsp;", unsafe_allow_html=True)
        applied = st.checkbox("Apply", key=apply_key)
    if with_period:
        with c_p:
            period = st.number_input("Period", min_value=1, step=1, key=period_key)
    with c_lo:
        lo = st.number_input("Min", key=keys["min"], step=0.1, format="%.2f")
    with c_hi:
        hi = st.number_input("Max", key=keys["max"], step=0.1, format="%.2f")
    with c_st:
        sp = st.number_input("Step", key=keys["step"], step=0.1, format="%.2f")
    if not applied:
        return None
    spec = {"min": lo, "max": hi, "step": sp}
    if with_period:
        spec = {"period": int(period), **spec}
    return spec


def _render_generator_form(generator, prefix, ema_periods):
    """Render the group set generator (elements, fixed values, exclusions,
    special candidates) with a live preview.

    Returns (generator_dict, errors, generated_candidates); candidates are []
    while the spec has errors."""
    generator = generator or {}
    ema_count = len(ema_periods)

    def fmt(name):
        return _fmt_ema_element(name, ema_periods)

    st.caption("Events (Cross/Close Above/Below…) are chosen at search time.")

    # ── Elements ─────────────────────────────────────────
    st.markdown("**Elements**")
    st.caption(
        "Every pair of ticked elements **within the same indicator group** is "
        "generated once (Cross Above / Cross Below at search time cover both "
        "directions). Pairs with **Price** are generated both ways, because "
        "Price as Element 1 uses the bar high/low. *These rules are "
        "assumptions — say if you want them different.*")
    initial = list(generator.get("elements") or [])
    elements = []
    unavailable = []
    for grp in GROUP_NAMES:
        opts = get_group_elements(grp, ema_count)
        key = f"{prefix}_el__{grp}"
        if key not in st.session_state:
            st.session_state[key] = [e for e in initial if e in opts]
        else:
            st.session_state[key] = [e for e in st.session_state[key] if e in opts]
        picked = st.multiselect(grp, opts, key=key, format_func=fmt)
        elements.extend(e for e in opts if e in picked)
        if grp == "Price & Indicators":
            # EMA N saved in the generator but beyond the strategy's EMA count:
            # kept so validation blocks the save instead of silently dropping it.
            unavailable = [e for e in initial
                           if isinstance(e, str) and e.startswith("EMA ") and e not in opts]
            if unavailable:
                drop = st.checkbox(
                    f"Drop unavailable elements ({', '.join(unavailable)}) — the "
                    f"selected strategy has {ema_count} EMA(s)",
                    key=f"{prefix}_drop_unavailable")
                if not drop:
                    elements.extend(unavailable)

    # ── Fixed values ─────────────────────────────────────
    saved_fv = generator.get("fixed_values") or {}
    fixed_values = {}
    with st.expander("Fixed values (optional)", expanded=bool(saved_fv)):
        st.caption("Per element: min / max / step → one *Element vs value* "
                   "candidate per value. Elements with a range still take "
                   "part in pairing.")
        if not elements:
            st.caption("Tick elements above first.")
        for el in elements:
            if el in unavailable:
                if el in saved_fv:
                    fixed_values[el] = saved_fv[el]
                continue
            spec = _render_range_inputs(fmt(el), f"{prefix}_fv__{el}",
                                        saved_fv.get(el), (20.0, 80.0, 10.0))
            if spec:
                fixed_values[el] = spec

    # ── Exclusions ───────────────────────────────────────
    st.markdown("**Do not run**")
    same_key = f"{prefix}_same_ind"
    cross_key = f"{prefix}_cross_group"
    if same_key not in st.session_state:
        st.session_state[same_key] = bool(generator.get("exclude_same_indicator", True))
    if cross_key not in st.session_state:
        st.session_state[cross_key] = bool(generator.get("allow_cross_group", False))
    xc1, xc2 = st.columns(2)
    with xc1:
        exclude_same = st.checkbox(
            "Exclude bands of the same indicator", key=same_key,
            help="Drops BB×BB, KC×KC, DC×DC, Supertrend×Supertrend, PSAR×PSAR, "
                 "LR×LR and Price Upper×Price Lower pairs. Ichimoku lines are "
                 "not grouped (Tenkan vs Kijun stays).")
    with xc2:
        allow_cross = st.checkbox(
            "Allow cross-group pairs", key=cross_key,
            help="Also pair elements from different indicator groups "
                 "(e.g. RSI vs BB Upper — usually different scales).")

    uids_key = f"{prefix}_excl_uids"
    next_key = f"{prefix}_excl_next"
    if uids_key not in st.session_state:
        uids = []
        for i, pair in enumerate(generator.get("exclusions") or []):
            st.session_state[f"{prefix}_excl_a__{i}"] = pair[0]
            st.session_state[f"{prefix}_excl_b__{i}"] = pair[1]
            uids.append(i)
        st.session_state[uids_key] = uids
        st.session_state[next_key] = len(uids)

    exclusions = []
    for uid in list(st.session_state[uids_key]):
        a_key, b_key = f"{prefix}_excl_a__{uid}", f"{prefix}_excl_b__{uid}"
        c_a, c_b, c_rm = st.columns([3, 3, 1])
        a_val, b_val = st.session_state.get(a_key), st.session_state.get(b_key)
        if a_val in elements and b_val in elements:
            with c_a:
                a_val = st.selectbox("Element 1", elements, key=a_key,
                                     format_func=fmt, label_visibility="collapsed")
            with c_b:
                b_val = st.selectbox("Element 2", elements, key=b_key,
                                     format_func=fmt, label_visibility="collapsed")
        else:
            # An element of this exclusion isn't ticked — keep the row as-is.
            with c_a:
                st.text(f"{fmt(a_val)} vs {fmt(b_val)}")
            with c_b:
                st.caption("(element not selected — no effect)")
        with c_rm:
            if st.button("✕", key=f"{prefix}_excl_rm__{uid}"):
                st.session_state[uids_key].remove(uid)
                st.session_state.pop(a_key, None)
                st.session_state.pop(b_key, None)
                st.rerun()
        exclusions.append([a_val, b_val])
    if st.button("+ Add exclusion", key=f"{prefix}_excl_add",
                 disabled=len(elements) < 2):
        uid = st.session_state[next_key]
        st.session_state[next_key] = uid + 1
        st.session_state[f"{prefix}_excl_a__{uid}"] = elements[0]
        st.session_state[f"{prefix}_excl_b__{uid}"] = elements[1]
        st.session_state[uids_key].append(uid)
        st.rerun()

    # ── Special candidates ───────────────────────────────
    has_special = any(generator.get(k) for k in ("r_profit", "r_loss", "atr_target", "atr_stop"))
    with st.expander("Special candidates (R Profit / R Loss / ATR)", expanded=has_special):
        r_profit = _render_range_inputs("R Profit", f"{prefix}_rp",
                                        generator.get("r_profit"), (1.0, 3.0, 0.5))
        r_loss = _render_range_inputs("R Loss", f"{prefix}_rl",
                                      generator.get("r_loss"), (0.5, 2.0, 0.5))
        st.caption("ATR: period, then multiplier min / max / step.")
        atr_target = _render_range_inputs("ATR Target", f"{prefix}_atrt",
                                          generator.get("atr_target"), (1.0, 3.0, 0.5),
                                          with_period=True)
        atr_stop = _render_range_inputs("ATR Stop", f"{prefix}_atrs",
                                        generator.get("atr_stop"), (1.0, 5.0, 0.2),
                                        with_period=True)

    gen = {
        "elements": elements,
        "fixed_values": fixed_values,
        "exclusions": exclusions,
        "exclude_same_indicator": exclude_same,
        "allow_cross_group": allow_cross,
        "r_profit": r_profit,
        "r_loss": r_loss,
        "atr_target": atr_target,
        "atr_stop": atr_stop,
    }

    # ── Preview ──────────────────────────────────────────
    errors = validate_generator(gen, ema_count)
    if errors:
        if errors == ["The selection generates no candidates."]:
            st.info("Tick elements (or special candidates) to generate candidates.")
        else:
            for e in errors:
                st.error(e)
        return gen, errors, []

    candidates = generate_candidates(gen, ema_count)
    counts = count_candidates(gen, ema_count)
    st.markdown(f"**Preview:** {counts['pairs']} pairs + {counts['fixed']} fixed + "
                f"{counts['r']} R + {counts['atr']} ATR = "
                f"**{counts['total']} candidates**")
    if counts["total"] > GENERATOR_WARN_CANDIDATES:
        st.warning(f"{counts['total']} candidates — each one runs for every event "
                   f"chosen at search time, so this search will be slow.")
    with st.expander("Generated candidates", expanded=False):
        st.dataframe(pd.DataFrame({"Candidate": [format_candidate_label(c)
                                                 for c in candidates]}),
                     width="stretch", hide_index=True)
    return gen, errors, candidates


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
    """User-added pattern rows (same modes as the Performance tab).

    All Patterns / All Bullish / All Bearish / Global are fixed columns of
    every result, so only the three "Specified…" modes are offered."""
    st.caption("All Patterns, All Bullish, All Bearish and Global are always shown; "
               "add rows for the individual patterns you want as extra columns "
               "(Global aggregates them). Changing rows re-aggregates the last "
               "run without re-running the search.")
    # Drop rows whose mode is no longer offered (e.g. a stale "All Patterns" row)
    selections = [sel for sel in st.session_state.get("gs_selections", [])
                  if sel.get("mode") in SELECTION_MODES]
    st.session_state["gs_selections"] = selections

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
            if st.button("X", key=f"{kp}_rm"):
                selections.pop(idx)
                st.session_state["gs_selections"] = selections
                # Bump generation so all widget keys are fresh on next render
                st.session_state["_gs_sel_gen"] = gen + 1
                st.rerun()

    if st.button("+ Add Selection", key="gs_add_sel"):
        selections.append(_new_user_selection())
        st.session_state["gs_selections"] = selections
        st.rerun()


# ======================================================================
# Dict-based aggregation (for multiprocessing — workers return dicts, not DataFrames)
# ======================================================================

def _aggregate_stats_dicts(all_stats_dicts):
    """Same logic as _aggregate_stats but works with lightweight dicts from workers."""
    import numpy as np

    all_trade_pnls = []
    all_r_dists = []
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
        # Pad to keep r_dists aligned with pnls (0.0 = unusable for MC sizing)
        n_sd = len(sd['trade_pnls_r'])
        all_r_dists.extend((list(sd.get('trade_r_distances', [])) + [0.0] * n_sd)[:n_sd])
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
        'trade_pnls_r': [float(p) for p in all_trade_pnls],
        'trade_r_distances': [float(r) for r in all_r_dists],
    }


# ======================================================================
# MC enrichment — parallel version
# ======================================================================

def _enrich_mc_parallel(results, balance, n_sims, target_dd=5.0,
                        point_value=None, margin=None, trades_per_sim=None):
    """Enrich Grid Search results (list of (label, all_patterns_agg, columns, strategy))
    with `mc_avg_profit` / `mc_margin_capped` for every agg dict (All
    Patterns + every other column).

    Thin wrapper around `enrich_aggs_with_mc` that flattens the nested shape
    (each distinct agg object is enriched once — columns["All Patterns"] is
    normally the same object as all_patterns_agg).
    """
    flat_aggs = []
    seen = set()
    for _label, all_patterns_agg, columns, _strategy in results:
        for agg in [all_patterns_agg, *columns.values()]:
            if id(agg) not in seen:
                seen.add(id(agg))
                flat_aggs.append(agg)
    enrich_aggs_with_mc(flat_aggs, balance, n_sims, target_dd=target_dd,
                        point_value=point_value, margin=margin,
                        trades_per_sim=trades_per_sim)


# ======================================================================
# Grid Search execution
# ======================================================================

def _run_grid_search(selected_strategy, search_group, search_set, selected_events,
                     condition_candidates, condition_event, sidebar_config,
                     effective_candidates=None, use_original_engine=False):
    """Run backtests for all candidate runs on every pattern combo.

    Returns (results, combo_results_list):
      results            — list of (label, all_patterns_agg, columns, strategy)
                           where columns is the build_pattern_columns OrderedDict
                           (All Patterns, All Bullish, All Bearish, Global,
                           one per user row; columns["All Patterns"] is
                           all_patterns_agg, which also carries correlation)
                           and strategy is the candidate's full strategy dict
                           (deep copy, variant indicator offsets applied).
      combo_results_list — per candidate {combo_key: [stats dicts]} (aligned
                           with results) so the pattern columns can be
                           re-aggregated when the user rows change.
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
        return [], []

    # Every candidate runs on every pattern combo (all 86); the fixed
    # columns and the user rows are aggregated from the per-combo results.
    user_selections = st.session_state.get("gs_selections", [])
    drm_bullish = st.session_state.get("drm_bullish")
    drm_bearish = st.session_state.get("drm_bearish")

    # Build period slices per combo
    combo_slices = OrderedDict()  # combo_key -> [(df_slice, ps, pe), ...]
    global_combo_keys = all_combos()

    for combo_key in global_combo_keys:
        pattern_type, primary, secondary = combo_key
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
                    show_pc=False,
                    show_psar=False)
                if not df_slice.empty:
                    slices.append((df_slice, ps, pe))
        combo_slices[combo_key] = slices

    if not any(combo_slices[ck] for ck in global_combo_keys):
        st.warning("No valid DRM periods found for any pattern.")
        return [], []

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
        return [], []

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

    # Candidate strategies kept with each result (Shortlist / Save as strategy)
    result_strategies = [result_strategy_for_variant(strategy, vid, variant_groups)
                         for _label, strategy, vid in run_configs]

    if use_original_engine:
        raw = _run_grid_search_original_engine(
            run_configs, variant_combo_slices, global_combo_keys, result_strategies)
    else:
        raw = _run_grid_search_multiprocessing(
            run_configs, variant_combo_slices, global_combo_keys, result_strategies)

    results, combo_results_list = _assemble_results(
        raw, user_selections, global_combo_keys, ref_directions)

    # Enrich every agg dict with MC Avg Profit @ 5% avg max DD
    # Bootstraps each candidate's own trades (pnl_r + stop distance) in
    # dollars, contracts sized per trade and capped by margin; binary-searches
    # the risk % that yields 5% avg max DD. Parallelised across workers.
    mc_cfg = mc_settings()
    _enrich_mc_parallel(results, mc_cfg['balance'], GS_MC_N_SIMULATIONS, target_dd=5.0,
                        point_value=mc_cfg['point_value'], margin=mc_cfg['margin'],
                        trades_per_sim=mc_cfg['trades_per_sim'])

    # Diagnostic: if all candidates produced zero trades, tell the user
    if results and all(r[1].get('num_trades', 0) == 0 for r in results):
        st.info(
            f"All {len(results)} candidates ran successfully but produced 0 trades. "
            f"Check entry/exit conditions, DRM periods, or date range."
        )

    return results, combo_results_list


# ----------------------------------------------------------------------
# Pattern-column assembly (pure; shared by the run path and the
# re-aggregation path taken when the user rows change)
# ----------------------------------------------------------------------

def _selections_json(selections):
    """Stable JSON of the user pattern rows (cache comparison key)."""
    import json
    return json.dumps(selections or [], sort_keys=True, default=str)


def build_candidate_columns(combo_results, user_selections):
    """Fixed + user pattern columns for one candidate from its per-combo
    stats dicts ({combo_key: [stats dicts]})."""
    return build_pattern_columns(user_selections, [], combo_results,
                                 _aggregate_stats_dicts, _empty_agg)


def _assemble_results(raw, user_selections, global_combo_keys, ref_directions=None):
    """Turn engine output [(label, combo_results, strategy)] into
    (results, combo_results_list):
    results = [(label, all_patterns_agg, columns, strategy)].
    Correlation is computed over global_combo_keys and written onto
    all_patterns_agg (== columns["All Patterns"])."""
    results = []
    combo_results_list = []
    for label, combo_results, strategy in raw:
        columns = build_candidate_columns(combo_results, user_selections)
        all_patterns_agg = columns["All Patterns"]

        # Correlation with reference strategies
        if ref_directions is not None:
            corr_value = _compute_candidate_correlation(
                combo_results, global_combo_keys, ref_directions)
            all_patterns_agg['correlation'] = corr_value
            all_patterns_agg['abs_correlation'] = abs(corr_value) if corr_value is not None else None
        else:
            all_patterns_agg['correlation'] = None
            all_patterns_agg['abs_correlation'] = None

        results.append((label, all_patterns_agg, columns, strategy))
        combo_results_list.append(combo_results)
    return results, combo_results_list


def reaggregate_results(results, combo_results_list, user_selections):
    """Rebuild every candidate's pattern columns from its cached per-combo
    stats after the user rows changed — without re-running the search.

    The three fixed "All …" aggs do not depend on the user rows, so the
    cached objects are kept (they already carry correlation + MC values).
    Global and the user columns are rebuilt. Returns (new_results,
    changed_aggs) where changed_aggs are the freshly built aggs that still
    need MC enrichment."""
    new_results = []
    changed_aggs = []
    for (label, all_patterns_agg, old_columns, strategy), combo_results in zip(results, combo_results_list):
        columns = build_candidate_columns(combo_results, user_selections)
        for fixed_label in FIXED_SELECTIONS:
            if fixed_label in old_columns:
                columns[fixed_label] = old_columns[fixed_label]
        all_patterns_agg = columns["All Patterns"]
        for col_label, agg in columns.items():
            if col_label not in FIXED_SELECTIONS:
                changed_aggs.append(agg)
        new_results.append((label, all_patterns_agg, columns, strategy))
    return new_results, changed_aggs


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
                                     result_strategies=None):
    """Run grid search using the ORIGINAL (non-numpy) engine, single-process.
    Used for debugging to compare results with the numpy engine.
    Returns [(label, combo_results, strategy)] — see _assemble_results;
    strategy is result_strategies[idx] (default: a deep copy of the run config's).
    """
    from strategies.first_strategy import execute_custom_strategy

    def _extract_stats(stats_df):
        return {
            'win_pnl': float(stats_df.loc['Winning trades P&L (R)', 'value']),
            'lose_pnl': float(stats_df.loc['Losing trades P&L (R)', 'value']),
            'trade_pnls_r': list(stats_df.attrs.get('trade_pnls_r', [])),
            'trade_holding_periods': list(stats_df.attrs.get('trade_holding_periods', [])),
            'trade_r_distances': list(stats_df.attrs.get('trade_r_distances', [])),
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

        results.append((label, combo_results,
                        result_strategies[idx] if result_strategies is not None
                        else copy.deepcopy(strategy)))
        progress.progress((idx + 1) / n_candidates,
                          text=f"Completed {idx + 1}/{n_candidates} candidates")

    progress.empty()
    return results


def _run_grid_search_multiprocessing(run_configs, variant_combo_slices, global_combo_keys,
                                     result_strategies=None):
    """Run grid search using multiprocessing Pool.
    Each worker process handles one candidate across all combo/period slices.
    Returns [(label, combo_results, strategy)] — see _assemble_results;
    strategy is result_strategies[idx] (default: a deep copy of the run config's).
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
        strategy = (result_strategies[idx] if result_strategies is not None
                    else copy.deepcopy(run_configs[idx][1]))
        results.append((label, combo_results, strategy))

    progress.empty()
    return results


# ======================================================================
# Results display with filtering, sorting, pagination
# ======================================================================

def _display_results(results, strategy_name, thresholds, sort_key, sort_descending,
                     scope="All Patterns", search_group="", group_set=""):
    """Display filtered, sorted, paginated results with the fixed pattern
    columns (All Patterns | All Bullish | All Bearish | Global) plus one per
    user row.

    `scope` picks whose metrics drive filtering, sorting and the results table:
    one of FIXED_COLUMNS or a user-row label.
    """

    if scope != "All Patterns" and not any(scope in (columns or {})
                                           for _, _, columns, _ in results):
        st.info(f"'{scope}' is not in the last calculated results — showing All Patterns. "
                "Click Calculate to refresh.")
        scope = "All Patterns"

    # Filter + sort on the scoped metrics (None sort values go to bottom)
    filtered = filter_and_sort_results(results, thresholds, sort_key,
                                       sort_descending, scope)

    st.subheader(f"Results — {strategy_name}")
    st.caption(f"{len(filtered)} of {len(results)} candidates pass filters")

    if not filtered:
        st.warning("No candidates pass the threshold filters.")
        return

    # Build scoped results table (rows = candidates, columns = metrics)
    rows = []
    for label, metric_agg, _all_patterns_agg, _columns, _strategy in filtered:
        rows.append({
            "Candidate": label,
            "Trades": metric_agg["num_trades"],
            "Win%": f"{metric_agg['win_pct']:.0f}%",
            "Lose%": f"{metric_agg['lose_pct']:.0f}%",
            "Avg Profit": f"{metric_agg['avg_win_pnl']:.2f}R",
            "Avg Loss": f"{metric_agg['avg_lose_pnl']:.2f}R",
            "Total P&L": f"{metric_agg['total_pnl']:.2f}R",
            "EV": f"{metric_agg['expected_value']:.2f}R",
            "Target%": f"{metric_agg['target_exit_pct']:.0f}%",
            "Static%": f"{metric_agg['static_exit_pct']:.0f}%",
            "Dynamic%": f"{metric_agg['dynamic_exit_pct']:.0f}%",
            "RR": f"{metric_agg.get('rr_ratio', 0):.2f}",
            "MC": format_mc_value(metric_agg),
            "Corr": f"{metric_agg['correlation']:.0f}%" if metric_agg.get('correlation') is not None else "\u2014",
            "Hold": f"{metric_agg.get('avg_holding_period', 0):.1f}",
        })

    df_results = pd.DataFrame(rows)

    # MC context caption
    st.caption(mc_caption())

    # Copy to clipboard (all filtered results, TSV)
    tsv_data = df_results.to_csv(sep='\t', index=False, header=False)
    _copy_to_clipboard(tsv_data, key="gs_copy_results")

    st.caption(f"{len(filtered)} results ({scope} Performance)")
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

    shortlist = st.session_state.setdefault("gs_shortlist", [])
    for idx in range(detail_start, detail_end):
        label, _metric_agg, all_patterns_agg, columns, strategy = filtered[idx]
        with st.expander(f"**{label}**", expanded=False):
            identity = {"label": label, "base_strategy": strategy_name,
                        "search_group": search_group, "group_set": group_set}
            if shortlist_has(shortlist, identity):
                st.caption("Already in Shortlist")
            elif st.button("➕ Shortlist", key=f"gs_short_add_{idx}"):
                entry = make_shortlist_entry(
                    label, all_patterns_agg, columns, strategy,
                    base_strategy=strategy_name, search_group=search_group,
                    group_set=group_set, scope=scope, mc_settings=mc_settings())
                st.session_state["gs_shortlist"] = shortlist_add(shortlist, entry)
                st.rerun()
            table = _build_metrics_table(detail_table_columns(columns, scope))
            st.table(table)
            st.caption(PATTERN_COLUMNS_CAPTION)
            _copy_to_clipboard(
                table.to_csv(sep='\t', header=False, index=False),
                key=f"gs_sel_detail_{idx}")


def _render_shortlist():
    """Shortlist section: candidates kept from any Detailed Performance run
    (snapshots; survive new searches and cache clears until the browser
    session ends)."""
    from strategies.strategy_manager import save_strategies_to_file

    shortlist = st.session_state.setdefault("gs_shortlist", [])
    component_labels = dict(SEARCH_COMPONENTS)

    st.markdown("---")
    st.subheader(f"Shortlist ({len(shortlist)})")
    st.caption("Kept until the browser session ends. Entries keep the numbers they had when added.")

    if not shortlist:
        st.caption("Use ➕ Shortlist on a Detailed Performance candidate to add it here.")
        return

    for entry in shortlist:
        eid = entry["id"]
        component = component_labels.get(entry["search_group"], entry["search_group"])
        title = f"**{entry['label']}** — {entry['base_strategy']} · {component} · {entry['group_set']}"
        with st.expander(title, expanded=False):
            mc = entry.get("mc_settings") or {}
            st.caption(
                f"Added {entry['added_at'].replace('T', ' ')} · Scope: {entry['scope']} · "
                f"{mc.get('instrument', '?')} / ${mc.get('balance', 0):,.0f} / "
                f"{mc.get('trades_per_sim', '?')} trades per sim")
            # Same column order as Detailed Performance showed under the entry's scope
            table = _build_metrics_table(detail_table_columns(entry["columns"], entry["scope"]))
            st.table(table)

            name = st.text_input(
                "Strategy name", value=f"{entry['base_strategy']} — {entry['label']}",
                key=f"gs_short_name_{eid}")
            b1, b2, b3 = st.columns(3)
            with b1:
                if st.button("➖ Remove", key=f"gs_short_rm_{eid}"):
                    st.session_state["gs_shortlist"] = shortlist_remove(shortlist, eid)
                    st.rerun()
            with b2:
                _copy_to_clipboard(table.to_csv(sep='\t', header=False, index=False),
                                   key=f"gs_short_copy_{eid}")
            with b3:
                save_clicked = st.button("💾 Save as strategy", key=f"gs_short_save_{eid}")
            if save_clicked:
                saved = st.session_state.setdefault("saved_strategies", [])
                existing = [s.get("strategy_name") for s in saved]
                strategy, errors = shortlist_strategy_for_save(entry, name, existing)
                if strategy is None:
                    st.error("Cannot save strategy:\n\n" + "\n".join(f"- {e}" for e in errors))
                else:
                    saved.append(strategy)
                    save_strategies_to_file()
                    st.success(f"Saved strategy '{strategy['strategy_name']}'.")

    c1, c2 = st.columns([1, 4])
    with c1:
        confirm = st.checkbox("Confirm", key="gs_short_clear_confirm")
    with c2:
        if st.button("Clear Shortlist", key="gs_short_clear", disabled=not confirm):
            st.session_state["gs_shortlist"] = []
            st.session_state.pop("gs_short_clear_confirm", None)
            st.rerun()


def detail_table_columns(columns, scope="All Patterns"):
    """Column order for a Detailed Performance table: `▶ scope` first only
    when scope is not "All Patterns"; then the fixed columns in order
    (skipping the scoped one), then the user columns in their row order."""
    table_data = OrderedDict()
    marked = scope != "All Patterns" and scope in columns
    if marked:
        table_data[f"▶ {scope}"] = columns[scope]
    for col_label in FIXED_COLUMNS:
        if col_label in columns and not (marked and col_label == scope):
            table_data[col_label] = columns[col_label]
    for col_label, agg in columns.items():
        if col_label not in FIXED_COLUMNS and not (marked and col_label == scope):
            table_data[col_label] = agg
    return table_data


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

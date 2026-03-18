"""
Grid Search tab — systematic strategy development via iterative component optimization.
"""
import streamlit as st
import streamlit.components.v1 as _components
import pandas as pd
import copy

from data.loader import parse_drm_periods
from data.helpers import (PRIMARY_SECONDARY_MAP, PRIMARY_LIST,
                          ALL_UNIQUE_SECONDARIES, expand_selection, selection_label)
from indicators.calculate_indicators import slice_for_graph, migrate_indicator_settings
from strategies.first_strategy import execute_custom_strategy
from strategies.strategy_manager import save_strategies_to_file
from ui.charting_tab import _aggregate_stats, _get_or_calculate
from ui.performance_tab import (_DEFAULT_INDICATOR_PARAMS, SELECTION_MODES,
                                 _build_metrics_table, _copy_to_clipboard, _empty_agg)
from ui.grid_search_helpers import (build_skeleton_strategy, build_candidate_strategy,
                                     candidate_label, collect_gs_indicator_settings)
from config.constants import (GROUP_NAMES, EVENT_TYPES, STOP_EVENT_TYPES,
                               CONDITION_OPERATORS, get_group_elements,
                               R_PROFIT_LOSS_ELEMENTS, ATR_TARGET_ELEMENTS)


STEP_NAMES = {
    1: "Step 1: Develop Trigger",
    2: "Step 2: Develop Condition",
    3: "Step 3: Develop Dynamic Stop",
    4: "Step 4: Develop Static Stop",
    5: "Step 5: Develop Target",
}

STEP_COMPONENT_KEY = {
    1: "trigger",
    2: "conditions",
    3: "dynamic_stop",
    4: "static_stop",
    5: "target",
}


# ======================================================================
# Main entry point
# ======================================================================

def render_grid_search_tab(sidebar_config):
    """Render the Grid Search tab."""

    st.subheader("Grid Search — Strategy Development")

    # ── Direction ────────────────────────────────────────────────────
    direction = st.radio("Direction", ["Long", "Short"],
                         horizontal=True, key="gs_direction_radio")
    if st.session_state.get("gs_direction") != direction:
        st.session_state["gs_direction"] = direction
        # Reset everything when direction changes
        _reset_all_steps()

    # ── Data checks ─────────────────────────────────────────────────
    show_1h = sidebar_config["analysis_mode"] == "1H"
    df_key = "df_1h" if show_1h else "df_15m"
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

    # ── Pattern selection ───────────────────────────────────────────
    _render_pattern_selection()

    st.markdown("---")

    # ── Indicator settings ──────────────────────────────────────────
    with st.expander("Indicator Settings", expanded=False):
        _render_indicator_settings()

    st.markdown("---")

    # ── Threshold filters ───────────────────────────────────────────
    st.markdown("**Global Performance Filters**")
    fc1, fc2, fc3, fc4, fc5 = st.columns(5)
    with fc1:
        min_trades = st.number_input("Min Trades", value=0, step=1, key="gs_thresh_min_trades")
    with fc2:
        min_ev = st.number_input("Min EV (R)", value=-999.0, step=0.01,
                                 format="%.2f", key="gs_thresh_min_ev")
    with fc3:
        min_sharpe = st.number_input("Min Sharpe", value=-999.0, step=0.01,
                                     format="%.2f", key="gs_thresh_min_sharpe")
    with fc4:
        max_dd = st.number_input("Max Drawdown (R)", value=999.0, step=0.1,
                                 format="%.2f", key="gs_thresh_max_dd")
    with fc5:
        min_sqn = st.number_input("Min SQN", value=-999.0, step=0.01,
                                  format="%.2f", key="gs_thresh_min_sqn")

    thresholds = {
        "min_trades": min_trades,
        "min_ev": min_ev,
        "min_sharpe": min_sharpe,
        "max_dd": max_dd,
        "min_sqn": min_sqn,
    }

    st.markdown("---")

    # ── Steps ───────────────────────────────────────────────────────
    current_step = st.session_state.get("gs_current_step", 1)

    for step in range(1, 6):
        _render_step(step, current_step, sidebar_config, show_1h, df_key, thresholds)

    # ── Save final strategy ─────────────────────────────────────────
    if current_step > 5:
        st.markdown("---")
        st.success("All components locked! Strategy is complete.")
        save_name = st.text_input("Strategy Name", key="gs_save_name")
        if st.button("Save Strategy", key="gs_save_btn", type="primary"):
            strategy = copy.deepcopy(st.session_state["gs_training_strategy"])
            strategy["strategy_name"] = save_name or f"GridSearch_{len(st.session_state['saved_strategies']) + 1}"
            st.session_state["saved_strategies"].append(strategy)
            save_strategies_to_file()
            st.success(f"Strategy **{strategy['strategy_name']}** saved!")


# ======================================================================
# Pattern selection UI (reuses Performance tab pattern)
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
# Indicator settings UI
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
# Candidate definition UI
# ======================================================================

def _render_trigger_candidate(idx, prefix):
    """Render a single trigger candidate row. Returns config dict or None."""
    ema_count = len(st.session_state.get("gs_ema_periods", []))

    c_grp, c_e1, c_ev, c_cmp, c_e2 = st.columns([2, 2, 2, 1.5, 2])

    with c_grp:
        group = st.selectbox("Group", GROUP_NAMES, key=f"{prefix}_grp")
    with c_e1:
        elements = get_group_elements(group, ema_count)
        element1 = st.selectbox("Element 1", elements, key=f"{prefix}_e1")
    with c_ev:
        event = st.selectbox("Event", EVENT_TYPES, key=f"{prefix}_ev")
    with c_cmp:
        compare = st.radio("Compare", ["Indicator", "Fixed Value"],
                           key=f"{prefix}_cmp", horizontal=True)
    with c_e2:
        if compare == "Indicator":
            all_elements = []
            for g in GROUP_NAMES:
                all_elements.extend(get_group_elements(g, ema_count))
            element2 = st.selectbox("Element 2", all_elements, key=f"{prefix}_e2")
            value = None
        else:
            element2 = None
            value = st.number_input("Value", value=0.0, step=0.01,
                                    format="%.4f", key=f"{prefix}_val")

    return {
        "group": group,
        "element1": element1,
        "event": event,
        "compare_type": compare,
        "element2": element2,
        "value": value,
    }


def _render_condition_candidate(idx, prefix):
    """Render a single condition candidate row. Returns config dict or None."""
    ema_count = len(st.session_state.get("gs_ema_periods", []))

    c_grp, c_e1, c_op, c_cmp, c_e2 = st.columns([2, 2, 1.5, 1.5, 2])

    with c_grp:
        group = st.selectbox("Group", GROUP_NAMES, key=f"{prefix}_grp")
    with c_e1:
        elements = get_group_elements(group, ema_count)
        element1 = st.selectbox("Element 1", elements, key=f"{prefix}_e1")
    with c_op:
        operator = st.selectbox("Operator", CONDITION_OPERATORS, key=f"{prefix}_op")
    with c_cmp:
        compare = st.radio("Compare", ["Indicator", "Fixed Value"],
                           key=f"{prefix}_cmp", horizontal=True)
    with c_e2:
        if compare == "Indicator":
            all_elements = []
            for g in GROUP_NAMES:
                all_elements.extend(get_group_elements(g, ema_count))
            element2 = st.selectbox("Element 2", all_elements, key=f"{prefix}_e2")
            value = None
        else:
            element2 = None
            value = st.number_input("Value", value=0.0, step=0.01,
                                    format="%.4f", key=f"{prefix}_val")

    return {
        "group": group,
        "element1": element1,
        "operator": operator,
        "compare_type": compare,
        "element2": element2,
        "value": value,
    }


def _render_exit_trigger_candidate(idx, prefix, include_r=False, include_atr_target=False):
    """Render a single exit trigger candidate row (dynamic stop or target)."""
    ema_count = len(st.session_state.get("gs_ema_periods", []))

    # Build element list — include R Profit/Loss and ATR Target for targets
    extra_elements = []
    if include_r:
        extra_elements.extend(R_PROFIT_LOSS_ELEMENTS)
    if include_atr_target:
        extra_elements.extend(ATR_TARGET_ELEMENTS)

    c_grp, c_e1, c_ev, c_cmp, c_e2 = st.columns([2, 2, 2, 1.5, 2])

    with c_grp:
        group = st.selectbox("Group", GROUP_NAMES, key=f"{prefix}_grp")
    with c_e1:
        elements = get_group_elements(group, ema_count) + extra_elements
        element1 = st.selectbox("Element 1", elements, key=f"{prefix}_e1")
    with c_ev:
        event = st.selectbox("Event", STOP_EVENT_TYPES, key=f"{prefix}_ev")

    is_r_element = element1 in R_PROFIT_LOSS_ELEMENTS
    is_atr_target = element1 in ATR_TARGET_ELEMENTS

    with c_cmp:
        if is_r_element:
            compare = "Fixed Value"
            st.radio("Compare", ["Fixed Value"], key=f"{prefix}_cmp", disabled=True)
        elif is_atr_target:
            compare = "Indicator"
            st.radio("Compare", ["Indicator"], key=f"{prefix}_cmp_d", disabled=True)
        else:
            compare = st.radio("Compare", ["Indicator", "Fixed Value"],
                               key=f"{prefix}_cmp", horizontal=True)

    with c_e2:
        if is_atr_target:
            atr_period = st.number_input("ATR Period", 1, 200, value=14,
                                         step=1, key=f"{prefix}_atr_p")
            atr_mult = st.number_input("ATR Multiplier", 0.1, 20.0, value=2.0,
                                       step=0.1, format="%.1f", key=f"{prefix}_atr_m")
            return {
                "group": group,
                "element1": element1,
                "event": event,
                "compare_type": "Indicator",
                "element2": None,
                "value": None,
                "atr_period": atr_period,
                "atr_multiplier": atr_mult,
            }
        elif compare == "Indicator":
            all_elements = []
            for g in GROUP_NAMES:
                all_elements.extend(get_group_elements(g, ema_count))
            element2 = st.selectbox("Element 2", all_elements, key=f"{prefix}_e2")
            value = None
        else:
            element2 = None
            value = st.number_input("Value", value=0.0, step=0.01,
                                    format="%.4f", key=f"{prefix}_val")

    return {
        "group": group,
        "element1": element1,
        "event": event,
        "compare_type": compare,
        "element2": element2,
        "value": value,
    }


def _render_static_stop_candidate(idx, prefix):
    """Render a single static / initial stop candidate row."""
    direction = st.session_state.get("gs_direction", "Long")

    c_type, c_detail = st.columns([1.5, 3])
    with c_type:
        stop_type = st.selectbox("Stop Type", ["ATR", "Indicator"], key=f"{prefix}_stype")

    with c_detail:
        if stop_type == "ATR":
            ac1, ac2 = st.columns(2)
            with ac1:
                atr_period = st.number_input("ATR Period", 1, 200, value=14,
                                             step=1, key=f"{prefix}_atr_p")
            with ac2:
                atr_mult = st.number_input("ATR Multiplier", 0.1, 20.0, value=2.0,
                                           step=0.1, format="%.1f", key=f"{prefix}_atr_m")
            return {
                "stop_type": "ATR",
                "atr_period": atr_period,
                "atr_multiplier": atr_mult,
            }
        else:
            ema_count = len(st.session_state.get("gs_ema_periods", []))
            all_elements = []
            for g in GROUP_NAMES:
                all_elements.extend(get_group_elements(g, ema_count))
            element2 = st.selectbox("Stop Element", all_elements, key=f"{prefix}_e2")
            default_event = "Cross Below" if direction == "Long" else "Cross Above"
            event_idx = STOP_EVENT_TYPES.index(default_event) if default_event in STOP_EVENT_TYPES else 0
            event = st.selectbox("Event", STOP_EVENT_TYPES, index=event_idx,
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
# Step rendering
# ======================================================================

def _render_step(step, current_step, sidebar_config, show_1h, df_key, thresholds):
    """Render a single step section."""
    locked = st.session_state.get("gs_locked_components", {})
    component_key = STEP_COMPONENT_KEY[step]
    is_locked = locked.get(component_key) is not None
    is_active = step <= current_step
    step_label = STEP_NAMES[step]

    if is_locked:
        step_label += "  ✓"

    with st.expander(step_label, expanded=(step == current_step)):
        if not is_active and not is_locked:
            st.info(f"Complete Step {step - 1} first.")
            return

        if is_locked:
            locked_cfg = locked[component_key]
            st.success(f"Locked: **{candidate_label(locked_cfg, step)}**")
            if st.button("Unlock", key=f"gs_unlock_{step}"):
                _unlock_from_step(step)
                st.rerun()
            # Still show cached results if available
            cached = st.session_state.get(f"gs_step{step}_results")
            if cached:
                _display_step_results(cached, thresholds, step)
            return

        # ── Candidate definition ────────────────────────────────
        candidates_key = f"gs_step{step}_count"
        if candidates_key not in st.session_state:
            st.session_state[candidates_key] = 1

        n_candidates = st.session_state[candidates_key]
        candidate_configs = []

        for i in range(n_candidates):
            prefix = f"gs_s{step}_{i}"
            st.markdown(f"**Candidate {i + 1}**")

            if step == 1:
                cfg = _render_trigger_candidate(i, prefix)
            elif step == 2:
                cfg = _render_condition_candidate(i, prefix)
            elif step == 3:
                cfg = _render_exit_trigger_candidate(i, prefix,
                                                     include_r=True, include_atr_target=False)
            elif step == 4:
                cfg = _render_static_stop_candidate(i, prefix)
            elif step == 5:
                cfg = _render_exit_trigger_candidate(i, prefix,
                                                     include_r=True, include_atr_target=True)
            candidate_configs.append(cfg)

            if n_candidates > 1:
                if st.button("Remove", key=f"gs_rm_s{step}_{i}"):
                    st.session_state[candidates_key] -= 1
                    st.rerun()

            st.markdown("---")

        if st.button("+ Add Candidate", key=f"gs_add_s{step}"):
            st.session_state[candidates_key] += 1
            st.rerun()

        # ── Calculate ───────────────────────────────────────────
        if st.button("Calculate", key=f"gs_calc_{step}", type="primary"):
            results = _run_step_search(
                step, candidate_configs, sidebar_config, show_1h, df_key)
            st.session_state[f"gs_step{step}_results"] = results
            st.session_state[f"gs_step{step}_configs"] = candidate_configs

        # ── Results ─────────────────────────────────────────────
        cached = st.session_state.get(f"gs_step{step}_results")
        if cached:
            _display_step_results(cached, thresholds, step)

            # Lock selection
            cached_configs = st.session_state.get(f"gs_step{step}_configs", candidate_configs)
            candidate_names = list(cached.keys())
            if candidate_names:
                lock_choice = st.selectbox("Select candidate to lock",
                                           candidate_names, key=f"gs_lock_sel_{step}")
                if st.button("Lock and Proceed", key=f"gs_lock_btn_{step}", type="primary"):
                    lock_idx = candidate_names.index(lock_choice)
                    locked_cfg = cached_configs[lock_idx]
                    _lock_step(step, locked_cfg)
                    st.rerun()


# ======================================================================
# Search execution
# ======================================================================

def _run_step_search(step, candidate_configs, sidebar_config, show_1h, df_key):
    """Run backtests for all candidates at the given step. Returns {label: agg_dict}."""

    # Build indicator params
    indicator_settings = collect_gs_indicator_settings(st.session_state)

    # Build or update training strategy
    direction = st.session_state.get("gs_direction", "Long")
    training_strategy = st.session_state.get("gs_training_strategy")
    if training_strategy is None or training_strategy["direction"] != direction:
        training_strategy = build_skeleton_strategy(direction, indicator_settings)
        st.session_state["gs_training_strategy"] = training_strategy
    # Always update indicator settings
    training_strategy["indicator_settings"] = dict(indicator_settings)

    # Calculate indicators
    g_start = sidebar_config.get("global_start_date")
    g_end = sidebar_config.get("global_end_date")

    df_full = _get_or_calculate(
        df_key, f"_gs_{df_key}_features", f"_gs_{df_key}_params",
        indicator_settings, global_start_date=g_start, global_end_date=g_end)

    if df_full.empty:
        st.warning("No data available for the selected date range.")
        return {}

    # Expand pattern selections
    selections = st.session_state.get("gs_selections", [])
    all_combos = []
    seen = set()
    for sel in selections:
        for combo in expand_selection(sel):
            if combo not in seen:
                seen.add(combo)
                all_combos.append(combo)

    if not all_combos:
        st.warning("No pattern combos selected.")
        return {}

    # Precompute period slices
    drm_bullish = st.session_state.get("drm_bullish")
    drm_bearish = st.session_state.get("drm_bearish")

    period_slices = []
    for pattern_type, primary, secondary in all_combos:
        drm_df = drm_bullish if pattern_type == "Bullish" else drm_bearish
        if drm_df is None:
            continue
        periods = parse_drm_periods(drm_df, pattern_type, primary, secondary)
        for start_dt, end_dt in periods:
            df_slice, ps, pe = slice_for_graph(
                df=df_full, start_date=start_dt, end_date=end_dt,
                show_ichimoku=True, show_bb=True, show_kc=True,
                show_donchian=True, show_psar=True)
            if not df_slice.empty:
                period_slices.append((df_slice, ps, pe))

    if not period_slices:
        st.warning("No valid DRM periods found for selected patterns.")
        return {}

    # Run each candidate
    total = len(candidate_configs) * len(period_slices)
    progress = st.progress(0, text="Running grid search...")
    done = 0

    results = {}
    for i, cfg in enumerate(candidate_configs):
        strategy = build_candidate_strategy(training_strategy, step, cfg)
        all_stats = []

        for df_slice, ps, pe in period_slices:
            try:
                _, stats = execute_custom_strategy(df_slice.copy(), strategy, ps, pe)
                if stats is not None:
                    all_stats.append(stats)
            except Exception:
                pass  # Skip invalid configs silently
            done += 1
            progress.progress(done / total, text=f"Candidate {i+1}/{len(candidate_configs)}")

        label = candidate_label(cfg, step)
        # Deduplicate labels
        base_label = label
        counter = 2
        while label in results:
            label = f"{base_label} ({counter})"
            counter += 1

        if all_stats:
            results[label] = _aggregate_stats(all_stats)
        else:
            results[label] = _empty_agg()

    progress.empty()
    return results


# ======================================================================
# Results display
# ======================================================================

def _display_step_results(results, thresholds, step):
    """Display filtered + sorted results table for a step."""
    # Filter by thresholds
    filtered = {}
    for label, agg in results.items():
        if (agg["num_trades"] >= thresholds["min_trades"]
                and agg["expected_value"] >= thresholds["min_ev"]
                and agg["sharpe_ratio"] >= thresholds["min_sharpe"]
                and agg["max_drawdown"] <= thresholds["max_dd"]
                and agg["sqn"] >= thresholds["min_sqn"]):
            filtered[label] = agg

    if not filtered:
        st.warning("No candidates pass the threshold filters.")
        return

    # Sort by EV descending
    sorted_results = dict(sorted(filtered.items(),
                                 key=lambda x: x[1]["expected_value"], reverse=True))

    table = _build_metrics_table(sorted_results)
    st.table(table)
    _copy_to_clipboard(
        table.to_csv(sep='\t', header=False, index=False),
        key=f"gs_copy_s{step}")


# ======================================================================
# Lock / unlock / reset
# ======================================================================

def _lock_step(step, candidate_config):
    """Lock a candidate at the given step and advance."""
    component_key = STEP_COMPONENT_KEY[step]
    locked = st.session_state.get("gs_locked_components", {})
    locked[component_key] = candidate_config
    st.session_state["gs_locked_components"] = locked

    # Update training strategy
    indicator_settings = collect_gs_indicator_settings(st.session_state)
    direction = st.session_state.get("gs_direction", "Long")
    training = st.session_state.get("gs_training_strategy")
    if training is None:
        training = build_skeleton_strategy(direction, indicator_settings)

    training = build_candidate_strategy(training, step, candidate_config)
    st.session_state["gs_training_strategy"] = training
    st.session_state["gs_current_step"] = step + 1


def _unlock_from_step(step):
    """Unlock the given step and all downstream steps."""
    locked = st.session_state.get("gs_locked_components", {})
    for s in range(step, 6):
        component_key = STEP_COMPONENT_KEY[s]
        locked[component_key] = None
        st.session_state.pop(f"gs_step{s}_results", None)
        st.session_state.pop(f"gs_step{s}_configs", None)
    st.session_state["gs_locked_components"] = locked
    st.session_state["gs_current_step"] = step

    # Rebuild training strategy from remaining locks
    indicator_settings = collect_gs_indicator_settings(st.session_state)
    direction = st.session_state.get("gs_direction", "Long")
    training = build_skeleton_strategy(direction, indicator_settings)
    for s in range(1, step):
        comp_key = STEP_COMPONENT_KEY[s]
        cfg = locked.get(comp_key)
        if cfg is not None:
            training = build_candidate_strategy(training, s, cfg)
    st.session_state["gs_training_strategy"] = training


def _reset_all_steps():
    """Reset all grid search state (e.g. when direction changes)."""
    st.session_state["gs_current_step"] = 1
    st.session_state["gs_training_strategy"] = None
    st.session_state["gs_locked_components"] = {
        "trigger": None,
        "conditions": None,
        "dynamic_stop": None,
        "static_stop": None,
        "target": None,
    }
    for s in range(1, 6):
        st.session_state.pop(f"gs_step{s}_results", None)
        st.session_state.pop(f"gs_step{s}_configs", None)
        st.session_state.pop(f"gs_step{s}_count", None)

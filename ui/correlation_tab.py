"""
Correlation Analysis tab — compare multiple strategies' trade overlap
across DRM periods to identify correlated and uncorrelated strategies.
"""
import json
import streamlit as st
import pandas as pd

from data.loader import parse_drm_periods
from data.helpers import (
    PRIMARY_SECONDARY_MAP, PRIMARY_LIST, ALL_UNIQUE_SECONDARIES,
    expand_selection,
)
from indicators.calculate_indicators import (
    slice_for_graph, migrate_indicator_settings, strategy_indicator_flags,
)
from strategies.first_strategy import execute_custom_strategy
from ui.charting_tab import _get_or_calculate

SELECTION_MODES = [
    "All Patterns",
    "All Bullish",
    "All Bearish",
    "Specified Primary",
    "Specified Secondary",
    "Secondary Across Primaries",
]

# Default indicator params (same as other tabs)
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
    'lr_period': 100, 'lr_multiplier': 2.0,
}


def render_correlation_tab(sidebar_config):
    """Render the Correlation Analysis tab."""

    st.subheader("Correlation Analysis")

    with st.expander("How correlation is calculated", expanded=False):
        st.markdown("""
**Directional Pearson Correlation** measures whether two strategies win and lose
at the same times across DRM periods.

For each strategy, every DRM period is scored as:
- **+1** if the strategy had a net profit (winning period)
- **-1** if the strategy had a net loss (losing period)
- **0** if the strategy did not trade in that period

The Pearson correlation of these directional signals is computed for each pair:

| Correlation | Meaning |
|---|---|
| **+100%** | Strategies win and lose in the exact same periods (redundant) |
| **0%** | No relationship between outcomes |
| **-100%** | One wins when the other loses (ideal hedge) |
        """)

    # --------------------------------------------------
    # Prerequisites
    # --------------------------------------------------
    saved = st.session_state.get('saved_strategies', [])
    if not saved:
        st.info("No saved strategies. Create some in the Strategy Builder tab.")
        return

    df_key = 'df_ohlc'
    if df_key not in st.session_state:
        st.info("Please upload OHLC data in the Charting tab first.")
        return

    drm_bullish = st.session_state.get('drm_bullish')
    drm_bearish = st.session_state.get('drm_bearish')
    if drm_bullish is None and drm_bearish is None:
        st.info("Please upload a DRM file in the Charting tab first.")
        return

    if not sidebar_config.get('date_range_applied', False):
        st.info("Please set a date range in the sidebar and click **Apply Training Set** to proceed.")
        return

    date_start = sidebar_config.get('global_start_date')
    date_end = sidebar_config.get('global_end_date')

    # --------------------------------------------------
    # Strategy multi-select
    # --------------------------------------------------
    st.markdown("**Select Strategies to Compare**")
    strategy_names = [s.get('strategy_name', f'Strategy_{i+1}') for i, s in enumerate(saved)]

    selected_indices = st.multiselect(
        "Strategies",
        options=range(len(strategy_names)),
        format_func=lambda x: strategy_names[x],
        default=list(range(min(len(strategy_names), 4))),
        key="corr_strategy_select",
    )

    if len(selected_indices) < 2:
        st.warning("Select at least 2 strategies to compare.")
        return

    # --------------------------------------------------
    # Pattern Selection UI
    # --------------------------------------------------
    st.markdown("---")
    st.markdown("**Pattern Selections**")

    if 'corr_selections' not in st.session_state:
        st.session_state['corr_selections'] = [{
            "mode": "All Patterns",
            "pattern_type": "Bullish",
            "primary": None,
            "secondary": None,
        }]

    selections = st.session_state['corr_selections']
    gen = st.session_state.get("_corr_sel_gen", 0)

    h_mode, h_ptype, h_primary, h_secondary, h_del = st.columns([2, 2, 2, 2, 0.5])
    h_mode.markdown("**Mode**")
    h_ptype.markdown("**Pattern Type**")
    h_primary.markdown("**Primary**")
    h_secondary.markdown("**Secondary**")

    rows_to_remove = []
    for idx, sel in enumerate(selections):
        c_mode, c_ptype, c_primary, c_secondary, c_delete = st.columns([2, 2, 2, 2, 0.5])
        kp = f"corr_sel_g{gen}_{idx}"

        with c_mode:
            current_mode = sel.get("mode", "All Patterns")
            mode = st.selectbox(
                "Mode", SELECTION_MODES,
                index=SELECTION_MODES.index(current_mode) if current_mode in SELECTION_MODES else 0,
                key=f"{kp}_mode", label_visibility="collapsed",
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
                    "Pattern Type", ptype_options,
                    index=ptype_options.index(current_ptype) if current_ptype in ptype_options else 0,
                    key=f"{kp}_pattern_type", label_visibility="collapsed",
                )
                selections[idx]["pattern_type"] = ptype

        with c_primary:
            if show_primary:
                current_primary = sel.get("primary")
                primary_idx = PRIMARY_LIST.index(current_primary) if current_primary in PRIMARY_LIST else 0
                primary = st.selectbox(
                    "Primary", PRIMARY_LIST, index=primary_idx,
                    key=f"{kp}_primary", label_visibility="collapsed",
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
                        "Secondary", sec_options, index=sec_idx,
                        key=f"{kp}_secondary", label_visibility="collapsed",
                    )
                    selections[idx]["secondary"] = secondary

        with c_delete:
            if st.button("X", key=f"{kp}_remove", type="primary"):
                rows_to_remove.append(idx)

    if rows_to_remove:
        for idx in sorted(rows_to_remove, reverse=True):
            st.session_state['corr_selections'].pop(idx)
        if not st.session_state['corr_selections']:
            st.session_state['corr_selections'] = [{
                "mode": "All Patterns", "pattern_type": "Bullish",
                "primary": None, "secondary": None,
            }]
        st.session_state["_corr_sel_gen"] = gen + 1
        st.rerun()

    if st.button("+ Add Selection", key="corr_add_selection"):
        st.session_state['corr_selections'].append({
            "mode": "All Patterns", "pattern_type": "Bullish",
            "primary": None, "secondary": None,
        })
        st.rerun()

    # --------------------------------------------------
    # Expand pattern combos
    # --------------------------------------------------
    seen = set()
    all_combos = []
    for sel in selections:
        for combo in expand_selection(sel):
            if combo not in seen:
                seen.add(combo)
                all_combos.append(combo)

    st.markdown("---")

    # --------------------------------------------------
    # Calculate button
    # --------------------------------------------------
    calculate = st.button("Calculate Correlations", key="corr_calculate", type="primary")

    cache = st.session_state.get('_corr_cached_results')
    fingerprint = json.dumps(
        [selected_indices, [saved[i] for i in selected_indices]],
        sort_keys=True, default=str,
    )
    if cache and cache.get('_fingerprint') != fingerprint:
        st.session_state.pop('_corr_cached_results', None)
        cache = None

    if calculate:
        _run_correlation(
            df_key, saved, selected_indices, strategy_names,
            all_combos, drm_bullish, drm_bearish, date_start, date_end,
            fingerprint,
        )
    elif cache:
        _display_results(cache)


# ------------------------------------------------------------------
# Core computation
# ------------------------------------------------------------------

def _run_correlation(df_key, saved, selected_indices, strategy_names,
                     all_combos, drm_bullish, drm_bearish,
                     date_start, date_end, fingerprint):
    """Run each selected strategy on all DRM periods and compute trade overlap."""

    range_start = pd.Timestamp(date_start)
    range_end = pd.Timestamp(date_end) + pd.Timedelta(days=1)

    # Pre-parse all DRM periods
    period_keys = []   # (ptype, primary, secondary, start_dt, end_dt)
    for ptype, primary, secondary in all_combos:
        drm_df = drm_bullish if ptype == "Bullish" else drm_bearish
        if drm_df is None:
            continue
        periods = parse_drm_periods(drm_df, ptype, primary, secondary)
        for start_dt, end_dt in periods:
            if start_dt >= range_start and start_dt < range_end:
                period_keys.append((ptype, primary, secondary, start_dt, end_dt))

    if not period_keys:
        st.warning("No DRM periods found in the selected date range.")
        return

    n_strategies = len(selected_indices)
    progress = st.progress(0, text="Running strategies...")

    # For each strategy, record per-period P&L and which periods had trades
    # period_pnls[strategy_idx] = {period_idx: total_pnl_r, ...}
    # trade_sets[strategy_idx] = set of period indices that had trades
    import numpy as np

    period_pnls = {}
    trade_sets = {}

    for step, s_idx in enumerate(selected_indices):
        strategy = saved[s_idx]
        name = strategy_names[s_idx]
        progress.progress(step / n_strategies, text=f"Running {name}...")

        # Build indicator params for this strategy
        indicator_params = dict(_DEFAULT_INDICATOR_PARAMS)
        strat_settings = strategy.get('indicator_settings')
        if strat_settings:
            strat_settings = migrate_indicator_settings(strat_settings)
            indicator_params.update(strat_settings)

        df_full = _get_or_calculate(
            df_key, f'_corr_features_{s_idx}', f'_corr_params_{s_idx}',
            indicator_params,
            global_start_date=str(range_start.date()),
            global_end_date=str(range_end.date()),
        )

        ind_flags = strategy_indicator_flags(strategy)
        traded_periods = set()
        pnls = {}

        for p_idx, (ptype, primary, secondary, start_dt, end_dt) in enumerate(period_keys):
            try:
                df_slice, ps, pe = slice_for_graph(
                    df_full, start_dt, end_dt, **ind_flags,
                )
                if df_slice.empty:
                    continue
                result_df, stats = execute_custom_strategy(
                    df_slice, strategy, ps, pe,
                )
                num_trades = int(stats.loc["Number of trades", "value"])
                total_pnl = float(stats.loc["Total P&L (R)", "value"])
                pnls[p_idx] = total_pnl
                if num_trades > 0:
                    traded_periods.add(p_idx)
            except Exception:
                continue

        period_pnls[s_idx] = pnls
        trade_sets[s_idx] = traded_periods

    progress.empty()

    # --------------------------------------------------
    # Compute pairwise Pearson correlation on directional outcome
    # --------------------------------------------------
    # For each strategy, build a direction signal per period:
    #   +1 = net winning period (P&L > 0)
    #   -1 = net losing period (P&L < 0)
    #    0 = no trade in this period
    # Pearson correlation on these signals captures whether strategies
    # win and lose at the same times (directional overlap for portfolio risk).
    n_periods = len(period_keys)

    def _direction_array(pnls_dict):
        arr = np.zeros(n_periods)
        for i in range(n_periods):
            pnl = pnls_dict.get(i, 0.0)
            if pnl > 0:
                arr[i] = 1.0
            elif pnl < 0:
                arr[i] = -1.0
        return arr

    correlations = {}
    for a_idx in selected_indices:
        a_arr = _direction_array(period_pnls.get(a_idx, {}))
        row = {}
        for b_idx in selected_indices:
            if a_idx == b_idx:
                row[b_idx] = 100.0
                continue
            b_arr = _direction_array(period_pnls.get(b_idx, {}))

            # Pearson correlation: requires variance in both series
            if np.std(a_arr) > 0 and np.std(b_arr) > 0:
                corr = np.corrcoef(a_arr, b_arr)[0, 1]
                row[b_idx] = corr * 100  # scale to percentage
            elif np.std(a_arr) == 0 and np.std(b_arr) == 0:
                # Both flat (e.g. both no trades or all same outcome)
                row[b_idx] = 0.0
            else:
                # One has variance, other is flat
                row[b_idx] = 0.0
        correlations[a_idx] = row

    # Average correlation per strategy (excluding self)
    avg_corr = {}
    for a_idx in selected_indices:
        others = [correlations[a_idx][b] for b in selected_indices if b != a_idx]
        avg_corr[a_idx] = sum(others) / len(others) if others else 0.0

    # Sort by avg correlation descending
    sorted_indices = sorted(selected_indices, key=lambda x: avg_corr[x], reverse=True)

    cache = {
        '_fingerprint': fingerprint,
        'strategy_names': strategy_names,
        'selected_indices': selected_indices,
        'sorted_indices': sorted_indices,
        'correlations': correlations,
        'avg_corr': avg_corr,
        'trade_counts': {idx: len(trade_sets.get(idx, set())) for idx in selected_indices},
        'total_periods': len(period_keys),
    }
    st.session_state['_corr_cached_results'] = cache
    _display_results(cache)


# ------------------------------------------------------------------
# Display
# ------------------------------------------------------------------

def _display_results(cache):
    """Display correlation results with expandable detail per strategy."""
    strategy_names = cache['strategy_names']
    sorted_indices = cache['sorted_indices']
    correlations = cache['correlations']
    avg_corr = cache['avg_corr']
    trade_counts = cache['trade_counts']
    total_periods = cache['total_periods']

    st.caption(f"Total DRM periods analysed: **{total_periods}** | "
               f"Sorted: most correlated to least correlated")

    # Summary table
    summary_rows = []
    for s_idx in sorted_indices:
        summary_rows.append({
            "Strategy": strategy_names[s_idx],
            "Traded Periods": trade_counts[s_idx],
            "Avg Correlation": f"{avg_corr[s_idx]:.1f}%",
        })

    st.table(pd.DataFrame(summary_rows))

    # Expandable details per strategy
    st.markdown("---")
    st.subheader("Pairwise Detail")

    for s_idx in sorted_indices:
        name = strategy_names[s_idx]
        avg = avg_corr[s_idx]

        with st.expander(f"**{name}** (Avg Correlation: {avg:.1f}%)", expanded=False):
            detail_rows = []
            # Sort other strategies by correlation to this one (descending)
            others = [(b, correlations[s_idx][b])
                      for b in sorted_indices if b != s_idx]
            others.sort(key=lambda x: x[1], reverse=True)

            for b_idx, corr_pct in others:
                detail_rows.append({
                    "Compared To": strategy_names[b_idx],
                    "Correlation": f"{corr_pct:.1f}%",
                    "Shared Periods": int(
                        trade_counts[s_idx] * corr_pct / 100
                    ) if trade_counts[s_idx] else 0,
                })

            st.table(pd.DataFrame(detail_rows))

    # Copy to clipboard
    _copy_to_clipboard(_build_clipboard_text(cache), key="corr_copy")


def _build_clipboard_text(cache):
    """Build TSV text for clipboard (numbers only)."""
    strategy_names = cache['strategy_names']
    sorted_indices = cache['sorted_indices']
    correlations = cache['correlations']

    lines = []
    for a_idx in sorted_indices:
        vals = []
        for b_idx in sorted_indices:
            vals.append(f"{correlations[a_idx][b_idx]:.1f}")
        lines.append("\t".join(vals))
    return "\n".join(lines)


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

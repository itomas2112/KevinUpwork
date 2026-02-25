"""
Performance tab (Tab 3) - Aggregated backtest metrics across multiple Elliott Wave patterns.
Rows = metrics, Columns = pattern combinations.
"""
import streamlit as st
import pandas as pd

from data.loader import parse_drm_periods
from data.helpers import PRIMARY_SECONDARY_MAP
from indicators.calculate_indicators import calculate_indicators, slice_for_graph
from strategies.first_strategy import execute_custom_strategy
from ui.charting_tab import _aggregate_stats


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
    if 'df_15m' not in st.session_state:
        st.info("Please upload 15m OHLC data in the Charting tab first.")
        return

    drm_bullish = st.session_state.get('drm_bullish')
    drm_bearish = st.session_state.get('drm_bearish')
    if drm_bullish is None and drm_bearish is None:
        st.info("Please upload a DRM file in the Charting tab first.")
        return

    # --------------------------------------------------
    # Calculate indicators once on full DataFrame
    # --------------------------------------------------
    display_only_keys = {'rsi_upper_1', 'rsi_upper_2', 'rsi_lower_1', 'rsi_lower_2', 'cmb_lines'}
    indicator_params = {k: v for k, v in sidebar_config['params_15m'].items() if k not in display_only_keys}
    df_full = calculate_indicators(df=st.session_state['df_15m'], **indicator_params)

    # --------------------------------------------------
    # Determine which pattern combinations to iterate
    # --------------------------------------------------
    pattern_type = sidebar_config['pattern']            # "Bullish" or "Bearish"
    primary_choice = sidebar_config['primary_choice']   # None or e.g. "W.(1)"
    secondary_choice = sidebar_config['secondary_choice']  # None or e.g. "W.1 Impulse"

    combos = _get_pattern_combos(pattern_type, primary_choice, secondary_choice)

    if not combos:
        st.warning("No pattern combinations match the current sidebar filters.")
        return

    # Pick the DRM sheet for current pattern type
    drm_df = drm_bullish if pattern_type == 'Bullish' else drm_bearish
    if drm_df is None:
        st.warning(f"No DRM data found for '{pattern_type}' sheet.")
        return

    # --------------------------------------------------
    # Strategy market parameters
    # --------------------------------------------------
    strategy_with_params = selected_strategy.copy()
    strategy_with_params['tick_size'] = sidebar_config['tick_size']
    strategy_with_params['minimal_change'] = sidebar_config['minimal_change']

    # --------------------------------------------------
    # Run backtest for each pattern combination
    # --------------------------------------------------
    results = {}  # pattern_label -> aggregated dict

    progress_bar = st.progress(0)
    total = len(combos)

    for idx, (primary, secondary) in enumerate(combos):
        label = f"{primary} → {secondary}"

        # Parse DRM periods for this specific combo
        periods = parse_drm_periods(drm_df, pattern_type, primary, secondary)

        if not periods:
            results[label] = _empty_agg()
            progress_bar.progress((idx + 1) / total)
            continue

        all_stats = []
        for start_dt, end_dt in periods:
            df_slice, period_start, period_end = slice_for_graph(
                df=df_full, start_date=start_dt, end_date=end_dt,
                show_ichimoku=sidebar_config['show_ichimoku'],
                show_bb=sidebar_config['show_bb'],
                show_kc=sidebar_config['show_kc'],
            )
            if df_slice.empty:
                continue

            _, stats = execute_custom_strategy(df_slice, strategy_with_params, period_start, period_end)
            all_stats.append(stats)

        if all_stats:
            results[label] = _aggregate_stats(all_stats)
        else:
            results[label] = _empty_agg()

        progress_bar.progress((idx + 1) / total)

    progress_bar.empty()

    # --------------------------------------------------
    # Build and display table
    # --------------------------------------------------
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
    for label, agg in results.items():
        table_data[label] = [
            f"{agg['num_trades']}",
            f"{agg['win_pct']:.0f}%",
            f"{agg['lose_pct']:.0f}%",
            f"${agg['avg_win_pnl']:.2f}",
            f"${agg['avg_lose_pnl']:.2f}",
            f"${agg['total_pnl']:.2f}",
            f"${agg['expected_value']:.2f}",
            f"{agg['target_exit_pct']:.0f}%",
            f"{agg['stop_exit_pct']:.0f}%",
        ]

    perf_df = pd.DataFrame(table_data, index=metric_names)

    st.subheader("Performance by Pattern")
    st.caption(f"Strategy: **{selected_strategy.get('strategy_name', 'Custom')}**")
    st.table(perf_df)


# ------------------------------------------------------------------
# Helpers
# ------------------------------------------------------------------

def _get_pattern_combos(pattern_type, primary_choice, secondary_choice):
    """
    Return list of (primary, secondary) tuples based on sidebar filters.
    - Primary=None  → all primaries × all their secondaries
    - Primary set, Secondary=None → that primary × all its secondaries
    - Both set → just that one combo
    """
    if primary_choice is not None and secondary_choice is not None:
        return [(primary_choice, secondary_choice)]

    if primary_choice is not None:
        # All secondaries under this primary
        secondaries = PRIMARY_SECONDARY_MAP.get(primary_choice, [])
        return [(primary_choice, sec) for sec in secondaries]

    # All combos
    combos = []
    for primary, secondaries in PRIMARY_SECONDARY_MAP.items():
        for sec in secondaries:
            combos.append((primary, sec))
    return combos


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

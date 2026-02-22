"""
Charting tab (Tab 1) UI and logic
"""
import streamlit as st
from data.loader import load_ohlc, load_drm, parse_drm_periods
from indicators.calculate_indicators import calculate_indicators, slice_for_graph
from graphs.graph import render_charts
from strategies.first_strategy import ichimoku_tenkan_kijun_strategy, execute_custom_strategy
import pandas as pd


def render_charting_tab(sidebar_config):
    """Render the charting tab content"""

    show_1h = sidebar_config['analysis_mode'] == "1H + 15m"

    # File uploaders
    render_file_uploaders(show_1h)

    # Check if data is loaded
    if not check_data_loaded(show_1h):
        return

    if sidebar_config['primary_choice'] is None or sidebar_config['secondary_choice'] is None:
        st.info("Please select Pattern, Primary setup, and Secondary setup to display charts.")
        return

    # Calculate indicators (exclude display-only params: RSI zones and CMB lines)
    display_only_keys = {'rsi_upper_1', 'rsi_upper_2', 'rsi_lower_1', 'rsi_lower_2',
                         'cmb_line_1', 'cmb_line_2', 'cmb_line_3', 'cmb_line_4'}
    indicator_params_15m = {k: v for k, v in sidebar_config['params_15m'].items() if k not in display_only_keys}

    df_features_1h = None
    if show_1h:
        indicator_params_1h = {k: v for k, v in sidebar_config['params_1h'].items() if k not in display_only_keys}
        df_features_1h = calculate_indicators(
            df=st.session_state["df_1h"],
            **indicator_params_1h
        )

    df_features_15m = calculate_indicators(
        df=st.session_state["df_15m"],
        **indicator_params_15m
    )

    # Determine if custom strategy is selected
    show_custom_strategy = False
    selected_custom_strategy = None

    # Use the actual strategy index stored in session state
    if st.session_state.get('selected_custom_strategy_idx', 0) > 0:
        actual_idx = st.session_state.get('selected_custom_strategy_actual_idx')
        if actual_idx is not None and actual_idx < len(st.session_state['saved_strategies']):
            show_custom_strategy = True
            selected_custom_strategy = st.session_state['saved_strategies'][actual_idx]

    # Parse DRM periods
    drm_periods = parse_drm_periods(
        st.session_state["drm"],
        sidebar_config['pattern'],
        sidebar_config['primary_choice'],
        sidebar_config['secondary_choice']
    )

    if not drm_periods:
        st.warning("No valid date ranges found in DRM.")
        return

    # Collect stats from all periods for global aggregation
    all_stats_1h = []
    all_stats_15m = []
    strategy_label = None

    # Reserve a container at the top for the global performance summary
    global_perf_container = st.container()

    # Render each period
    for i, (start_dt, end_dt) in enumerate(drm_periods, start=1):
        stats_1h, stats_15m = render_period(
            i, start_dt, end_dt,
            df_features_1h, df_features_15m,
            sidebar_config,
            show_custom_strategy,
            selected_custom_strategy,
            show_1h,
        )

        if stats_1h is not None:
            all_stats_1h.append(stats_1h)
        if stats_15m is not None:
            all_stats_15m.append(stats_15m)

    if show_custom_strategy and selected_custom_strategy is not None:
        strategy_label = selected_custom_strategy.get('strategy_name', 'Custom Strategy')

    # Render global performance summary at the top
    has_stats = all_stats_15m and (all_stats_1h or not show_1h)
    if has_stats:
        with global_perf_container:
            render_global_performance(all_stats_1h, all_stats_15m, strategy_label, len(drm_periods), show_1h)


def render_file_uploaders(show_1h=True):
    """Render file upload section"""
    if show_1h:
        col_u1, col_u2, col_u3 = st.columns([1, 1, 1], gap="small")
    else:
        col_u2, col_u3 = st.columns([1, 1], gap="small")

    if show_1h:
        with col_u1:
            uploaded_file_1h = st.file_uploader(
                "1H OHLC", type=["csv"], key="1h", label_visibility="collapsed"
            )
            st.caption("1H OHLC (.csv)")

            if uploaded_file_1h is not None:
                df_1h = load_ohlc(uploaded_file_1h)
                st.session_state["df_1h"] = df_1h
                st.success("1H data loaded")

    with col_u2:
        uploaded_file_15m = st.file_uploader(
            "15m OHLC", type=["csv"], key="15m", label_visibility="collapsed"
        )
        st.caption("15m OHLC (.csv)")

        if uploaded_file_15m is not None:
            df_15m = load_ohlc(uploaded_file_15m)
            st.session_state["df_15m"] = df_15m
            st.success("15m data loaded")

    with col_u3:
        uploaded_drm = st.file_uploader(
            "Date Range Manager", type=["xlsx"], key="drm_", label_visibility="collapsed"
        )
        st.caption("DRM (.xlsx)")

        if uploaded_drm is not None:

            # Just get the pattern from session state or use a default
            # Since sidebar is already rendered, we need to get pattern differently
            drm = load_drm(uploaded_drm, st.session_state.get('pattern', 'Bullish'))
            st.session_state['drm'] = drm
            st.success("Date Range Manager loaded")

def check_data_loaded(show_1h=True):
    """Check if all required data is loaded"""
    if show_1h:
        if ("df_1h" not in st.session_state or
                "df_15m" not in st.session_state or
                "drm" not in st.session_state):
            st.info("Please upload 1H, 15m data files and DRM file.")
            return False
    else:
        if ("df_15m" not in st.session_state or
                "drm" not in st.session_state):
            st.info("Please upload 15m data file and DRM file.")
            return False
    return True


def render_period(period_num, start_dt, end_dt, df_features_1h, df_features_15m,
                  sidebar_config, show_custom_strategy, selected_custom_strategy, show_1h=True):
    """Render a single period with charts and stats. Returns (stats_1h, stats_15m) or (None, None)."""

    st.markdown(f"### Period {period_num}: {start_dt} → {end_dt}")

    # Slice data
    df_slice_1h, period_start_1h, period_end_1h = None, None, None
    if show_1h:
        df_slice_1h, period_start_1h, period_end_1h = slice_for_graph(
            df=df_features_1h, start_date=start_dt, end_date=end_dt,
            show_ichimoku=sidebar_config['show_ichimoku'],
            show_bb=sidebar_config['show_bb'],
            show_kc=sidebar_config['show_kc']
        )

    df_slice_15m, period_start_15m, period_end_15m = slice_for_graph(
        df=df_features_15m, start_date=start_dt, end_date=end_dt,
        show_ichimoku=sidebar_config['show_ichimoku'],
        show_bb=sidebar_config['show_bb'],
        show_kc=sidebar_config['show_kc']
    )

    if df_slice_15m.empty or (show_1h and df_slice_1h.empty):
        st.info("No data for this period.")
        return None, None

    # Execute strategies
    stats_1h, stats_15m = None, None
    strategy_label = None

    if show_custom_strategy and selected_custom_strategy is not None:
        # Add market parameters to strategy config
        strategy_with_params = selected_custom_strategy.copy()
        strategy_with_params['tick_size'] = sidebar_config['tick_size']
        strategy_with_params['minimal_change'] = sidebar_config['minimal_change']

        if show_1h:
            df_slice_1h, stats_1h = execute_custom_strategy(df_slice_1h, strategy_with_params, period_start_1h, period_end_1h)
        df_slice_15m, stats_15m = execute_custom_strategy(df_slice_15m, strategy_with_params, period_start_15m, period_end_15m)
        strategy_label = selected_custom_strategy.get('strategy_name', 'Custom Strategy')

    # RSI zone and CMB line parameters per timeframe
    chart_param_keys = ['rsi_upper_1', 'rsi_upper_2', 'rsi_lower_1', 'rsi_lower_2',
                        'cmb_line_1', 'cmb_line_2', 'cmb_line_3', 'cmb_line_4']
    rsi_zones_1h = {k: sidebar_config['params_1h'][k] for k in chart_param_keys} if show_1h else None
    rsi_zones_15m = {k: sidebar_config['params_15m'][k] for k in chart_param_keys}

    # Render charts
    show_strategy = show_custom_strategy and stats_15m is not None
    if show_strategy:
        col_charts, col_stats = st.columns([3, 1], gap="medium")

        with col_charts:
            render_charts(
                df_slice_1h, df_slice_15m,
                period_start_1h, period_end_1h,
                period_start_15m, period_end_15m,
                sidebar_config['show_ichimoku'],
                sidebar_config['show_bb'],
                sidebar_config['show_kc'],
                True,
                rsi_zones_1h,
                rsi_zones_15m,
                show_1h,
            )

        with col_stats:
            render_strategy_stats(stats_1h, stats_15m, strategy_label, show_1h)
    else:
        render_charts(
            df_slice_1h, df_slice_15m,
            period_start_1h, period_end_1h,
            period_start_15m, period_end_15m,
            sidebar_config['show_ichimoku'],
            sidebar_config['show_bb'],
            sidebar_config['show_kc'],
            False,
            rsi_zones_1h,
            rsi_zones_15m,
            show_1h,
        )

    st.divider()

    return stats_1h, stats_15m


def render_strategy_stats(stats_1h, stats_15m, strategy_label, show_1h=True):
    """Render strategy statistics table"""
    st.subheader("Strategy Statistics")

    if strategy_label:
        st.caption(f"**{strategy_label}**")

    def _format_stats(stats):
        return [
            f"{int(stats.loc['Number of trades', 'value'])}",
            f"{round(stats.loc['Win rate (%)', 'value']):.0f}%",
            f"{round(stats.loc['Loss rate (%)', 'value']):.0f}%",
            f"${stats.loc['Winning trades P&L ($)', 'value']:.2f}",
            f"${stats.loc['Losing trades P&L ($)', 'value']:.2f}",
            f"${stats.loc['Total P&L ($)', 'value']:.2f}",
            f"{round(stats.loc['Target exit (%)', 'value']):.0f}%",
            f"{round(stats.loc['Stop exit (%)', 'value']):.0f}%",
        ]

    data = {"15m": _format_stats(stats_15m)}
    if show_1h and stats_1h is not None:
        data["1H"] = _format_stats(stats_1h)
        data = {"1H": data["1H"], "15m": data["15m"]}  # 1H first

    stats_table = pd.DataFrame(
        data,
        index=[
            "Number of trades",
            "Win %",
            "Lose %",
            "Win $",
            "Lose $",
            "Total P&L",
            "Target Exit %",
            "Stop Exit %",
        ],
    )
    st.table(stats_table)


def _aggregate_stats(all_stats):
    """
    Aggregate stats across multiple periods.

    Each stats_df has these rows:
        Number of trades, Win rate (%), Loss rate (%),
        Total return (%), Total P&L ($), Avg P&L per trade ($),
        Winning trades P&L ($), Losing trades P&L ($)

    We sum the raw counts and dollars, then recompute rates.
    """
    total_trades = 0
    total_wins = 0
    total_win_pnl = 0.0
    total_lose_pnl = 0.0
    total_target_exits = 0
    total_stop_exits = 0

    for stats_df in all_stats:
        n = int(stats_df.loc['Number of trades', 'value'])
        win_rate = stats_df.loc['Win rate (%)', 'value'] / 100.0

        wins = round(n * win_rate)

        total_trades += n
        total_wins += wins
        total_win_pnl += stats_df.loc['Winning trades P&L ($)', 'value']
        total_lose_pnl += stats_df.loc['Losing trades P&L ($)', 'value']
        total_target_exits += round(n * stats_df.loc['Target exit (%)', 'value'] / 100.0)
        total_stop_exits += round(n * stats_df.loc['Stop exit (%)', 'value'] / 100.0)

    total_losses = total_trades - total_wins
    total_pnl = total_win_pnl + total_lose_pnl

    if total_trades > 0:
        win_pct = (total_wins / total_trades) * 100
        lose_pct = (total_losses / total_trades) * 100
        target_exit_pct = (total_target_exits / total_trades) * 100
        stop_exit_pct = (total_stop_exits / total_trades) * 100

        # Expected Value = (Win% x Avg Win $) - (Lose% x Avg Lose $)
        expected_value = (win_pct/100 * total_win_pnl) + (lose_pct / 100 * total_lose_pnl)
    else:
        win_pct = 0.0
        lose_pct = 0.0
        total_pnl = 0.0
        expected_value = 0.0
        target_exit_pct = 0.0
        stop_exit_pct = 0.0

    return {
        'num_trades': total_trades,
        'win_pct': win_pct,
        'lose_pct': lose_pct,
        'win_pnl': total_win_pnl,
        'lose_pnl': total_lose_pnl,
        'total_pnl': total_pnl,
        'expected_value': expected_value,
        'target_exit_pct': target_exit_pct,
        'stop_exit_pct': stop_exit_pct,
    }


def render_global_performance(all_stats_1h, all_stats_15m, strategy_label, num_periods, show_1h=True):
    """Render the global performance summary across all date ranges"""

    st.markdown("---")
    st.header("Global Performance Metric of Group of Date Ranges")
    st.caption(f"Performance metrics across **{num_periods}** date range(s)")

    if strategy_label:
        st.caption(f"Strategy: **{strategy_label}**")

    agg_15m = _aggregate_stats(all_stats_15m)

    def _format_agg(agg):
        return [
            f"{agg['num_trades']}",
            f"{agg['win_pct']:.0f}%",
            f"{agg['lose_pct']:.0f}%",
            f"${agg['win_pnl']:.2f}",
            f"${agg['lose_pnl']:.2f}",
            f"${agg['total_pnl']:.2f}",
            f"${agg['expected_value']:.2f}",
            f"{agg['target_exit_pct']:.0f}%",
            f"{agg['stop_exit_pct']:.0f}%",
        ]

    data = {"15m": _format_agg(agg_15m)}
    if show_1h and all_stats_1h:
        agg_1h = _aggregate_stats(all_stats_1h)
        data = {"1H": _format_agg(agg_1h), "15m": data["15m"]}

    global_table = pd.DataFrame(
        data,
        index=[
            "Number of Trades",
            "Win %",
            "Lose %",
            "Win $",
            "Lose $",
            "Total P&L",
            "Expected Value",
            "Target Exit %",
            "Stop Exit %",
        ],
    )

    st.table(global_table)
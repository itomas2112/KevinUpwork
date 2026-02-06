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

    # File uploaders
    render_file_uploaders()

    # Check if data is loaded
    if not check_data_loaded():
        return

    if sidebar_config['primary_choice'] is None or sidebar_config['secondary_choice'] is None:
        st.info("Please select Pattern, Primary setup, and Secondary setup to display charts.")
        return

    # Calculate indicators
    df_features_1h = calculate_indicators(
        df=st.session_state["df_1h"],
        **sidebar_config['params_1h']
    )

    df_features_15m = calculate_indicators(
        df=st.session_state["df_15m"],
        **sidebar_config['params_15m']
    )

    # Determine if custom strategy is selected
    show_custom_strategy = False
    selected_custom_strategy = None

    if st.session_state.get('selected_custom_strategy_idx', 0) > 0:
        show_custom_strategy = True
        strategy_idx = st.session_state['selected_custom_strategy_idx'] - 1
        selected_custom_strategy = st.session_state['saved_strategies'][strategy_idx]

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

    # Render each period
    for i, (start_dt, end_dt) in enumerate(drm_periods, start=1):
        render_period(
            i, start_dt, end_dt,
            df_features_1h, df_features_15m,
            sidebar_config,
            show_custom_strategy,
            selected_custom_strategy
        )


def render_file_uploaders():
    """Render file upload section"""
    col_u1, col_u2, col_u3 = st.columns([1, 1, 1], gap="small")

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

def check_data_loaded():
    """Check if all required data is loaded"""
    if ("df_1h" not in st.session_state or
            "df_15m" not in st.session_state or
            "drm" not in st.session_state):
        st.info("Please upload both 1H and 15m data files. Pls upload DRM file.")
        return False
    return True


def render_period(period_num, start_dt, end_dt, df_features_1h, df_features_15m,
                  sidebar_config, show_custom_strategy, selected_custom_strategy):
    """Render a single period with charts and stats"""

    st.markdown(f"### Period {period_num}: {start_dt} → {end_dt}")

    # Slice data
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

    if df_slice_1h.empty or df_slice_15m.empty:
        st.info("No data for this period.")
        return

    # Execute strategies
    stats_1h, stats_15m = None, None
    strategy_label = None

    if sidebar_config['show_tenkan_kijun']:
        df_slice_1h, stats_1h = ichimoku_tenkan_kijun_strategy(df_slice_1h)
        df_slice_15m, stats_15m = ichimoku_tenkan_kijun_strategy(df_slice_15m)
        strategy_label = "Tenkan Kijun Strategy"

    if show_custom_strategy and selected_custom_strategy is not None:
        df_slice_1h, custom_stats_1h = execute_custom_strategy(df_slice_1h, selected_custom_strategy)
        df_slice_15m, custom_stats_15m = execute_custom_strategy(df_slice_15m, selected_custom_strategy)

        if not sidebar_config['show_tenkan_kijun']:
            stats_1h = custom_stats_1h
            stats_15m = custom_stats_15m
            strategy_label = selected_custom_strategy.get('strategy_name', 'Custom Strategy')

    # Render charts
    if sidebar_config['show_tenkan_kijun'] or show_custom_strategy:
        col_charts, col_stats = st.columns([3, 1], gap="medium")

        with col_charts:
            render_charts(
                df_slice_1h, df_slice_15m,
                period_start_1h, period_end_1h,
                period_start_15m, period_end_15m,
                sidebar_config['show_ichimoku'],
                sidebar_config['show_bb'],
                sidebar_config['show_kc'],
                True
            )

        with col_stats:
            render_strategy_stats(stats_1h, stats_15m, strategy_label)
    else:
        render_charts(
            df_slice_1h, df_slice_15m,
            period_start_1h, period_end_1h,
            period_start_15m, period_end_15m,
            sidebar_config['show_ichimoku'],
            sidebar_config['show_bb'],
            sidebar_config['show_kc'],
            False
        )

    st.divider()


def render_strategy_stats(stats_1h, stats_15m, strategy_label):
    """Render strategy statistics table"""
    st.subheader("Strategy Statistics")

    if strategy_label:
        st.caption(f"**{strategy_label}**")

    stats_table = pd.DataFrame(
        {
            "1H": [
                f"{int(stats_1h.loc['Number of trades', 'value'])}",
                f"{round(stats_1h.loc['Win rate (%)', 'value']):.0f}%",
                f"{round(stats_1h.loc['Loss rate (%)', 'value']):.0f}%",
                f"{stats_1h.loc['Total return (%)', 'value']:.2f}%",
            ],
            "15m": [
                f"{int(stats_15m.loc['Number of trades', 'value'])}",
                f"{round(stats_15m.loc['Win rate (%)', 'value']):.0f}%",
                f"{round(stats_15m.loc['Loss rate (%)', 'value']):.0f}%",
                f"{stats_15m.loc['Total return (%)', 'value']:.2f}%",
            ],
        },
        index=["Number of trades", "Win rate (%)", "Loss rate (%)", "Total return (%)"],
    )
    st.table(stats_table)
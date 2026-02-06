"""
Sidebar UI components
"""
import streamlit as st
from data.helpers import on_primary_change, PRIMARY_SECONDARY_MAP


def render_sidebar():
    """Render the complete sidebar with all controls"""

    # Pattern Parameters
    st.sidebar.header("Pattern Parameters")
    pattern = st.sidebar.selectbox("Pattern", ['Bullish', 'Bearish'], index=0)

    # Store pattern in session state so other functions can access it
    st.session_state['pattern'] = pattern

    primary_choice = st.sidebar.selectbox(
        "Primary wave",
        options=[None] + list(PRIMARY_SECONDARY_MAP.keys()),
        format_func=lambda x: "Select..." if x is None else x,
        key="primary_choice",
        on_change=on_primary_change,
    )

    secondary_choice = None
    if primary_choice is not None:
        secondary_choice = st.sidebar.selectbox(
            "Secondary wave",
            options=[None] + PRIMARY_SECONDARY_MAP[primary_choice],
            format_func=lambda x: "Select..." if x is None else x,
            key="secondary_choice",
        )

    # Global Overlays
    st.sidebar.header("Global Overlays")
    show_ichimoku = st.sidebar.checkbox("Show Ichimoku Cloud", value=False)
    show_bb = st.sidebar.checkbox("Show Bollinger Bands", value=False)
    show_kc = st.sidebar.checkbox("Show Keltner Channel", value=False)

    # Strategy Overlays
    st.sidebar.header("Strategy Overlays")
    show_tenkan_kijun = st.sidebar.checkbox("Show Tenkan Kijun Strategy", value=False)

    # Custom Strategies (single selection)
    if st.session_state['saved_strategies']:
        st.sidebar.markdown("**Custom Strategies:**")

        strategy_options = ["None"] + [
            strategy.get('strategy_name', f'Strategy_{idx + 1}')
            for idx, strategy in enumerate(st.session_state['saved_strategies'])
        ]

        if 'selected_custom_strategy_idx' not in st.session_state:
            st.session_state['selected_custom_strategy_idx'] = 0

        # Radio button for single selection
        selected_option = st.sidebar.radio(
            "Select one strategy:",
            options=range(len(strategy_options)),
            format_func=lambda x: strategy_options[x],
            index=st.session_state['selected_custom_strategy_idx'],
            key="custom_strategy_radio"
        )

        # Check if selection changed and update
        if selected_option != st.session_state['selected_custom_strategy_idx']:
            st.session_state['selected_custom_strategy_idx'] = selected_option
            st.rerun()

    # Indicator Parameters
    params_1h = render_timeframe_parameters("1H")
    params_15m = render_timeframe_parameters("15m")

    return {
        'pattern': pattern,
        'primary_choice': primary_choice,
        'secondary_choice': secondary_choice,
        'show_ichimoku': show_ichimoku,
        'show_bb': show_bb,
        'show_kc': show_kc,
        'show_tenkan_kijun': show_tenkan_kijun,
        'params_1h': params_1h,
        'params_15m': params_15m
    }


def render_timeframe_parameters(timeframe):
    """Render indicator parameters for a specific timeframe"""
    st.sidebar.header(f"{timeframe} Parameters")

    key_prefix = timeframe.lower().replace('h', '_h').replace('m', '_m')

    params = {
        'rsi_window': st.sidebar.slider(
            f"RSI window ({timeframe})", 5, 50, 14,
            key=f"rsi_{key_prefix}"
        ),
        'bb_period': st.sidebar.number_input(
            f"BB Period ({timeframe})", 5, 100, 20, step=1,
            key=f"bb_p_{key_prefix}"
        ),
        'bb_stdev': st.sidebar.number_input(
            f"BB StdDev ({timeframe})", 0.5, 5.0, 2.0, step=0.1,
            key=f"bb_s_{key_prefix}"
        ),
        'kc_ema_period': st.sidebar.number_input(
            f"KC EMA Period ({timeframe})", 5, 100, 20, step=1,
            key=f"kc_ema_{key_prefix}"
        ),
        'kc_atr_period': st.sidebar.number_input(
            f"KC ATR Period ({timeframe})", 5, 100, 10, step=1,
            key=f"kc_atr_{key_prefix}"
        ),
        'kc_atr_mult': st.sidebar.number_input(
            f"KC ATR Mult ({timeframe})", 0.5, 5.0, 2.0, step=0.1,
            key=f"kc_mult_{key_prefix}"
        ),
    }

    return params
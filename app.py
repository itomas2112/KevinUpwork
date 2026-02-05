#%% Imports
import streamlit as st
import pandas as pd
import numpy as np

from data.helpers import on_primary_change

from data.loader import load_ohlc, load_drm, parse_drm_periods

from indicators.calculate_indicators import calculate_indicators, slice_for_graph

from graphs.graph import build_main_chart

#%% Code

# -------------------------------------------------
# Start
# -------------------------------------------------

st.title("Charting")

st.set_page_config(
    layout="wide",
    initial_sidebar_state="expanded"
)

# -------------------------------------------------
# Sidebar
# -------------------------------------------------

PRIMARY_SECONDARY_MAP = {
    "W.(1)": [
        "W.1 Impulse",
        "W.3 Impulse",
        "W.5 Impulse",
    ],

    "W.(2)": [
        "W.A Impulse",
        "W.C Impulse",
        "W.A/W Zigzag",
        "W.Y Zigzag",
    ],

    "W.(3)": [
        "W.1 Impulse",
        "W.3 Impulse",
        "W.5 Impulse",
    ],

    "W.(4)": [
        "W.A Impulse",
        "W.C Impulse",
        "W.A/W Zigzag",
        "W.Y Zigzag",
    ],

    "W.(5)": [
        "W.1 Impulse",
        "W.3 Impulse",
        "W.5 Impulse",
    ],

    "W.(A)": [
        "W.1 Impulse",
        "W.3 Impulse",
        "W.5 Impulse",
        "W.A Impulse",
        "W.C Impulse",
        "W.A/W Zigzag",
        "W.Y Zigzag",
    ],

    "W.(B)": [
        "W.A Impulse",
        "W.C Impulse",
        "W.A/W Zigzag",
        "W.Y Zigzag",
    ],

    "W.(C)": [
        "W.1 Impulse",
        "W.3 Impulse",
        "W.5 Impulse",
    ],

    "W.(W)": [
        "W.A Impulse",
        "W.C Impulse",
        "W.A/W Zigzag",
        "W.Y Zigzag",
    ],

    "W.(X)": [
        "W.A Impulse",
        "W.C Impulse",
        "W.A/W Zigzag",
        "W.Y Zigzag",
    ],

    "W.(Y)": [
        "W.A Impulse",
        "W.C Impulse",
        "W.A/W Zigzag",
        "W.Y Zigzag",
    ],
}


st.sidebar.header("Pattern Parameters")
pattern = st.sidebar.selectbox("Pattern", ['Bullish', 'Bearish'], index = 0)

primary_choice = st.sidebar.selectbox(
    "Primary wave",
    options=[None] + list(PRIMARY_SECONDARY_MAP.keys()),
    format_func=lambda x: "Select..." if x is None else x,
    key="primary_choice",
    on_change=on_primary_change,   # 👈 THIS
)

secondary_choice = None

if primary_choice is not None:
    secondary_choice = st.sidebar.selectbox(
        "Secondary wave",
        options=[None] + PRIMARY_SECONDARY_MAP[primary_choice],
        format_func=lambda x: "Select..." if x is None else x,
        key="secondary_choice",
    )

st.sidebar.header("Global Overlays")
show_ichimoku = st.sidebar.checkbox("Show Ichimoku Cloud", value=False)
show_bb = st.sidebar.checkbox("Show Bollinger Bands", value=False)
show_kc = st.sidebar.checkbox("Show Keltner Channel", value=False)

st.sidebar.header("1H Parameters")
rsi_window_1h = st.sidebar.slider("RSI window (1H)", 5, 50, 14, key="rsi_1h")
bb_period_1h = st.sidebar.number_input("BB Period (1H)", 5, 100, 20, step=1, key="bb_p_1h")
bb_stdev_1h = st.sidebar.number_input("BB StdDev (1H)", 0.5, 5.0, 2.0, step=0.1, key="bb_s_1h")
kc_ema_period_1h = st.sidebar.number_input("KC EMA Period (1H)", 5, 100, 20, step=1, key="kc_ema_1h")
kc_atr_period_1h = st.sidebar.number_input("KC ATR Period (1H)", 5, 100, 10, step=1, key="kc_atr_1h")
kc_atr_mult_1h = st.sidebar.number_input("KC ATR Mult (1H)", 0.5, 5.0, 2.0, step=0.1, key="kc_mult_1h")

st.sidebar.header("15m Parameters")
rsi_window_15m = st.sidebar.slider("RSI window (15m)", 5, 50, 14, key="rsi_15m")
bb_period_15m = st.sidebar.number_input("BB Period (15m)", 5, 100, 20, step=1, key="bb_p_15m")
bb_stdev_15m = st.sidebar.number_input("BB StdDev (15m)", 0.5, 5.0, 2.0, step=0.1, key="bb_s_15m")
kc_ema_period_15m = st.sidebar.number_input("KC EMA Period (15m)", 5, 100, 20, step=1, key="kc_ema_15m")
kc_atr_period_15m = st.sidebar.number_input("KC ATR Period (15m)", 5, 100, 10, step=1, key="kc_atr_15m")
kc_atr_mult_15m = st.sidebar.number_input("KC ATR Mult (15m)", 0.5, 5.0, 2.0, step=0.1, key="kc_mult_15m")


# -------------------------------------------------
# Upload
# -------------------------------------------------
col_u1, col_u2, col_u3 = st.columns([1, 1, 1], gap="small")

with col_u1:
    uploaded_file_1h = st.file_uploader(
        "1H OHLC",
        type=["csv"],
        key="1h",
        label_visibility="collapsed"
    )
    st.caption("1H OHLC (.csv)")

with col_u2:
    uploaded_file_15m = st.file_uploader(
        "15m OHLC",
        type=["csv"],
        key="15m",
        label_visibility="collapsed"
    )
    st.caption("15m OHLC (.csv)")

with col_u3:
    uploaded_drm = st.file_uploader(
        "Date Range Manager",
        type=["xlsx"],
        key="drm_",
        label_visibility="collapsed"
    )
    st.caption("DRM (.xlsx)")

if uploaded_file_1h is not None:
    df_1h = load_ohlc(uploaded_file_1h)
    st.session_state["df_1h"] = df_1h
    st.success("1H data loaded")

if uploaded_file_15m is not None:
    df_15m = load_ohlc(uploaded_file_15m)
    st.session_state["df_15m"] = df_15m
    st.success("15m data loaded")

if uploaded_drm is not None:
    drm = load_drm(uploaded_drm, pattern)
    st.session_state['drm'] = drm
    st.success("Date Range Manager loaded")

# -------------------------------------------------
# Main logic
# -------------------------------------------------
if "df_1h" not in st.session_state or "df_15m" not in st.session_state or "drm" not in st.session_state:
    st.info("Please upload both 1H and 15m data files. Pls upload DRM file.")
    st.stop()

if st.session_state.get("primary_choice") is None or st.session_state.get("secondary_choice") is None:
    st.info("Please select Pattern, Primary setup, and Secondary setup to display charts.")
    st.stop()

# Calculate features
df_features_1h = calculate_indicators(
    df=st.session_state["df_1h"],
    rsi_window=rsi_window_1h,
    bb_period=bb_period_1h,
    bb_stdev=bb_stdev_1h,
    kc_ema_period=kc_ema_period_1h,
    kc_atr_period=kc_atr_period_1h,
    kc_atr_mult=kc_atr_mult_1h,
)

df_features_15m = calculate_indicators(
    df=st.session_state["df_15m"],
    rsi_window=rsi_window_15m,
    bb_period=bb_period_15m,
    bb_stdev=bb_stdev_15m,
    kc_ema_period=kc_ema_period_15m,
    kc_atr_period=kc_atr_period_15m,
    kc_atr_mult=kc_atr_mult_15m,
)

# Date selection (from existing data only)
drm_periods = parse_drm_periods(st.session_state["drm"], pattern, primary_choice, secondary_choice)

if not drm_periods:
    st.warning("No valid date ranges found in DRM.")
    st.stop()

for i, (start_dt, end_dt) in enumerate(drm_periods, start=1):

    st.markdown(f"### Period {i}: {start_dt} → {end_dt}")

    df_slice_1h, period_start_1h, period_end_1h = slice_for_graph(
        df=df_features_1h,
        start_date=start_dt,
        end_date=end_dt,
        show_ichimoku=show_ichimoku,
        show_bb=show_bb,
        show_kc=show_kc
    )

    df_slice_15m, period_start_15m, period_end_15m = slice_for_graph(
        df=df_features_15m,
        start_date=start_dt,
        end_date=end_dt,
        show_ichimoku=show_ichimoku,
        show_bb=show_bb,
        show_kc=show_kc
    )

    if df_slice_1h.empty or df_slice_15m.empty:
        st.info("No data for this period.")
        continue

    col_left, col_right = st.columns([1, 1], gap="small")

    with col_left:
        st.subheader("1H Chart")
        fig_1h = build_main_chart(
            df_slice=df_slice_1h,
            period_start=period_start_1h,
            period_end=period_end_1h,
            show_ichimoku=show_ichimoku,
            show_bb=show_bb,
            show_kc=show_kc,
        )
        st.plotly_chart(fig_1h, use_container_width=True, key = f"chart_1h_{i}")

    with col_right:
        st.subheader("15m Chart")
        fig_15m = build_main_chart(
            df_slice=df_slice_15m,
            period_start=period_start_15m,
            period_end=period_end_15m,
            show_ichimoku=show_ichimoku,
            show_bb=show_bb,
            show_kc=show_kc,
        )
        st.plotly_chart(fig_15m, use_container_width=True, key = f"chart_15m_{i}")

    st.divider()



#%% Testing code
# import pandas as pd
# df = pd.read_excel(rf'C:\Users\MI\Downloads\DateRangeManager.xlsx')
# sheet_name = 'Bullish'
# df[sheet_name] = df[sheet_name].ffill().copy()
#
# primary_choice = "W.(1)"
# secondary_choice = "W.1 Impulse"
#
# df[(df[sheet_name] == primary_choice) & (df.iloc[:,1] == secondary_choice)].iloc[:,2:]
import streamlit as st
def on_primary_change():
    st.session_state.secondary_choice = None

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
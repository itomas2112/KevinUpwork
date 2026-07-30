"""
Bidirectional Streamlit component rendering the MotiveWave-style wave chart.

The frontend is plain, hand-written JS (no npm / no build step) plus a vendored
copy of TradingView Lightweight Charts.
"""
import os
import streamlit.components.v1 as components

_component = components.declare_component(
    "wave_chart",
    path=os.path.join(os.path.dirname(__file__), "frontend"),
)


def wave_chart(payload, config, patterns, wave_defs, ack=0, key=None):
    """Render the chart. Returns whatever the frontend last sent back.

    ``patterns`` is the authoritative pattern list for the payload's timeframe;
    ``wave_defs`` carries the pattern/degree definitions the toolbar is built from.
    ``ack`` is the highest event seq Python has already folded in -- the
    frontend prunes its outbox against it.
    """
    return _component(
        payload=payload,
        config=config,
        patterns=patterns,
        wave_defs=wave_defs,
        ack=ack,
        key=key,
        default=None,
    )

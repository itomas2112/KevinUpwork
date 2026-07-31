"""
Wave Analysis tab (Tab 8) UI and logic
"""
import copy

import numpy as np
import pandas as pd
import streamlit as st

from config.wave_analysis import apply_wave_event, validate_patterns, wave_defs
from indicators.rsi import rsi
from indicators.cmb import cmb_composite
from strategies.wave_marking_manager import load_wave_markings, save_wave_markings
from ui.components.wave_chart import wave_chart

# Static for the life of the process -- built once, handed to every render.
WAVE_DEFS = wave_defs()

# The component's Streamlit key. Also the session-state key its return value is
# stored under, which is where the held fingerprint is read from.
CHART_KEY = "wave_chart_main"

# How many mutations Ctrl+Z can walk back. Deep copies of a timeframe's whole
# pattern list, so the ceiling is about memory, not about the client's patience.
UNDO_LIMIT = 50


def _close_column(df):
    """Close prices. The main pipeline names this column 'latest'."""
    return df["latest"] if "latest" in df.columns else df["close"]


def _floats(series):
    """Series -> plain Python floats with None in place of NaN (JSON-safe)."""
    return [None if pd.isna(v) else float(v) for v in series]


def build_fingerprint(df, timeframe, dataset_key):
    """Identity of the chart data. Changes iff the data or timeframe changes.

    The dataset key leads: row count and date range alone are not identity --
    two instruments exported over the same period collide on them, and the
    cached payload (plus the frontend's own fingerprint check) would then keep
    the previous upload's bars on screen after the new one lands.
    """
    if len(df) == 0:
        return f"{dataset_key}|{timeframe}|0||"
    return (f"{dataset_key}|{timeframe}|{len(df)}"
            f"|{df.index[0].isoformat()}|{df.index[-1].isoformat()}")


def dedupe_bars(df):
    """One row per timestamp, strictly increasing in time.

    Lightweight Charts' ``setData`` rejects the *entire* dataset on the first
    non-strictly-increasing time, and it does so silently: a blank price pane,
    a live crosshair over nothing. Real exports hit this -- the client's
    ten-year Gold file repeats 1,985 timestamps, every one of them a verbatim
    second copy of a bar. Repeats need not be verbatim in general, though: on a
    contract-roll day two contracts can both print at the same minute, and the
    one that was actually trading is the one carrying the volume. So the
    highest-volume row wins the timestamp, and with no usable volume the last
    row does, which is the export's own ordering.

    Indicators are computed downstream of this, so a bar that lost the contest
    never reaches a rolling window either.

    The frame is only copied when there is something to fix -- this runs on
    every payload build, over a quarter of a million rows.
    """
    if df.index.is_unique and df.index.is_monotonic_increasing:
        return df

    frame = df
    if not frame.index.is_unique:
        if "volume" in frame.columns:
            volume = pd.to_numeric(frame["volume"], errors="coerce").to_numpy(dtype="float64")
            # A row with no readable volume must lose to any row that has one,
            # and must not beat its own duplicates on a NaN comparison.
            volume = np.where(np.isnan(volume), -np.inf, volume)
        else:
            volume = np.zeros(len(frame))
        # Both sorts are stable, so rows of equal volume -- and every row of a
        # timestamp with no usable volume at all -- keep the frame's original
        # order and ``keep="last"`` resolves the tie in favour of the last one.
        frame = frame.iloc[np.argsort(volume, kind="stable")]
        frame = frame.sort_index(kind="mergesort")
        frame = frame[~frame.index.duplicated(keep="last")]

    # The invariant, not an optimisation: a payload whose times are not
    # strictly increasing blanks the chart, so pathological input (unsorted
    # rows, an unparsable timestamp) loses the offending rows here rather than
    # taking the whole pane down in the browser.
    if not (frame.index.is_unique and frame.index.is_monotonic_increasing):
        frame = frame[frame.index.notna()]
        frame = frame.sort_index(kind="mergesort")
        frame = frame[~frame.index.duplicated(keep="last")]
    return frame


def build_wave_payload(df, timeframe, dataset_key):
    """Build the JSON-serializable payload the wave chart frontend consumes."""
    # Identity of the *input* frame: the tab computes the same fingerprint from
    # the same frame to decide whether this payload is still current, and it
    # must not pay for a dedup pass to do it. The dedup is deterministic, so
    # one identity implies the other.
    fingerprint = build_fingerprint(df, timeframe, dataset_key)
    df = dedupe_bars(df)
    close = _close_column(df)

    rsi_series = rsi(close, 14)
    rsi_13 = rsi_series.rolling(13).mean()
    rsi_33 = rsi_series.rolling(33).mean()

    ci, ci_13, ci_33 = cmb_composite(close)

    return {
        "time": [int(ts.timestamp()) for ts in df.index],
        "open": _floats(df["open"]),
        "high": _floats(df["high"]),
        "low": _floats(df["low"]),
        "close": _floats(close),
        "rsi": _floats(rsi_series),
        "rsi_13": _floats(rsi_13),
        "rsi_33": _floats(rsi_33),
        "ci": _floats(ci),
        "ci_13": _floats(ci_13),
        "ci_33": _floats(ci_33),
        "timeframe": timeframe,
        "fingerprint": fingerprint,
    }


def choose_payload(payload, held):
    """The payload to hand the component: the real one, or a stub.

    A decade of 15m bars serialises to tens of megabytes, and Streamlit ships
    every component argument to the iframe again on every single rerun -- a
    widget toggled in another tab would otherwise re-send the whole chart. The
    frontend reports the fingerprint it currently holds, so once that matches
    what we would send there is nothing to send: a stub carrying the same
    fingerprint says "your data is still current" in a few dozen bytes.

    Anything else -- no value yet, a stale fingerprint, a frontend that lost
    its state to a hard refresh -- gets the full payload.
    """
    fingerprint = payload.get("fingerprint")
    if held is not None and held == fingerprint:
        return {"fingerprint": fingerprint, "stub": True}
    return payload


def held_fingerprint(value):
    """The fingerprint the frontend last reported holding, or None."""
    return value.get("held") if isinstance(value, dict) else None


def _is_int(value):
    return isinstance(value, int) and not isinstance(value, bool)


# --------------------------------------------------------------------- undo
#
# Session-only history: a restart clears what can be undone but never the
# markings themselves, which live on disk. One snapshot per applied batch, per
# timeframe, so an accidental delete of a nested pattern is always one Ctrl+Z
# away even when the frontend fired several events between two reruns.


def _timeframe_list(patterns_by_tf, timeframe):
    existing = patterns_by_tf.get(timeframe) if isinstance(patterns_by_tf, dict) else None
    return existing if isinstance(existing, list) else []


def _undo_stack():
    stack = st.session_state.get("_wa_undo")
    if not isinstance(stack, list):
        stack = []
        st.session_state["_wa_undo"] = stack
    return stack


def _push_undo(patterns_by_tf, timeframe):
    """Snapshot a timeframe's list before it is mutated. Oldest drops at the cap."""
    stack = _undo_stack()
    stack.append((timeframe, copy.deepcopy(_timeframe_list(patterns_by_tf, timeframe))))
    if len(stack) > UNDO_LIMIT:
        del stack[:len(stack) - UNDO_LIMIT]


def _pop_undo(timeframe):
    """The most recent snapshot taken on ``timeframe``, or None.

    Other timeframes' entries are stepped over rather than consumed: the chart
    shows one timeframe at a time, so an undo must revert what the user is
    actually looking at, not whatever happened to be mutated last.
    """
    stack = _undo_stack()
    for index in range(len(stack) - 1, -1, -1):
        if stack[index][0] == timeframe:
            return stack.pop(index)[1]
    return None


def apply_event_batch(patterns_by_tf, timeframe, value, last_seq):
    """Fold every not-yet-seen event of a component value into the state.

    Protocol v2: the frontend posts its whole outbox as
    ``{"seq": n, "events": [{"eseq": ..., "type": ...}, ...]}`` and prunes it
    against the ``ack`` we hand back, so a burst of mutations fired between two
    reruns can never overwrite each other.

    Returns ``(state, seq, changed)``. ``seq`` advances past every event that
    was *processed*, not merely the ones that changed anything -- an event this
    module rejects must still be acked or the frontend would replay it forever.

    Same-degree overlap validation runs once at the end of a batch that changed
    something, which is the only place that catches every mutation path: the
    reducers preserve a pattern's existing colour, so a ``shift_degree`` or a
    ``move_point`` that creates -- or resolves -- a collision would otherwise
    leave a stale colour behind.

    ``undo`` is handled here rather than in the reducers because its history is
    session state and ``config.wave_analysis`` stays free of Streamlit. One
    snapshot is pushed for the whole batch, before anything is folded, and only
    when the batch carries a mutation: an ``undo`` never records history of its
    own, which keeps the stack one-directional and makes an ``undo`` arriving in
    the same batch as its mutation land exactly on that batch's snapshot.
    """
    if not isinstance(value, dict):
        return patterns_by_tf, last_seq, False
    events = value.get("events")
    if not isinstance(events, list):
        return patterns_by_tf, last_seq, False

    fresh = [e for e in events
             if isinstance(e, dict) and _is_int(e.get("eseq")) and e["eseq"] > last_seq]
    fresh.sort(key=lambda e: e["eseq"])

    if any(event.get("type") != "undo" for event in fresh):
        _push_undo(patterns_by_tf, timeframe)

    state = patterns_by_tf
    seq = last_seq
    for event in fresh:
        if event.get("type") == "undo":
            restored = _pop_undo(timeframe) if isinstance(state, dict) else None
            if restored is not None:
                state = dict(state)
                state[timeframe] = restored
        else:
            state = apply_wave_event(state, timeframe, event)
        seq = event["eseq"]

    changed = state is not patterns_by_tf
    if changed:
        state = dict(state)
        state[timeframe] = validate_patterns(state.get(timeframe, []))

    return state, seq, changed


def render_wave_analysis_tab(sidebar_config):
    """Render the Wave Analysis tab"""

    df = st.session_state.get("df_ohlc")
    if df is None or df.empty:
        st.info("Please upload OHLC data in the Charting tab first.")
        return

    timeframe = st.session_state.get("_agg_timeframe",
                                     st.session_state.get("base_timeframe", "15m"))

    # Raised by a save that failed on an earlier run and stays up until one
    # succeeds -- an unwritable markings file is a condition, not an incident,
    # and the ``st.rerun()`` below would throw away a banner drawn inline.
    # The markings themselves are safe in memory throughout.
    save_error = st.session_state.get("_wa_save_error")
    if save_error:
        st.warning(f"Could not save wave markings: {save_error}")

    with st.expander("Chart settings", expanded=False):
        col_h, col_rsi, col_cmb, col_bs = st.columns(4)
        with col_h:
            height = st.number_input("Chart height (px)", min_value=400, max_value=1600,
                                     value=820, step=20, key="_wa_height")
        with col_rsi:
            rsi_height = st.number_input("RSI pane height (px)", min_value=60, max_value=600,
                                         value=120, step=10, key="_wa_rsi_height")
        with col_cmb:
            cmb_height = st.number_input("CMB pane height (px)", min_value=60, max_value=600,
                                         value=120, step=10, key="_wa_cmb_height")
        with col_bs:
            bar_spacing = st.number_input("Bar spacing (px)", min_value=1, max_value=20,
                                          value=3, step=1, key="_wa_bar_spacing")

    # Markings are keyed by the uploaded file's name: they are snapped to one
    # instrument's bars and mean nothing on another's. The key is also part of
    # the payload's identity, so it has to be known before the payload is built.
    dataset_key = st.session_state.get("data_file_name", "default")

    # Rebuild the payload only when the data actually changed -- this tab runs on
    # every rerun like all the others, so it must be cheap when idle.
    fingerprint = build_fingerprint(df, timeframe, dataset_key)
    payload = st.session_state.get("_wa_payload")
    if payload is None or payload.get("fingerprint") != fingerprint:
        payload = build_wave_payload(df, timeframe, dataset_key)
        st.session_state["_wa_payload"] = payload

    config = {
        "height": int(height),
        # Pane 0 (price) takes whatever is left over.
        "pane_heights": [None, int(rsi_height), int(cmb_height)],
        "bar_spacing": int(bar_spacing),
        # Bars panned per standard wheel notch (|deltaY| == 100). No UI control yet.
        "wheel_speed": 8,
    }

    # Reload from disk on the first run and whenever the dataset changed. A
    # base-timeframe switch sweeps every ``_wa_`` key, including this one, so
    # the reload also restores markings after that sweep. Colours are always
    # recomputed -- the persisted ``color`` is never trusted.
    if ("_wa_patterns" not in st.session_state
            or st.session_state.get("_wa_dataset_key") != dataset_key):
        stored = load_wave_markings().get(dataset_key, {})
        st.session_state["_wa_patterns"] = {
            tf: validate_patterns(pattern_list) for tf, pattern_list in stored.items()
        }
        st.session_state["_wa_dataset_key"] = dataset_key
        # History belongs to the dataset it was recorded on -- undoing into
        # another instrument's pattern list would be worse than no undo at all.
        st.session_state["_wa_undo"] = []

    patterns = st.session_state["_wa_patterns"].get(timeframe, [])

    # The ack tells the frontend which events it may drop from its outbox.
    last_seq = st.session_state.get("_wa_event_seq", 0)

    # Read what the frontend is holding *before* the component call: the call's
    # own return value only lands afterwards, and deciding one rerun late would
    # cost a full resend every time the data changed. The component's value
    # lives in session state under its key -- its widget identity is keyed on
    # the key alone, so swapping the payload for a stub never disturbs it.
    stored = st.session_state.get(CHART_KEY)
    if not isinstance(stored, dict):
        stored = st.session_state.get("_wa_last_value")
    held = held_fingerprint(stored)

    value = wave_chart(choose_payload(payload, held), config, patterns, WAVE_DEFS,
                       last_seq, key=CHART_KEY)
    st.session_state["_wa_last_value"] = value

    # Streamlit re-delivers the last component value on every rerun, so the
    # monotonic event seq -- not the value itself -- decides what is new.
    updated, new_seq, changed = apply_event_batch(
        st.session_state["_wa_patterns"], timeframe, value, last_seq)
    if new_seq != last_seq:
        st.session_state["_wa_event_seq"] = new_seq
        if changed:
            st.session_state["_wa_patterns"] = updated
            # A failed write must not take the rerun down with it: the client
            # runs on Windows, where a file left open by a viewer or a backup
            # agent is routinely unwritable for a few seconds. In-memory state
            # is already correct, so the next mutation simply retries.
            try:
                # Read-modify-write so the other datasets' markings survive.
                markings = load_wave_markings()
                markings[dataset_key] = updated
                save_wave_markings(markings)
            except OSError as error:
                # Recorded rather than warned inline: the ``st.rerun()`` below
                # discards everything this run drew. The banner at the top of
                # the tab raises it from here on the run that follows.
                st.session_state["_wa_save_error"] = str(error)
            else:
                st.session_state.pop("_wa_save_error", None)
        # Rerun even when nothing changed: the frontend needs the ack to clear
        # the optimistic overlay of an event we rejected.
        st.rerun()

    st.caption(f"{len(patterns)} pattern(s) on {timeframe} · dataset: {dataset_key} "
               "· saved to saved_wave_markings.json")

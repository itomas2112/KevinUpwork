"""
Wave Analysis tab (Tab 8) UI and logic
"""
import copy
import json

import numpy as np
import pandas as pd
import streamlit as st

from config.wave_analysis import (LEG_VALUE_FIELDS, PATTERN_DEFS, apply_wave_event,
                                  settle, wave_defs)
from config.wave_projection import (CANONICAL, DISPLAY, RESAMPLE_RULES, period_map,
                                    pivot_magnets, project_patterns, refine_event)
from data.loader import resample_ohlc
from indicators.rsi import rsi
from indicators.cmb import cmb_composite
from strategies.drm_export import (PERIOD_FORMAT, build_drm_rows, build_drm_sheets,
                                   build_drm_workbook, build_wave_json, summary_lines)
from strategies.wave_marking_manager import (WAVE_MARKINGS_FILE, load_wave_documents,
                                             migrate_document, save_wave_documents)
from strategies.wave_study import (ALL_WAVES, GRID_COMBINATIONS, GRID_DEGREES,
                                   GRID_VALUE_PAIRS, INTER, OPERATORS,
                                   PARENT_WAVE_LABELS, SKIP_ORDER, field_family,
                                   run_grid, run_pair_study, run_study,
                                   wave_labels)
from ui.components.wave_chart import wave_chart

# Static for the life of the process -- built once, handed to every render.
WAVE_DEFS = wave_defs()

# The component's Streamlit key. Also the session-state key its return value is
# stored under, which is where the held fingerprint is read from.
CHART_KEY = "wave_chart_main"

# How many mutations Ctrl+Z can walk back. Deep copies of the whole canonical
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
# markings themselves, which live on disk. One snapshot of the whole canonical
# list per applied batch, so an accidental delete of a nested pattern is always
# one Ctrl+Z away even when the frontend fired several events between two
# reruns.
#
# The stack is per *dataset*, not per aggregation: there is one pattern list
# now, so an edit made while looking at 1D is undoable while looking at 15m.
# The alternative -- a stack that silently forgets work the moment the client
# changes aggregation -- would be worse than no undo at all.


def _undo_stack():
    stack = st.session_state.get("_wa_undo")
    if not isinstance(stack, list):
        stack = []
        st.session_state["_wa_undo"] = stack
    return stack


def _push_undo(patterns):
    """Snapshot the canonical list before it is mutated. Oldest drops at the cap."""
    stack = _undo_stack()
    stack.append(copy.deepcopy(patterns if isinstance(patterns, list) else []))
    if len(stack) > UNDO_LIMIT:
        del stack[:len(stack) - UNDO_LIMIT]


def _pop_undo():
    """The most recent snapshot, or None when there is nothing left to undo."""
    stack = _undo_stack()
    return stack.pop() if stack else None


def apply_event_batch(patterns, value, last_seq, pmap=None):
    """Fold every not-yet-seen event of a component value into the state.

    Protocol v2: the frontend posts its whole outbox as
    ``{"seq": n, "events": [{"eseq": ..., "type": ...}, ...]}`` and prunes it
    against the ``ack`` we hand back, so a burst of mutations fired between two
    reruns can never overwrite each other.

    ``patterns`` is the canonical list and ``pmap`` the map of the aggregation
    on screen -- None when that aggregation *is* the base timeframe, where
    display and canonical coincide. Every event is refined through ``pmap``
    before it reaches a reducer, so nothing in display coordinates is ever
    stored. An event that cannot be refined is refused, and refused **still
    advances the seq**: the frontend prunes its outbox against the ack and
    replays anything unacked forever, so a silent drop that is not acked is an
    infinite loop. The optimistic overlay snaps back on the next render, which
    is the same path an event rejected by a reducer already takes.

    Refinement is *magnetic*: a click landing on a display period that already
    carries a pivot of the same kind attaches to that pivot rather than to the
    period's own extreme. That is what lets a child drawn at 1D find a parent
    drawn at 15m, whose pivots are usually not their days' extremes. Only a
    projection earns a magnet -- at the base timeframe ``pmap`` is None and the
    click already names the exact bar the client meant.

    The canonical list goes to ``refine_event`` alongside the magnets, because in
    a dense count one display period holds several pivots of the same kind and
    price distance alone cannot say which was meant. Given the list, refinement
    chooses every point of the new marking together, for the reading that
    expresses the most structure: the marking slotting into an existing pattern
    as a child, and each of its legs adopting an existing pattern as one. The
    client marks bottom-up -- small degrees first, the larger degree drawn over
    them afterwards -- so it is a new pattern's *interior* points that decide
    which of the counts already on screen it takes under it.

    Returns ``(state, seq, changed)``. ``seq`` advances past every event that
    was *processed*, not merely the ones that changed anything.

    ``settle`` runs once at the end of a batch that changed something, on the
    canonical list -- one call, not one per aggregation. That is the only place
    that catches every mutation path: the reducers preserve a pattern's existing
    colour *and* its stored degree, so a ``shift_degree`` or a ``move_point``
    that creates -- or resolves -- a collision, or that makes or breaks a
    parent/child link, would otherwise leave a stale colour or a child sitting
    at a degree its parent no longer implies.

    ``undo`` is handled here rather than in the reducers because its history is
    session state and ``config.wave_analysis`` stays free of Streamlit. One
    snapshot is pushed for the whole batch, before anything is folded, and only
    when the batch carries a mutation: an ``undo`` never records history of its
    own, which keeps the stack one-directional and makes an ``undo`` arriving in
    the same batch as its mutation land exactly on that batch's snapshot.
    """
    if not isinstance(value, dict):
        return patterns, last_seq, False
    events = value.get("events")
    if not isinstance(events, list):
        return patterns, last_seq, False

    fresh = [e for e in events
             if isinstance(e, dict) and _is_int(e.get("eseq")) and e["eseq"] > last_seq]
    fresh.sort(key=lambda e: e["eseq"])

    if any(event.get("type") != "undo" for event in fresh):
        _push_undo(patterns)

    state = patterns
    seq = last_seq
    for event in fresh:
        if event.get("type") == "undo":
            restored = _pop_undo()
            if restored is not None:
                state = restored
        else:
            # Rebuilt per event, not once for the batch: a pattern completed by
            # the first event creates pivots the second may legitimately want to
            # attach to. Pattern counts are in the tens, so this is a few hundred
            # dict lookups -- the expensive thing, the PeriodMap, is not rebuilt.
            refined = refine_event(event, pmap, pivot_magnets(state, pmap), state)
            if refined is not None:
                state = apply_wave_event(state, refined)
        seq = event["eseq"]

    changed = state is not patterns
    if changed:
        state = settle(state)

    return state, seq, changed


# ------------------------------------------------------------ frames and maps
#
# One grid, one frame. The base frame is the uploaded data deduped exactly as
# the payload builder leaves it, and every display aggregation -- the bars the
# chart holds *and* the period map a projection is built on -- is derived from
# that same frame. The sidebar's own ``df_ohlc`` resamples the *raw* frame,
# duplicate timestamps and all; the two paths agree on the client's current
# file only because its duplicates happen to be byte-identical repeats. Deriving
# both ends here makes them agree by construction instead of by luck.
#
# All three are cached because they run on a quarter of a million rows and this
# tab is re-rendered on every widget touch anywhere in the app.


def _cached(slot, key, build):
    """``build()`` memoised in one session-state slot, replaced on a key change.

    Exactly one value per slot, never a growing dict: the period map over the
    client's ten-year file is tens of megabytes, so a cache that accumulated one
    entry per aggregation would cost more than it saves.
    """
    entry = st.session_state.get(slot)
    if isinstance(entry, tuple) and len(entry) == 2 and entry[0] == key:
        return entry[1]
    value = build()
    st.session_state[slot] = (key, value)
    return value


def _base_frame(df_raw, dataset_key):
    """The canonical frame: one row per timestamp, strictly increasing.

    Keyed without a timeframe -- it does not depend on the aggregation, and
    re-deduping a quarter of a million rows every time the client switches
    aggregation would be pure waste.
    """
    return _cached("_wa_base_df", build_fingerprint(df_raw, "", dataset_key),
                   lambda: dedupe_bars(df_raw))


def _display_frame(base_df, timeframe, base_timeframe, key):
    """The bars on screen at this aggregation, derived from the base frame."""
    if timeframe == base_timeframe:
        # Canonical *is* display: no copy, and no stale coarser frame left behind.
        st.session_state.pop("_wa_display_df", None)
        return base_df
    return _cached("_wa_display_df", key,
                   lambda: resample_ohlc(base_df, timeframe, base_timeframe))


def _display_map(base_df, timeframe, base_timeframe, key):
    """The period map for this aggregation, or None when there is nothing to map.

    At the base timeframe canonical *is* display, so projection is skipped
    entirely rather than run through an identity map: over the client's 262k-row
    file that map retains ~140 MB to do nothing, on top of the payload the tab
    already holds. None means "the display is canonical" and every call site
    handles it explicitly.
    """
    if timeframe == base_timeframe:
        st.session_state.pop("_wa_pmap", None)
        return None
    return _cached("_wa_pmap", key,
                   lambda: period_map(base_df, timeframe, base_timeframe))


# ------------------------------------------------------------- load / migrate


# Finest first. Only a *coarser* stored base timeframe can be re-snapped onto the
# current one: 1H markings land on the 15m bars carrying each hour's extreme, but
# 15m bars cannot be recovered from an hourly upload -- they were never uploaded.
TIMEFRAME_ORDER = ("15m", "1H", "4H", "1D", "1W", "1M")


def _coarser(timeframe, base_timeframe):
    """True when ``timeframe`` is a strictly larger period than ``base_timeframe``."""
    if timeframe not in TIMEFRAME_ORDER or base_timeframe not in TIMEFRAME_ORDER:
        return False
    return TIMEFRAME_ORDER.index(timeframe) > TIMEFRAME_ORDER.index(base_timeframe)


def _canonicaliser(pmap):
    """A ``migrate_document`` projector: one legacy timeframe's list -> canonical.

    A factory rather than a lambda in the loop, so each projector closes over
    its own map instead of over whatever the loop variable ended up holding.
    """
    return lambda patterns: project_patterns(patterns, pmap, CANONICAL)


def _load_canonical(dataset_key, base_df, base_timeframe, path=WAVE_MARKINGS_FILE):
    """The dataset's canonical pattern list, migrated if it needed it.

    Returns ``(patterns, note)``; the note is whatever the caption has to own
    up to about this load, and is empty when nothing happened.

    ``path`` exists so a test can point the whole load-migrate-write-back cycle
    at a throwaway file; the tab always takes the default.
    """
    documents = load_wave_documents(path)
    document = documents.get(dataset_key)
    if document is None:
        return [], ""

    note = ""
    if document.get("schema") != 2:
        # One projector per legacy timeframe the file actually holds. The base
        # timeframe's patterns are already canonical; anything not expressible
        # through RESAMPLE_RULES gets no projector and migrate_document drops
        # and counts it rather than guessing at bars it does not have.
        projectors = {}
        by_timeframe = document.get("by_timeframe") or {}
        for timeframe in by_timeframe:
            if timeframe == base_timeframe:
                projectors[timeframe] = lambda patterns: patterns
            elif timeframe in RESAMPLE_RULES:
                projectors[timeframe] = _canonicaliser(
                    period_map(base_df, timeframe, base_timeframe))

        document, dropped = migrate_document(document, base_timeframe, projectors)
        projectors = None               # release the temporary maps

        # Written back immediately so the migration runs once for this dataset
        # rather than on every switch away and back. The other datasets ride
        # along untouched, still at whatever schema they came out as.
        documents[dataset_key] = document
        try:
            save_wave_documents(documents, path)
        except OSError as error:
            st.session_state["_wa_save_error"] = str(error)
        if dropped:
            note = f"{dropped} pattern(s) dropped in migration"

    patterns = document.get("patterns") or []

    # The client moved the base-timeframe radio: what is stored is canonical at
    # a resolution that no longer exists. Re-snap it through the same machinery
    # when the pair is expressible -- and when it is not (the stored resolution
    # is *finer* than the new base, so those bars are simply gone) say so and
    # leave the markings alone rather than guess.
    stored_base = document.get("base_timeframe", base_timeframe)
    if stored_base != base_timeframe:
        if stored_base in RESAMPLE_RULES and _coarser(stored_base, base_timeframe):
            patterns = project_patterns(
                patterns, period_map(base_df, stored_base, base_timeframe), CANONICAL)
        else:
            note = ((note + " · ") if note else "") + (
                f"stored at {stored_base}, which {base_timeframe} bars cannot express")

    return patterns, note


def wave_caption(total, shown, timeframe, dataset_key, note=None):
    """The line under the chart: what exists, what is on screen, where it lives.

    The client asked in as many words to be able to see that his wave counts are
    always saved, and a marking that cannot be drawn at this aggregation -- its
    points share a bar there -- must never read as lost work. So the count that
    leads is the canonical one, and anything not on screen is named and
    explained rather than simply missing.
    """
    parts = [f"{total} pattern(s)", f"{shown} shown on {timeframe}"]
    hidden = total - shown
    if hidden > 0:
        parts.append(f"{hidden} hidden (points share a {timeframe} bar)")
    if note:
        parts.append(note)
    parts.append(f"dataset: {dataset_key}")
    parts.append("saved to saved_wave_markings.json")
    return " · ".join(parts)


# ------------------------------------------------------------------- export
#
# The client's Date Range Manager, generated from the wave counts instead of
# typed by hand. Everything that decides *what* a row says lives in
# ``strategies.drm_export``; what lives here is the one thing it deliberately
# does not know -- which bar a canonical pivot falls in at the chosen export
# timeframe.


def export_timeframes(base_timeframe):
    """The aggregations an export may be timestamped at: the base, and coarser.

    A finer one is not on offer because those bars were never uploaded, which is
    the same rule ``_coarser`` already enforces when re-snapping stored markings.
    """
    if base_timeframe not in TIMEFRAME_ORDER:
        return [base_timeframe]
    return [tf for tf in TIMEFRAME_ORDER
            if tf == base_timeframe or (tf in RESAMPLE_RULES and _coarser(tf, base_timeframe))]


def default_export_index(timeframes):
    """Where the timeframe selector starts: 4H, which is how he logs the DRM.

    His own file sits on a 4H grid (373 of 382 timestamps), so 4H is the habit
    the export should match. A base timeframe that cannot offer it falls back to
    the coarsest one that exists rather than to nothing.
    """
    if "4H" in timeframes:
        return timeframes.index("4H")
    return max(len(timeframes) - 1, 0)


def time_formatter(pmap):
    """Canonical point time -> the period-cell string for the bar containing it.

    What a period cell names is the *containing* bar, not the extreme the pivot
    was snapped to, so this reads ``bucket_of`` directly rather than going
    through ``to_display`` -- which would also demand a ``kind`` the caller has
    no reason to carry, and hand back a price nobody wants.

    ``pmap`` of None means the export timeframe *is* the canonical one, where
    every point already sits on its own bar. None comes back for a time the
    frame no longer has, and the export counts that rather than guessing.
    """
    def format_time(time):
        if not _is_int(time):
            return None
        bucket = time if pmap is None else pmap.bucket_of.get(time)
        if bucket is None:
            return None
        # The inverse of ``config.wave_projection._epoch_seconds``, so the string
        # is the bar's own index timestamp rather than a re-interpretation of it.
        return pd.Timestamp(bucket, unit="s").strftime(PERIOD_FORMAT)

    return format_time


def export_stem(dataset_key):
    """The download file names' common stem: the dataset without its extension."""
    stem = str(dataset_key or "waves")
    for suffix in (".csv", ".CSV", ".xlsx", ".XLSX"):
        if stem.endswith(suffix):
            stem = stem[:-len(suffix)]
            break
    return stem.strip() or "waves"


def _export_key(dataset_key, base_timeframe, export_timeframe, patterns):
    """Identity of a built export: rebuild it when any of this moved.

    The patterns go in through their JSON, not their length -- moving a pivot
    changes every period without changing the count, and serving him yesterday's
    bytes would be the one failure mode worse than making him press the button
    again.
    """
    return (dataset_key, base_timeframe, export_timeframe,
            json.dumps(patterns, sort_keys=True, default=str))


def build_export(patterns, dataset_key, base_timeframe, pmap):
    """Everything the Export panel serves, built once per press of the button."""
    rows, summary = build_drm_rows(patterns, time_formatter(pmap))
    sheets = build_drm_sheets(rows)
    return {
        "sheets": sheets,
        "workbook": build_drm_workbook(sheets),
        "json": json.dumps(build_wave_json(patterns, dataset_key, base_timeframe),
                           indent=2, allow_nan=False).encode("utf-8"),
        "summary": summary,
    }


def render_export(canonical, base_df, dataset_key, base_timeframe, timeframe, pmap):
    """The Export expander: choose a resolution, build, download or use in place."""
    with st.expander("Export", expanded=False):
        timeframes = export_timeframes(base_timeframe)
        export_timeframe = st.selectbox(
            "Period timestamps at", timeframes,
            index=default_export_index(timeframes), key="_wa_export_tf")

        key = _export_key(dataset_key, base_timeframe, export_timeframe, canonical)
        stored = st.session_state.get("_wa_export")
        built = stored[1] if isinstance(stored, tuple) and stored[0] == key else None
        if built is None:
            # Stale bytes are worse than no bytes: drop them the moment anything
            # they were built from moved.
            st.session_state.pop("_wa_export", None)

        if st.button("Build export", key="_wa_export_build"):
            # The period map is the expensive part -- tens of megabytes and a
            # full pass over the base frame -- so it is built only on demand and
            # released again, and the chart's own map is borrowed when the two
            # resolutions happen to agree. Caching it in the tab's single map
            # slot instead would evict the chart's map on every rerun and rebuild
            # both, which is the opposite of what the cache is for.
            if export_timeframe == base_timeframe:
                export_pmap = None
            elif export_timeframe == timeframe:
                export_pmap = pmap
            else:
                export_pmap = period_map(base_df, export_timeframe, base_timeframe)
            built = build_export(canonical, dataset_key, base_timeframe, export_pmap)
            export_pmap = None          # drops a map built just for this press
            st.session_state["_wa_export"] = (key, built)

        if built is None:
            st.caption("Press **Build export** to generate the DRM from these "
                       "wave counts.")
            return

        for line in summary_lines(built["summary"]):
            st.caption(line)

        stem = export_stem(dataset_key)
        col_use, col_xlsx, col_json = st.columns(3)
        with col_use:
            if st.button("Use as DRM now", key="_wa_use_drm"):
                # Straight into the keys every consuming tab reads, so the
                # markings drive the Charting tab without a download and
                # re-upload round trip.
                st.session_state["drm_bullish"] = built["sheets"]["Bullish"]
                st.session_state["drm_bearish"] = built["sheets"]["Bearish"]
                st.session_state["drm"] = built["sheets"]["Bullish"]
                st.success("Date Range Manager replaced for this session")
        with col_xlsx:
            st.download_button(
                "Download .xlsx", data=built["workbook"],
                file_name=f"{stem} Date Range.xlsx",
                mime="application/vnd.openxmlformats-officedocument."
                     "spreadsheetml.sheet",
                key="_wa_export_xlsx")
        with col_json:
            st.download_button(
                "Download .json", data=built["json"],
                file_name=f"{stem} wave counts.json", mime="application/json",
                key="_wa_export_json")


# ------------------------------------------------------------------ analysis
#
# The screen the whole feature exists for: one comparison, counted across every
# pattern he has marked, expressed as a percentage. Everything that decides
# *what* is compared lives in ``strategies.wave_study``; what lives here is the
# form he fills in and the sentence that reads it back to him.

ALL_VARIATIONS = "All variations"
ANY_PARENT_WAVE = "Any"

# His own notation: "+/- 1" beside each wave count, only one of them active.
SAME_DEGREE = "same"
DEGREE_CHOICES = (SAME_DEGREE, "+1", "-1")
DEGREE_OFFSETS = {SAME_DEGREE: 0, "+1": 1, "-1": -1}

OPERATOR_CHOICES = [">", "<", "="]

# Every analysis offers both. "Single" is Phase 22's form, unchanged; "Grid
# search" keeps the same side filters and sweeps his three factors across them.
GRID_MODES = ["Single", "Grid search"]

# The leading entry that turns a Wave box into a fourth axis of the sweep.
ALL_WAVES_CHOICE = "All waves"

# (row key, column header), in the order the grid table draws them. "Pairs" is
# dropped for the Inter-Pattern grid, which discovers no pairs.
GRID_TABLE_COLUMNS = (
    ("wave_a", "Wave A"), ("wave_b", "Wave B"),
    ("field_a", "Value A"), ("field_b", "Value B"),
    ("operator", "Op"), ("relative_degree", "Rel deg"),
    ("pairs", "Pairs"), ("samples", "Samples"),
    ("true", "True"), ("false", "False"), ("ties", "Ties"),
    ("pct_true", "% True"), ("pct_false", "% False"),
)

# What a row that was never evaluated must not show. Zeros in these columns
# would read as "nothing measured yet" rather than "this question has no
# answer", which is the one confusion the note exists to prevent.
GRID_COUNT_KEYS = ("pairs", "samples", "true", "false", "ties",
                   "pct_true", "pct_false")

# One knob, wave A relative to wave B, which is how the client wrote it in his
# grid-search factors. "-1" reads wave A from the children of its pair member,
# "+1" does the same to wave B; neither ever climbs up.
RELATIVE_DEGREE_CHOICES = ["Same degree", "Wave A +1", "Wave A -1"]
RELATIVE_DEGREES = {"Same degree": 0, "Wave A +1": 1, "Wave A -1": -1}

# The three pair analyses, in the order they appear below the chart:
# (analysis key, expander title, session/widget prefix, the pairing rule in his
# own terms). The rule line is drawn above each form so the section still
# explains itself when he comes back to it in a month.
PAIR_SECTIONS = (
    ("parallel", "Parallel Analysis", "_wa_par",
     "Pairs two same-role waves -- both actionary or both reactionary -- "
     "inside one marked parent pattern."),
    ("inverse_parallel", "Inverse Parallel Analysis", "_wa_invpar",
     "Pairs an actionary wave in one parent pattern with a reactionary wave in "
     "another, separated by an actionary wave and running the same way."),
    ("adjacent", "Adjacent Analysis", "_wa_adj",
     "Pairs two actionary waves in different parent patterns that meet on the "
     "same pivot and run opposite ways."),
)


def _pattern_phrase(pattern_type, variation):
    return f"{pattern_type} ({variation or 'all variations'})"


def study_restatement(spec):
    """The line that reads a study back in words: how he checks it asked what he meant.

    Worth more than it looks. Every control on the form is a dropdown, and a
    stale one is invisible -- the restatement is the only place a study he did
    not intend shows itself before he acts on the number beside it. So it names
    the second value field, and the second pattern filter, whenever they differ
    from the first -- and stays short when they do not.
    """
    field_a = spec.get("field_a", spec.get("field"))
    field_b = spec.get("field_b", spec.get("field"))
    pattern_type, variation = spec["pattern_type"], spec["variation"]
    pattern_type_b = spec.get("pattern_type_b", pattern_type)
    variation_b = spec.get("variation_b", variation)

    if spec["offset_a"] == 1 or spec["offset_b"] == -1:
        relation = (f"Wave {spec['wave_a']}'s pattern is the parent of "
                    f"Wave {spec['wave_b']}'s")
    elif spec["offset_a"] == -1 or spec["offset_b"] == 1:
        relation = (f"Wave {spec['wave_a']}'s pattern is a child of "
                    f"Wave {spec['wave_b']}'s")
    else:
        relation = "same degree"

    patterns = _pattern_phrase(pattern_type, variation)
    if (pattern_type_b, variation_b) != (pattern_type, variation):
        patterns += f" vs {_pattern_phrase(pattern_type_b, variation_b)}"
    return (f"Wave {spec['wave_a']} {field_a} {spec['operator']} "
            f"Wave {spec['wave_b']} {field_b}, {patterns}, {relation}")


def pair_restatement(spec, title):
    """The same service for a pair analysis: the whole form in one sentence."""
    side_a = (f"{_pattern_phrase(spec['type_a'], spec['variation_a'])} as parent "
              f"wave {spec['parent_wave_a'] or ANY_PARENT_WAVE}")
    side_b = (f"{_pattern_phrase(spec['type_b'], spec['variation_b'])} as parent "
              f"wave {spec['parent_wave_b'] or ANY_PARENT_WAVE}")
    degree = {0: "same degree",
              1: "wave A one degree senior",
              -1: "wave A one degree junior"}[spec["relative_degree"]]
    return (f"Wave {spec['wave_a']} {spec['field_a']} {spec['operator']} "
            f"Wave {spec['wave_b']} {spec['field_b']} · {side_a} vs {side_b} · "
            f"{degree} · {title}")


def skip_summary(skipped):
    """What did not become a sample, in plain language.

    Always drawn, even when nothing was skipped: "90%" means one thing when
    everything he marked was counted and another when most of it was not, and
    silence would read as the first.
    """
    if not skipped:
        return "Skipped: nothing"
    return "Skipped: " + ", ".join(f"{count} {reason}"
                                   for reason, count in skipped.items())


def _dependent_key(prefix, *parts):
    """A widget key that changes when the thing the widget is populated from does.

    A variation of the old pattern type is not a variation of the new one, and a
    wave label usually does not survive the change either. Popping the old key
    out of session state does *not* clear the widget: Streamlit keeps the value
    the frontend last sent under the widget's own identity, so an Impulse's
    "Extended Impulse" comes straight back as the selection for a Zigzag, and
    the study is then run on a label the chosen patterns do not have.

    Changing the key makes it a different widget, which starts at its own first
    option. Coming back to a combination restores what was chosen there before,
    which is what a client flicking between two types would expect anyway.
    """
    return "_".join([prefix] + [str(part) for part in parts])


def _study_key(patterns):
    """Identity of the markings a stored result was counted from.

    Through the JSON, not the count: a value typed onto a wave changes what the
    percentage should be without changing how many patterns there are, and
    serving him the previous answer would be worse than making him press Run.
    """
    return json.dumps(patterns, sort_keys=True, default=str)


def _variation_box(prefix, pattern_type, label):
    """A Variation selectbox for a type, returning (display name, spec value).

    Not mandatory, in his words: with nothing named the study pools every
    variation of the type.
    """
    names = [ALL_VARIATIONS] + [n for n, _seq in PATTERN_DEFS[pattern_type]]
    variation_name = st.selectbox(label, names,
                                  key=_dependent_key(prefix, pattern_type))
    return variation_name, (None if variation_name == ALL_VARIATIONS else variation_name)


def _value_boxes(prefix, col_a, col_b):
    """The Value A / Value B pair, B locked to A's family. Returns both fields.

    His rule: a CMB is only ever compared with another CMB. Rather than letting
    him choose a comparison the engine will refuse, Value B is only ever offered
    the fields of Value A's family -- and its key hangs off that *family* rather
    than off the field, so moving Peak CMB to Terminating CMB leaves his choice
    of B alone while moving to an RSI field resets it. (A popped key would not:
    Streamlit hands the widget back the value the frontend last sent.)
    """
    with col_a:
        field_a = st.selectbox("Value A", LEG_VALUE_FIELDS, key=f"{prefix}_field_a")
    family = field_family(field_a)
    options = [field for field in LEG_VALUE_FIELDS if field_family(field) == family]
    with col_b:
        field_b = st.selectbox("Value B", options,
                               key=_dependent_key(f"{prefix}_field_b", family))
    return field_a, field_b


def _stored_study(session_key, key):
    """A stored (key, spec, result), or None once the markings moved under it.

    A percentage counted off the previous list is not an answer about this one,
    so a stale entry is dropped rather than redrawn.
    """
    stored = st.session_state.get(session_key)
    if isinstance(stored, tuple) and len(stored) == 3 and stored[0] == key:
        return stored
    st.session_state.pop(session_key, None)
    return None


def _render_result(result, restatement, operator):
    """The four lines every study answers in: headline, restatement, counts, skips."""
    # Only a pair study carries a pair count, and it leads: it is the
    # denominator context that makes the percentage beside it readable.
    pairs = f"{result['pairs']} pairs · " if "pairs" in result else ""
    st.markdown(f"### {pairs}{result['samples']} samples · "
                f"{result['pct_true']:.1f}% true · "
                f"{result['pct_false']:.1f}% false")
    st.caption(restatement)

    counts = f"{result['true']} true · {result['false']} false"
    if result["ties"]:
        # Surfaced only when they exist. Under ``=`` they *are* the true count,
        # which is the one operator where he wants to see them.
        verdict = "true" if operator == "=" else "false"
        counts += f" · {result['ties']} tie(s), counted {verdict}"
    st.caption(counts)
    st.caption(skip_summary(result["skipped"]))


# ---------------------------------------------------------------- the grid
#
# The same four analyses, asked every way at once. The manual form answers the
# question he already has; the grid is what he trawls with -- his three factors
# (operator, value pair, relative degree) crossed, and the wave labels too when
# he asks for them.
#
# Everything below is drawing and reading a table. Which combinations exist and
# what each one counts is ``strategies.wave_study.run_grid`` and nothing here.


def _grid_mode(prefix):
    """True when this analysis is showing its grid rather than its manual form.

    One radio per analysis, and its state is all that separates the two: the
    side filters are the same filters, so flicking between the modes keeps them.
    """
    key = f"{prefix}_mode"
    # Mirrored into a plain session key, and the mirror decides the default.
    # Streamlit discards a widget's state on any run that does not draw it, and
    # applying a chart event reruns the tab before it reaches these expanders --
    # so a marking change would drop the mode back to Single while the radio,
    # which keeps whatever the frontend last sent, went on showing "Grid
    # search". Every other control in these forms resets for the same reason;
    # this one had to be held because it decides which form is drawn at all.
    held = st.session_state.get(f"{key}_held", GRID_MODES[0])
    if key not in st.session_state and held in GRID_MODES:
        st.session_state[key] = held
    chosen = st.radio("Mode", GRID_MODES, horizontal=True, key=key)
    st.session_state[f"{key}_held"] = chosen
    return chosen == GRID_MODES[1]


def _grid_side(prefix, side, with_parent):
    """One side of a grid form: the filters, and a Wave box that offers All waves.

    The operator, the value and the degree are absent by design -- those are the
    grid. What is left is exactly what the sweep does *not* enumerate, which is
    why it is the manual form's filters and nothing else.

    Its own widget keys, not the manual form's: the Wave box offers an extra
    option the manual one must never carry, and a shared key would hand
    ``All waves`` to a single study the engine would then reject.
    """
    label = side.upper()
    boxes = st.columns(4 if with_parent else 3)
    with boxes[0]:
        pattern_type = st.selectbox(f"Pattern {label}", list(PATTERN_DEFS),
                                    key=f"{prefix}_grid_type_{side}")
    with boxes[1]:
        variation_name, variation = _variation_box(
            f"{prefix}_grid_variation_{side}", pattern_type, f"Variation {label}")

    parent_wave = ANY_PARENT_WAVE
    if with_parent:
        with boxes[2]:
            parent_wave = st.selectbox(
                f"Parent wave {label}", [ANY_PARENT_WAVE] + list(PARENT_WAVE_LABELS),
                key=f"{prefix}_grid_parent_{side}")
    with boxes[-1]:
        wave = st.selectbox(
            f"Wave {label}", [ALL_WAVES_CHOICE] + wave_labels(pattern_type, variation),
            key=_dependent_key(f"{prefix}_grid_wave_{side}", pattern_type,
                               variation_name))

    return {f"type_{side}": pattern_type,
            f"variation_{side}": variation,
            f"parent_wave_{side}": (None if parent_wave == ANY_PARENT_WAVE
                                    else parent_wave),
            f"wave_{side}": (ALL_WAVES if wave == ALL_WAVES_CHOICE else wave)}


def grid_size(spec):
    """(rows the sweep will produce, labels on side A, labels on side B).

    Quoted back to him before he presses Run: an impulse against an impulse with
    both sides on All waves is 4,050 rows, and a number that large is worth
    seeing before it is computed rather than after.
    """
    counts = []
    for side in ("a", "b"):
        labels = wave_labels(spec[f"type_{side}"], spec[f"variation_{side}"])
        counts.append(len(labels) if spec[f"wave_{side}"] == ALL_WAVES else 1)
    return counts[0] * counts[1] * GRID_COMBINATIONS, counts[0], counts[1]


def grid_caption(spec):
    """The sweep, multiplied out in words, before anything is counted."""
    sweep, count_a, count_b = grid_size(spec)
    caption = (f"{sweep} combinations — {len(OPERATORS)} operators × "
               f"{len(GRID_VALUE_PAIRS)} value pairs × "
               f"{len(GRID_DEGREES)} relative degrees")
    if count_a * count_b > 1:
        caption += f" × {count_a} Wave A × {count_b} Wave B labels"
    return caption


def grid_frame(rows):
    """The sweep as the table he reads: named columns, notes in place of counts.

    Two things the raw rows do not decide. ``Pairs`` is dropped where the
    analysis discovers none, rather than drawn as a column of blanks. And a row
    carrying a note shows the note instead of its counts: it was never
    evaluated, so zeros there would read as "nothing measured yet" instead of
    "that question has no answer".
    """
    if not rows:
        return pd.DataFrame(columns=[header for _key, header in GRID_TABLE_COLUMNS])

    has_pairs = any("pairs" in row for row in rows)
    columns = [(key, header) for key, header in GRID_TABLE_COLUMNS
               if key != "pairs" or has_pairs]
    noted = any(row.get("note") for row in rows)

    records = []
    for row in rows:
        note = row.get("note")
        record = {}
        for key, header in columns:
            if note and key in GRID_COUNT_KEYS:
                record[header] = None
            elif key in ("pct_true", "pct_false"):
                record[header] = round(row[key], 1)
            else:
                record[header] = row.get(key)
        if noted:
            record["Note"] = note or ""
        records.append(record)

    frame = pd.DataFrame(records)
    # Strongest claims on top, and the fuller sample first where two agree.
    # A stable sort, so anything still tied stays in enumeration order rather
    # than shuffling between runs of the same sweep.
    return frame.sort_values(["% True", "Samples"], ascending=False,
                             kind="mergesort")


def grid_view(frame, hide_empty, min_true):
    """The rows he chose to look at: the table's two view filters, composed.

    Neither filter changes what was computed. The summary's denominators and
    the CSV stay whole below, so a hidden row is still a counted one.

    The threshold is applied only where it is positive, and that is the whole
    subtlety of it: ``% True`` is blank on a noted row, ``NaN >= 0`` is False,
    and a comparison left switched on at zero would therefore delete every note
    from the table without ever being asked to.
    """
    shown = frame
    if hide_empty:
        shown = shown[shown["Samples"] > 0]
    if min_true > 0:
        shown = shown[shown["% True"] >= min_true]
    return shown


def grid_skips(rows):
    """The sweep's skip reasons, counted once per evaluation rather than per row.

    The fifty-four rows that share a wave pair and a degree share its skips --
    neither the value pair nor the operator changes which waves were dropped --
    so summing every row would multiply the same unmeasured waves by fifty-four
    and turn the one honest denominator on the screen into a made-up number.
    """
    totals = {}
    seen = set()
    for row in rows:
        key = (row["wave_a"], row["wave_b"], row["relative_degree"])
        if key in seen:
            continue
        seen.add(key)
        for reason, count in row["skipped"].items():
            totals[reason] = totals.get(reason, 0) + count
    return {reason: totals[reason] for reason in SKIP_ORDER if totals.get(reason)}


def grid_summary(rows, analysis):
    """The denominator context under the table: what was swept, what was counted.

    A table of percentages with no denominators is exactly what this feature
    must not hand him, and a sweep hides its denominators better than a single
    study does -- the strongest-looking row on screen may be one sample out of
    six hundred combinations that found nothing.
    """
    with_samples = sum(1 for row in rows if row["samples"])
    parts = [f"{len(rows)} combinations · {with_samples} with samples"]
    if analysis != INTER:
        # Said outright, because the table invites the opposite reading: a side
        # shifted a degree reads every matching child of a pair member, so one
        # pair can yield several samples.
        parts.append("a ±1 combination reads every matching child of a pair, so "
                     "Samples can exceed Pairs")
    parts.append(skip_summary(grid_skips(rows)))
    return " · ".join(parts)


def render_grid(canonical, analysis, prefix):
    """The grid mode of one analysis: the filters, the sweep, the table.

    Four separate grids, one per analysis, which is the client's own line:
    "They serve 4 different purposes so I do not want a single Grid Search
    combining all 4." So this draws one of them and never pools them.
    """
    spec = {"analysis": analysis}
    spec.update(_grid_side(prefix, "a", analysis != INTER))
    spec.update(_grid_side(prefix, "b", analysis != INTER))

    st.caption(grid_caption(spec))
    # As everywhere in this tab: nothing is counted until he asks, because
    # Streamlit reruns the whole app on every widget touch anywhere in it.
    run = st.button("Run grid", key=f"{prefix}_grid_run")

    session_key = f"_wa_grid_{analysis}_result"
    key = _study_key(canonical)
    stored = _stored_study(session_key, key)

    if run:
        try:
            rows = run_grid(canonical, spec)
        except ValueError as error:
            st.error(str(error))
            return
        stored = (key, spec, rows)
        st.session_state[session_key] = stored

    if stored is None:
        st.caption("Press **Run grid** to sweep every combination across every "
                   "marked pattern.")
        return

    _key, _ran_spec, rows = stored
    frame = grid_frame(rows)
    # The two view filters as one toolbar: both narrow what is on screen, and
    # neither narrows what was counted.
    col_hide, col_true = st.columns([3, 1])
    with col_hide:
        hide = st.checkbox("Hide combinations with no samples", value=True,
                           key=f"{prefix}_grid_hide")
    with col_true:
        min_true = st.number_input(
            "Min % True", min_value=0, max_value=100, value=0, step=5,
            key=f"{prefix}_grid_min_true",
            help="Show only combinations whose % True is at least this. "
                 "0 shows all.")
    shown = grid_view(frame, hide, min_true)
    st.dataframe(shown, hide_index=True)
    st.caption(grid_summary(rows, analysis))
    if len(shown) != len(frame):
        # Said under the denominators, not inside them: the filters change what
        # he is looking at, never what the sweep found.
        st.caption(f"Showing {len(shown)} of {len(frame)} combinations")
    # The whole sweep, including everything the checkbox is hiding: a row with
    # no samples is a fact about his markings, and one he may want to sort
    # through outside the app.
    st.download_button("Download CSV", frame.to_csv(index=False),
                       file_name=f"wave_grid_{analysis}.csv", mime="text/csv",
                       key=f"{prefix}_grid_csv")


def render_study(canonical):
    """The Inter-Pattern Analysis expander: configure one comparison, run it.

    Named for what it actually does: it compares wave counts within the same
    parent pattern. The Parallel, Inverse Parallel and Adjacent analyses beside
    it compare counts across two *different* marked patterns.
    """
    with st.expander("Inter-Pattern Analysis", expanded=False):
        if _grid_mode("_wa_study"):
            render_grid(canonical, INTER, "_wa_study")
            return

        col_type, col_variation = st.columns(2)
        with col_type:
            pattern_type = st.selectbox("Pattern", list(PATTERN_DEFS),
                                        key="_wa_study_type")
        with col_variation:
            variation_name, variation = _variation_box(
                "_wa_study_variation", pattern_type, "Variation")

        # Whether side B needs a pattern filter of its own is decided from what
        # the two degree boxes hold, before either is drawn: they sit further
        # down the form, but a widget's value is already in session state by the
        # time the rerun it caused runs. At the same degree the two sides are
        # one pattern, so a second filter there is not merely redundant -- the
        # engine refuses it -- and the boxes are not drawn at all.
        cross = (st.session_state.get("_wa_study_deg_a", SAME_DEGREE) != SAME_DEGREE
                 or st.session_state.get("_wa_study_deg_b", SAME_DEGREE) != SAME_DEGREE)
        if cross:
            col_type_b, col_variation_b = st.columns(2)
            with col_type_b:
                pattern_type_b = st.selectbox("Pattern B", list(PATTERN_DEFS),
                                              key="_wa_study_type_b")
            with col_variation_b:
                variation_name_b, variation_b = _variation_box(
                    "_wa_study_variation_b", pattern_type_b, "Variation B")
        else:
            pattern_type_b = pattern_type
            variation_name_b, variation_b = variation_name, variation

        labels = wave_labels(pattern_type, variation)
        labels_b = wave_labels(pattern_type_b, variation_b)

        col_a, col_deg_a, col_b, col_deg_b = st.columns([3, 2, 3, 2])
        with col_a:
            wave_a = st.selectbox(
                "Wave A", labels,
                key=_dependent_key("_wa_study_wave_a", pattern_type, variation_name))
        # Mutually exclusive, and decided before either box is drawn: the value
        # a disabled selectbox displays is whatever sits in session state, so
        # the locked side is forced back to "same" rather than merely greyed out
        # over a stale +1 that would still reach the spec.
        lock_a = st.session_state.get("_wa_study_deg_b", SAME_DEGREE) != SAME_DEGREE
        if lock_a:
            st.session_state["_wa_study_deg_a"] = SAME_DEGREE
        with col_deg_a:
            degree_a = st.selectbox("Degree", DEGREE_CHOICES, key="_wa_study_deg_a",
                                    disabled=lock_a)
        with col_b:
            wave_b = st.selectbox(
                "Wave B", labels_b,
                key=_dependent_key("_wa_study_wave_b", pattern_type_b,
                                   variation_name_b))
        lock_b = degree_a != SAME_DEGREE
        if lock_b:
            st.session_state["_wa_study_deg_b"] = SAME_DEGREE
        with col_deg_b:
            degree_b = st.selectbox("Degree", DEGREE_CHOICES, key="_wa_study_deg_b",
                                    disabled=lock_b)

        col_field_a, col_field_b, col_op, col_run = st.columns([3, 3, 2, 2])
        field_a, field_b = _value_boxes("_wa_study", col_field_a, col_field_b)
        with col_op:
            operator = st.selectbox("Operator", OPERATOR_CHOICES, key="_wa_study_op")
        with col_run:
            # Streamlit reruns the whole app on every widget touch anywhere, and
            # a study walks every pattern and every parent lookup. It runs when
            # he asks for it, not when he is still choosing.
            st.markdown("&nbsp;", unsafe_allow_html=True)
            run = st.button("Run", key="_wa_study_run")

        spec = {
            "pattern_type": pattern_type,
            "variation": variation,
            "pattern_type_b": pattern_type_b,
            "variation_b": variation_b,
            "wave_a": wave_a,
            "offset_a": DEGREE_OFFSETS[degree_a],
            "wave_b": wave_b,
            "offset_b": DEGREE_OFFSETS[degree_b],
            "field_a": field_a,
            "field_b": field_b,
            "operator": operator,
        }

        key = _study_key(canonical)
        stored = _stored_study("_wa_study_result", key)

        if run:
            try:
                result = run_study(canonical, spec)
            except ValueError as error:
                st.error(str(error))
                return
            stored = (key, spec, result)
            st.session_state["_wa_study_result"] = stored

        if stored is None:
            st.caption("Press **Run** to count this comparison across every "
                       "marked pattern.")
            return

        _key, ran_spec, result = stored
        _render_result(result, study_restatement(ran_spec), ran_spec["operator"])


def _side_controls(prefix, side):
    """One side of a pair analysis: Pattern, Variation, Parent wave, Wave.

    Parent wave is the label the pattern occupies in *its own* parent -- his
    "Parent wave (1)" -- and defaults to Any, which pools every pair the
    discovery found rather than making him name a role before he can ask
    anything at all.
    """
    label = side.upper()
    col_type, col_variation, col_parent, col_wave = st.columns(4)
    with col_type:
        pattern_type = st.selectbox(f"Pattern {label}", list(PATTERN_DEFS),
                                    key=f"{prefix}_type_{side}")
    with col_variation:
        variation_name, variation = _variation_box(
            f"{prefix}_variation_{side}", pattern_type, f"Variation {label}")
    with col_parent:
        parent_wave = st.selectbox(
            f"Parent wave {label}", [ANY_PARENT_WAVE] + list(PARENT_WAVE_LABELS),
            key=f"{prefix}_parent_{side}")
    with col_wave:
        wave = st.selectbox(
            f"Wave {label}", wave_labels(pattern_type, variation),
            key=_dependent_key(f"{prefix}_wave_{side}", pattern_type, variation_name))

    return {f"type_{side}": pattern_type,
            f"variation_{side}": variation,
            f"parent_wave_{side}": (None if parent_wave == ANY_PARENT_WAVE
                                    else parent_wave),
            f"wave_{side}": wave}


def render_pair_study(canonical, analysis, title, prefix, rule):
    """One of the three pair-analysis expanders: configure one comparison, run it.

    All three share this function because they differ in exactly one thing --
    which discovery backs them -- and three near-identical copies of a form this
    size would drift apart within a phase.
    """
    with st.expander(title, expanded=False):
        st.caption(rule)
        if _grid_mode(prefix):
            render_grid(canonical, analysis, prefix)
            return

        spec = {"analysis": analysis}
        spec.update(_side_controls(prefix, "a"))
        spec.update(_side_controls(prefix, "b"))

        col_degree, col_field_a, col_field_b, col_op, col_run = st.columns(
            [3, 3, 3, 2, 2])
        with col_degree:
            degree_name = st.selectbox("Relative degree", RELATIVE_DEGREE_CHOICES,
                                       key=f"{prefix}_degree")
        field_a, field_b = _value_boxes(prefix, col_field_a, col_field_b)
        with col_op:
            operator = st.selectbox("Operator", OPERATOR_CHOICES, key=f"{prefix}_op")
        with col_run:
            # As with the study above: nothing is counted until he asks, because
            # the tab reruns on every widget touch anywhere in the app.
            st.markdown("&nbsp;", unsafe_allow_html=True)
            run = st.button("Run", key=f"{prefix}_run")

        spec.update({"relative_degree": RELATIVE_DEGREES[degree_name],
                     "field_a": field_a, "field_b": field_b,
                     "operator": operator})

        session_key = f"{prefix}_result"
        key = _study_key(canonical)
        stored = _stored_study(session_key, key)

        if run:
            try:
                result = run_pair_study(canonical, spec)
            except ValueError as error:
                st.error(str(error))
                return
            stored = (key, spec, result)
            st.session_state[session_key] = stored

        if stored is None:
            st.caption("Press **Run** to count this comparison across every "
                       "pair this analysis finds.")
            return

        _key, ran_spec, result = stored
        _render_result(result, pair_restatement(ran_spec, title),
                       ran_spec["operator"])


def render_wave_analysis_tab(sidebar_config):
    """Render the Wave Analysis tab"""

    df_raw = st.session_state.get("df_raw")
    if df_raw is None or df_raw.empty:
        st.info("Please upload OHLC data in the Charting tab first.")
        return

    base_timeframe = st.session_state.get("base_timeframe", "15m")
    timeframe = st.session_state.get("_agg_timeframe", base_timeframe)

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

    # Identity of the *raw* upload at this aggregation. Everything derived from
    # it -- the deduped base frame, the display bars, the period map -- is
    # cached under this, because dedup and resample are both deterministic and
    # this tab is re-rendered on every widget touch anywhere in the app.
    frame_key = build_fingerprint(df_raw, timeframe, dataset_key)
    base_df = _base_frame(df_raw, dataset_key)
    display_df = _display_frame(base_df, timeframe, base_timeframe, frame_key)
    pmap = _display_map(base_df, timeframe, base_timeframe, frame_key)

    # Rebuild the payload only when the data actually changed -- this tab runs on
    # every rerun like all the others, so it must be cheap when idle.
    fingerprint = build_fingerprint(display_df, timeframe, dataset_key)
    payload = st.session_state.get("_wa_payload")
    if payload is None or payload.get("fingerprint") != fingerprint:
        payload = build_wave_payload(display_df, timeframe, dataset_key)
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
    # the reload also restores markings after that sweep. Colours *and* degrees
    # are always recomputed -- a persisted ``degree`` is no more trustworthy
    # than a persisted ``color``, and a file written before the parent/child
    # relation existed can hold a child at the degree it was drawn with.
    if ("_wa_patterns" not in st.session_state
            or st.session_state.get("_wa_dataset_key") != dataset_key):
        canonical, load_note = _load_canonical(dataset_key, base_df, base_timeframe)
        st.session_state["_wa_patterns"] = settle(canonical)
        st.session_state["_wa_dataset_key"] = dataset_key
        st.session_state["_wa_load_note"] = load_note
        # History belongs to the dataset it was recorded on -- undoing into
        # another instrument's pattern list would be worse than no undo at all.
        st.session_state["_wa_undo"] = []

    canonical = st.session_state["_wa_patterns"]

    # What the frontend gets is the projection, never the canonical list. It is
    # rebuilt every render and stored nowhere: colours and degrees were decided
    # canonically and ride along untouched, so ``validate_patterns`` must *not*
    # run again here -- two patterns legally chained at 15m can look overlapped
    # at 1D, and repainting them red for that would be a lie about the count.
    patterns = canonical if pmap is None else project_patterns(canonical, pmap, DISPLAY)

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
    updated, new_seq, changed = apply_event_batch(canonical, value, last_seq, pmap)
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
                documents = load_wave_documents()
                documents[dataset_key] = {"schema": 2,
                                          "base_timeframe": base_timeframe,
                                          "patterns": updated}
                save_wave_documents(documents)
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

    st.caption(wave_caption(len(canonical), len(patterns), timeframe, dataset_key,
                            st.session_state.get("_wa_load_note")))

    # Above the export: these are the questions the markings were made to
    # answer, and the export is the older, coarser way of getting at the same
    # list. Four separate sections, in the client's own order -- the
    # within-a-pattern study first, then the three that pair two patterns.
    render_study(canonical)
    for analysis, title, prefix, rule in PAIR_SECTIONS:
        render_pair_study(canonical, analysis, title, prefix, rule)
    render_export(canonical, base_df, dataset_key, base_timeframe, timeframe, pmap)

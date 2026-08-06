"""
Tests for the canonical wave model: one pattern list per dataset, stored at the
base timeframe, projected onto whatever aggregation is on screen.

Frames are small and their extremes are placed by hand, so every expected
answer can be read off the construction rather than recomputed by the test.
"""

import copy
import json

import pandas as pd
import pytest

from config.wave_analysis import child_leg_index, find_parent, settle
from config.wave_projection import (DISPLAY, period_map, pivot_magnets,
                                    project_patterns, refine_event)
from data.loader import resample_ohlc
from strategies.wave_marking_manager import load_wave_documents, save_wave_documents
from ui import wave_analysis_tab
from ui.wave_analysis_tab import _load_canonical, apply_event_batch, wave_caption

BARS_PER_DAY = 96                       # 15m bars
START = "2024-01-01 00:00"

# Where each day's extremes sit, in bars from the day's first. Day 0's high is
# bar 20 (05:00), its low bar 60 (15:00) -- far enough apart that a pattern can
# alternate high/low inside a single day, which is what makes it collapse at 1D.
HIGH_BAR = 20
LOW_BAR = 60


def make_frame(days=5, start=START):
    """15m bars whose daily high and low sit on one deliberate bar each.

    Every other bar carries the same flat high and low, so within any *hour*
    that holds no extreme all four bars tie -- which is what the tie-break has
    to resolve, and it resolves to the last of them.
    """
    count = days * BARS_PER_DAY
    index = pd.date_range(start, periods=count, freq="15min")
    highs = [100.0] * count
    lows = [90.0] * count
    for day in range(days):
        highs[day * BARS_PER_DAY + HIGH_BAR] = 200.0 + day
        lows[day * BARS_PER_DAY + LOW_BAR] = 10.0 - day
    return pd.DataFrame({"open": highs, "high": highs, "low": lows,
                         "close": highs, "volume": [1.0] * count}, index=index)


def bar_time(df, position):
    return int(df.index[position].timestamp())


def times_of(df):
    return [int(ts.timestamp()) for ts in df.index]


def day_bar(df, day):
    """The display time of a whole day, as the 1D resample produces it."""
    return times_of(resample_ohlc(df, "1D", "15m"))[day]


def point(time, price, kind):
    return {"time": time, "price": price, "kind": kind}


def pattern_of(points, pattern_id="p1", degree="Subminuette", color="yellow"):
    """A structurally valid Zigzag over the four points given."""
    return {
        "id": pattern_id,
        "pattern_type": "Zigzag",
        "variation": "Zigzag",
        "degree": degree,
        "color": color,
        "points": [dict(p) for p in points],
    }


def daily_points(df, days, kinds):
    """The points the frontend sends for clicks on those daily bars."""
    highs = resample_ohlc(df, "1D", "15m")["high"]
    lows = resample_ohlc(df, "1D", "15m")["low"]
    return [point(day_bar(df, day), float(highs.iloc[day] if kind == "high"
                                          else lows.iloc[day]), kind)
            for day, kind in zip(days, kinds)]


def completed(pattern):
    return {"type": "pattern_completed", "pattern": pattern}


def move(pattern_id, index, time, price, kind):
    return {"type": "move_point", "id": pattern_id, "point_index": index,
            "time": time, "price": price, "kind": kind}


def batch_from(last_seq, *events):
    numbered = [dict(event, eseq=last_seq + i + 1) for i, event in enumerate(events)]
    return {"seq": last_seq + len(numbered), "events": numbered}


def batch(*events):
    return batch_from(0, *events)


def document(patterns, base_timeframe="15m"):
    return {"schema": 2, "base_timeframe": base_timeframe, "patterns": patterns}


@pytest.fixture(autouse=True)
def session(monkeypatch):
    """A plain dict standing in for Streamlit's session state.

    Undo history and the save-error flag are session state, and there is none
    outside a ``streamlit run`` script thread.
    """
    state = {}
    monkeypatch.setattr(wave_analysis_tab.st, "session_state", state)
    return state


# ------------------------------------------------------------- 1. refine_event


def test_refine_pattern_completed_takes_times_and_prices_from_the_base_bars():
    df = make_frame(days=4)
    pmap = period_map(df, "1D", "15m")

    # The frontend drew four clicks on four daily bars. The prices here are
    # deliberately wrong: the refined ones must come from the map, not the event.
    drawn = pattern_of([point(day_bar(df, 0), 1.0, "high"),
                        point(day_bar(df, 1), 2.0, "low"),
                        point(day_bar(df, 2), 3.0, "high"),
                        point(day_bar(df, 3), 4.0, "low")])
    event = completed(drawn)

    refined = refine_event(event, pmap)
    points = refined["pattern"]["points"]

    assert [p["time"] for p in points] == [
        bar_time(df, HIGH_BAR),
        bar_time(df, BARS_PER_DAY + LOW_BAR),
        bar_time(df, 2 * BARS_PER_DAY + HIGH_BAR),
        bar_time(df, 3 * BARS_PER_DAY + LOW_BAR),
    ]
    assert [p["price"] for p in points] == [200.0, 9.0, 202.0, 7.0]
    assert [p["kind"] for p in points] == ["high", "low", "high", "low"]
    # Everything that is not a coordinate rides along untouched.
    assert refined["pattern"]["id"] == "p1"
    assert refined["pattern"]["degree"] == "Subminuette"


def test_refine_move_point_lands_on_the_base_bar_carrying_the_periods_extreme():
    df = make_frame(days=3)
    pmap = period_map(df, "1D", "15m")

    refined = refine_event(move("p1", 2, day_bar(df, 2), 12345.0, "low"), pmap)

    assert refined["time"] == bar_time(df, 2 * BARS_PER_DAY + LOW_BAR)
    assert refined["price"] == 8.0                  # the day's low, not the event's
    assert refined["kind"] == "low"
    assert refined["point_index"] == 2 and refined["id"] == "p1"


def test_refine_is_pure():
    df = make_frame(days=2)
    pmap = period_map(df, "1D", "15m")
    event = completed(pattern_of(daily_points(df, [0, 0, 1, 1],
                                              ["high", "low", "high", "low"])))
    snapshot = copy.deepcopy(event)

    refine_event(event, pmap)

    assert event == snapshot


def test_refine_refuses_a_point_that_is_on_no_display_bar():
    df = make_frame(days=2)
    pmap = period_map(df, "1D", "15m")

    # A 15m bar time is not a 1D bar time (except the day's first, so pick one
    # in the middle of a day).
    stray = bar_time(df, 37)
    assert refine_event(move("p1", 0, stray, 100.0, "high"), pmap) is None

    drawn = pattern_of([point(day_bar(df, 0), 200.0, "high"),
                        point(stray, 100.0, "low"),
                        point(day_bar(df, 1), 201.0, "high"),
                        point(day_bar(df, 1), 9.0, "low")])
    assert refine_event(completed(drawn), pmap) is None


@pytest.mark.parametrize("event", [
    {"type": "delete_pattern", "id": "p1"},
    {"type": "shift_degree", "id": "p1", "delta": 1},
    {"type": "undo"},
    {"type": "something_else", "id": "p1"},
])
def test_refine_passes_id_addressed_events_through_untouched(event):
    df = make_frame(days=2)
    pmap = period_map(df, "1D", "15m")
    assert refine_event(event, pmap) == event


@pytest.mark.parametrize("event", [
    {"type": "pattern_completed", "pattern": {"points": [{"time": 7, "price": 1.0,
                                                          "kind": "high"}]}},
    {"type": "move_point", "id": "p1", "point_index": 0, "time": 7, "price": 1.0,
     "kind": "high"},
    {"type": "delete_pattern", "id": "p1"},
    {"type": "undo"},
])
def test_a_pmap_of_none_is_the_identity_for_every_event_type(event):
    assert refine_event(event, None) is event


# --------------------------------------------------------- 2. refused-but-acked


def test_an_unrefinable_event_advances_the_seq_and_leaves_the_list_alone():
    # The infinite-replay guard: the frontend prunes its outbox against the ack
    # and replays anything unacked forever, so an event we refuse must still be
    # acked. The optimistic overlay snaps back on the next render instead.
    df = make_frame(days=2)
    pmap = period_map(df, "1D", "15m")
    before = [pattern_of([point(bar_time(df, 0), 100.0, "high"),
                          point(bar_time(df, 1), 90.0, "low"),
                          point(bar_time(df, 2), 100.0, "high"),
                          point(bar_time(df, 3), 90.0, "low")])]

    stray = bar_time(df, 37)                        # on no 1D bar
    state, seq, changed = apply_event_batch(before, batch(move("p1", 1, stray,
                                                              100.0, "high")), 0, pmap)

    assert state is before
    assert seq == 1
    assert changed is False


def test_a_refused_event_does_not_stop_the_rest_of_its_batch():
    df = make_frame(days=4)
    pmap = period_map(df, "1D", "15m")
    good = pattern_of(daily_points(df, [0, 1, 2, 3], ["high", "low", "high", "low"]))

    state, seq, changed = apply_event_batch(
        [], batch(move("nope", 0, bar_time(df, 37), 1.0, "high"), completed(good)),
        0, pmap)

    assert seq == 2
    assert changed is True
    assert [p["id"] for p in state] == ["p1"]


# ---------------------------------------------------------- 3. display round trip


def test_a_pattern_completed_at_1d_projects_back_to_the_points_that_were_drawn():
    df = make_frame(days=4)
    pmap = period_map(df, "1D", "15m")
    drawn = daily_points(df, [0, 1, 2, 3], ["high", "low", "high", "low"])

    stored, _seq, _changed = apply_event_batch(
        [], batch(completed(pattern_of(drawn))), 0, pmap)

    # Stored canonically at 15m...
    assert [p["time"] for p in stored[0]["points"]] != [p["time"] for p in drawn]
    assert all(t in pmap.bucket_of for t in (p["time"] for p in stored[0]["points"]))

    # ...and identical again on the chart it was drawn on.
    back = project_patterns(stored, pmap, DISPLAY)
    assert back[0]["points"] == drawn


# ----------------------------------------------------- 4. cross-aggregation view


@pytest.mark.parametrize("timeframe", ["1H", "4H", "1D"])
def test_a_marking_made_at_15m_shows_at_every_coarser_period_extreme(timeframe):
    df = make_frame(days=5)
    # Four pivots on four different days, none of them its day's extreme.
    canonical = [pattern_of([point(bar_time(df, 5), 100.0, "high"),
                             point(bar_time(df, BARS_PER_DAY + 5), 90.0, "low"),
                             point(bar_time(df, 2 * BARS_PER_DAY + 5), 100.0, "high"),
                             point(bar_time(df, 3 * BARS_PER_DAY + 5), 90.0, "low")])]

    pmap = period_map(df, timeframe, "15m")
    shown = project_patterns(canonical, pmap, DISPLAY)
    assert len(shown) == 1

    periods = resample_ohlc(df, timeframe, "15m")
    for moved in shown[0]["points"]:
        row = periods.loc[pd.Timestamp(moved["time"], unit="s")]
        assert moved["price"] == float(row["high" if moved["kind"] == "high" else "low"])
        assert moved["time"] in pmap.display_times


def test_a_marking_made_at_1d_sits_on_the_15m_bars_carrying_those_extremes():
    df = make_frame(days=4)
    pmap = period_map(df, "1D", "15m")
    drawn = daily_points(df, [0, 1, 2, 3], ["high", "low", "high", "low"])

    canonical, _seq, _changed = apply_event_batch(
        [], batch(completed(pattern_of(drawn))), 0, pmap)

    # At 15m there is no map at all -- canonical *is* display.
    assert [p["time"] for p in canonical[0]["points"]] == [
        bar_time(df, HIGH_BAR),
        bar_time(df, BARS_PER_DAY + LOW_BAR),
        bar_time(df, 2 * BARS_PER_DAY + HIGH_BAR),
        bar_time(df, 3 * BARS_PER_DAY + LOW_BAR),
    ]


def test_a_tied_extreme_refines_to_the_latest_bar_that_carries_it():
    # The client's rule, verbatim: "if there are 2 periods in the smaller time
    # frames with the same high or low then the period that occurred last will
    # be the one chosen". Every bar of an hour holding no extreme ties at 100.0.
    df = make_frame(days=1)
    pmap = period_map(df, "1H", "15m")
    hours = times_of(resample_ohlc(df, "1H", "15m"))

    # Hour 0 covers bars 0..3, none of which is the day's high bar (20).
    refined = refine_event(move("p1", 0, hours[0], 100.0, "high"), pmap)
    assert refined["time"] == bar_time(df, 3)
    assert refined["price"] == 100.0

    refined = refine_event(move("p1", 0, hours[0], 90.0, "low"), pmap)
    assert refined["time"] == bar_time(df, 3)


# --------------------------------------------------------- 5. migration on load


def legacy_file(tmp_path, entries):
    path = str(tmp_path / "saved_wave_markings.json")
    with open(path, "w") as handle:
        json.dump(entries, handle)
    return path


def test_a_v1_document_migrates_on_load_and_is_written_back_once(tmp_path):
    df = make_frame(days=4)
    drawn = daily_points(df, [0, 1, 2, 3], ["high", "low", "high", "low"])
    path = legacy_file(tmp_path, {"gold.csv": {"1D": [pattern_of(drawn, "daily")]}})

    patterns, note = _load_canonical("gold.csv", df, "15m", path)

    assert [p["id"] for p in patterns] == ["daily"]
    assert [p["time"] for p in patterns[0]["points"]] == [
        bar_time(df, HIGH_BAR),
        bar_time(df, BARS_PER_DAY + LOW_BAR),
        bar_time(df, 2 * BARS_PER_DAY + HIGH_BAR),
        bar_time(df, 3 * BARS_PER_DAY + LOW_BAR),
    ]
    assert note == ""                               # nothing was lost

    # The file itself is now v2, so the next load has nothing left to migrate.
    with open(path) as handle:
        on_disk = json.load(handle)
    assert on_disk["gold.csv"]["schema"] == 2
    assert on_disk["gold.csv"]["base_timeframe"] == "15m"
    assert on_disk["gold.csv"]["patterns"] == patterns

    again, again_note = _load_canonical("gold.csv", df, "15m", path)
    assert again == patterns
    assert again_note == ""


def test_migration_leaves_the_other_datasets_documents_intact(tmp_path):
    df = make_frame(days=4)
    other = pattern_of(daily_points(df, [0, 1, 2, 3], ["high", "low", "high", "low"]),
                       "other")
    path = legacy_file(tmp_path, {
        "gold.csv": {"1D": [pattern_of(daily_points(df, [0, 1, 2, 3],
                                                    ["high", "low", "high", "low"]))]},
        "silver.csv": {"1D": [other]},
    })

    _load_canonical("gold.csv", df, "15m", path)

    # silver was written back exactly as it came out -- still unmigrated, because
    # only the dataset on screen has bars to migrate against.
    reloaded = load_wave_documents(path)
    assert reloaded["silver.csv"]["schema"] == 1
    assert reloaded["silver.csv"]["by_timeframe"]["1D"] == [other]


def test_a_migration_that_loses_a_pattern_says_so(tmp_path):
    df = make_frame(days=4)
    # "5m" is not a timeframe this app can resample to, so it gets no projector.
    path = legacy_file(tmp_path, {"gold.csv": {
        "5m": [pattern_of([point(bar_time(df, 0), 100.0, "high"),
                           point(bar_time(df, 1), 90.0, "low"),
                           point(bar_time(df, 2), 100.0, "high"),
                           point(bar_time(df, 3), 90.0, "low")], "orphan")],
    }})

    patterns, note = _load_canonical("gold.csv", df, "15m", path)

    assert patterns == []
    assert note == "1 pattern(s) dropped in migration"


def test_a_v2_document_stored_at_another_base_timeframe_is_re_snapped(tmp_path):
    df = make_frame(days=4)
    hourly = pattern_of([point(t, 100.0, "high") if i % 2 == 0
                         else point(t, 90.0, "low")
                         for i, t in enumerate(times_of(
                             resample_ohlc(df, "1H", "15m"))[4:12:2])], "hourly")
    path = str(tmp_path / "saved_wave_markings.json")
    save_wave_documents({"gold.csv": document([hourly], base_timeframe="1H")}, path)

    patterns, note = _load_canonical("gold.csv", df, "15m", path)

    assert note == ""
    assert len(patterns) == 1
    # Every point now sits on a 15m bar of the frame.
    assert all(p["time"] in times_of(df) for p in patterns[0]["points"])


def test_a_base_timeframe_that_cannot_be_expressed_is_captioned_not_guessed(tmp_path):
    # Stored at 15m, now looking at data whose base is 1H: those 15m bars are
    # simply gone, so the markings are left exactly as they are and said so.
    df = make_frame(days=4)
    stored = pattern_of([point(bar_time(df, 0), 100.0, "high"),
                         point(bar_time(df, 1), 90.0, "low"),
                         point(bar_time(df, 2), 100.0, "high"),
                         point(bar_time(df, 3), 90.0, "low")], "fine")
    path = str(tmp_path / "saved_wave_markings.json")
    save_wave_documents({"gold.csv": document([stored], base_timeframe="15m")}, path)

    patterns, note = _load_canonical("gold.csv", df, "1H", path)

    assert patterns == [stored]
    assert "15m" in note and "1H" in note


def test_an_unknown_dataset_loads_as_an_empty_list(tmp_path):
    df = make_frame(days=2)
    path = str(tmp_path / "saved_wave_markings.json")
    save_wave_documents({"gold.csv": document([])}, path)

    assert _load_canonical("silver.csv", df, "15m", path) == ([], "")
    assert _load_canonical("gold.csv", df, "15m", str(tmp_path / "nope.json")) == ([], "")


# ------------------------------------------------------ 6. undo across the board


def test_an_edit_made_at_1d_is_undone_while_looking_at_15m():
    # The deliberate behaviour change of this phase: one canonical list means one
    # undo stack, so history survives a change of aggregation instead of being
    # silently forgotten by it.
    df = make_frame(days=6)
    daily = period_map(df, "1D", "15m")
    drawn = daily_points(df, [0, 1, 2, 3], ["high", "low", "high", "low"])

    marked, seq, changed = apply_event_batch(
        [], batch(completed(pattern_of(drawn))), 0, daily)
    assert changed is True and len(marked) == 1

    # The client switches the aggregation back to the base timeframe and hits
    # Ctrl+Z. There is no map at 15m at all.
    reverted, seq, changed = apply_event_batch(
        marked, batch_from(seq, {"type": "undo"}), seq, None)

    assert changed is True
    assert reverted == []


def test_an_edit_made_at_15m_is_undone_while_looking_at_1d():
    df = make_frame(days=6)
    daily = period_map(df, "1D", "15m")
    canonical = pattern_of([point(bar_time(df, 5), 100.0, "high"),
                            point(bar_time(df, BARS_PER_DAY + 5), 90.0, "low"),
                            point(bar_time(df, 2 * BARS_PER_DAY + 5), 100.0, "high"),
                            point(bar_time(df, 3 * BARS_PER_DAY + 5), 90.0, "low")])

    marked, seq, _changed = apply_event_batch(
        [], batch(completed(canonical)), 0, None)

    reverted, _seq, changed = apply_event_batch(
        marked, batch_from(seq, {"type": "undo"}), seq, daily)

    assert changed is True
    assert reverted == []


# ------------------------------------------------------------- 7. hidden count


def test_a_pattern_whose_points_share_a_display_bar_is_hidden_and_counted():
    df = make_frame(days=6)
    pmap = period_map(df, "1D", "15m")

    # Inside day 0: high bar then low bar. Two 15m pivots, one 1D bar.
    collapsing = pattern_of([point(bar_time(df, HIGH_BAR), 200.0, "high"),
                             point(bar_time(df, LOW_BAR), 10.0, "low"),
                             point(bar_time(df, BARS_PER_DAY + HIGH_BAR), 201.0, "high"),
                             point(bar_time(df, BARS_PER_DAY + LOW_BAR), 9.0, "low")],
                            "tight")
    spread = pattern_of([point(bar_time(df, 2 * BARS_PER_DAY + HIGH_BAR), 202.0, "high"),
                         point(bar_time(df, 3 * BARS_PER_DAY + LOW_BAR), 7.0, "low"),
                         point(bar_time(df, 4 * BARS_PER_DAY + HIGH_BAR), 204.0, "high"),
                         point(bar_time(df, 5 * BARS_PER_DAY + LOW_BAR), 5.0, "low")],
                        "wide")
    canonical = [collapsing, spread]

    shown = project_patterns(canonical, pmap, DISPLAY)

    assert [p["id"] for p in shown] == ["wide"]
    assert wave_caption(len(canonical), len(shown), "1D", "gold.csv") == (
        "2 pattern(s) · 1 shown on 1D · 1 hidden (points share a 1D bar) "
        "· dataset: gold.csv · saved to saved_wave_markings.json")

    # ...and at 15m both are on screen, so the clause goes away entirely.
    assert wave_caption(2, 2, "15m", "gold.csv") == (
        "2 pattern(s) · 2 shown on 15m · dataset: gold.csv "
        "· saved to saved_wave_markings.json")


def test_the_caption_carries_a_migrations_dropped_count_once():
    caption = wave_caption(3, 2, "1D", "gold.csv", "1 pattern(s) dropped in migration")
    assert caption == (
        "3 pattern(s) · 2 shown on 1D · 1 hidden (points share a 1D bar) "
        "· 1 pattern(s) dropped in migration · dataset: gold.csv "
        "· saved to saved_wave_markings.json")
    assert caption.count("dropped") == 1


# ------------------------------------------------------ 8. canonical settling


def chained_pair(df):
    """Two patterns of the same degree, chained on one shared 15m pivot.

    Both live inside days 0 and 1, so at 1D each of them collapses -- and their
    spans, coarsened, would look like one bar sitting on top of another.
    """
    shared = point(bar_time(df, 40), 100.0, "high")
    first = pattern_of([point(bar_time(df, 10), 90.0, "low"),
                        point(bar_time(df, 20), 100.0, "high"),
                        point(bar_time(df, 30), 90.0, "low"),
                        dict(shared)], "first", degree="Minor")
    second = pattern_of([dict(shared),
                         point(bar_time(df, 50), 90.0, "low"),
                         point(bar_time(df, 60), 100.0, "high"),
                         point(bar_time(df, 70), 90.0, "low")], "second", degree="Minor")
    return [first, second]


@pytest.mark.parametrize("timeframe", ["15m", "1H", "4H", "1D"])
def test_colours_and_degrees_are_the_same_whichever_aggregation_is_current(timeframe):
    df = make_frame(days=3)
    canonical = settle(chained_pair(df))

    # Chained, not overlapping: both stay yellow and keep their own degree.
    assert {p["id"]: p["color"] for p in canonical} == {"first": "yellow",
                                                       "second": "yellow"}

    pmap = None if timeframe == "15m" else period_map(df, timeframe, "15m")
    shown = canonical if pmap is None else project_patterns(canonical, pmap, DISPLAY)

    # Whatever survives the projection carries canonical colours and degrees --
    # validate_patterns must not run again on a projected list.
    for moved in shown:
        original = next(p for p in canonical if p["id"] == moved["id"])
        assert moved["color"] == original["color"]
        assert moved["degree"] == original["degree"]


def test_a_nest_keeps_its_derived_degrees_at_every_aggregation():
    df = make_frame(days=6)
    parent = pattern_of([point(bar_time(df, 0 * BARS_PER_DAY + HIGH_BAR), 200.0, "high"),
                         point(bar_time(df, 1 * BARS_PER_DAY + LOW_BAR), 9.0, "low"),
                         point(bar_time(df, 4 * BARS_PER_DAY + HIGH_BAR), 204.0, "high"),
                         point(bar_time(df, 5 * BARS_PER_DAY + LOW_BAR), 5.0, "low")],
                        "parent", degree="Minor")
    # Spans exactly leg 1 of the parent, endpoint for endpoint.
    child = pattern_of([point(bar_time(df, 1 * BARS_PER_DAY + LOW_BAR), 9.0, "low"),
                        point(bar_time(df, 2 * BARS_PER_DAY + HIGH_BAR), 202.0, "high"),
                        point(bar_time(df, 3 * BARS_PER_DAY + LOW_BAR), 7.0, "low"),
                        point(bar_time(df, 4 * BARS_PER_DAY + HIGH_BAR), 204.0, "high")],
                       "child", degree="Minor")

    canonical = settle([parent, child])
    assert {p["id"]: p["degree"] for p in canonical} == {"parent": "Minor",
                                                        "child": "Minute"}
    assert {p["color"] for p in canonical} == {"yellow"}

    for timeframe in ("1H", "4H", "1D"):
        pmap = period_map(df, timeframe, "15m")
        shown = project_patterns(canonical, pmap, DISPLAY)
        assert {p["id"]: p["degree"] for p in shown} == {"parent": "Minor",
                                                         "child": "Minute"}
        assert {p["color"] for p in shown} == {"yellow"}
        # The child still hangs off leg 1 of the parent on screen, which is what
        # the frontend's own childLegIndex uses to hide its origin glyph.
        by_id = {p["id"]: p for p in shown}
        assert child_leg_index(by_id["parent"], by_id["child"]) == 1


def test_a_settle_on_the_canonical_list_never_sees_the_display_timeframe():
    # Two patterns that share a 1D bar at either end but are legally chained at
    # 15m. Settling the *projection* would be a different answer; settling
    # canonically is the only one this tab computes.
    df = make_frame(days=3)
    canonical = settle(chained_pair(df))
    pmap = period_map(df, "1D", "15m")

    state, _seq, _changed = apply_event_batch(
        canonical, batch({"type": "shift_degree", "id": "first", "delta": 1}), 0, pmap)

    assert {p["id"]: p["degree"] for p in state} == {"first": "Intermediate",
                                                     "second": "Minor"}
    assert {p["color"] for p in state} == {"yellow"}


# --------------------------------------------------------- 9. pivot magnetism
#
# The defect this closes: Phase 11b made a count survive an aggregation switch
# and Phase 10 made a child derive its degree from its parent, but the two did
# not cooperate. A parent pivot drawn at 15m is usually not its day's extreme,
# so a child drawn at 1D over one of its legs refined to *different* base bars
# and the parent/child relation -- which matches on exact (time, kind) -- never
# saw it. Refinement now attaches to an existing pivot in the clicked period.


def offset_parent(df, pattern_id="parent", degree="Subminuette"):
    """A four-point count drawn at 15m, no pivot of which is its day's extreme.

    Leg 1 (day 1 -> day 5) is deliberately wide enough to hold a child of its
    own drawn on daily bars.
    """
    return pattern_of([point(bar_time(df, 0 * BARS_PER_DAY + 5), 100.0, "high"),
                       point(bar_time(df, 1 * BARS_PER_DAY + 50), 90.0, "low"),
                       point(bar_time(df, 5 * BARS_PER_DAY + 5), 100.0, "high"),
                       point(bar_time(df, 6 * BARS_PER_DAY + 50), 90.0, "low")],
                      pattern_id, degree=degree)


def child_drawn_at_1d(df):
    """The four daily clicks that draw a child over the parent's leg 1."""
    return completed(pattern_of(
        daily_points(df, [1, 2, 3, 5], ["low", "high", "low", "high"]), "child"))


def test_a_child_drawn_at_1d_attaches_to_a_parent_drawn_at_15m():
    df = make_frame(days=12)
    pmap = period_map(df, "1D", "15m")
    parent = offset_parent(df)

    state, seq, changed = apply_event_batch([parent], batch(child_drawn_at_1d(df)),
                                            0, pmap)

    assert changed is True and seq == 1
    by_id = {p["id"]: p for p in state}
    child = by_id["child"]

    # The child's endpoints are the parent's own pivots, to the bar and to the
    # price -- neither of which is its day's extreme.
    assert child["points"][0] == parent["points"][1]
    assert child["points"][-1] == parent["points"][2]
    assert child["points"][0]["time"] != pmap.extreme_time[(day_bar(df, 1), "low")]

    # ...so the relation resolves, the degree is derived, and nothing goes red.
    found, leg = find_parent(state, child)
    assert (found["id"], leg) == ("parent", 1)
    assert by_id["parent"]["degree"] == "Subminuette"
    assert child["degree"] == "Micro"               # one step junior
    assert {p["color"] for p in state} == {"yellow"}


def test_the_same_child_without_magnetism_lands_nowhere_near_the_parent(monkeypatch):
    # The guard that stops the test above from silently passing on nothing: with
    # the magnet index emptied, the very same batch produces an unrelated pattern
    # at the parent's own degree, which is exactly the red pair the client hit.
    monkeypatch.setattr(wave_analysis_tab, "pivot_magnets", lambda patterns, pmap: {})

    df = make_frame(days=12)
    pmap = period_map(df, "1D", "15m")
    parent = offset_parent(df)

    state, _seq, _changed = apply_event_batch([parent], batch(child_drawn_at_1d(df)),
                                              0, pmap)

    by_id = {p["id"]: p for p in state}
    assert by_id["child"]["points"][0] != parent["points"][1]
    assert find_parent(state, by_id["child"]) == (None, None)
    assert by_id["child"]["degree"] == "Subminuette"
    assert {p["color"] for p in state} == {"red"}


def test_the_magnetised_child_shows_on_the_parents_pivot_bars_back_at_15m():
    # Viewed at the base timeframe -- where there is no map at all -- the child's
    # ends sit on the same 15m bars the parent was drawn on, which is what the
    # frontend's own childLegIndex reads to hide the child's origin glyph.
    df = make_frame(days=12)
    parent = offset_parent(df)

    state, _seq, _changed = apply_event_batch(
        [parent], batch(child_drawn_at_1d(df)), 0, period_map(df, "1D", "15m"))

    by_id = {p["id"]: p for p in state}
    assert child_leg_index(by_id["parent"], by_id["child"]) == 1
    for moved in state:
        assert all(p["time"] in times_of(df) for p in moved["points"])


def test_a_click_at_the_base_timeframe_is_never_pulled_onto_a_neighbour():
    # There is no map at 15m, so the client clicked the exact bar he meant and a
    # magnet would turn a deliberate mark into a silent correction.
    df = make_frame(days=4)
    existing = pattern_of([point(bar_time(df, 5), 100.0, "high"),
                           point(bar_time(df, 10), 90.0, "low"),
                           point(bar_time(df, 15), 100.0, "high"),
                           point(bar_time(df, 20), 90.0, "low")], "existing")
    drawn = pattern_of([point(bar_time(df, 6), 100.0, "high"),
                        point(bar_time(df, 11), 90.0, "low"),
                        point(bar_time(df, 16), 100.0, "high"),
                        point(bar_time(df, 21), 90.0, "low")], "next to it")

    state, _seq, _changed = apply_event_batch([existing], batch(completed(drawn)),
                                              0, None)

    marked = next(p for p in state if p["id"] == "next to it")
    assert [p["time"] for p in marked["points"]] == [p["time"] for p in drawn["points"]]


def test_a_magnetised_pattern_that_cannot_be_ordered_is_refused_and_acked():
    # Both clicks land on day 0 and both magnet onto one bar; the fallback puts
    # each back on its own extreme, and day 0's low bar comes *after* its high
    # bar, so the low-then-high order asked for is unreachable either way. The
    # seq still advances -- the frontend replays anything unacked forever.
    df = make_frame(days=4)
    pmap = period_map(df, "1D", "15m")
    shared = bar_time(df, 40)
    existing = [pattern_of([point(shared, 100.0, "high"),
                            point(bar_time(df, BARS_PER_DAY + 5), 90.0, "low"),
                            point(bar_time(df, 2 * BARS_PER_DAY + 5), 100.0, "high"),
                            point(bar_time(df, 3 * BARS_PER_DAY + 5), 90.0, "low")],
                           "highs"),
                pattern_of([point(shared, 90.0, "low"),
                            point(bar_time(df, BARS_PER_DAY + 6), 100.0, "high"),
                            point(bar_time(df, 2 * BARS_PER_DAY + 6), 90.0, "low"),
                            point(bar_time(df, 3 * BARS_PER_DAY + 6), 100.0, "high")],
                           "lows")]
    assert pivot_magnets(existing, pmap)[(day_bar(df, 0), "high")][0][0] == shared

    drawn = pattern_of([point(day_bar(df, 0), 10.0, "low"),
                        point(day_bar(df, 0), 200.0, "high"),
                        point(day_bar(df, 2), 8.0, "low"),
                        point(day_bar(df, 3), 203.0, "high")], "unorderable")
    state, seq, changed = apply_event_batch(existing, batch(completed(drawn)), 0, pmap)

    assert seq == 1
    assert changed is False
    assert [p["id"] for p in state] == ["highs", "lows"]


# ------------------------------------------------- 10. relation-aware magnets
#
# What Phase 13 left open: the choice *between* two pivots sharing a display bar
# was made on price distance, which says nothing about intent. In a dense count
# that picks the wrong one, the child spans no leg, no degree is derived, and the
# pair collides at the same degree and goes red -- the very defect magnetism was
# introduced to remove. The choice is now made on what the finished pattern would
# mean instead: the combination of endpoints that makes it a child wins.


def dense_frame(days=8):
    """Days whose extremes sit at bars 80 and 90, clear of every pivot drawn.

    Bar 20 is where a count drawn at 15m sits and bar 60 where an older,
    unrelated one does -- and bar 60 is deliberately the nearer of the two to
    each day's extreme, so the price-distance rule prefers the wrong count.
    """
    count = days * BARS_PER_DAY
    index = pd.date_range(START, periods=count, freq="15min")
    highs = [100.0] * count
    lows = [90.0] * count
    for day in range(days):
        highs[day * BARS_PER_DAY + 80] = 200.0 + day
        lows[day * BARS_PER_DAY + 90] = 10.0 - day
        highs[day * BARS_PER_DAY + 60] = 150.0 + day
        lows[day * BARS_PER_DAY + 60] = 50.0 - day
    return pd.DataFrame({"open": highs, "high": highs, "low": lows,
                         "close": highs, "volume": [1.0] * count}, index=index)


def dense_counts(df):
    """A parent drawn at 15m and an unrelated older count sharing its days.

    The unrelated count touches both of the contended days, but with its *first*
    and *last* points -- indices 0 and 3, which are not consecutive -- so it
    offers no leg between them and cannot be mistaken for the parent. It sits at
    its own degree, as a second count in the same price action must, or the two
    would overlap at one degree and be red before this phase is even reached.
    """
    parent = pattern_of([point(bar_time(df, 20), 100.0, "high"),
                         point(bar_time(df, 1 * BARS_PER_DAY + 20), 90.0, "low"),
                         point(bar_time(df, 5 * BARS_PER_DAY + 20), 100.0, "high"),
                         point(bar_time(df, 6 * BARS_PER_DAY + 20), 90.0, "low")],
                        "parent", degree="Minor")
    unrelated = pattern_of([point(bar_time(df, 1 * BARS_PER_DAY + 60), 49.0, "low"),
                            point(bar_time(df, 2 * BARS_PER_DAY + 60), 152.0, "high"),
                            point(bar_time(df, 3 * BARS_PER_DAY + 60), 47.0, "low"),
                            point(bar_time(df, 5 * BARS_PER_DAY + 60), 155.0, "high")],
                           "unrelated", degree="Intermediate")
    return parent, unrelated


def dense_child_at_1d(df):
    """The four daily clicks that draw a child over the parent's leg 1.

    Completed at the parent's own degree, which is what the frontend sends: the
    junior degree is derived here, not chosen there.
    """
    return completed(pattern_of(
        daily_points(df, [1, 2, 3, 5], ["low", "high", "low", "high"]),
        "child", degree="Minor"))


def heads_only(patterns, pmap):
    """Phase 13's magnet index: the head of each of this phase's lists."""
    return {key: candidates[:1]
            for key, candidates in pivot_magnets(patterns, pmap).items()}


def test_the_contended_day_attaches_to_the_unrelated_count_under_the_old_rule(monkeypatch):
    # The defect, reproduced through the tab. The child's first click lands in a
    # day whose low is 9.0 at bar 90; the unrelated count's pivot there is 49.0
    # and the parent's is 90.0, so the nearer-the-extreme rule takes the
    # unrelated one. The child then spans no leg, keeps the degree it arrived
    # with, and collides with the parent.
    monkeypatch.setattr(wave_analysis_tab, "pivot_magnets", heads_only)

    df = dense_frame()
    pmap = period_map(df, "1D", "15m")
    parent, unrelated = dense_counts(df)

    state, _seq, _changed = apply_event_batch([parent, unrelated],
                                              batch(dense_child_at_1d(df)), 0, pmap)

    by_id = {p["id"]: p for p in state}
    assert by_id["child"]["points"][0] == unrelated["points"][0]
    assert find_parent(state, by_id["child"]) == (None, None)
    assert by_id["child"]["degree"] == "Minor"
    assert by_id["parent"]["color"] == "red" and by_id["child"]["color"] == "red"


def test_the_same_contended_day_attaches_to_the_parent_under_the_new_rule():
    df = dense_frame()
    pmap = period_map(df, "1D", "15m")
    parent, unrelated = dense_counts(df)

    state, seq, changed = apply_event_batch([parent, unrelated],
                                            batch(dense_child_at_1d(df)), 0, pmap)

    assert changed is True and seq == 1
    by_id = {p["id"]: p for p in state}
    child = by_id["child"]

    # Both ends landed on the parent's own pivots, which are the *farther* of the
    # two candidates their days offered.
    assert child["points"][0] == parent["points"][1]
    assert child["points"][-1] == parent["points"][2]

    # ...so the relation resolves, the degree is derived, and nothing is red.
    found, leg = find_parent(state, child)
    assert (found["id"], leg) == ("parent", 1)
    assert child["degree"] == "Minute"              # one step junior to Minor
    assert child_leg_index(by_id["parent"], child) == 1
    assert {p["color"] for p in state} == {"yellow"}
    # The unrelated count is left exactly where it was.
    assert by_id["unrelated"] == unrelated


def test_deleting_the_unrelated_count_leaves_the_simple_path_working():
    # The uncontended case must not have been traded away for the contended one.
    df = dense_frame()
    pmap = period_map(df, "1D", "15m")
    parent, _unrelated = dense_counts(df)

    state, _seq, _changed = apply_event_batch([parent], batch(dense_child_at_1d(df)),
                                              0, pmap)

    by_id = {p["id"]: p for p in state}
    assert by_id["child"]["points"][0] == parent["points"][1]
    assert by_id["child"]["degree"] == "Minute"
    assert {p["color"] for p in state} == {"yellow"}


def test_dragging_a_childs_endpoint_at_1d_keeps_it_on_the_parent():
    # The child already hangs off the parent; the client nudges its first point
    # in the same contended day. It must come back to the parent's pivot rather
    # than to the unrelated count's nearer one.
    df = dense_frame()
    pmap = period_map(df, "1D", "15m")
    parent, unrelated = dense_counts(df)
    nested, _seq, _changed = apply_event_batch([parent, unrelated],
                                               batch(dense_child_at_1d(df)), 0, pmap)

    day1 = day_bar(df, 1)
    lows = resample_ohlc(df, "1D", "15m")["low"]
    dragged, seq, changed = apply_event_batch(
        nested, batch(move("child", 0, day1, float(lows.iloc[1]), "low")), 0, pmap)

    assert changed is True and seq == 1
    by_id = {p["id"]: p for p in dragged}
    assert by_id["child"]["points"][0] == parent["points"][1]
    assert find_parent(dragged, by_id["child"])[0]["id"] == "parent"
    assert {p["color"] for p in dragged} == {"yellow"}


def test_a_nest_built_in_the_dense_frame_survives_every_aggregation():
    # Nothing about the new choice depends on 1D in particular, and viewing the
    # result back at the base timeframe is where the client sees the pivots.
    df = dense_frame()
    parent, unrelated = dense_counts(df)
    state, _seq, _changed = apply_event_batch(
        [parent, unrelated], batch(dense_child_at_1d(df)), 0,
        period_map(df, "1D", "15m"))

    for moved in state:
        assert all(p["time"] in times_of(df) for p in moved["points"])

    for timeframe in ("1H", "4H", "1D"):
        shown = project_patterns(state, period_map(df, timeframe, "15m"), DISPLAY)
        by_id = {p["id"]: p for p in shown}
        assert child_leg_index(by_id["parent"], by_id["child"]) == 1
        assert {p["id"]: p["degree"] for p in shown} == {"parent": "Minor",
                                                        "child": "Minute",
                                                        "unrelated": "Intermediate"}
        assert {p["color"] for p in shown} == {"yellow"}


def test_an_existing_nest_is_untouched_by_this_phase_at_every_aggregation():
    df = make_frame(days=6)
    parent = pattern_of([point(bar_time(df, 0 * BARS_PER_DAY + HIGH_BAR), 200.0, "high"),
                         point(bar_time(df, 1 * BARS_PER_DAY + LOW_BAR), 9.0, "low"),
                         point(bar_time(df, 4 * BARS_PER_DAY + HIGH_BAR), 204.0, "high"),
                         point(bar_time(df, 5 * BARS_PER_DAY + LOW_BAR), 5.0, "low")],
                        "parent", degree="Minor")
    child = pattern_of([point(bar_time(df, 1 * BARS_PER_DAY + LOW_BAR), 9.0, "low"),
                        point(bar_time(df, 2 * BARS_PER_DAY + HIGH_BAR), 202.0, "high"),
                        point(bar_time(df, 3 * BARS_PER_DAY + LOW_BAR), 7.0, "low"),
                        point(bar_time(df, 4 * BARS_PER_DAY + HIGH_BAR), 204.0, "high")],
                       "child", degree="Minor")
    canonical = settle([parent, child])

    for timeframe in ("1H", "4H", "1D"):
        pmap = period_map(df, timeframe, "15m")
        # Every pivot of the nest is a magnet at this aggregation -- six of them,
        # not eight, because the child shares both ends of the parent's leg 1.
        assert len(pivot_magnets(canonical, pmap)) == 6

        # ...and none of that moves anything: an event carrying no coordinates
        # leaves the list exactly as settled, relations and degrees included.
        state, _seq, _changed = apply_event_batch(
            canonical, batch({"type": "delete_pattern", "id": "nobody"}), 0, pmap)
        assert state == canonical

        shown = project_patterns(canonical, pmap, DISPLAY)
        by_id = {p["id"]: p for p in shown}
        assert child_leg_index(by_id["parent"], by_id["child"]) == 1
        assert {p["id"]: p["degree"] for p in shown} == {"parent": "Minor",
                                                         "child": "Minute"}
        assert {p["color"] for p in shown} == {"yellow"}


# --------------------------------------------- 11. whole-pattern selection
#
# The mirror of section 10, and the direction the client actually works in:
# "Okay, so I've notated all the sub minuets. Now I can notate the minuet
# degree." A pattern drawn over counts already marked adopts them through its
# *interior* points -- a child hangs off a consecutive pair of its parent's
# points -- so steering only the two ends left his primary workflow broken. Every
# point of an incoming pattern is now chosen together, for the reading that
# expresses the most structure.


def bottom_up_frame(days=7):
    """Days whose extremes sit at bars 80 and 90, clear of every pivot drawn.

    Bar 10 carries the client's own 15m pivots at the frame's flat 100/90, bar 50
    two older rival counts' at 140/20 -- deliberately the nearer of the two to
    every day's extreme, so price distance prefers a rival in every single day.
    """
    count = days * BARS_PER_DAY
    index = pd.date_range(START, periods=count, freq="15min")
    highs = [100.0] * count
    lows = [90.0] * count
    for day in range(days):
        highs[day * BARS_PER_DAY + 80] = 150.0 + day
        lows[day * BARS_PER_DAY + 90] = 10.0 - day
        highs[day * BARS_PER_DAY + 50] = 140.0
        lows[day * BARS_PER_DAY + 50] = 20.0
    return pd.DataFrame({"open": highs, "high": highs, "low": lows,
                         "close": highs, "volume": [1.0] * count}, index=index)


def bar_point(df, position, kind):
    """The canonical point a 15m click on that bar would produce."""
    price = df["high" if kind == "high" else "low"].iloc[position]
    return point(bar_time(df, position), float(price), kind)


def bottom_up_counts(df):
    """(corners, children, rivals) for the bottom-up frame.

    The three children are chained end to end over bar 10 of days 0, 2, 4 and 6,
    which are the four pivots the larger degree drawn over them has to find. The
    rivals sit at their own degrees, as second counts in the same price action
    must, and between them cover all four of the clicked days on bar 50.
    """
    corners = [bar_point(df, day * BARS_PER_DAY + 10, kind)
               for day, kind in zip((0, 2, 4, 6), ("high", "low", "high", "low"))]
    interiors = [(0 * BARS_PER_DAY + 60, 1 * BARS_PER_DAY + 30),
                 (2 * BARS_PER_DAY + 60, 3 * BARS_PER_DAY + 30),
                 (4 * BARS_PER_DAY + 60, 5 * BARS_PER_DAY + 30)]
    children = []
    for index, (start, end) in enumerate(zip(corners, corners[1:])):
        first, second = interiors[index]
        kinds = ("low", "high") if start["kind"] == "high" else ("high", "low")
        children.append(pattern_of(
            [start, bar_point(df, first, kinds[0]), bar_point(df, second, kinds[1]),
             end], f"child {index}", degree="Minor"))

    rivals = [pattern_of([bar_point(df, 0 * BARS_PER_DAY + 50, "high"),
                          bar_point(df, 2 * BARS_PER_DAY + 50, "low"),
                          bar_point(df, 5 * BARS_PER_DAY + 50, "high"),
                          bar_point(df, 6 * BARS_PER_DAY + 50, "low")],
                         "rival", degree="Intermediate"),
              pattern_of([bar_point(df, 1 * BARS_PER_DAY + 50, "high"),
                          bar_point(df, 3 * BARS_PER_DAY + 50, "low"),
                          bar_point(df, 4 * BARS_PER_DAY + 50, "high"),
                          bar_point(df, 5 * BARS_PER_DAY + 50, "low")],
                         "older", degree="Primary")]
    return corners, children, rivals


def bottom_up_parent_at_1d(df):
    """The four daily clicks that draw the larger degree over the children.

    Completed at the degree the children carry, which is what the frontend sends:
    the junior degree is derived here, not chosen there.
    """
    return completed(pattern_of(
        daily_points(df, [0, 2, 4, 6], ["high", "low", "high", "low"]),
        "parent", degree="Minor"))


def test_a_parent_drawn_at_1d_over_marked_children_adopts_all_three():
    df = bottom_up_frame()
    pmap = period_map(df, "1D", "15m")
    corners, children, rivals = bottom_up_counts(df)

    state, seq, changed = apply_event_batch(rivals + children,
                                            batch(bottom_up_parent_at_1d(df)), 0, pmap)

    assert changed is True and seq == 1
    by_id = {p["id"]: p for p in state}

    # All four of the parent's points landed on the pivots the children share,
    # every one of which is the *farther* candidate its day offered.
    assert by_id["parent"]["points"] == corners

    # ...so all three children are adopted, one degree junior, and nothing is red.
    for index, child in enumerate(children):
        found, leg = find_parent(state, by_id[child["id"]])
        assert (found["id"], leg) == ("parent", index)
        assert by_id[child["id"]]["degree"] == "Minute"
    assert by_id["parent"]["degree"] == "Minor"
    assert {p["color"] for p in state} == {"yellow"}


def test_the_same_parent_adopts_nothing_under_the_head_of_list_rule(monkeypatch):
    # The guard that stops the test above from silently passing on nothing. Every
    # one of the parent's four clicks lands in a day where a rival's pivot is
    # nearer that day's extreme, so the head-of-list rule takes all four, the
    # children are left behind at the parent's own degree, and the whole nest goes
    # red -- which is exactly what the client sees today.
    monkeypatch.setattr(wave_analysis_tab, "pivot_magnets", heads_only)

    df = bottom_up_frame()
    pmap = period_map(df, "1D", "15m")
    corners, children, rivals = bottom_up_counts(df)

    state, _seq, _changed = apply_event_batch(rivals + children,
                                              batch(bottom_up_parent_at_1d(df)), 0, pmap)

    by_id = {p["id"]: p for p in state}
    assert by_id["parent"]["points"] != corners
    for child in children:
        assert find_parent(state, by_id[child["id"]]) == (None, None)
        assert by_id[child["id"]]["degree"] == "Minor"
        assert by_id[child["id"]]["color"] == "red"
    assert by_id["parent"]["color"] == "red"


def test_the_adopted_children_sit_on_the_parents_pivot_bars_at_every_aggregation():
    df = bottom_up_frame()
    _corners, children, rivals = bottom_up_counts(df)
    state, _seq, _changed = apply_event_batch(
        rivals + children, batch(bottom_up_parent_at_1d(df)), 0,
        period_map(df, "1D", "15m"))

    for moved in state:
        assert all(p["time"] in times_of(df) for p in moved["points"])

    # 1D is left out on purpose: each child alternates twice inside a single day,
    # so at 1D it collapses onto one bar and Phase 11's hidden count -- not this
    # phase -- is what removes it from the chart.
    for timeframe in ("1H", "4H"):
        shown = project_patterns(state, period_map(df, timeframe, "15m"), DISPLAY)
        by_id = {p["id"]: p for p in shown}
        for index, child in enumerate(children):
            assert child_leg_index(by_id["parent"], by_id[child["id"]]) == index
        assert {p["color"] for p in shown} == {"yellow"}


def test_dragging_an_interior_point_at_1d_does_not_shake_a_child_loose():
    df = bottom_up_frame()
    pmap = period_map(df, "1D", "15m")
    corners, children, rivals = bottom_up_counts(df)
    nested, _seq, _changed = apply_event_batch(
        rivals + children, batch(bottom_up_parent_at_1d(df)), 0, pmap)

    day2 = day_bar(df, 2)
    lows = resample_ohlc(df, "1D", "15m")["low"]
    dragged, seq, changed = apply_event_batch(
        nested, batch(move("parent", 1, day2, float(lows.iloc[2]), "low")), 0, pmap)

    assert changed is True and seq == 1
    by_id = {p["id"]: p for p in dragged}
    assert by_id["parent"]["points"][1] == corners[1]
    for index, child in enumerate(children):
        assert find_parent(dragged, by_id[child["id"]])[0]["id"] == "parent"
        assert child_leg_index(by_id["parent"], by_id[child["id"]]) == index
    assert {p["color"] for p in dragged} == {"yellow"}


def both_directions_counts(df):
    """A grandparent and a grandchild marked at 15m, plus a rival on bar 50.

    The middle degree drawn over them at 1D slots into the grandparent's leg 1
    *and* takes the grandchild under its own leg 1 -- two relations at once.
    """
    grandparent = pattern_of([bar_point(df, 0 * BARS_PER_DAY + 10, "high"),
                              bar_point(df, 2 * BARS_PER_DAY + 10, "low"),
                              bar_point(df, 5 * BARS_PER_DAY + 10, "high"),
                              bar_point(df, 7 * BARS_PER_DAY + 10, "low")],
                             "grandparent", degree="Minor")
    grandchild = pattern_of([bar_point(df, 3 * BARS_PER_DAY + 10, "high"),
                             bar_point(df, 3 * BARS_PER_DAY + 40, "low"),
                             bar_point(df, 3 * BARS_PER_DAY + 70, "high"),
                             bar_point(df, 4 * BARS_PER_DAY + 10, "low")],
                            "grandchild", degree="Minor")
    rival = pattern_of([bar_point(df, 2 * BARS_PER_DAY + 50, "low"),
                        bar_point(df, 3 * BARS_PER_DAY + 50, "high"),
                        bar_point(df, 4 * BARS_PER_DAY + 50, "low"),
                        bar_point(df, 5 * BARS_PER_DAY + 50, "high")],
                       "rival", degree="Intermediate")
    return grandparent, grandchild, rival


def test_a_middle_degree_drawn_at_1d_slots_in_and_adopts_at_once():
    df = bottom_up_frame(days=8)
    pmap = period_map(df, "1D", "15m")
    grandparent, grandchild, rival = both_directions_counts(df)
    drawn = completed(pattern_of(
        daily_points(df, [2, 3, 4, 5], ["low", "high", "low", "high"]),
        "middle", degree="Minor"))

    state, seq, changed = apply_event_batch([grandparent, grandchild, rival],
                                            batch(drawn), 0, pmap)

    assert changed is True and seq == 1
    by_id = {p["id"]: p for p in state}

    # Slotted into the grandparent's leg 1, and the grandchild slotted into its
    # own leg 1 -- three consecutive degrees derived from one completion.
    assert child_leg_index(by_id["grandparent"], by_id["middle"]) == 1
    assert child_leg_index(by_id["middle"], by_id["grandchild"]) == 1
    assert {p["id"]: p["degree"] for p in state} == {"grandparent": "Minor",
                                                     "middle": "Minute",
                                                     "grandchild": "Minuette",
                                                     "rival": "Intermediate"}
    assert {p["color"] for p in state} == {"yellow"}

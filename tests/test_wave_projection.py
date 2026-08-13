"""
Tests for projecting wave markings between the canonical (base) timeframe and
a displayed aggregation.

Frames are small and their extremes are placed by hand, so every expected
answer can be read off the construction rather than recomputed by the test.
"""

import copy

import pandas as pd
import pytest

from config.wave_analysis import is_valid_pattern
from config.wave_projection import (
    _candidates_for,
    _point_candidates,
    _relation_pairs,
    MAX_CANDIDATES_PER_POINT,
    period_map,
    pivot_magnets,
    project_pattern,
    project_patterns,
    projection_collapses,
    refine_event,
    to_canonical,
    to_display,
    CANONICAL,
)
from data.loader import resample_ohlc

START = "2024-01-01 00:00"
BARS_PER_DAY = 96                       # 15m bars


def make_frame(highs, lows, start=START, freq="15min"):
    """A base-resolution OHLC frame with the highs and lows given, in order."""
    index = pd.date_range(start, periods=len(highs), freq=freq)
    return pd.DataFrame({"open": list(highs), "high": list(highs),
                         "low": list(lows), "close": list(highs),
                         "volume": [1.0] * len(highs)}, index=index)


def flat_frame(days=1, high=100.0, low=90.0):
    count = days * BARS_PER_DAY
    return make_frame([high] * count, [low] * count)


def times_of(df):
    return [int(ts.timestamp()) for ts in df.index]


def bar_time(df, position):
    return int(df.index[position].timestamp())


def make_pattern(points, pattern_id="p1"):
    """A structurally valid Zigzag over the four points given."""
    return {
        "id": pattern_id,
        "pattern_type": "Zigzag",
        "variation": "Zigzag",
        "degree": "Subminuette",
        "color": "yellow",
        "points": list(points),
    }


def point_at(df, position, kind):
    """The canonical point a frontend click on that bar would produce."""
    price = df["high" if kind == "high" else "low"].iloc[position]
    return {"time": bar_time(df, position), "price": float(price), "kind": kind}


# ------------------------------------------------------------ grid agreement


def gappy_frame():
    """Three days of 15m bars with a whole hour and a whole day missing.

    The missing day empties a 1D bin and the missing hour empties a 1H bin, so
    ``dropna(subset=['high'])`` actually bites on more than one timeframe.
    """
    index = pd.date_range(START, periods=3 * BARS_PER_DAY, freq="15min")
    index = index[(index.day != 2) & ~((index.day == 1) & (index.hour == 5))]
    highs = [100.0 + i for i in range(len(index))]
    lows = [h - 5.0 for h in highs]
    return pd.DataFrame({"open": highs, "high": highs, "low": lows,
                         "close": highs, "volume": [1.0] * len(index)},
                        index=index)


@pytest.mark.parametrize("timeframe", ["1H", "4H", "1D"])
def test_display_times_match_resample_ohlc_exactly(timeframe):
    df = gappy_frame()
    pmap = period_map(df, timeframe, "15m")

    assert pmap.display_times == times_of(resample_ohlc(df, timeframe, "15m"))


@pytest.mark.parametrize("timeframe", ["1H", "4H", "1D"])
def test_empty_bins_are_absent_from_the_grid(timeframe):
    # Guards the assertion above against passing vacuously: the gap has to be
    # visible as a shorter grid than an ungapped frame of the same span.
    df = gappy_frame()
    ungapped = flat_frame(days=3)

    assert len(period_map(df, timeframe, "15m").display_times) \
        < len(period_map(ungapped, timeframe, "15m").display_times)


def test_a_bucket_is_never_a_dropped_period(timeframe="1D"):
    df = gappy_frame()
    pmap = period_map(df, timeframe, "15m")

    assert set(pmap.bucket_of.values()) <= set(pmap.display_times)
    assert set(pmap.bucket_of) == set(times_of(df))


def test_an_unsupported_timeframe_is_rejected():
    with pytest.raises(ValueError):
        period_map(flat_frame(), "3W", "15m")


# The invariant that lets the map hold one price table instead of two:
# ``to_canonical`` reads a refined point's price out of ``extreme_price``, which
# is only sound because the bar ``extreme_time`` names is the one whose own high
# (or low) achieves that period's extreme. Break ``_extreme_rows`` and the price
# a refined point carries would silently stop belonging to the bar it names, so
# this is pinned rather than left implied.
@pytest.mark.parametrize("timeframe", ["1H", "4H", "1D"])
def test_the_extreme_bar_owns_the_periods_extreme_price(timeframe):
    highs = [100.0 + (i % 7) for i in range(BARS_PER_DAY * 3)]
    lows = [90.0 - (i % 5) for i in range(BARS_PER_DAY * 3)]
    df = make_frame(highs, lows)
    pmap = period_map(df, timeframe, "15m")

    by_time = {int(ts.timestamp()): row for ts, row in df.iterrows()}
    for (display_time, kind), base_time in pmap.extreme_time.items():
        assert by_time[base_time][kind] == pmap.extreme_price[(display_time, kind)]


# --------------------------------------------------------------- identity map


def test_identity_map_is_well_formed():
    df = make_frame([100.0, 101.0, 99.0, 105.0], [90.0, 91.0, 80.0, 95.0])
    pmap = period_map(df, "15m", "15m")

    assert pmap.display_times == times_of(resample_ohlc(df, "15m", "15m"))
    assert pmap.bucket_of == {t: t for t in times_of(df)}
    for position, time in enumerate(times_of(df)):
        assert pmap.extreme_time[(time, "high")] == time
        assert pmap.extreme_time[(time, "low")] == time
        assert pmap.extreme_price[(time, "high")] == df["high"].iloc[position]
        assert pmap.extreme_price[(time, "low")] == df["low"].iloc[position]


def test_every_projection_through_the_identity_map_is_a_no_op():
    df = make_frame([100.0, 101.0, 99.0, 105.0], [90.0, 91.0, 80.0, 95.0])
    pmap = period_map(df, "15m", "15m")
    pattern = make_pattern([point_at(df, 0, "low"), point_at(df, 1, "high"),
                            point_at(df, 2, "low"), point_at(df, 3, "high")])

    assert project_pattern(pattern, pmap) == pattern
    assert project_pattern(pattern, pmap, CANONICAL) == pattern


# -------------------------------------------------------------------- coarsen


def coarsen_frame():
    """One day whose high sits at bar 40 and whose low sits at bar 70."""
    highs = [100.0] * BARS_PER_DAY
    lows = [90.0] * BARS_PER_DAY
    highs[40] = 150.0
    lows[70] = 50.0
    return make_frame(highs, lows)


def test_a_high_point_coarsens_to_its_periods_high():
    df = coarsen_frame()
    pmap = period_map(df, "1D", "15m")
    day = pmap.display_times[0]

    # Bar 10 is nowhere near the day's high, which is exactly the point: the
    # pivot moves to the period's extreme.
    assert to_display(point_at(df, 10, "high"), pmap) == {
        "time": day, "price": 150.0, "kind": "high"}


def test_a_low_point_coarsens_to_its_periods_low():
    df = coarsen_frame()
    pmap = period_map(df, "1D", "15m")
    day = pmap.display_times[0]

    assert to_display(point_at(df, 10, "low"), pmap) == {
        "time": day, "price": 50.0, "kind": "low"}


def test_kind_decides_which_extreme_is_taken():
    df = coarsen_frame()
    pmap = period_map(df, "1D", "15m")
    at_the_high = point_at(df, 40, "high")

    # Same bar, asked for as a low: it takes the day's low, not the day's high.
    as_low = dict(at_the_high, kind="low")
    assert to_display(as_low, pmap)["price"] == 50.0


def test_points_in_different_periods_coarsen_to_different_bars():
    df = flat_frame(days=2)
    pmap = period_map(df, "1D", "15m")

    first = to_display(point_at(df, 3, "high"), pmap)
    second = to_display(point_at(df, BARS_PER_DAY + 3, "high"), pmap)
    assert [first["time"], second["time"]] == pmap.display_times


# ---------------------------------------------------------- refine, tie-break


def frame_with_high_at(positions, days=1):
    """A flat day whose high (150.0) is achieved at exactly these bars."""
    count = days * BARS_PER_DAY
    highs = [100.0] * count
    for position in positions:
        highs[position] = 150.0
    return make_frame(highs, [90.0] * count)


def frame_with_low_at(positions, days=1):
    count = days * BARS_PER_DAY
    lows = [90.0] * count
    for position in positions:
        lows[position] = 50.0
    return make_frame([100.0] * count, lows)


def refine(df, timeframe, kind, period_index=0):
    pmap = period_map(df, timeframe, "15m")
    display = pmap.display_times[period_index]
    point = {"time": display, "price": pmap.extreme_price[(display, kind)],
             "kind": kind}
    return to_canonical(point, pmap)


def test_a_tied_high_refines_to_the_later_bar():
    df = frame_with_high_at([40, 77])

    assert refine(df, "1D", "high") == {"time": bar_time(df, 77),
                                        "price": 150.0, "kind": "high"}


def test_a_tied_low_refines_to_the_later_bar():
    df = frame_with_low_at([5, 60])

    assert refine(df, "1D", "low") == {"time": bar_time(df, 60),
                                       "price": 50.0, "kind": "low"}


def test_a_three_way_tie_refines_to_the_last_of_the_three():
    df = frame_with_high_at([40, 77, 80])

    assert refine(df, "1D", "high")["time"] == bar_time(df, 80)


def test_a_tie_between_the_first_and_last_bar_of_the_period_refines_to_the_last():
    df = frame_with_high_at([0, BARS_PER_DAY - 1])

    assert refine(df, "1D", "high")["time"] == bar_time(df, BARS_PER_DAY - 1)


def test_an_untied_extreme_on_the_first_bar_still_refines_to_it():
    df = frame_with_high_at([0])

    assert refine(df, "1D", "high")["time"] == bar_time(df, 0)


def test_the_tie_break_is_per_period_not_across_the_whole_frame():
    # Both days carry the same high; each day must resolve inside itself.
    df = frame_with_high_at([10, 30, BARS_PER_DAY + 12, BARS_PER_DAY + 50],
                            days=2)

    assert refine(df, "1D", "high", 0)["time"] == bar_time(df, 30)
    assert refine(df, "1D", "high", 1)["time"] == bar_time(df, BARS_PER_DAY + 50)


def test_a_point_that_is_not_on_a_display_bar_does_not_refine():
    df = flat_frame()
    pmap = period_map(df, "1D", "15m")

    stray = {"time": pmap.display_times[0] + 7, "price": 100.0, "kind": "high"}
    assert to_canonical(stray, pmap) is None


# ----------------------------------------------------------------- round trip


@pytest.mark.parametrize("timeframe", ["1H", "4H", "1D"])
@pytest.mark.parametrize("kind", ["high", "low"])
def test_refining_then_coarsening_returns_the_same_display_point(timeframe, kind):
    df = gappy_frame()
    pmap = period_map(df, timeframe, "15m")

    for display in pmap.display_times:
        point = {"time": display, "price": pmap.extreme_price[(display, kind)],
                 "kind": kind}
        assert to_display(to_canonical(point, pmap), pmap) == point


def test_coarsening_then_refining_is_deliberately_not_the_identity():
    # Documented, not a defect: bar 10 is not the day's high, so it coarsens to
    # the day's high, which refines to bar 40. Nothing may "fix" this -- the
    # canonical list is simply never rewritten from a projection.
    df = coarsen_frame()
    pmap = period_map(df, "1D", "15m")
    original = point_at(df, 10, "high")

    round_tripped = to_canonical(to_display(original, pmap), pmap)
    assert round_tripped == {"time": bar_time(df, 40), "price": 150.0,
                             "kind": "high"}
    assert round_tripped != original


# -------------------------------------------------------------------- collapse


def five_day_frame():
    """Five days, each with its own distinct high and low, all at bar 12."""
    count = 5 * BARS_PER_DAY
    highs = [100.0] * count
    lows = [90.0] * count
    for day in range(5):
        highs[day * BARS_PER_DAY + 12] = 150.0 + day
        lows[day * BARS_PER_DAY + 12] = 50.0 - day
    return make_frame(highs, lows)


def spanning_pattern(df, positions, pattern_id="wide"):
    kinds = ["low", "high", "low", "high"]
    return make_pattern([point_at(df, position, kind)
                         for position, kind in zip(positions, kinds)],
                        pattern_id)


def test_a_pattern_whose_points_share_a_display_bar_collapses():
    df = five_day_frame()
    pmap = period_map(df, "1D", "15m")
    # Points 1 and 2 both land inside day 1.
    squashed = spanning_pattern(df, [3, BARS_PER_DAY + 3, BARS_PER_DAY + 40,
                                     2 * BARS_PER_DAY + 3], "squashed")

    assert projection_collapses(squashed, pmap)
    assert project_pattern(squashed, pmap) is None


def test_a_collapsing_pattern_is_dropped_and_its_siblings_survive():
    df = five_day_frame()
    pmap = period_map(df, "1D", "15m")
    squashed = spanning_pattern(df, [3, BARS_PER_DAY + 3, BARS_PER_DAY + 40,
                                     2 * BARS_PER_DAY + 3], "squashed")
    wide = spanning_pattern(df, [3, BARS_PER_DAY + 3, 2 * BARS_PER_DAY + 3,
                                 3 * BARS_PER_DAY + 3], "wide")

    projected = project_patterns([wide, squashed, wide | {"id": "wide2"}], pmap)

    assert [p["id"] for p in projected] == ["wide", "wide2"]
    assert not projection_collapses(wide, pmap)


def test_a_surviving_pattern_keeps_everything_but_its_points():
    df = five_day_frame()
    pmap = period_map(df, "1D", "15m")
    wide = spanning_pattern(df, [3, BARS_PER_DAY + 3, 2 * BARS_PER_DAY + 3,
                                 3 * BARS_PER_DAY + 3])
    wide["degree"] = "Minute"
    wide["color"] = "red"

    projected = project_pattern(wide, pmap)

    assert {k: v for k, v in projected.items() if k != "points"} \
        == {k: v for k, v in wide.items() if k != "points"}
    assert [p["time"] for p in projected["points"]] == pmap.display_times[:4]


def test_the_same_pattern_survives_at_a_finer_aggregation():
    # The squashed pattern above is only squashed on 1D: on 1H its two middle
    # points are hours apart, which is the whole reason a marking is projected
    # rather than stored per timeframe.
    df = five_day_frame()
    pmap = period_map(df, "1H", "15m")
    squashed = spanning_pattern(df, [3, BARS_PER_DAY + 3, BARS_PER_DAY + 40,
                                     2 * BARS_PER_DAY + 3], "squashed")

    assert not projection_collapses(squashed, pmap)


# ---------------------------------------------------------- unmappable points


def test_a_point_on_an_unknown_bar_drops_the_whole_pattern():
    df = five_day_frame()
    pmap = period_map(df, "1D", "15m")
    wide = spanning_pattern(df, [3, BARS_PER_DAY + 3, 2 * BARS_PER_DAY + 3,
                                 3 * BARS_PER_DAY + 3])
    # The data changed under the marking: this bar is no longer in the frame.
    wide["points"][2] = dict(wide["points"][2], time=wide["points"][2]["time"] + 1)

    assert project_pattern(wide, pmap) is None
    assert projection_collapses(wide, pmap)
    assert project_patterns([wide], pmap) == []


def test_a_malformed_point_drops_the_pattern_rather_than_half_projecting_it():
    df = five_day_frame()
    pmap = period_map(df, "1D", "15m")
    wide = spanning_pattern(df, [3, BARS_PER_DAY + 3, 2 * BARS_PER_DAY + 3,
                                 3 * BARS_PER_DAY + 3])
    wide["points"][1] = dict(wide["points"][1], kind="sideways")

    assert project_pattern(wide, pmap) is None
    assert project_patterns([wide], pmap) == []


def test_junk_in_a_pattern_list_is_dropped_not_raised():
    df = five_day_frame()
    pmap = period_map(df, "1D", "15m")

    assert project_patterns([None, "nope", {}, {"points": []}], pmap) == []
    assert project_patterns("not a list", pmap) == []


# ---------------------------------------------------------------------- purity


def test_no_projection_function_mutates_its_input():
    df = five_day_frame()
    pmap = period_map(df, "1D", "15m")
    wide = spanning_pattern(df, [3, BARS_PER_DAY + 3, 2 * BARS_PER_DAY + 3,
                                 3 * BARS_PER_DAY + 3])
    patterns = [wide]
    before = copy.deepcopy(patterns)
    point = point_at(df, 3, "high")
    point_before = copy.deepcopy(point)

    to_display(point, pmap)
    to_canonical(to_display(point, pmap), pmap)
    project_pattern(wide, pmap)
    project_patterns(patterns, pmap)
    projection_collapses(wide, pmap)

    assert patterns == before
    assert point == point_before


def test_a_projected_pattern_does_not_share_its_points_with_the_original():
    df = five_day_frame()
    pmap = period_map(df, "1D", "15m")
    wide = spanning_pattern(df, [3, BARS_PER_DAY + 3, 2 * BARS_PER_DAY + 3,
                                 3 * BARS_PER_DAY + 3])

    projected = project_pattern(wide, pmap)
    projected["points"][0]["price"] = -1.0

    assert wide["points"][0]["price"] != -1.0


# -------------------------------------------------------------------- ordering


@pytest.mark.parametrize("timeframe", ["1H", "4H", "1D"])
def test_every_projected_pattern_is_still_a_valid_pattern(timeframe):
    df = five_day_frame()
    pmap = period_map(df, timeframe, "15m")
    patterns = [
        spanning_pattern(df, [3, BARS_PER_DAY + 3, 2 * BARS_PER_DAY + 3,
                              3 * BARS_PER_DAY + 3], "wide"),
        spanning_pattern(df, [1, 5, 9, 13], "tight"),
        spanning_pattern(df, [12, BARS_PER_DAY + 12, BARS_PER_DAY + 13,
                              4 * BARS_PER_DAY + 12], "mixed"),
    ]

    projected = project_patterns(patterns, pmap)

    assert projected                     # not vacuous on any timeframe
    for pattern in projected:
        assert is_valid_pattern(pattern)
        times = [p["time"] for p in pattern["points"]]
        assert times == sorted(set(times))


@pytest.mark.parametrize("timeframe", ["1H", "4H", "1D"])
def test_refining_a_display_pattern_keeps_it_valid(timeframe):
    df = five_day_frame()
    pmap = period_map(df, timeframe, "15m")
    kinds = ["low", "high", "low", "high"]
    display = make_pattern([
        {"time": pmap.display_times[i], "kind": kind,
         "price": pmap.extreme_price[(pmap.display_times[i], kind)]}
        for i, kind in zip((0, 1, 2, 3), kinds)])

    refined = project_pattern(display, pmap, CANONICAL)

    assert is_valid_pattern(refined)
    assert project_pattern(refined, pmap) == display


# --------------------------------------------------------------- leg values
#
# A field the projection does not carry is silently dropped the moment the
# client changes aggregation -- the exact class of bug Phase 11b existed to fix,
# and a far worse one for the per-leg study values, which are typed by hand and
# cannot be recomputed from the chart.


LEG_VALUES = {"0": {"CMB": 12.34, "RSI": 45.6, "timeframe": "1D"},
              "2": {"CMB": -5.1, "timeframe": "15m"}}


@pytest.mark.parametrize("timeframe", ["1H", "4H", "1D"])
def test_leg_values_survive_a_projection_in_both_directions(timeframe):
    df = five_day_frame()
    pmap = period_map(df, timeframe, "15m")
    kinds = ["low", "high", "low", "high"]
    display = make_pattern([
        {"time": pmap.display_times[i], "kind": kind,
         "price": pmap.extreme_price[(pmap.display_times[i], kind)]}
        for i, kind in zip((0, 1, 2, 3), kinds)])
    display["leg_values"] = copy.deepcopy(LEG_VALUES)

    refined = project_pattern(display, pmap, CANONICAL)
    coarsened = project_pattern(refined, pmap)

    assert refined["leg_values"] == LEG_VALUES
    assert coarsened["leg_values"] == LEG_VALUES


def test_a_projected_pattern_does_not_share_its_leg_values_with_the_original():
    df = five_day_frame()
    pmap = period_map(df, "1D", "15m")
    wide = spanning_pattern(df, [3, BARS_PER_DAY + 3, 2 * BARS_PER_DAY + 3,
                                 3 * BARS_PER_DAY + 3])
    wide["leg_values"] = copy.deepcopy(LEG_VALUES)

    projected = project_pattern(wide, pmap)
    projected["leg_values"]["0"]["CMB"] = -1.0

    assert wide["leg_values"] == LEG_VALUES


def test_refine_passes_set_leg_values_through_untouched():
    # It carries no chart coordinates: a leg index means the same thing at every
    # aggregation, so there is nothing to refine.
    df = five_day_frame()
    pmap = period_map(df, "1D", "15m")
    event = {"type": "set_leg_values", "id": "p1", "leg_index": 3,
             "values": {"CMB": 12.34, "RSI": None}, "timeframe": "1D"}
    snapshot = copy.deepcopy(event)

    assert refine_event(event, pmap) == snapshot
    assert refine_event(event, pmap, pivot_magnets([], pmap), []) == snapshot
    assert refine_event(event, None) is event
    assert event == snapshot


# ------------------------------------------------------------------ magnetism
#
# A click lands on a display period, and refinement has to decide which base bar
# inside it the client meant. The period's extreme is the answer only when there
# is nothing better: a pivot he has already marked in that period, on that side
# of the bar, is what a wave count actually shares -- chained siblings share an
# endpoint and every child shares both endpoints with a parent leg.


def frame_with(highs=None, lows=None, days=1):
    """Flat days at 100/90 with the named bars overridden.

    Every expected magnet can then be read straight off the override table
    rather than recomputed from a price series.
    """
    count = days * BARS_PER_DAY
    high_values = [100.0] * count
    low_values = [90.0] * count
    for position, value in (highs or {}).items():
        high_values[position] = value
    for position, value in (lows or {}).items():
        low_values[position] = value
    return make_frame(high_values, low_values)


def extremes_frame(days=1):
    """Days whose high sits at bar 40 and whose low sits at bar 70."""
    highs = {day * BARS_PER_DAY + 40: 150.0 + day for day in range(days)}
    lows = {day * BARS_PER_DAY + 70: 50.0 - day for day in range(days)}
    return frame_with(highs, lows, days=days)


def one_point_pattern(df, position, kind, pattern_id):
    """A pattern carrying a single pivot -- all ``pivot_magnets`` ever reads.

    Structurally short of a real Zigzag on purpose: the magnet index is built
    from points alone, and a four-point stand-in would put three pivots into
    the frame that the test then has to reason about.
    """
    return {"id": pattern_id, "pattern_type": "Zigzag", "variation": "Zigzag",
            "degree": "Subminuette", "color": "yellow",
            "points": [point_at(df, position, kind)]}


def test_a_pivot_that_is_not_its_periods_extreme_still_magnetises_its_bar():
    # The whole defect in one assertion: a 15m pivot at bar 10 is nowhere near
    # the day's high at bar 40, and a 1D click on that day must still find it.
    df = extremes_frame()
    pmap = period_map(df, "1D", "15m")
    day = pmap.display_times[0]

    magnets = pivot_magnets([one_point_pattern(df, 10, "high", "p")], pmap)

    # A candidate names the bar, its price, and which point of which pattern it
    # is -- that last pair is what lets a caller recognise a leg.
    assert magnets == {(day, "high"): [(bar_time(df, 10), 100.0, "p", 0)]}
    assert pmap.extreme_time[(day, "high")] == bar_time(df, 40)


def test_a_pattern_spread_over_several_periods_magnetises_each_of_them():
    df = extremes_frame(days=4)
    pmap = period_map(df, "1D", "15m")
    positions = [10, BARS_PER_DAY + 20, 2 * BARS_PER_DAY + 30, 3 * BARS_PER_DAY + 50]
    kinds = ["high", "low", "high", "low"]
    pattern = make_pattern([point_at(df, position, kind)
                            for position, kind in zip(positions, kinds)])

    magnets = pivot_magnets([pattern], pmap)

    assert magnets == {
        (pmap.display_times[day], kind): [(bar_time(df, position),
                                           100.0 if kind == "high" else 90.0,
                                           "p1", day)]
        for day, (position, kind) in enumerate(zip(positions, kinds))}


def test_a_red_pattern_magnetises_exactly_like_a_yellow_one():
    # A red pattern's pivot is still a real pivot the client counts from, and
    # excluding it would make magnetism come and go as an unrelated overlap
    # somewhere else in the list is created and resolved.
    df = extremes_frame()
    pmap = period_map(df, "1D", "15m")
    pattern = one_point_pattern(df, 10, "high", "p")

    assert pivot_magnets([dict(pattern, color="red")], pmap) \
        == pivot_magnets([pattern], pmap)


def test_the_base_timeframe_has_no_magnets_at_all():
    df = extremes_frame()
    assert pivot_magnets([one_point_pattern(df, 10, "high", "p")], None) == {}


def test_a_pivot_the_frame_no_longer_has_magnetises_nothing():
    df = extremes_frame()
    pmap = period_map(df, "1D", "15m")
    stray = one_point_pattern(df, 10, "high", "p")
    stray["points"][0]["time"] += 1

    assert pivot_magnets([stray], pmap) == {}


@pytest.mark.parametrize("patterns", [
    "not a list", [None, "nope", {}], [{"points": "nope"}],
    [{"points": [None, {"time": 1}, {"time": 1, "kind": "sideways"}]}],
])
def test_junk_produces_no_magnets_rather_than_an_error(patterns):
    df = extremes_frame()
    assert pivot_magnets(patterns, period_map(df, "1D", "15m")) == {}


def test_a_pivot_with_an_unusable_price_never_wins_a_contest():
    # NaN loses every comparison it takes part in, so a pivot carrying one would
    # otherwise be impossible to displace once it was in.
    df = extremes_frame()
    pmap = period_map(df, "1D", "15m")
    day = pmap.display_times[0]
    broken = one_point_pattern(df, 10, "high", "broken")
    broken["points"][0]["price"] = float("nan")
    good = one_point_pattern(df, 30, "high", "good")

    assert pivot_magnets([broken, good], pmap) == {
        (day, "high"): [(bar_time(df, 30), 100.0, "good", 0)]}
    assert pivot_magnets([broken], pmap) == {}


def test_the_pivot_nearest_the_periods_extreme_wins_the_display_bar():
    # The display bar is drawn *at* its extreme, so of two pivots sharing it the
    # one closest to that extreme is the one the client is looking at.
    df = frame_with(highs={40: 150.0, 10: 120.0, 30: 140.0})
    pmap = period_map(df, "1D", "15m")
    day = pmap.display_times[0]

    magnets = pivot_magnets([one_point_pattern(df, 10, "high", "far"),
                             one_point_pattern(df, 30, "high", "near")], pmap)

    # Both are offered -- a caller that can tell them apart by intent needs the
    # loser too -- but the nearer one heads the list, which is the whole of the
    # behaviour for a caller that cannot.
    assert magnets == {(day, "high"): [(bar_time(df, 30), 140.0, "near", 0),
                                       (bar_time(df, 10), 120.0, "far", 0)]}


def test_two_pivots_at_the_same_price_resolve_to_the_later_one():
    df = frame_with(highs={40: 150.0, 10: 120.0, 30: 120.0})
    pmap = period_map(df, "1D", "15m")
    day = pmap.display_times[0]

    for order in ([10, 30], [30, 10]):       # input order must not decide it
        magnets = pivot_magnets(
            [one_point_pattern(df, position, "high", f"p{position}")
             for position in order], pmap)
        assert list(magnets) == [(day, "high")]
        assert magnets[(day, "high")][0] == (bar_time(df, 30), 120.0, "p30", 0)
        assert magnets[(day, "high")][1] == (bar_time(df, 10), 120.0, "p10", 0)


def test_a_high_pivot_never_magnetises_a_low_click():
    # A bar's high and its low are two different pivots; conflating them would
    # hang a child off the wrong one.
    df = extremes_frame()
    pmap = period_map(df, "1D", "15m")
    day = pmap.display_times[0]

    magnets = pivot_magnets([one_point_pattern(df, 10, "high", "p")], pmap)

    assert (day, "low") not in magnets


def test_a_pivot_shared_by_two_patterns_is_offered_once_for_each_of_them():
    # The normal case in a nest: one bar is the parent's leg end and the child's
    # terminus. Collapsing those two into one entry would throw away exactly the
    # information that says which of them forms a leg.
    df = extremes_frame(days=3)
    pmap = period_map(df, "1D", "15m")
    shared = point_at(df, BARS_PER_DAY + 10, "low")
    parent = make_pattern([point_at(df, 5, "high"), shared,
                           point_at(df, 2 * BARS_PER_DAY + 10, "high"),
                           point_at(df, 2 * BARS_PER_DAY + 80, "low")], "parent")
    child = make_pattern([shared,
                          point_at(df, BARS_PER_DAY + 20, "high"),
                          point_at(df, BARS_PER_DAY + 30, "low"),
                          point_at(df, 2 * BARS_PER_DAY + 10, "high")], "child")

    magnets = pivot_magnets([parent, child], pmap)

    # Day 1's low is 49.0 at bar 70, so all three pivots at 90.0 tie on distance
    # and resolve by time -- the later bar first, then the two on bar 10 in the
    # order their patterns appear.
    assert magnets[(pmap.display_times[1], "low")] == [
        (bar_time(df, BARS_PER_DAY + 30), 90.0, "child", 2),
        (bar_time(df, BARS_PER_DAY + 10), 90.0, "parent", 1),
        (bar_time(df, BARS_PER_DAY + 10), 90.0, "child", 0),
    ]
    # ...and the pivot the two patterns share at the far end likewise.
    assert [c[2:] for c in magnets[(pmap.display_times[2], "high")]] \
        == [("parent", 2), ("child", 3)]


# ------------------------------------------------------- magnetised refinement


def move_event(time, price, kind):
    return {"type": "move_point", "id": "p1", "point_index": 0,
            "time": time, "price": price, "kind": kind}


def completed_event(points):
    return {"type": "pattern_completed", "pattern": make_pattern(points)}


def test_a_click_on_a_period_holding_a_pivot_refines_to_that_pivot():
    df = frame_with(highs={40: 150.0, 10: 120.0})
    pmap = period_map(df, "1D", "15m")
    day = pmap.display_times[0]
    magnets = pivot_magnets([one_point_pattern(df, 10, "high", "p")], pmap)

    # The click carries the day's own extreme, which is what the chart drew.
    event = move_event(day, 150.0, "high")
    magnetised = refine_event(event, pmap, magnets)

    # Both the time *and* the price come from the pivot, not from the period.
    assert magnetised["time"] == bar_time(df, 10)
    assert magnetised["price"] == 120.0

    # Without magnets the same click still lands on the day's extreme.
    plain = refine_event(event, pmap)
    assert plain["time"] == bar_time(df, 40)
    assert plain["price"] == 150.0


def test_an_empty_magnet_index_changes_nothing():
    df = frame_with(highs={40: 150.0, 10: 120.0})
    pmap = period_map(df, "1D", "15m")
    event = move_event(pmap.display_times[0], 150.0, "high")

    assert refine_event(event, pmap, {}) == refine_event(event, pmap)
    assert refine_event(event, pmap, None) == refine_event(event, pmap)


def test_a_click_on_a_period_with_no_pivot_of_that_kind_takes_the_extreme():
    df = extremes_frame(days=2)
    pmap = period_map(df, "1D", "15m")
    magnets = pivot_magnets([one_point_pattern(df, 10, "high", "p")], pmap)

    # Day 1 holds no pivot at all, and day 0 holds no *low* pivot.
    assert refine_event(move_event(pmap.display_times[1], 151.0, "high"),
                        pmap, magnets)["time"] == bar_time(df, BARS_PER_DAY + 40)
    assert refine_event(move_event(pmap.display_times[0], 50.0, "low"),
                        pmap, magnets)["time"] == bar_time(df, 70)


def test_magnetism_never_applies_at_the_base_timeframe():
    # Display *is* canonical there: the client clicked the exact bar he meant,
    # and pulling it onto a neighbour would turn a mark into a correction.
    df = extremes_frame()
    daily = period_map(df, "1D", "15m")
    magnets = pivot_magnets([one_point_pattern(df, 10, "high", "p")], daily)
    assert magnets                                  # not vacuous

    adjacent = move_event(bar_time(df, 11), 100.0, "high")
    assert refine_event(adjacent, None, magnets) is adjacent


def test_a_completed_pattern_magnetises_every_point_it_can():
    df = extremes_frame(days=4)
    pmap = period_map(df, "1D", "15m")
    existing = [one_point_pattern(df, 10, "high", "a"),
                one_point_pattern(df, 2 * BARS_PER_DAY + 30, "high", "b")]
    magnets = pivot_magnets(existing, pmap)

    drawn = [{"time": pmap.display_times[day],
              "price": pmap.extreme_price[(pmap.display_times[day], kind)],
              "kind": kind}
             for day, kind in enumerate(["high", "low", "high", "low"])]
    refined = refine_event(completed_event(drawn), pmap, magnets)

    assert [p["time"] for p in refined["pattern"]["points"]] == [
        bar_time(df, 10),                           # magnetised
        bar_time(df, BARS_PER_DAY + 70),            # day 1's low, no pivot there
        bar_time(df, 2 * BARS_PER_DAY + 30),        # magnetised
        bar_time(df, 3 * BARS_PER_DAY + 70),        # day 3's low, no pivot there
    ]


def test_refining_with_magnets_is_pure():
    df = extremes_frame(days=2)
    pmap = period_map(df, "1D", "15m")
    magnets = pivot_magnets([one_point_pattern(df, 10, "high", "p")], pmap)
    event = move_event(pmap.display_times[0], 150.0, "high")
    snapshot = copy.deepcopy(event)

    refine_event(event, pmap, magnets)

    assert event == snapshot


# ------------------------------------------------------ magnetism and ordering
#
# Magnetism can pull two consecutive points of one pattern onto the same base
# bar, and a pattern whose times do not strictly increase is not a pattern.


def clashing_magnets(df, pmap, position=55):
    """Two existing patterns whose high and low pivots share one base bar."""
    return pivot_magnets([one_point_pattern(df, position, "high", "h"),
                          one_point_pattern(df, position, "low", "l")], pmap)


def test_two_points_magnetised_onto_one_bar_keep_one_magnet_between_them():
    # Phase 13 gave *both* offenders back their extremes. The search gives back
    # only as much as ordering costs: the period's extreme is the last candidate
    # on every list, so the earliest ordered assignment keeps point 0 on the pivot
    # it was drawn at and retreats only point 1. Strictly closer to the click than
    # the old fallback, and the marking still lands either way -- which is the
    # part that must never change.
    df = extremes_frame(days=3)
    pmap = period_map(df, "1D", "15m")
    magnets = clashing_magnets(df, pmap)

    # Day 0 is clicked twice -- once for its high, once for its low -- and both
    # clicks would magnet onto bar 55.
    drawn = [{"time": pmap.display_times[0], "price": 150.0, "kind": "high"},
             {"time": pmap.display_times[0], "price": 50.0, "kind": "low"},
             {"time": pmap.display_times[1], "price": 151.0, "kind": "high"},
             {"time": pmap.display_times[2], "price": 48.0, "kind": "low"}]
    refined = refine_event(completed_event(drawn), pmap, magnets)

    times = [p["time"] for p in refined["pattern"]["points"]]
    assert times[:2] == [bar_time(df, 55), bar_time(df, 70)]
    assert times == sorted(set(times))
    assert is_valid_pattern(refined["pattern"])


def test_only_the_magnetised_offender_gives_up_its_magnet():
    df = extremes_frame(days=3)
    pmap = period_map(df, "1D", "15m")
    # Only the high side clashes: bar 75 is *after* day 0's low bar (70), so a
    # magnetised high would sit behind the low that follows it.
    magnets = pivot_magnets([one_point_pattern(df, 75, "high", "h"),
                             one_point_pattern(df, BARS_PER_DAY + 10, "high", "k")],
                            pmap)

    drawn = [{"time": pmap.display_times[0], "price": 150.0, "kind": "high"},
             {"time": pmap.display_times[0], "price": 50.0, "kind": "low"},
             {"time": pmap.display_times[1], "price": 151.0, "kind": "high"},
             {"time": pmap.display_times[2], "price": 48.0, "kind": "low"}]
    refined = refine_event(completed_event(drawn), pmap, magnets)

    times = [p["time"] for p in refined["pattern"]["points"]]
    # Point 0 gave up its magnet; point 2, which was not part of the clash,
    # keeps the one it had.
    assert times[0] == bar_time(df, 40)
    assert times[2] == bar_time(df, BARS_PER_DAY + 10)
    assert times == sorted(set(times))


def test_an_order_that_cannot_be_saved_either_way_is_refused():
    # Day 0's low bar (70) comes after its high bar (40), so asking for the low
    # first is unorderable with or without a magnet in play.
    df = extremes_frame(days=3)
    pmap = period_map(df, "1D", "15m")
    magnets = clashing_magnets(df, pmap)

    drawn = [{"time": pmap.display_times[0], "price": 50.0, "kind": "low"},
             {"time": pmap.display_times[0], "price": 150.0, "kind": "high"},
             {"time": pmap.display_times[1], "price": 49.0, "kind": "low"},
             {"time": pmap.display_times[2], "price": 152.0, "kind": "high"}]

    assert refine_event(completed_event(drawn), pmap, magnets) is None


# ------------------------------------------------------ relation-aware magnets
#
# Price distance carries no information about intent. In a dense count two
# same-kind pivots inside one display bar draw at the same point and their
# glyphs stack, so the click cannot separate them -- and the nearer-the-extreme
# rule then picks whichever happens to be nearer, which in a dense count is
# routinely the wrong one. What *can* separate them is what the finished pattern
# would mean: a combination of endpoints that spans exactly one leg of an
# existing pattern is a child, and a child is what is being drawn.


def phase13(magnets):
    """The magnet index as Phase 13 built it: one winner per (bar, kind).

    Truncating each list to its head *is* the old rule -- the head is the pivot
    nearest that period's extreme -- so a test can assert the old behaviour
    without hardcoding a bar the frame's construction might later move.
    """
    return {key: candidates[:1] for key, candidates in magnets.items()}


def contended_frame(days=4):
    """Days whose extremes sit at bars 80 and 90, where no pivot is ever drawn.

    Day 1's low is 9.0 at bar 90, and two low pivots sit inside that day: an
    unrelated older count's at bar 60 (89.0) and a parent's at bar 20 (90.0).
    The unrelated one is *nearer* that 9.0, so price distance prefers it. Day 3's
    high mirrors the shape on the other side of the bar: 153.0 at bar 80, the
    unrelated count at bar 60 (149.0), the parent at bar 20 (100.0).

    Each day's high and low extremes are on different bars so that a click with
    no magnet at all still refines to a bar of its own -- otherwise two adjacent
    magnetless clicks in one day would collapse and the ordering fallback, not
    the choice under test, would decide the answer.
    """
    highs = {day * BARS_PER_DAY + 80: 150.0 + day for day in range(days)}
    lows = {day * BARS_PER_DAY + 90: 10.0 - day for day in range(days)}
    highs[3 * BARS_PER_DAY + 60] = 149.0
    lows[BARS_PER_DAY + 60] = 89.0
    return frame_with(highs, lows, days=days)


def contended_counts(df):
    """The parent whose leg 1 is about to be counted, and an unrelated count.

    The unrelated one deliberately touches the same two contended bars with its
    *first* and *last* points -- indices 0 and 3, which are not consecutive, so
    it offers no leg between them and cannot be mistaken for the parent.
    """
    parent = make_pattern([point_at(df, 20, "high"),
                           point_at(df, BARS_PER_DAY + 20, "low"),
                           point_at(df, 3 * BARS_PER_DAY + 20, "high"),
                           point_at(df, 3 * BARS_PER_DAY + 90, "low")], "parent")
    unrelated = make_pattern([point_at(df, BARS_PER_DAY + 60, "low"),
                              point_at(df, 2 * BARS_PER_DAY + 30, "high"),
                              point_at(df, 2 * BARS_PER_DAY + 70, "low"),
                              point_at(df, 3 * BARS_PER_DAY + 60, "high")], "unrelated")
    return parent, unrelated


def child_clicks(pmap):
    """The four daily clicks that draw a child over the parent's leg 1."""
    return [{"time": pmap.display_times[day],
             "price": pmap.extreme_price[(pmap.display_times[day], kind)],
             "kind": kind}
            for day, kind in ((1, "low"), (2, "high"), (2, "low"), (3, "high"))]


def test_the_contended_day_goes_to_the_unrelated_pivot_and_now_to_the_parents():
    # The whole contract in one test. Both endpoints of the child land in a day
    # where the unrelated count's pivot is nearer the day's extreme than the
    # parent's, so the price-distance rule attaches the child to the unrelated
    # count -- it then spans no leg of anything, derives no degree, and collides
    # with the parent. Reading the same lists for a *leg* instead picks the
    # parent's pivot at both ends, even though both are the farther one.
    df = contended_frame()
    pmap = period_map(df, "1D", "15m")
    parent, unrelated = contended_counts(df)
    magnets = pivot_magnets([parent, unrelated], pmap)

    day1_low = magnets[(pmap.display_times[1], "low")]
    day3_high = magnets[(pmap.display_times[3], "high")]
    assert [c[2:] for c in day1_low] == [("unrelated", 0), ("parent", 1)]
    assert [c[2:] for c in day3_high] == [("unrelated", 3), ("parent", 2)]

    drawn = child_clicks(pmap)
    old = refine_event(completed_event(drawn), pmap, phase13(magnets))
    assert old["pattern"]["points"][0] == unrelated["points"][0]
    assert old["pattern"]["points"][-1] == unrelated["points"][-1]

    new = refine_event(completed_event(drawn), pmap, magnets, [parent, unrelated])
    assert new["pattern"]["points"][0] == parent["points"][1]
    assert new["pattern"]["points"][-1] == parent["points"][2]


def test_an_interior_point_stays_put_when_it_has_no_structure_to_express():
    # Interior points are steered now too, but only by the score: with no child
    # to adopt in the middle days they tie at zero and the tie-break -- earliest
    # in the list, which is the price-distance head -- leaves them exactly where
    # Phase 13 put them. Preferring a leg at the ends never makes the middle of a
    # pattern wander on its own account.
    df = contended_frame()
    pmap = period_map(df, "1D", "15m")
    patterns = list(contended_counts(df))
    magnets = pivot_magnets(patterns, pmap)
    drawn = child_clicks(pmap)

    old = refine_event(completed_event(drawn), pmap, phase13(magnets))
    new = refine_event(completed_event(drawn), pmap, magnets, patterns)

    interior = slice(1, -1)
    assert [p["time"] for p in new["pattern"]["points"]][interior] \
        == [p["time"] for p in old["pattern"]["points"]][interior]


def test_with_no_leg_to_form_every_point_takes_the_head_of_its_list():
    # The fallback is Phase 13 exactly, asserted against the head-of-list choice
    # rather than a hardcoded bar. The parent is removed, so the only pivots left
    # in the contended days are the unrelated count's non-consecutive ends.
    df = contended_frame()
    pmap = period_map(df, "1D", "15m")
    _parent, unrelated = contended_counts(df)
    magnets = pivot_magnets([unrelated], pmap)
    drawn = child_clicks(pmap)
    # Not vacuous: every click still finds a magnet, they just never make a leg.
    assert all(_candidates_for(point, magnets) for point in drawn)

    assert refine_event(completed_event(drawn), pmap, magnets, [unrelated]) \
        == refine_event(completed_event(drawn), pmap, phase13(magnets))


def test_two_patterns_offering_the_same_leg_resolve_to_the_earlier_candidates():
    # Both rivals expose a leg between the same two display bars, so both
    # combinations are leg-forming. The one whose candidates come earliest in
    # their lists wins -- earliest meaning nearest each day's extreme, which is
    # the tie-break the client can already reason about -- and it wins the same
    # way every time.
    df = contended_frame()
    pmap = period_map(df, "1D", "15m")
    far, _unrelated = contended_counts(df)
    near = make_pattern([point_at(df, 30, "high"),
                         point_at(df, BARS_PER_DAY + 60, "low"),
                         point_at(df, 3 * BARS_PER_DAY + 60, "high"),
                         point_at(df, 3 * BARS_PER_DAY + 91, "low")], "near")
    magnets = pivot_magnets([far, near], pmap)
    assert [c[2] for c in magnets[(pmap.display_times[1], "low")]] == ["near", "parent"]

    drawn = child_clicks(pmap)
    refined = [refine_event(completed_event(drawn), pmap, magnets, [far, near])
               for _ in range(3)]

    assert all(r == refined[0] for r in refined)
    assert refined[0]["pattern"]["points"][0] == near["points"][1]
    assert refined[0]["pattern"]["points"][-1] == near["points"][2]


def unorderable_leg_frame(low_extreme):
    """A day whose only leg-forming low pivot sits *after* the next click.

    The head of that day's low list is bar 10; the pivot that would form a leg is
    bar 90, and the pattern's second click magnets to bar 50 in the same day. So
    taking the leg puts point 0 behind point 1. ``low_extreme`` places the day's
    own low, which is where the ordering fallback would send point 0.
    """
    return frame_with(highs={80: 160.0}, lows={low_extreme: 50.0, 10: 55.0}, days=3)


def unorderable_leg_counts(df):
    host = make_pattern([point_at(df, 5, "high"), point_at(df, 90, "low"),
                         point_at(df, 2 * BARS_PER_DAY + 20, "high"),
                         point_at(df, 2 * BARS_PER_DAY + 90, "low")], "host")
    other = make_pattern([point_at(df, 10, "low"), point_at(df, 50, "high"),
                          point_at(df, BARS_PER_DAY + 40, "low"),
                          point_at(df, 2 * BARS_PER_DAY + 30, "high")], "other")
    return host, other


def unorderable_leg_clicks(pmap):
    return [{"time": pmap.display_times[day],
             "price": pmap.extreme_price[(pmap.display_times[day], kind)],
             "kind": kind}
            for day, kind in ((0, "low"), (0, "high"), (1, "low"), (2, "high"))]


@pytest.mark.parametrize("low_extreme", [20, 95])
def test_a_leg_that_cannot_be_ordered_never_costs_the_client_the_marking(low_extreme):
    # Ordering is a constraint inside the search, so the scoring assignment is
    # never even considered and there is nothing left to rescue afterwards. Both
    # ways Phase 14's preferred-then-checked leg could fail end at the plain
    # answer instead of at a bar the client never clicked (low_extreme=20) or at a
    # refusal that would have cost him the marking (low_extreme=95).
    df = unorderable_leg_frame(low_extreme)
    pmap = period_map(df, "1D", "15m")
    host, other = unorderable_leg_counts(df)
    patterns = [host, other]
    magnets = pivot_magnets(patterns, pmap)
    drawn = unorderable_leg_clicks(pmap)

    # Not vacuous: the host really does expose a leg between two bars that are
    # both on offer, and taking it really would put point 0 after point 1.
    consecutive, _endpoints = _relation_pairs(patterns)
    leg = ((host["points"][1]["time"], "low"), (host["points"][2]["time"], "high"))
    assert leg in consecutive
    offered = [_point_candidates(point, pmap, magnets) for point in drawn]
    assert leg[0] in [(c["time"], c["kind"]) for c in offered[0]]
    assert leg[1] in [(c["time"], c["kind"]) for c in offered[-1]]
    assert all(c["time"] < leg[0][0] for c in offered[1])

    refined = refine_event(completed_event(drawn), pmap, magnets, patterns)
    plain = refine_event(completed_event(drawn), pmap, phase13(magnets))

    assert plain is not None
    assert refined == plain
    # The Phase 14 regression probe: the bars the client's clicks land on.
    assert [p["time"] for p in refined["pattern"]["points"]] \
        == [bar_time(df, position) for position in (10, 50, 136, 222)]


# ------------------------------------------------- whole-pattern selection
#
# The client marks bottom-up: "Okay, so I've notated all the sub minuets. Now I
# can notate the minuet degree." A pattern drawn over counts already marked
# adopts them through its *interior* points -- a child hangs off a consecutive
# pair of its parent's points -- so choosing only the endpoints by intent leaves
# the ordinary direction of his workflow broken. Every point is chosen together
# now, for the assignment that expresses the most structure.


def bottom_up_frame(days=7):
    """Days whose extremes sit at bars 80 and 90, where no pivot is ever drawn.

    Bar 10 is where the client's own 15m pivots sit, at the frame's flat 100/90,
    and bar 50 where two older rival counts' do, at 140/20 -- deliberately the
    nearer of the two to every day's extreme, so price distance prefers a rival
    in every single day.
    """
    highs = {}
    lows = {}
    for day in range(days):
        highs[day * BARS_PER_DAY + 80] = 150.0 + day
        lows[day * BARS_PER_DAY + 90] = 10.0 - day
        highs[day * BARS_PER_DAY + 50] = 140.0
        lows[day * BARS_PER_DAY + 50] = 20.0
    return frame_with(highs, lows, days=days)


def rival_counts(df):
    """Two older counts on bar 50, between them covering all four clicked days.

    Two rather than one so that the head-of-list answer is not simply an existing
    pattern's own point list -- the defect has to be the parent adopting nothing,
    not the parent being a duplicate.
    """
    rival = make_pattern([point_at(df, 0 * BARS_PER_DAY + 50, "high"),
                          point_at(df, 2 * BARS_PER_DAY + 50, "low"),
                          point_at(df, 5 * BARS_PER_DAY + 50, "high"),
                          point_at(df, 6 * BARS_PER_DAY + 50, "low")], "rival")
    older = make_pattern([point_at(df, 1 * BARS_PER_DAY + 50, "high"),
                          point_at(df, 3 * BARS_PER_DAY + 50, "low"),
                          point_at(df, 4 * BARS_PER_DAY + 50, "high"),
                          point_at(df, 5 * BARS_PER_DAY + 50, "low")], "older")
    return rival, older


def marked_children(df):
    """Three counts marked at 15m, chained end to end over bar 10 of days 0-6.

    Their four shared endpoints are exactly the pivots a larger-degree pattern
    drawn over them has to find, and not one of them is its day's extreme.
    """
    corners = [point_at(df, day * BARS_PER_DAY + 10, kind)
               for day, kind in zip((0, 2, 4, 6), ("high", "low", "high", "low"))]
    interiors = [(0 * BARS_PER_DAY + 60, 1 * BARS_PER_DAY + 30),
                 (2 * BARS_PER_DAY + 60, 3 * BARS_PER_DAY + 30),
                 (4 * BARS_PER_DAY + 60, 5 * BARS_PER_DAY + 30)]
    children = []
    for index, (start, end) in enumerate(zip(corners, corners[1:])):
        first, second = interiors[index]
        kinds = ("low", "high") if start["kind"] == "high" else ("high", "low")
        children.append(make_pattern(
            [dict(start), point_at(df, first, kinds[0]), point_at(df, second, kinds[1]),
             dict(end)], f"child {index}"))
    return corners, children


def parent_clicks(pmap):
    """The four daily clicks that draw the larger degree over those children."""
    return [{"time": pmap.display_times[day],
             "price": pmap.extreme_price[(pmap.display_times[day], kind)],
             "kind": kind}
            for day, kind in ((0, "high"), (2, "low"), (4, "high"), (6, "low"))]


def test_a_parent_drawn_over_marked_children_adopts_all_three():
    # The headline, both ways. Every one of the parent's four clicks lands in a
    # day where a rival's pivot is nearer that day's extreme, so the head-of-list
    # rule puts all four on rivals and the parent adopts nothing. Scoring the
    # whole assignment instead picks the four pivots the children actually hang
    # off, because doing so forms three adoptions rather than none.
    df = bottom_up_frame()
    pmap = period_map(df, "1D", "15m")
    rival, older = rival_counts(df)
    corners, children = marked_children(df)
    patterns = [rival, older] + children
    magnets = pivot_magnets(patterns, pmap)
    drawn = parent_clicks(pmap)

    old = refine_event(completed_event(drawn), pmap, phase13(magnets))
    assert [p["time"] for p in old["pattern"]["points"]] == [
        rival["points"][0]["time"], rival["points"][1]["time"],
        older["points"][2]["time"], rival["points"][3]["time"]]

    new = refine_event(completed_event(drawn), pmap, magnets, patterns)
    assert new["pattern"]["points"] == corners

    # ...which is three adoptions: every child's two ends are now a consecutive
    # pair of the new pattern's points, which is what ``child_leg_index`` reads.
    legs = list(zip(new["pattern"]["points"], new["pattern"]["points"][1:]))
    for child in children:
        assert (child["points"][0], child["points"][-1]) in legs


def test_the_bottom_up_parent_is_not_dragged_anywhere_by_the_head_rule():
    # The guard on the test above: the head-of-list assignment really is the one
    # Phase 14 produced, endpoints included. Phase 14 steered the two ends and
    # nothing else, and neither end has a leg to slot into here, so all four
    # points took the head and the children were left behind.
    df = bottom_up_frame()
    pmap = period_map(df, "1D", "15m")
    rival, older = rival_counts(df)
    _corners, children = marked_children(df)
    patterns = [rival, older] + children
    magnets = pivot_magnets(patterns, pmap)
    drawn = parent_clicks(pmap)

    heads = [_point_candidates(point, pmap, magnets)[0] for point in drawn]
    assert refine_event(completed_event(drawn), pmap, phase13(magnets)) \
        == refine_event(completed_event(drawn), pmap, magnets, [rival, older])
    assert [p["time"] for p in
            refine_event(completed_event(drawn), pmap,
                         phase13(magnets))["pattern"]["points"]] \
        == [head["time"] for head in heads]


def test_with_nothing_to_adopt_the_parent_lands_on_the_head_of_every_list():
    # The no-regression case, asserted against the head-of-list choice rather
    # than a hardcoded bar: delete the children and the same four clicks fall
    # back to exactly what Phase 13 always did.
    df = bottom_up_frame()
    pmap = period_map(df, "1D", "15m")
    patterns = list(rival_counts(df))
    magnets = pivot_magnets(patterns, pmap)
    drawn = parent_clicks(pmap)
    assert all(_candidates_for(point, magnets) for point in drawn)   # not vacuous

    refined = refine_event(completed_event(drawn), pmap, magnets, patterns)

    assert refined["pattern"]["points"] == [
        _point_candidates(point, pmap, magnets)[0] for point in drawn]


def both_directions_frame():
    """The bottom-up frame, one day longer, for a middle degree drawn at 1D."""
    return bottom_up_frame(days=8)


def both_directions_counts(df):
    """A grandparent and a grandchild marked at 15m, plus a rival on bar 50.

    The middle degree drawn over them slots into the grandparent's leg 1 *and*
    takes the grandchild under its own leg 1 -- two relations from one pattern.
    """
    grandparent = make_pattern([point_at(df, 0 * BARS_PER_DAY + 10, "high"),
                                point_at(df, 2 * BARS_PER_DAY + 10, "low"),
                                point_at(df, 5 * BARS_PER_DAY + 10, "high"),
                                point_at(df, 7 * BARS_PER_DAY + 10, "low")],
                               "grandparent")
    grandchild = make_pattern([point_at(df, 3 * BARS_PER_DAY + 10, "high"),
                               point_at(df, 3 * BARS_PER_DAY + 40, "low"),
                               point_at(df, 3 * BARS_PER_DAY + 70, "high"),
                               point_at(df, 4 * BARS_PER_DAY + 10, "low")],
                              "grandchild")
    rival = make_pattern([point_at(df, 2 * BARS_PER_DAY + 50, "low"),
                          point_at(df, 3 * BARS_PER_DAY + 50, "high"),
                          point_at(df, 4 * BARS_PER_DAY + 50, "low"),
                          point_at(df, 5 * BARS_PER_DAY + 50, "high")], "rival")
    return grandparent, grandchild, rival


def both_directions_clicks(pmap):
    return [{"time": pmap.display_times[day],
             "price": pmap.extreme_price[(pmap.display_times[day], kind)],
             "kind": kind}
            for day, kind in ((2, "low"), (3, "high"), (4, "low"), (5, "high"))]


def test_a_middle_degree_slots_into_a_parent_and_adopts_a_child_at_once():
    df = both_directions_frame()
    pmap = period_map(df, "1D", "15m")
    grandparent, grandchild, rival = both_directions_counts(df)
    patterns = [grandparent, grandchild, rival]
    magnets = pivot_magnets(patterns, pmap)
    drawn = both_directions_clicks(pmap)

    refined = refine_event(completed_event(drawn), pmap, magnets, patterns)
    points = refined["pattern"]["points"]

    # Slotted in: the two ends are a consecutive pair of the grandparent's.
    assert (points[0], points[-1]) == (grandparent["points"][1], grandparent["points"][2])
    # Adopted: leg 1 is the grandchild's two ends.
    assert (points[1], points[2]) == (grandchild["points"][0], grandchild["points"][-1])


def test_two_relations_beat_one_whose_candidates_come_earlier():
    # Scoring precedence, spelled out in list positions. The assignment that only
    # slots into the grandparent sits earlier in the candidate lists at both of
    # the points that differ, and still loses to the one that also adopts.
    df = both_directions_frame()
    pmap = period_map(df, "1D", "15m")
    grandparent, grandchild, rival = both_directions_counts(df)
    patterns = [grandparent, grandchild, rival]
    magnets = pivot_magnets(patterns, pmap)
    drawn = both_directions_clicks(pmap)
    offered = [_point_candidates(point, pmap, magnets) for point in drawn]

    chosen = refine_event(completed_event(drawn), pmap,
                          magnets, patterns)["pattern"]["points"]
    taken = [candidates.index(point) for candidates, point in zip(offered, chosen)]

    # The single-relation rival assignment: the same two ends, both interior
    # points on the head of their lists.
    single = [taken[0], 0, 0, taken[-1]]
    assert single < taken                       # earlier in the lists, and it lost
    assert taken == [1, 2, 1, 1]


def test_the_same_input_always_gives_the_same_answer():
    # The tie-break is a total order over list positions, so nothing is left to
    # dict iteration order -- which is what decides two candidates naming the
    # very same bar at the very same price.
    df = both_directions_frame()
    pmap = period_map(df, "1D", "15m")
    patterns = list(both_directions_counts(df))
    magnets = pivot_magnets(patterns, pmap)
    drawn = both_directions_clicks(pmap)

    answers = [refine_event(completed_event(drawn), pmap, magnets, patterns)
               for _ in range(5)]
    # The same index, rebuilt with its keys inserted in the opposite order.
    reversed_keys = {key: magnets[key] for key in reversed(list(magnets))}
    answers.append(refine_event(completed_event(drawn), pmap, reversed_keys, patterns))

    assert all(answer == answers[0] for answer in answers)


def test_the_candidate_list_is_bounded_and_always_offers_the_extreme():
    # A pathological pile-up of pivots in one display bar cannot blow the search
    # up, and the period's own extreme is on the end of every list -- which is
    # what the search retreats to when magnets cannot be ordered.
    df = extremes_frame()
    pmap = period_map(df, "1D", "15m")
    crowd = [one_point_pattern(df, position, "high", f"p{position}")
             for position in range(1, 30)]
    magnets = pivot_magnets(crowd, pmap)
    assert len(magnets[(pmap.display_times[0], "high")]) == 29

    click = {"time": pmap.display_times[0], "price": 150.0, "kind": "high"}
    offered = _point_candidates(click, pmap, magnets)

    assert len(offered) == MAX_CANDIDATES_PER_POINT
    assert offered[-1] == to_canonical(click, pmap)


def test_relation_pairs_reads_both_directions_off_the_list():
    df = bottom_up_frame()
    _corners, children = marked_children(df)
    consecutive, endpoints = _relation_pairs(children)

    first = children[0]["points"]
    assert ((first[0]["time"], "high"), (first[1]["time"], "low")) in consecutive
    assert ((first[0]["time"], "high"), (first[-1]["time"], "low")) in endpoints
    # ...and dropping one pattern drops both of its contributions.
    without, without_ends = _relation_pairs(children, exclude_id="child 0")
    assert ((first[0]["time"], "high"), (first[1]["time"], "low")) not in without
    assert ((first[0]["time"], "high"), (first[-1]["time"], "low")) not in without_ends


# --------------------------------------------------- relation-aware move_point


def child_on_the_wrong_pivot(df, parent, unrelated):
    """A child whose last point is the parent's but whose first is not.

    Exactly the state the price-distance rule leaves behind, and what dragging
    the loose end at 1D has to repair.
    """
    return make_pattern([dict(unrelated["points"][0]),
                         point_at(df, 2 * BARS_PER_DAY + 30, "high"),
                         point_at(df, 2 * BARS_PER_DAY + 70, "low"),
                         dict(parent["points"][2])], "child")


def drag(pattern_id, index, time, price, kind):
    return {"type": "move_point", "id": pattern_id, "point_index": index,
            "time": time, "price": price, "kind": kind}


def test_dragging_a_first_point_lands_on_the_pivot_that_makes_the_leg_exact():
    df = contended_frame()
    pmap = period_map(df, "1D", "15m")
    parent, unrelated = contended_counts(df)
    child = child_on_the_wrong_pivot(df, parent, unrelated)
    patterns = [parent, unrelated, child]
    magnets = pivot_magnets(patterns, pmap)

    day1 = pmap.display_times[1]
    event = drag("child", 0, day1, pmap.extreme_price[(day1, "low")], "low")

    # Without the canonical list there is nothing to test a candidate against,
    # so the head wins and the child stays hung off the unrelated count.
    assert refine_event(event, pmap, magnets)["time"] == unrelated["points"][0]["time"]

    # With it, the candidate whose *next* point is the child's own far end -- the
    # parent's leg end -- is the one taken.
    repaired = refine_event(event, pmap, magnets, patterns)
    assert repaired["time"] == parent["points"][1]["time"]
    assert repaired["price"] == parent["points"][1]["price"]


def test_dragging_a_last_point_looks_backwards_from_the_candidate():
    df = contended_frame()
    pmap = period_map(df, "1D", "15m")
    parent, unrelated = contended_counts(df)
    # This time the *first* point is already right and the last one is loose.
    child = make_pattern([dict(parent["points"][1]),
                          point_at(df, 2 * BARS_PER_DAY + 30, "high"),
                          point_at(df, 2 * BARS_PER_DAY + 70, "low"),
                          dict(unrelated["points"][3])], "child")
    patterns = [parent, unrelated, child]
    magnets = pivot_magnets(patterns, pmap)

    day3 = pmap.display_times[3]
    event = drag("child", 3, day3, pmap.extreme_price[(day3, "high")], "high")

    assert refine_event(event, pmap, magnets)["time"] == unrelated["points"][3]["time"]
    assert refine_event(event, pmap, magnets, patterns)["time"] \
        == parent["points"][2]["time"]


def test_dragging_an_interior_point_takes_the_head_like_it_always_did():
    # An interior point does not decide whether this pattern spans a leg, so it
    # is left exactly where the price-distance rule puts it.
    df = contended_frame()
    pmap = period_map(df, "1D", "15m")
    parent, unrelated = contended_counts(df)
    patterns = [parent, unrelated, child_on_the_wrong_pivot(df, parent, unrelated)]
    magnets = pivot_magnets(patterns, pmap)

    day2 = pmap.display_times[2]
    event = drag("child", 1, day2, pmap.extreme_price[(day2, "high")], "high")

    assert refine_event(event, pmap, magnets, patterns) \
        == refine_event(event, pmap, magnets)


def test_a_pattern_is_never_dragged_onto_a_leg_of_itself():
    df = contended_frame()
    pmap = period_map(df, "1D", "15m")
    _parent, unrelated = contended_counts(df)
    # Points 2 and 3 of this count are a leg whose far end is the very point that
    # stays still, so without the guard dragging point 0 into day 1 would land on
    # its own point 2 and declare the pattern a child of itself.
    lone = make_pattern([point_at(df, 20, "high"),
                         point_at(df, 50, "low"),
                         point_at(df, BARS_PER_DAY + 20, "low"),
                         point_at(df, 2 * BARS_PER_DAY + 20, "high")], "lone")
    patterns = [lone, unrelated]
    magnets = pivot_magnets(patterns, pmap)
    assert [c[2:] for c in magnets[(pmap.display_times[1], "low")]] \
        == [("unrelated", 0), ("lone", 2)]

    day1 = pmap.display_times[1]
    event = drag("lone", 0, day1, pmap.extreme_price[(day1, "low")], "low")

    assert refine_event(event, pmap, magnets, patterns)["time"] \
        == unrelated["points"][0]["time"]


def test_a_leg_past_the_next_point_is_passed_over_rather_than_swallowed():
    # ``_apply_move_point`` refuses a point that does not stay strictly between
    # its neighbours, so steering the drag onto a leg beyond the next point would
    # attach nothing at all -- the reducer would drop the move and the client
    # would watch his point snap back. The plain choice, which the reducer does
    # accept, is worth more than a leg that never lands.
    df = unorderable_leg_frame(20)
    pmap = period_map(df, "1D", "15m")
    host, other = unorderable_leg_counts(df)
    # This child's far end is the host's leg end, so the host's bar-90 pivot is
    # leg-forming for its first point -- but its own second point is at bar 50.
    child = make_pattern([point_at(df, 30, "low"), point_at(df, 50, "high"),
                          point_at(df, BARS_PER_DAY + 40, "low"),
                          dict(host["points"][2])], "child")
    patterns = [host, other, child]
    magnets = pivot_magnets(patterns, pmap)

    day0 = pmap.display_times[0]
    event = drag("child", 0, day0, pmap.extreme_price[(day0, "low")], "low")
    refined = refine_event(event, pmap, magnets, patterns)

    assert refined["time"] < child["points"][1]["time"]
    assert refined == refine_event(event, pmap, magnets)


def test_an_unknown_pattern_id_falls_back_to_the_head():
    df = contended_frame()
    pmap = period_map(df, "1D", "15m")
    patterns = list(contended_counts(df))
    magnets = pivot_magnets(patterns, pmap)

    day1 = pmap.display_times[1]
    event = drag("nobody", 0, day1, pmap.extreme_price[(day1, "low")], "low")

    assert refine_event(event, pmap, magnets, patterns) \
        == refine_event(event, pmap, magnets)


@pytest.mark.parametrize("patterns", ["not a list", [], [None], [{"id": "child"}],
                                      [{"id": "child", "points": "nope"}],
                                      [{"id": "child", "points": [{"time": 1}]}]])
def test_a_junk_canonical_list_refines_exactly_as_no_list_at_all(patterns):
    df = contended_frame()
    pmap = period_map(df, "1D", "15m")
    magnets = pivot_magnets(list(contended_counts(df)), pmap)

    day1 = pmap.display_times[1]
    event = drag("child", 0, day1, pmap.extreme_price[(day1, "low")], "low")

    assert refine_event(event, pmap, magnets, patterns) \
        == refine_event(event, pmap, magnets)


def test_dragging_an_interior_point_keeps_the_adoptions_hanging_off_it():
    # An interior point is what an existing child hangs off, so dragging one is
    # exactly how a parent shakes its children loose. Nudging the parent's second
    # point inside its own day must bring it back to the pivot two children share,
    # not to the rival's nearer one.
    df = bottom_up_frame()
    pmap = period_map(df, "1D", "15m")
    rival, older = rival_counts(df)
    corners, children = marked_children(df)
    parent = make_pattern([dict(corner) for corner in corners], "parent")
    patterns = [rival, older, parent] + children
    magnets = pivot_magnets(patterns, pmap)

    day2 = pmap.display_times[2]
    event = drag("parent", 1, day2, pmap.extreme_price[(day2, "low")], "low")

    # Phase 14 left an interior point on the head of its list, which here is the
    # rival's pivot -- and that shakes two of the three children loose.
    assert refine_event(event, pmap, magnets)["time"] == rival["points"][1]["time"]

    held = refine_event(event, pmap, magnets, patterns)
    assert held["time"] == corners[1]["time"]
    assert held["price"] == corners[1]["price"]


def skipped_neighbour_counts(df):
    """A count being edited, one whose adoption sits past its neighbour, one not.

    ``kid`` would be adopted by putting ``big``'s second point on bar 80 of day 1
    -- but ``big``'s own third point is bar 60 of that day, so the reducer would
    throw the move out and the client would watch his point snap back.
    """
    big = make_pattern([point_at(df, 0 * BARS_PER_DAY + 10, "high"),
                        point_at(df, 1 * BARS_PER_DAY + 30, "low"),
                        point_at(df, 1 * BARS_PER_DAY + 60, "high"),
                        point_at(df, 2 * BARS_PER_DAY + 10, "low")], "big")
    kid = make_pattern([point_at(df, 0 * BARS_PER_DAY + 10, "high"),
                        point_at(df, 0 * BARS_PER_DAY + 50, "low"),
                        point_at(df, 1 * BARS_PER_DAY + 20, "high"),
                        point_at(df, 1 * BARS_PER_DAY + 80, "low")], "kid")
    other = make_pattern([point_at(df, 1 * BARS_PER_DAY + 45, "low"),
                          point_at(df, 1 * BARS_PER_DAY + 70, "high"),
                          point_at(df, 2 * BARS_PER_DAY + 40, "low"),
                          point_at(df, 2 * BARS_PER_DAY + 70, "high")], "other")
    return big, kid, other


def test_an_interior_candidate_past_its_neighbour_is_skipped_and_the_drag_lands():
    df = bottom_up_frame(days=3)
    pmap = period_map(df, "1D", "15m")
    big, kid, other = skipped_neighbour_counts(df)
    patterns = [big, kid, other]
    magnets = pivot_magnets(patterns, pmap)

    day1 = pmap.display_times[1]
    event = drag("big", 1, day1, pmap.extreme_price[(day1, "low")], "low")

    # The adoption really is on offer, and it really is past the next point.
    _consecutive, endpoints = _relation_pairs(patterns, exclude_id="big")
    tempting = kid["points"][-1]
    assert ((big["points"][0]["time"], "high"), (tempting["time"], "low")) in endpoints
    assert tempting["time"] > big["points"][2]["time"]

    # So Phase 13's head-of-list choice would hand the reducer a move it refuses,
    # and the choice made instead is the earliest candidate that actually fits.
    assert refine_event(event, pmap, magnets)["time"] == tempting["time"]
    landed = refine_event(event, pmap, magnets, patterns)
    assert landed["time"] == other["points"][0]["time"]
    assert big["points"][0]["time"] < landed["time"] < big["points"][2]["time"]


def test_refining_with_the_canonical_list_is_still_pure():
    df = contended_frame()
    pmap = period_map(df, "1D", "15m")
    parent, unrelated = contended_counts(df)
    patterns = [parent, unrelated, child_on_the_wrong_pivot(df, parent, unrelated)]
    magnets = pivot_magnets(patterns, pmap)
    day1 = pmap.display_times[1]
    event = drag("child", 0, day1, pmap.extreme_price[(day1, "low")], "low")
    snapshot = copy.deepcopy((event, patterns))

    refine_event(event, pmap, magnets, patterns)
    refine_event(completed_event(child_clicks(pmap)), pmap, magnets, patterns)

    assert (event, patterns) == snapshot


# ------------------------------------------------------------------- pickling


def test_a_period_map_pickles():
    import pickle

    df = five_day_frame()
    pmap = period_map(df, "4H", "15m")

    assert pickle.loads(pickle.dumps(pmap)) == pmap

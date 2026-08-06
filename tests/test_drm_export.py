"""
Tests for the Date Range Manager export built from Wave Analysis markings.

The acceptance test is the round trip: whatever ``strategies.drm_export`` writes
has to come back out of ``data.loader.load_drm`` / ``parse_drm_periods``
unchanged, because those two functions are what every consuming tab reads a DRM
with and neither of them is allowed to move for this.
"""
import copy
import json

import pandas as pd
import pytest

from config.wave_analysis import point_labels
from config.wave_projection import period_map
from data.helpers import PRIMARY_SECONDARY_MAP
from data.loader import load_drm, parse_drm_periods
from strategies import drm_export
from strategies.drm_export import (DRM_SKELETON, PERIOD_FORMAT,
                                   TRACKED_PRIMARY_POSITIONS,
                                   TRACKED_SECONDARY_POSITIONS, build_drm_rows,
                                   build_drm_sheets, build_drm_workbook,
                                   build_wave_json, dropped_total,
                                   not_applicable_total, primary_name,
                                   secondary_name, summary_lines)
from ui import wave_analysis_tab


# --------------------------------------------------------------- builders
#
# Canonical pattern dicts as the tab holds them. Times are plain ints -- the
# projection has already happened by the time the export sees a pattern -- except
# where a real timestamp is needed to exercise the DRM's own format.


def make_pattern(pattern_id, pattern_type, variation, points,
                 color="yellow", degree="Minor"):
    return {
        "id": pattern_id,
        "pattern_type": pattern_type,
        "variation": variation,
        "degree": degree,
        "color": color,
        "points": points,
    }


def alternating(pattern_id, pattern_type, variation, start_time, step,
                prices=None, first_kind="low", color="yellow"):
    """A pattern marching forward by ``step`` with alternating high/low pivots."""
    count = len(point_labels(pattern_type, variation))
    other = "high" if first_kind == "low" else "low"
    if prices is None:
        prices = [100.0 + i for i in range(count)]
    assert len(prices) == count
    points = [{"time": start_time + i * step,
               "price": float(prices[i]),
               "kind": first_kind if i % 2 == 0 else other}
              for i in range(count)]
    return make_pattern(pattern_id, pattern_type, variation, points, color=color)


def spanning(pattern_id, host, leg, pattern_type="Impulse", variation="Impulse",
             prices=None, color="yellow"):
    """A pattern spanning exactly ``host``'s leg ``leg``, endpoint for endpoint.

    Endpoints have to match on time *and* kind or the relation will not see it as
    a child at all, so both are taken straight off the host.
    """
    start, end = host["points"][leg], host["points"][leg + 1]
    count = len(point_labels(pattern_type, variation))
    gap = end["time"] - start["time"]
    assert gap >= count - 1, "leg too narrow to hold this pattern"

    times = [start["time"] + round(i * gap / (count - 1)) for i in range(count)]
    times[-1] = end["time"]
    assert all(b > a for a, b in zip(times, times[1:])), "times must increase"

    other = "high" if start["kind"] == "low" else "low"
    assert end["kind"] == other, "a leg's ends always alternate"

    if prices is None:
        prices = [start["price"] + i * (end["price"] - start["price"]) / (count - 1)
                  for i in range(count)]
        prices[0], prices[-1] = start["price"], end["price"]
    assert len(prices) == count

    points = [{"time": times[i], "price": float(prices[i]),
               "kind": start["kind"] if i % 2 == 0 else other}
              for i in range(count)]
    return make_pattern(pattern_id, pattern_type, variation, points, color=color)


def simple_nest(prefix, start_time, step=1000, parent_leg=0, child_leg=0,
                parent_type=("Impulse", "Impulse"), child_type=("Impulse", "Impulse"),
                grandchild_type=("Impulse", "Impulse")):
    """The three levels one DRM row needs: grandparent, child, grandchild."""
    parent = alternating(prefix + "P", parent_type[0], parent_type[1],
                         start_time, step)
    child = spanning(prefix + "C", parent, parent_leg, *child_type)
    grandchild = spanning(prefix + "G", child, child_leg, *grandchild_type)
    return [parent, child, grandchild]


def ints(seconds):
    """A formatter for int times: identity, so a period reads as its raw bounds."""
    return None if seconds is None else str(seconds)


def drm_time(seconds):
    """The real thing: an epoch second as the string a period cell carries."""
    return pd.Timestamp(seconds, unit="s").strftime(PERIOD_FORMAT)


def cells(sheet_frame, primary, secondary):
    """The period strings of one skeleton row, empties removed."""
    row = sheet_frame[(sheet_frame.iloc[:, 0] == primary)
                      & (sheet_frame.iloc[:, 1] == secondary)]
    return [v for v in row.iloc[:, 2:].values.flatten() if isinstance(v, str)]


# ------------------------------------------------------------- leg labelling


IMPULSE = ("Impulse", "Impulse")
ZIGZAG = ("Zigzag", "Zigzag")


@pytest.mark.parametrize("leg,expected,child_type", [
    (0, "W.(1)", IMPULSE), (1, "W.(2)", ZIGZAG), (2, "W.(3)", IMPULSE),
    (3, "W.(4)", ZIGZAG), (4, "W.(5)", IMPULSE)])
def test_a_child_on_leg_k_names_the_wave_ending_that_leg(leg, expected, child_type):
    # Leg k runs points[k] -> points[k+1] and is the wave labelled point k+1.
    # An off-by-one here is invisible in the output and wrong in every row.
    # The child alternates impulse/zigzag because the DRM's own secondaries do:
    # a motive leg's sub-waves are 1/3/5, a corrective leg's are A/C.
    patterns = simple_nest("n", 0, step=1000, parent_leg=leg, child_type=child_type)
    rows, _summary = build_drm_rows(patterns, ints)
    assert [row[1] for row in rows] == [expected]


@pytest.mark.parametrize("leg,expected,child_type", [
    (0, "W.(A)", ("Impulse", "Impulse")),
    (1, "W.(B)", ("Zigzag", "Zigzag")),
    (2, "W.(C)", ("Impulse", "Impulse")),
])
def test_a_zigzag_parents_legs_name_waves_a_b_and_c(leg, expected, child_type):
    patterns = simple_nest("z", 0, step=1000, parent_leg=leg,
                           parent_type=("Zigzag", "Zigzag"), child_type=child_type)
    rows, _summary = build_drm_rows(patterns, ints)
    assert [row[1] for row in rows] == [expected]


# ------------------------------------------------------------------- naming


def test_primary_name_round_trips_every_key_of_the_map():
    for primary in PRIMARY_SECONDARY_MAP:
        label = primary[3:-1]                   # 'W.(3)' -> '3'
        assert primary_name(label) == primary


@pytest.mark.parametrize("label", ["D", "E", "Z", "", "2b", None, 3])
def test_primary_name_rejects_a_leg_the_drm_cannot_hold(label):
    assert primary_name(label) is None


def test_secondary_name_writes_a_leg_and_the_pattern_forming_it():
    assert secondary_name("1", "Impulse") == "W.1 Impulse"
    assert secondary_name("C", "Impulse") == "W.C Impulse"
    assert secondary_name("Y", "Zigzag") == "W.Y Zigzag"


def test_secondary_name_merges_the_zigzags_a_with_the_double_zigzags_w():
    assert secondary_name("A", "Zigzag") == "W.A/W Zigzag"
    assert secondary_name("W", "Zigzag") == "W.A/W Zigzag"


@pytest.mark.parametrize("pattern_type",
                         ["Flat", "Triangle", "Combination", None, "impulse"])
def test_secondary_name_refuses_a_type_the_drm_has_no_word_for(pattern_type):
    # The format names exactly one corrective form, Zigzag. Calling a flat a
    # zigzag would be inventing data, so these are reported instead.
    assert secondary_name("1", pattern_type) is None


@pytest.mark.parametrize("label", ["1", "3", "5"])
def test_a_diagonal_is_logged_as_the_impulse_it_stands_in_for(label):
    # The DRM's type word is binary -- motive or corrective -- and the motive
    # primaries offer no secondary but Impulse, so a diagonal in a motive
    # position has exactly one name the format can express.
    assert secondary_name(label, "Diagonal") == f"W.{label} Impulse"
    assert drm_export.SECONDARY_TYPE_ALIASES == {"Diagonal": "Impulse"}


def test_every_name_secondary_name_can_build_for_a_real_leg_is_known_or_dropped():
    # Sanity on the vocabulary itself: nothing outside the map's 7 secondaries
    # may ever reach a sheet.
    known = {s for secs in PRIMARY_SECONDARY_MAP.values() for s in secs}
    assert secondary_name("2", "Impulse") == "W.2 Impulse"
    assert "W.2 Impulse" not in known


# ---------------------------------------------------------------- direction


@pytest.mark.parametrize("prices,sheet", [
    ([100.0, 110.0, 105.0, 120.0, 115.0, 130.0], "Bullish"),
    ([130.0, 120.0, 125.0, 110.0, 115.0, 100.0], "Bearish"),
])
def test_the_legs_own_direction_decides_the_sheet(prices, sheet):
    parent = alternating("P", "Impulse", "Impulse", 0, 1000)
    child = spanning("C", parent, 0, prices=prices)
    grandchild = spanning("G", child, 0)
    rows, _summary = build_drm_rows([parent, child, grandchild], ints)
    assert [row[0] for row in rows] == [sheet]


def test_a_flat_leg_produces_no_row_and_is_counted():
    parent = alternating("P", "Impulse", "Impulse", 0, 1000)
    child = spanning("C", parent, 0, prices=[100.0] * 6)
    grandchild = spanning("G", child, 0)
    rows, summary = build_drm_rows([parent, child, grandchild], ints)

    assert rows == []
    assert summary["dropped"]["flat_leg"] == 1
    assert summary["candidates"] == summary["rows"] + not_applicable_total(summary) \
        + dropped_total(summary)


# -------------------------------------------------------- the three-level walk


def test_a_three_level_nest_produces_exactly_the_expected_rows():
    parent = alternating("P", "Impulse", "Impulse", 0, 1000)
    child = spanning("C", parent, 2)                # leg 2 -> wave 3 -> W.(3)
    first = spanning("G1", child, 0)                # leg 0 -> wave 1 -> W.1 Impulse
    second = spanning("G2", child, 2)               # leg 2 -> wave 3 -> W.3 Impulse
    rows, summary = build_drm_rows([parent, child, first, second], ints)

    assert rows == [
        ("Bullish", "W.(3)", "W.1 Impulse",
         str(child["points"][0]["time"]), str(child["points"][1]["time"])),
        ("Bullish", "W.(3)", "W.3 Impulse",
         str(child["points"][2]["time"]), str(child["points"][3]["time"])),
    ]
    assert summary["candidates"] == 2
    assert dropped_total(summary) == 0


def test_a_two_level_nest_names_nothing():
    # No grandchild means no secondary, and a DRM row without one is not a row.
    parent = alternating("P", "Impulse", "Impulse", 0, 1000)
    child = spanning("C", parent, 0)
    rows, summary = build_drm_rows([parent, child], ints)

    assert rows == []
    assert summary["candidates"] == 0


def test_a_root_with_only_grandchildren_below_it_names_nothing():
    # The child needs a parent of its own -- that parent is what the primary is.
    parent = alternating("P", "Impulse", "Impulse", 0, 1000)
    child = spanning("C", parent, 0)
    grandchild = spanning("G", child, 0)
    rows, _summary = build_drm_rows([child, grandchild], ints)
    assert rows == []
    assert len(build_drm_rows([parent, child, grandchild], ints)[0]) == 1


# ------------------------------------------------------------ pair validation


def test_a_structurally_derivable_pair_the_drm_has_no_cell_for_is_dropped():
    # W.(2) is a corrective leg: its secondaries are A/C impulses and zigzags,
    # never 'W.1 Impulse'. The row is derivable and is still not a DRM cell.
    patterns = simple_nest("n", 0, step=1000, parent_leg=1)
    rows, summary = build_drm_rows(patterns, ints)

    assert rows == []
    assert summary["dropped"]["unknown_pair"] == 1
    sheets = build_drm_sheets(rows)
    assert len(sheets["Bullish"]) == 43 and len(sheets["Bearish"]) == 43


@pytest.mark.parametrize("grandchild_type", [("Flat", "Flat"),
                                             ("Triangle", "Contracting Triangle"),
                                             ("Combination", "Double Three")])
def test_a_grandchild_the_drm_cannot_name_is_reported_by_type(grandchild_type):
    patterns = simple_nest("n", 0, step=1000, grandchild_type=grandchild_type)
    rows, summary = build_drm_rows(patterns, ints)

    assert rows == []
    assert summary["dropped"]["no_secondary"] == {grandchild_type[0]: 1}


def test_a_diagonal_in_a_motive_position_now_produces_a_row():
    patterns = simple_nest("n", 0, step=1000,
                           grandchild_type=("Diagonal", "Leading Diagonal"))
    rows, summary = build_drm_rows(patterns, ints)

    assert [row[1:3] for row in rows] == [("W.(1)", "W.1 Impulse")]
    assert dropped_total(summary) == not_applicable_total(summary) == 0


def test_a_diagonals_period_survives_the_round_trip(tmp_path):
    step = 6 * 3600
    parent = alternating("P", "Impulse", "Impulse", epoch(1), step)
    child = spanning("C", parent, 0)
    grandchild = spanning("G", child, 0, "Diagonal", "Ending Diagonal")

    rows, summary = build_drm_rows([parent, child, grandchild], drm_time)
    assert summary["rows"] == 1

    path = tmp_path / "diagonal.xlsx"
    path.write_bytes(build_drm_workbook(build_drm_sheets(rows)))
    with open(path, "rb") as handle:
        frame = load_drm(handle, "Bullish")

    assert parse_drm_periods(frame, "Bullish", "W.(1)", "W.1 Impulse") == [
        (pd.to_datetime(rows[0][3], format=PERIOD_FORMAT),
         pd.to_datetime(rows[0][4], format=PERIOD_FORMAT))]


# ------------------------------------------------- tracked, versus lost
#
# Most unrepresentable candidates are not losses: the DRM simply never records
# an even impulse leg or a triangle's D. Reporting those as drops makes a
# correct export read as lossy, which is the opposite of what a summary is for.


def test_the_tracked_positions_are_derived_from_the_map_not_written_out():
    assert TRACKED_PRIMARY_POSITIONS == {"1", "2", "3", "4", "5",
                                         "A", "B", "C", "W", "X", "Y"}
    # 'W.A/W Zigzag' merges two structural positions and both of them count.
    assert TRACKED_SECONDARY_POSITIONS == {"1", "3", "5", "A", "C", "W", "Y"}
    assert TRACKED_PRIMARY_POSITIONS == drm_export._primary_positions(
        PRIMARY_SECONDARY_MAP)
    assert TRACKED_SECONDARY_POSITIONS == drm_export._secondary_positions(
        PRIMARY_SECONDARY_MAP)


def use_map(monkeypatch, mapping):
    """Point the whole module at another DRM vocabulary.

    The two position sets are re-derived by the module's own functions rather
    than listed here, which is the whole point: a row added to the client's
    format has to be followed automatically.
    """
    monkeypatch.setattr(drm_export, "PRIMARY_SECONDARY_MAP", mapping)
    monkeypatch.setattr(drm_export, "TRACKED_PRIMARY_POSITIONS",
                        drm_export._primary_positions(mapping))
    monkeypatch.setattr(drm_export, "TRACKED_SECONDARY_POSITIONS",
                        drm_export._secondary_positions(mapping))


def test_a_wave_2_child_leg_is_not_applicable_rather_than_dropped():
    parent = alternating("P", "Impulse", "Impulse", 0, 1000)
    child = spanning("C", parent, 0)                # W.(1)
    grandchild = spanning("G", child, 1)            # leg 1 -> wave 2
    rows, summary = build_drm_rows([parent, child, grandchild], ints)

    assert rows == []
    assert summary["not_applicable"]["untracked_secondary"] == 1
    assert dropped_total(summary) == 0


def test_a_triangles_d_parent_leg_is_not_applicable_rather_than_dropped():
    patterns = simple_nest("t", 0, step=1000, parent_leg=3,
                           parent_type=("Triangle", "Contracting Triangle"))
    rows, summary = build_drm_rows(patterns, ints)

    assert rows == []
    assert summary["not_applicable"]["untracked_primary"] == 1
    assert dropped_total(summary) == 0


def test_a_pair_absent_from_the_map_for_any_other_reason_is_still_dropped():
    # Wave 3 *is* a position the DRM records -- but only as an impulse, and only
    # under a motive primary. A zigzag there is real work the format cannot hold.
    parent = alternating("P", "Impulse", "Impulse", 0, 1000)
    child = spanning("C", parent, 0)                            # W.(1)
    grandchild = spanning("G", child, 2, "Zigzag", "Zigzag")    # 'W.3 Zigzag'
    rows, summary = build_drm_rows([parent, child, grandchild], ints)

    assert rows == []
    assert summary["dropped"]["unknown_pair"] == 1
    assert not_applicable_total(summary) == 0


def test_the_classification_follows_the_map_rather_than_a_fixed_list(monkeypatch):
    # Pretend the client's format grew a row for the very positions it skips
    # today: the same markings must stop being "not applicable" without a line
    # of classification code changing.
    parent = alternating("P", "Impulse", "Impulse", 0, 1000)
    child = spanning("C", parent, 1)                # leg 1 -> wave 2 -> W.(2)
    grandchild = spanning("G", child, 1)            # leg 1 -> wave 2

    _rows, before = build_drm_rows([parent, child, grandchild], ints)
    assert before["not_applicable"]["untracked_secondary"] == 1

    use_map(monkeypatch, {"W.(2)": ["W.2 Impulse"]})
    rows, after = build_drm_rows([parent, child, grandchild], ints)

    assert [row[1:3] for row in rows] == [("W.(2)", "W.2 Impulse")]
    assert not_applicable_total(after) == dropped_total(after) == 0


def test_a_primary_the_narrowed_map_no_longer_holds_becomes_not_applicable(monkeypatch):
    patterns = simple_nest("n", 0, step=1000)       # W.(1) / W.1 Impulse
    assert build_drm_rows(patterns, ints)[1]["rows"] == 1

    use_map(monkeypatch, {"W.(3)": ["W.1 Impulse"]})
    rows, summary = build_drm_rows(patterns, ints)

    assert rows == []
    assert summary["not_applicable"]["untracked_primary"] == 1
    assert dropped_total(summary) == 0


# ------------------------------------------------------------------- dropping


def test_a_degenerate_period_is_dropped_and_counted():
    patterns = simple_nest("n", 0, step=1000)
    # Both pivots of the leg land in one bar of the export timeframe.
    rows, summary = build_drm_rows(patterns, lambda time: "same bar")

    assert rows == []
    assert summary["dropped"]["degenerate_period"] == 1


def test_a_pivot_the_frame_no_longer_has_is_dropped_and_counted():
    patterns = simple_nest("n", 0, step=1000)
    rows, summary = build_drm_rows(patterns, lambda time: None)

    assert rows == []
    assert summary["dropped"]["unmappable_time"] == 1


def test_two_grandchildren_on_one_leg_keep_the_first_and_count_the_rest():
    parent = alternating("P", "Impulse", "Impulse", 0, 1000)
    child = spanning("C", parent, 0)
    first = spanning("G1", child, 0)
    second = spanning("G2", child, 0, "Zigzag", "Zigzag")
    rows, summary = build_drm_rows([parent, child, first, second], ints)

    assert [row[2] for row in rows] == ["W.1 Impulse"]
    assert summary["dropped"]["ambiguous_leg"] == 1
    assert summary["candidates"] == 2


def test_the_same_period_reached_twice_is_written_once():
    # Two markings over the same span, differing only inside: at an export
    # resolution coarse enough to swallow the difference they are one period.
    parent = alternating("P", "Impulse", "Impulse", 0, 1000)
    first = spanning("C1", parent, 0)
    second = spanning("C2", parent, 0, "Zigzag", "Zigzag")
    g_first = spanning("G1", first, 0)
    g_second = spanning("G2", second, 0, "Zigzag", "Zigzag")
    patterns = [parent, first, second, g_first, g_second]

    # Both children start on the parent's own pivot, so their first legs share a
    # start; the coarse formatter puts their ends in one bar too.
    coarse = lambda time: str((time // 500) * 500)
    rows, summary = build_drm_rows(patterns, coarse)

    assert len(rows) == len(set(rows))
    assert summary["dropped"]["duplicate"] + summary["dropped"]["degenerate_period"] >= 1
    assert summary["candidates"] == summary["rows"] + not_applicable_total(summary) \
        + dropped_total(summary)


def test_a_period_reachable_through_two_identical_cells_appears_once():
    # The plain case: two separate nests whose legs collapse onto the same bars.
    left = simple_nest("l", 0, step=100000)
    right = simple_nest("r", 1000, step=100000)
    coarse = lambda time: str((time // 10000) * 10000)
    rows, summary = build_drm_rows(left + right, coarse)

    assert len(rows) == 1
    assert summary["candidates"] == 2
    assert summary["dropped"]["duplicate"] == 1


# ---------------------------------------------------------------- red patterns


def test_a_red_pattern_removes_its_rows_and_is_reported():
    patterns = simple_nest("n", 0, step=1000)
    baseline, _summary = build_drm_rows(patterns, ints)
    assert len(baseline) == 1

    reddened = list(patterns)
    reddened[1] = dict(reddened[1], color="red")
    rows, summary = build_drm_rows(reddened, ints)

    assert rows == []
    assert summary["red_patterns"] == 1
    assert summary["candidates"] == 0
    assert any("red pattern" in line for line in summary_lines(summary))


# -------------------------------------------------------------- sheet shape


def test_both_sheets_carry_the_whole_skeleton_in_map_order():
    sheets = build_drm_sheets([])
    assert len(DRM_SKELETON) == 43

    for name in ("Bullish", "Bearish"):
        frame = sheets[name]
        assert list(frame.columns) == [name, ""]
        assert len(frame) == 43
        assert list(zip(frame.iloc[:, 0], frame.iloc[:, 1])) == DRM_SKELETON
        # Every row carries its primary, not just the first of each block.
        assert frame.iloc[:, 0].notna().all()


def test_periods_land_in_columns_two_onward_sorted_by_start_time():
    # Day-first strings, so lexical order is not chronological order: the later
    # period sorts *first* as text and must still be written second.
    rows = [
        ("Bullish", "W.(3)", "W.1 Impulse", "28.09.2025_17:00", "30.09.2025_19:00"),
        ("Bullish", "W.(3)", "W.1 Impulse", "03.10.2025_11:00", "05.10.2025_15:00"),
        ("Bullish", "W.(3)", "W.1 Impulse", "01.01.2025_03:00", "02.01.2025_07:00"),
    ]
    frame = build_drm_sheets(rows)["Bullish"]
    assert list(frame.columns) == ["Bullish", "", 1, 2, 3]
    assert cells(frame, "W.(3)", "W.1 Impulse") == [
        "01.01.2025_03:00, 02.01.2025_07:00",
        "28.09.2025_17:00, 30.09.2025_19:00",
        "03.10.2025_11:00, 05.10.2025_15:00",
    ]
    # Untouched cells stay empty rather than becoming an empty period.
    assert cells(frame, "W.(1)", "W.1 Impulse") == []


def test_a_sheet_with_no_rows_still_emits_the_skeleton():
    frame = build_drm_sheets([("Bearish", "W.(2)", "W.C Impulse", "a", "b")])["Bullish"]
    assert len(frame) == 43
    assert list(frame.columns) == ["Bullish", ""]


# ------------------------------------------------------- the round trip proper


def epoch(day, hour=3):
    return int(pd.Timestamp(f"2024-01-{day:02d} {hour:02d}:00").timestamp())


def test_the_workbook_round_trips_through_load_drm_and_parse_drm_periods(tmp_path):
    """The acceptance test: what we write is what every consuming tab reads."""
    # Three nests: two land in the same cell, one lands on the Bearish sheet.
    step = 6 * 3600
    first = simple_nest("a", epoch(1), step=step)
    second = simple_nest("b", epoch(11), step=step)
    falling = alternating("cP", "Impulse", "Impulse", epoch(21), step,
                          prices=[130.0, 120.0, 125.0, 110.0, 115.0, 100.0])
    falling_child = spanning("cC", falling, 0,
                             prices=[130.0, 128.0, 129.0, 126.0, 127.0, 120.0])
    falling_grandchild = spanning("cG", falling_child, 0)

    patterns = first + second + [falling, falling_child, falling_grandchild]
    rows, summary = build_drm_rows(patterns, drm_time)
    assert summary["rows"] == 3 and dropped_total(summary) == 0

    path = tmp_path / "generated.xlsx"
    path.write_bytes(build_drm_workbook(build_drm_sheets(rows)))

    expected = {}
    for sheet, primary, secondary, start, end in rows:
        expected.setdefault((sheet, primary, secondary), []).append(
            (pd.to_datetime(start, format=PERIOD_FORMAT),
             pd.to_datetime(end, format=PERIOD_FORMAT)))

    for sheet in ("Bullish", "Bearish"):
        with open(path, "rb") as handle:
            frame = load_drm(handle, sheet)
        assert frame[sheet].notna().all()
        for primary, secondary in DRM_SKELETON:
            got = parse_drm_periods(frame, sheet, primary, secondary)
            assert got == expected.get((sheet, primary, secondary), [])
            for start, end in got:
                assert isinstance(start, pd.Timestamp) and isinstance(end, pd.Timestamp)

    # The cell with several periods, and one with none.
    with open(path, "rb") as handle:
        bullish = load_drm(handle, "Bullish")
    assert len(parse_drm_periods(bullish, "Bullish", "W.(1)", "W.1 Impulse")) == 2
    assert parse_drm_periods(bullish, "Bullish", "W.(4)", "W.Y Zigzag") == []


def test_an_empty_export_still_round_trips(tmp_path):
    path = tmp_path / "empty.xlsx"
    path.write_bytes(build_drm_workbook(build_drm_sheets([])))
    with open(path, "rb") as handle:
        frame = load_drm(handle, "Bullish")
    assert len(frame) == 43
    assert parse_drm_periods(frame, "Bullish", "W.(1)", "W.1 Impulse") == []


# ---------------------------------------------------------------- the JSON


def test_the_json_export_is_dumpable_and_keeps_the_patterns_intact():
    patterns = simple_nest("n", 0, step=1000)
    payload = build_wave_json(patterns, "GC 15m.csv", "15m")
    text = json.dumps(payload, allow_nan=False)

    restored = json.loads(text)
    assert restored["dataset_key"] == "GC 15m.csv"
    assert restored["base_timeframe"] == "15m"
    assert restored["patterns"] == patterns


def test_the_json_export_survives_an_unusable_price():
    patterns = simple_nest("n", 0, step=1000)
    patterns[1]["points"][0]["price"] = float("nan")
    payload = build_wave_json(patterns, "GC 15m.csv", "15m")

    # One unusable pivot must not cost the client the backup of everything else.
    text = json.dumps(payload, allow_nan=False)
    assert json.loads(text)["patterns"][1]["points"][0]["price"] is None


def test_the_json_export_tolerates_junk():
    assert build_wave_json("not a list", "k", "15m")["patterns"] == []


# ------------------------------------------------------------- summary honesty


def mixed_nest():
    """One marked structure firing every bucket of both groups."""
    # A triangle root: leg 0 is wave A and leg 2 is wave C (both real primaries),
    # leg 3 is wave D (a position the DRM has no primary for).
    root = alternating("P", "Triangle", "Contracting Triangle", 0, 100000)

    good = spanning("C", root, 0)                       # -> W.(A)
    extra = spanning("E", root, 2)                      # -> W.(C)
    orphan = spanning("D", root, 3)                     # -> wave D, no primary

    patterns = [root, good, extra, orphan]
    # Under the good child, whose primary W.(A) holds motive secondaries: two
    # rows, a diagonal that is logged as the impulse it stands in for, two
    # counter-trend legs the DRM never records, and a leg marked twice.
    patterns.append(spanning("G1", good, 0))                            # W.1 Impulse
    patterns.append(spanning("G2", good, 1))                            # wave 2
    patterns.append(spanning("G3", good, 2, "Diagonal", "Leading Diagonal"))
    patterns.append(spanning("G4", good, 3, "Flat", "Flat"))            # wave 4
    patterns.append(spanning("G5", good, 4))                            # W.5 Impulse
    patterns.append(spanning("G6", good, 4, "Zigzag", "Zigzag"))        # the extra
    # Under the other child: a type the DRM cannot name in a position it does
    # record, and a well-formed pair that is not a cell of this primary.
    patterns.append(spanning("E1", extra, 0, "Flat", "Flat"))           # wave 1
    patterns.append(spanning("E2", extra, 2, "Zigzag", "Zigzag"))       # 'W.3 Zigzag'
    # Under the orphan: one candidate that dies on its primary alone.
    patterns.append(spanning("H1", orphan, 0))
    # A red marking takes no part in the relation at all.
    patterns.append(alternating("R", "Impulse", "Impulse", 900000, 1000, color="red"))
    return patterns


def test_the_summary_splits_what_was_never_tracked_from_what_was_lost():
    rows, summary = build_drm_rows(mixed_nest(), ints)

    assert summary["red_patterns"] == 1
    assert summary["candidates"] == 9
    assert summary["rows"] == len(rows) == 3        # G1, G3 (the diagonal), G5
    assert summary["not_applicable"] == {
        "untracked_primary": 1,         # H1, under the triangle's D leg
        "untracked_secondary": 2,       # G2 on wave 2, G4 on wave 4
    }
    assert summary["dropped"] == {
        "no_primary": 0,
        "no_secondary": {"Flat": 1},    # E1, a flat on a wave the DRM does record
        "unknown_pair": 1,              # E2, 'W.3 Zigzag' under W.(C)
        "flat_leg": 0,
        "unmappable_time": 0,
        "degenerate_period": 0,
        "ambiguous_leg": 1,             # G6 on a leg G5 already owns
        "duplicate": 0,
    }


@pytest.mark.parametrize("formatter", ["ints", "patchy", "coarse"])
def test_the_three_way_accounting_is_exact(formatter):
    # Nothing may vanish without landing in exactly one bucket of exactly one
    # group, whatever the export timeframe does to the timestamps.
    lost = mixed_nest()[1]["points"][1]["time"]

    def patchy(time):
        return None if time == lost else str(time // 250000)

    format_time = {"ints": ints, "patchy": patchy,
                   "coarse": lambda time: str(time // 250000)}[formatter]
    rows, summary = build_drm_rows(mixed_nest(), format_time)

    assert summary["rows"] == len(rows)
    assert summary["candidates"] == (summary["rows"] + not_applicable_total(summary)
                                     + dropped_total(summary))


def test_the_summary_counts_still_add_up_once_timestamps_start_failing():
    lost = mixed_nest()[1]["points"][1]["time"]

    def patchy(time):
        return None if time == lost else str(time // 250000)

    _rows, summary = build_drm_rows(mixed_nest(), patchy)
    assert summary["dropped"]["unmappable_time"] >= 1
    assert summary["dropped"]["degenerate_period"] >= 1


def test_the_summary_lines_report_every_reason_that_fired():
    _rows, summary = build_drm_rows(mixed_nest(), ints)
    lines = summary_lines(summary)
    text = " ".join(lines)

    assert "3 period(s) written" in text
    assert "9 candidate row(s)" in text
    assert "red pattern" in text
    assert "Flat" in text
    for phrase in ("no primary for", "never records", "has no cell for",
                   "already-used leg"):
        assert phrase in text

    # The two groups are headed separately, and every count sits under the one
    # it belongs to -- "the DRM does not track this" must not read as "your
    # marking was thrown away".
    heads = [i for i, line in enumerate(lines) if line.startswith("**")]
    assert len(heads) == 2
    assert lines[heads[0]].startswith("**Not applicable**")
    assert lines[heads[1]].startswith("**Dropped**")
    assert all("never records" not in line for line in lines[heads[1]:])
    assert all("already-used leg" not in line for line in lines[heads[0]:heads[1]])


def test_the_summary_line_stays_quiet_about_reasons_that_did_not_fire():
    _rows, summary = build_drm_rows(simple_nest("n", 0, step=1000), ints)
    lines = summary_lines(summary)
    assert lines == ["1 period(s) written from 1 candidate row(s)."]


def test_a_wholly_untracked_export_reads_as_lossless():
    # Every candidate in a position the DRM skips: not one word about dropping.
    parent = alternating("P", "Impulse", "Impulse", 0, 1000)
    child = spanning("C", parent, 0)
    grandchild = spanning("G", child, 1)                # wave 2
    _rows, summary = build_drm_rows([parent, child, grandchild], ints)
    lines = summary_lines(summary)

    assert not any(line.startswith("**Dropped**") for line in lines)
    assert any("nothing was lost" in line for line in lines)


# ------------------------------------------------------- the tab's own side
#
# Everything above is pure. What the tab adds is the one thing the export
# deliberately does not know: which bar of the chosen timeframe a canonical
# pivot falls in.


def ohlc_frame(bars, start="2024-01-01 00:00", freq="15min"):
    index = pd.date_range(start, periods=bars, freq=freq)
    highs = [100.0 + i for i in range(bars)]
    lows = [h - 5.0 for h in highs]
    return pd.DataFrame({"open": highs, "high": highs, "low": lows,
                         "close": highs, "volume": [1.0] * bars}, index=index)


def drm_nest_on(df):
    """A three-level nest snapped onto real bars of ``df``: W.(1) / W.1 Impulse."""
    kinds = ["low", "high"] * 3

    def impulse(pattern_id, positions):
        points = [{"time": int(df.index[p].timestamp()),
                   "price": float(df["high" if k == "high" else "low"].iloc[p]),
                   "kind": k}
                  for p, k in zip(positions, kinds)]
        return make_pattern(pattern_id, "Impulse", "Impulse", points)

    return [impulse("P", [0, 30, 40, 50, 60, 80]),      # leg 0 -> W.(1)
            impulse("C", [0, 6, 12, 18, 24, 30]),       # leg 0 -> W.1
            impulse("G", [0, 1, 2, 3, 4, 6])]           # ...an Impulse


def test_the_export_timeframe_list_offers_the_base_and_anything_coarser():
    assert wave_analysis_tab.export_timeframes("15m") == ["15m", "1H", "4H", "1D"]
    assert wave_analysis_tab.export_timeframes("1H") == ["1H", "4H", "1D"]
    assert wave_analysis_tab.export_timeframes("nonsense") == ["nonsense"]


def test_the_timeframe_selector_starts_on_4h():
    # His own DRM is logged at 4H, so that is where the selector opens.
    for base in ("15m", "1H"):
        timeframes = wave_analysis_tab.export_timeframes(base)
        index = wave_analysis_tab.default_export_index(timeframes)
        assert timeframes[index] == "4H"
    assert wave_analysis_tab.default_export_index([]) == 0


def test_a_period_timestamp_is_the_containing_bar_not_the_extreme():
    df = ohlc_frame(96)                                 # one day of 15m bars
    pmap = period_map(df, "4H", "15m")
    format_time = wave_analysis_tab.time_formatter(pmap)

    # A pivot at 05:30 belongs to the 04:00 four-hour bar, whatever its price.
    canonical = int(df.index[22].timestamp())
    assert str(df.index[22]) == "2024-01-01 05:30:00"
    assert format_time(canonical) == "01.01.2024_04:00"

    # And every bound the export writes is a bar this app actually has.
    assert all(format_time(int(ts.timestamp())) is not None for ts in df.index)


def test_a_canonical_export_formats_the_point_time_itself():
    df = ohlc_frame(8)
    format_time = wave_analysis_tab.time_formatter(None)
    assert format_time(int(df.index[3].timestamp())) == "01.01.2024_00:45"
    assert format_time(None) is None
    assert format_time("nope") is None


def test_a_pivot_outside_the_frame_has_no_timestamp():
    pmap = period_map(ohlc_frame(96), "4H", "15m")
    assert wave_analysis_tab.time_formatter(pmap)(1) is None


def test_a_coarser_export_collapses_a_short_leg_and_says_so():
    df = ohlc_frame(96)
    patterns = drm_nest_on(df)

    at_15m = build_drm_rows(patterns, wave_analysis_tab.time_formatter(None))[0]
    assert len(at_15m) == 1

    daily = period_map(df, "1D", "15m")
    rows, summary = build_drm_rows(patterns,
                                   wave_analysis_tab.time_formatter(daily))
    assert rows == []
    assert summary["dropped"]["degenerate_period"] == 1


def test_build_export_produces_a_workbook_and_a_json_together():
    df = ohlc_frame(96)
    built = wave_analysis_tab.build_export(drm_nest_on(df), "GC 15m.csv", "15m", None)

    assert set(built["sheets"]) == {"Bullish", "Bearish"}
    assert built["workbook"][:2] == b"PK"           # a real zip, i.e. a real xlsx
    assert json.loads(built["json"].decode("utf-8"))["dataset_key"] == "GC 15m.csv"
    assert built["summary"]["rows"] == 1


def test_the_export_key_moves_when_a_pivot_moves():
    df = ohlc_frame(96)
    patterns = drm_nest_on(df)
    before = wave_analysis_tab._export_key("k", "15m", "4H", patterns)

    moved = copy.deepcopy(patterns)
    moved[1]["points"][1]["time"] += 900
    # Same pattern count, different periods: a key built on the count alone
    # would keep serving the stale workbook.
    assert wave_analysis_tab._export_key("k", "15m", "4H", moved) != before
    assert wave_analysis_tab._export_key("k", "15m", "1D", patterns) != before


@pytest.mark.parametrize("dataset_key,stem", [
    ("GC 2015-2025 15m Barcharts.csv", "GC 2015-2025 15m Barcharts"),
    ("GC.XLSX", "GC"), ("plain", "plain"), ("", "waves"), (None, "waves"),
])
def test_download_names_come_from_the_dataset(dataset_key, stem):
    assert wave_analysis_tab.export_stem(dataset_key) == stem


def test_junk_input_produces_an_empty_export_rather_than_an_error():
    rows, summary = build_drm_rows("not a list", ints)
    assert rows == []
    assert summary["rows"] == 0
    assert build_drm_sheets(rows)["Bullish"].shape == (43, 2)

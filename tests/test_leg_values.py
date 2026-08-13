"""
Tests for the per-wave study values: the numbers the client attaches to one leg
of a marked pattern so he can later count them across many patterns.

The leg indexing is the whole risk here -- leg ``k`` runs from ``points[k]`` to
``points[k + 1]`` and is the wave labelled ``point_labels(...)[k + 1]`` -- so it
is pinned first and by name, before anything that depends on it.
"""

import json
from pathlib import Path

import pytest

from config import wave_analysis
from config.wave_analysis import (
    LEG_VALUE_FIELDS,
    apply_wave_event,
    is_valid_pattern,
    point_labels,
    settle,
)
from strategies.drm_export import build_wave_json
from strategies.wave_marking_manager import load_wave_documents, save_wave_documents

FRONTEND = Path(__file__).resolve().parents[1] / "ui" / "components" / "wave_chart" / "frontend"


def zigzag(pattern_id="z1", leg_values=None):
    """A structurally valid Zigzag: three legs, waves A, B and C."""
    pattern = {
        "id": pattern_id,
        "pattern_type": "Zigzag",
        "variation": "Zigzag",
        "degree": "Subminuette",
        "color": "yellow",
        "points": [
            {"time": 1719878400, "price": 2360.5, "kind": "low"},
            {"time": 1719879300, "price": 2371.0, "kind": "high"},
            {"time": 1719880200, "price": 2355.25, "kind": "low"},
            {"time": 1719881100, "price": 2380.75, "kind": "high"},
        ],
    }
    if leg_values is not None:
        pattern["leg_values"] = leg_values
    return pattern


def impulse(pattern_id="i1", leg_values=None):
    """A structurally valid Impulse: five legs, waves 1 to 5."""
    pattern = zigzag(pattern_id, leg_values)
    pattern.update(pattern_type="Impulse", variation="Impulse")
    pattern["points"] = [{"time": 1719878400 + 900 * i,
                          "price": 2360.0 + i,
                          "kind": "low" if i % 2 == 0 else "high"}
                         for i in range(6)]
    return pattern


def set_values(pattern_id, leg_index, values, timeframe="1D"):
    return {"type": "set_leg_values", "id": pattern_id, "leg_index": leg_index,
            "values": values, "timeframe": timeframe}


def wave_label(pattern, leg):
    """The wave a leg *is* -- what the popup's heading names it by."""
    return point_labels(pattern["pattern_type"], pattern["variation"])[leg + 1]


def values_of(state, pattern_id="z1"):
    return next(p for p in state if p["id"] == pattern_id).get("leg_values")


# ------------------------------------------------------- 1. leg -> wave label


def test_leg_zero_of_an_impulse_is_wave_one_and_leg_four_is_wave_five():
    pattern = impulse()

    assert wave_label(pattern, 0) == "1"
    assert wave_label(pattern, 4) == "5"
    # The origin keeps its place in the list even though it is no longer drawn:
    # every leg index below is read off it.
    assert point_labels("Impulse", "Impulse")[0] == "0"


def test_leg_zero_of_a_zigzag_is_wave_a():
    assert wave_label(zigzag(), 0) == "A"
    assert wave_label(zigzag(), 2) == "C"


def test_a_leg_is_named_by_the_point_it_ends_on():
    # The off-by-one stated as the property it is, not as two examples: leg k
    # ends on point k + 1, and that point's label is the wave's name.
    pattern = impulse()
    labels = point_labels(pattern["pattern_type"], pattern["variation"])
    for leg in range(len(pattern["points"]) - 1):
        assert wave_label(pattern, leg) == labels[leg + 1]


# ---------------------------------------------------------- 2. is_valid_pattern


def test_a_pattern_with_valid_leg_values_is_valid():
    assert is_valid_pattern(zigzag(leg_values={
        "0": {"CMB": 12.34, "RSI": 45.6, "timeframe": "1D"},
        "2": {"CMB": -5.1, "timeframe": "15m"},
    }))


def test_a_pattern_without_leg_values_stays_valid():
    # Every marking on the client's disk predates this field.
    pattern = zigzag()
    assert "leg_values" not in pattern
    assert is_valid_pattern(pattern)


def test_a_leg_value_of_none_is_valid():
    # A field the client cleared reads back as None rather than as a zero.
    assert is_valid_pattern(zigzag(leg_values={"1": {"CMB": None, "RSI": 30.0,
                                                     "timeframe": "1D"}}))


@pytest.mark.parametrize("leg_values", [
    "not a dict",
    [],
    {"3": {"CMB": 1.0}},                        # only three legs, so 3 is out of range
    {"-1": {"CMB": 1.0}},
    {"nope": {"CMB": 1.0}},
    {0: {"CMB": 1.0}},                          # keys are strings after a round trip
    {"0": "not a dict"},
    {"0": {"CMB": "cheap"}},
    {"0": {"CMB": True}},                       # a bool is not a reading
    {"0": {"Elephants": 1.0}},                  # not a configured field
    {"0": {"timeframe": 15}},
])
def test_a_malformed_leg_values_makes_the_pattern_invalid(leg_values):
    assert not is_valid_pattern(zigzag(leg_values=leg_values))


def test_the_leg_range_follows_the_pattern_that_carries_it():
    # Leg 4 exists on an impulse and does not on a zigzag; the same entry is
    # therefore valid on one and not on the other.
    entry = {"4": {"CMB": 1.0, "timeframe": "1D"}}
    assert is_valid_pattern(impulse(leg_values=entry))
    assert not is_valid_pattern(zigzag(leg_values=entry))


# ------------------------------------------------------ 3. set_leg_values


def test_set_leg_values_stores_an_entry_under_the_stringified_leg_index():
    state = apply_wave_event([zigzag()], set_values("z1", 1, {"CMB": 12.34}))

    assert values_of(state) == {"1": {"CMB": 12.34, "timeframe": "1D"}}


def test_set_leg_values_merges_rather_than_replacing():
    state = apply_wave_event([zigzag()], set_values("z1", 0, {"CMB": 12.34,
                                                              "RSI": 45.6}))
    state = apply_wave_event(state, set_values("z1", 0, {"CMB": -1.0}))

    # The submission carried no RSI, so the stored one stands.
    assert values_of(state) == {"0": {"CMB": -1.0, "RSI": 45.6, "timeframe": "1D"}}


def test_a_field_set_to_none_is_removed_from_the_entry():
    state = apply_wave_event([zigzag()], set_values("z1", 0, {"CMB": 12.34,
                                                              "RSI": 45.6}))
    state = apply_wave_event(state, set_values("z1", 0, {"RSI": None}))

    assert values_of(state) == {"0": {"CMB": 12.34, "timeframe": "1D"}}


def test_an_entry_left_with_no_fields_disappears_entirely():
    # Absent has to keep meaning "never measured", so an emptied entry must not
    # linger as a timeframe with nothing attached to it.
    state = apply_wave_event([zigzag()], set_values("z1", 2, {"CMB": 1.0}))
    state = apply_wave_event(state, set_values("z1", 2, {"CMB": None}))

    assert values_of(state) in (None, {})
    assert "2" not in (values_of(state) or {})


def test_clearing_one_leg_leaves_another_alone():
    state = apply_wave_event([zigzag()], set_values("z1", 0, {"CMB": 1.0}))
    state = apply_wave_event(state, set_values("z1", 2, {"CMB": 2.0}))
    state = apply_wave_event(state, set_values("z1", 0, {"CMB": None}))

    assert values_of(state) == {"2": {"CMB": 2.0, "timeframe": "1D"}}


def test_the_timeframe_is_stored_with_the_numbers():
    state = apply_wave_event([zigzag()], set_values("z1", 1, {"CMB": 1.0}, "15m"))

    assert values_of(state)["1"]["timeframe"] == "15m"


def test_set_leg_values_only_touches_the_named_pattern():
    other = zigzag("other")
    state = apply_wave_event([zigzag(), other], set_values("z1", 1, {"CMB": 1.0}))

    assert state[1] is other


@pytest.mark.parametrize("event", [
    set_values("nosuch", 0, {"CMB": 1.0}),                  # unknown id
    set_values("z1", 3, {"CMB": 1.0}),                      # only three legs
    set_values("z1", -1, {"CMB": 1.0}),
    set_values("z1", "0", {"CMB": 1.0}),                    # index must be an int
    set_values("z1", True, {"CMB": 1.0}),
    set_values("z1", 0, {"Elephants": 1.0}),                # unknown field
    set_values("z1", 0, {"CMB": "cheap"}),                  # not a number
    set_values("z1", 0, {"CMB": True}),
    set_values("z1", 0, "not a dict"),
    set_values("z1", 0, {"CMB": 1.0}, ""),                  # timeframe must be named
    set_values("z1", 0, {"CMB": 1.0}, None),
    {"type": "set_leg_values", "id": "z1", "leg_index": 0, "values": {"CMB": 1.0}},
])
def test_a_malformed_set_leg_values_leaves_the_state_unchanged(event):
    before = [zigzag()]

    assert apply_wave_event(before, event) is before


def test_a_rejected_field_does_not_store_the_ones_beside_it():
    # All or nothing: a submission is one reading of one leg, so half of it
    # landing would be worse than none of it.
    before = [zigzag()]
    state = apply_wave_event(before, set_values("z1", 0, {"CMB": 1.0,
                                                          "Elephants": 2.0}))

    assert state is before


def test_set_leg_values_does_not_mutate_its_input():
    before = [zigzag(leg_values={"0": {"CMB": 1.0, "timeframe": "1D"}})]
    snapshot = json.loads(json.dumps(before))

    apply_wave_event(before, set_values("z1", 0, {"CMB": 99.0}))

    assert before == snapshot


# ------------------------------------------------------------ 5. persistence


def test_values_survive_save_load_and_settle(tmp_path):
    path = str(tmp_path / "saved_wave_markings.json")
    stored = {"0": {"CMB": 12.34, "RSI": 45.6, "timeframe": "1D"},
              "2": {"CMB": -5.1, "timeframe": "15m"}}
    save_wave_documents({"gold.csv": {"schema": 2, "base_timeframe": "15m",
                                      "patterns": [zigzag(leg_values=stored)]}}, path)

    loaded = load_wave_documents(path)["gold.csv"]["patterns"]
    settled = settle(loaded)

    assert loaded[0]["leg_values"] == stored
    assert settled[0]["leg_values"] == stored


def test_a_pattern_whose_values_are_corrupt_on_disk_is_dropped_not_loaded(tmp_path):
    # The guard on the test above: the load path really does validate, so a
    # marking carrying a leg index its own shape cannot have does not come back.
    path = str(tmp_path / "saved_wave_markings.json")
    save_wave_documents({"gold.csv": {"schema": 2, "base_timeframe": "15m",
                                      "patterns": [zigzag(leg_values={"9": {"CMB": 1.0}})]}},
                        path)

    assert load_wave_documents(path)["gold.csv"]["patterns"] == []


# ------------------------------------------------------------ 7. the JSON export


def test_build_wave_json_carries_the_leg_values():
    stored = {"1": {"CMB": 12.34, "RSI": 45.6, "timeframe": "1D"}}
    exported = build_wave_json([zigzag(leg_values=stored)], "gold.csv", "15m")

    # Through a real dump: this file *is* the client's study data, so what
    # matters is that the values are there after serialisation, not merely in
    # the dict on the way to it.
    round_tripped = json.loads(json.dumps(exported, allow_nan=False))
    assert round_tripped["patterns"][0]["leg_values"] == stored


# ------------------------------------------------------- 8. config-driven fields


def test_a_field_added_to_the_table_is_accepted_with_no_other_change(monkeypatch):
    added = ("ADX", "adx")
    assert added not in LEG_VALUE_FIELDS         # not vacuous
    before = [zigzag()]

    # Rejected as things stand...
    assert apply_wave_event(before, set_values("z1", 0, {"ADX": 25.0})) is before

    # ...and accepted with one entry added to the table and nothing else.
    monkeypatch.setattr(wave_analysis, "LEG_VALUE_FIELDS",
                        list(LEG_VALUE_FIELDS) + [added])
    state = apply_wave_event(before, set_values("z1", 0, {"ADX": 25.0}))

    assert values_of(state) == {"0": {"ADX": 25.0, "timeframe": "1D"}}
    assert is_valid_pattern(state[0])


def test_the_frontend_is_told_which_fields_to_draw():
    # The popup builds one input per entry, in this order. Sent down rather
    # than duplicated in JavaScript, which is what makes the one-line promise
    # above hold for the UI as well as for the reducer.
    fields = wave_analysis.wave_defs()["leg_value_fields"]

    assert fields == [list(field) for field in LEG_VALUE_FIELDS]
    assert all(isinstance(field, list) and len(field) == 2 for field in fields)


# ------------------------------------------------------------- 9. no origin label


def frontend_source(name):
    return (FRONTEND / name).read_text()


def test_point_zero_carries_no_label_for_any_pattern():
    """The client asked for the 0 to go from every wave count.

    Asserted against the frontend source because that is where the rule lives
    and the component has no test harness of its own -- a weaker check than
    driving the canvas, but it does pin the two things that could regress: the
    skip being unconditional, and the per-child mechanism it replaced being
    gone rather than merely bypassed.
    """
    source = frontend_source("main.js")

    assert "for (var i = 1; i < pattern.points.length && i < labels.length; i++)" in source
    for dead in ("originHidden", "refreshOriginHidden", "pointsMatch", "childLegIndex"):
        assert dead not in source, f"{dead} should have been deleted, not left unused"


def test_point_zero_itself_stays():
    # Only the glyph goes. Point 0 is the origin of wave 1, it defines the
    # pattern's span, and it is half of the endpoint match that drives degree
    # derivation, the DRM export and magnetism.
    for pattern_type, variations in wave_analysis.PATTERN_DEFS.items():
        for variation, label_seq in variations:
            labels = point_labels(pattern_type, variation)
            assert labels[0] == "0"
            assert len(labels) == len(label_seq) + 1

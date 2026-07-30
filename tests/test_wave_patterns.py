"""
Tests for the Wave Analysis pattern/degree definitions and the event reducer.
"""

import copy
import pytest

from config.wave_analysis import (
    DEFAULT_DEGREE,
    DEGREES,
    PATTERN_DEFS,
    apply_wave_event,
    degree_component,
    pattern_span,
    point_labels,
    related,
    render_glyph,
    validate_patterns,
    wave_defs,
)
from ui import wave_analysis_tab
from ui.wave_analysis_tab import apply_event_batch


def make_pattern(**overrides):
    """A structurally valid pattern as the frontend would send it."""
    pattern = {
        "id": "abc123",
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
    pattern.update(overrides)
    return pattern


def completed(pattern):
    return {"type": "pattern_completed", "pattern": pattern}


def span_pattern(pattern_id, start, end, degree="Minor"):
    """A valid Zigzag whose points run from ``start`` to ``end`` inclusive."""
    assert end - start >= 3, "need room for four strictly increasing times"
    times = [start, start + 1, end - 1, end]
    return {
        "id": pattern_id,
        "pattern_type": "Zigzag",
        "variation": "Zigzag",
        "degree": degree,
        "color": "yellow",
        "points": [{"time": t, "price": 100.0 + i, "kind": "low" if i % 2 == 0 else "high"}
                   for i, t in enumerate(times)],
    }


def degrees_of(pattern_list):
    return {p["id"]: p["degree"] for p in pattern_list}


def colors_of(pattern_list):
    return {p["id"]: p["color"] for p in pattern_list}


# ---------------------------------------------------------------- point_labels

def test_point_labels_prefix_every_variation_with_zero():
    for pattern_type, variations in PATTERN_DEFS.items():
        for name, label_seq in variations:
            assert point_labels(pattern_type, name) == ["0"] + list(label_seq)


def test_point_labels_triple_zigzag_keeps_the_repeated_x():
    labels = point_labels("Zigzag", "Triple Zigzag")
    assert labels == ["0", "W", "X", "Y", "X", "Z"]
    assert len(labels) == 6


def test_point_labels_returns_a_fresh_list():
    labels = point_labels("Impulse", "Impulse")
    labels.append("junk")
    assert point_labels("Impulse", "Impulse") == ["0", "1", "2", "3", "4", "5"]


def test_point_labels_rejects_unknown_combinations():
    with pytest.raises(ValueError):
        point_labels("Impulse", "Zigzag")
    with pytest.raises(ValueError):
        point_labels("Nonsense", "Impulse")


# ---------------------------------------------------------------- render_glyph

@pytest.mark.parametrize("digit,upper,lower", [
    ("1", "I", "i"),
    ("2", "II", "ii"),
    ("3", "III", "iii"),
    ("4", "IV", "iv"),
    ("5", "V", "v"),
])
def test_render_glyph_numerals(digit, upper, lower):
    assert render_glyph(digit, "upper_sans", "arabic") == digit
    assert render_glyph(digit, "lower_serif", "roman_upper") == upper
    assert render_glyph(digit, "lower_serif", "roman_lower") == lower


@pytest.mark.parametrize("label", ["A", "B", "C", "D", "E", "W", "X", "Y", "Z"])
def test_render_glyph_letters_follow_the_letter_style(label):
    assert render_glyph(label, "upper_sans", "arabic") == label.upper()
    assert render_glyph(label, "lower_serif", "roman_lower") == label.lower()


def test_render_glyph_zero_is_never_transformed():
    for _name, letter_style, numeral_style, _decoration, _font in DEGREES:
        assert render_glyph("0", letter_style, numeral_style) == "0"


def test_render_glyph_rejects_unknown_labels():
    with pytest.raises(ValueError):
        render_glyph("Q", "upper_sans", "arabic")


# --------------------------------------------------------------------- DEGREES

def test_degrees_has_eighteen_unique_entries():
    assert len(DEGREES) == 18
    names = [d[0] for d in DEGREES]
    assert len(set(names)) == 18
    assert DEFAULT_DEGREE in names


def test_degree_decorations_cycle_within_every_triad():
    decorations = [d[3] for d in DEGREES]
    assert set(decorations) == {"circle", "parens", "plain"}
    for start in range(0, len(DEGREES), 3):
        assert decorations[start:start + 3] == ["circle", "parens", "plain"]


def test_degree_styles_are_from_the_known_vocabulary():
    for _name, letter_style, numeral_style, _decoration, font_px in DEGREES:
        assert letter_style in {"upper_sans", "lower_serif"}
        assert numeral_style in {"arabic", "roman_upper", "roman_lower"}
        assert isinstance(font_px, int) and font_px > 0


def test_wave_defs_is_plain_json_shaped():
    defs = wave_defs()
    assert defs["default_degree"] == DEFAULT_DEGREE
    assert len(defs["degrees"]) == 18
    assert all(isinstance(d, list) for d in defs["degrees"])
    assert set(defs["pattern_defs"]) == set(PATTERN_DEFS)
    for variations in defs["pattern_defs"].values():
        for entry in variations:
            assert isinstance(entry, list) and isinstance(entry[1], list)


# ------------------------------------------------------------ apply_wave_event

def test_apply_appends_under_the_right_timeframe():
    pattern = make_pattern()
    state = apply_wave_event({}, "15m", completed(pattern))
    assert state == {"15m": [pattern]}


def test_apply_leaves_other_timeframes_untouched():
    other = [make_pattern(id="other")]
    before = {"1h": other}
    state = apply_wave_event(before, "15m", completed(make_pattern()))

    assert state["1h"] is other
    assert len(state["15m"]) == 1
    assert before == {"1h": other}          # the input mapping is not mutated


def test_apply_ignores_a_duplicate_id():
    pattern = make_pattern()
    state = apply_wave_event({}, "15m", completed(pattern))
    again = apply_wave_event(state, "15m", completed(make_pattern(price_note="ignored")))
    assert again is state
    assert len(again["15m"]) == 1


@pytest.mark.parametrize("bad", [
    make_pattern(points=make_pattern()["points"][:3]),                      # too few points
    make_pattern(points=make_pattern()["points"] + [
        {"time": 1719882000, "price": 2390.0, "kind": "high"}]),            # too many points
    make_pattern(degree="Nonexistent"),
    make_pattern(variation="Impulse"),                                      # not a Zigzag variation
    make_pattern(variation="Triple Zigzag"),                                # wrong point count for it
    make_pattern(pattern_type="Nonsense"),
    make_pattern(color="chartreuse"),
    make_pattern(id=""),
    make_pattern(points="not a list"),
])
def test_apply_rejects_malformed_patterns(bad):
    before = {"15m": []}
    snapshot = copy.deepcopy(before)
    state = apply_wave_event(before, "15m", completed(bad))
    assert state is before
    assert before == snapshot


def test_apply_rejects_non_increasing_times():
    points = make_pattern()["points"]
    points[2] = dict(points[2], time=points[1]["time"])
    state = apply_wave_event({}, "15m", completed(make_pattern(points=points)))
    assert state == {}

    points[2] = dict(points[2], time=points[1]["time"] - 60)
    state = apply_wave_event({}, "15m", completed(make_pattern(points=points)))
    assert state == {}


def test_apply_rejects_bad_point_fields():
    for broken in [
        {"time": "1719878400", "price": 2360.5, "kind": "low"},
        {"time": 1719878400, "price": "cheap", "kind": "low"},
        {"time": 1719878400, "price": 2360.5, "kind": "middle"},
        "not a point",
    ]:
        points = make_pattern()["points"]
        points[0] = broken
        assert apply_wave_event({}, "15m", completed(make_pattern(points=points))) == {}


def test_apply_accepts_integer_prices():
    points = make_pattern()["points"]
    points[0] = dict(points[0], price=2360)      # JSON hands back an int for round prices
    state = apply_wave_event({}, "15m", completed(make_pattern(points=points)))
    assert len(state["15m"]) == 1


def test_apply_ignores_unknown_and_malformed_events():
    before = {"15m": []}
    assert apply_wave_event(before, "15m", {"type": "something_else"}) is before
    assert apply_wave_event(before, "15m", {}) is before
    assert apply_wave_event(before, "15m", None) is before
    assert apply_wave_event(before, "15m", {"type": "pattern_completed"}) is before


# ------------------------------------------------------- spans / containment

def test_pattern_span_is_the_first_and_last_point_times():
    assert pattern_span(span_pattern("a", 100, 200)) == (100, 200)
    assert pattern_span({"points": []}) is None
    assert pattern_span({"points": [{"time": "100"}]}) is None
    assert pattern_span("not a pattern") is None


def test_related_is_containment_in_either_direction():
    parent = span_pattern("parent", 0, 100)
    child = span_pattern("child", 10, 40)

    assert related(parent, child)
    assert related(child, parent)          # symmetric
    assert related(parent, parent)


def test_related_accepts_a_child_flush_with_its_parents_ends():
    parent = span_pattern("parent", 0, 100)
    flush = span_pattern("flush", 0, 100)
    assert related(parent, flush)


def test_related_rejects_siblings_that_only_share_an_endpoint():
    first = span_pattern("first", 0, 50)
    second = span_pattern("second", 50, 100)   # starts exactly where the first ends
    assert not related(first, second)

    overlapping = span_pattern("overlap", 40, 140)   # crosses, contains neither
    assert not related(first, overlapping)


def test_degree_component_covers_a_transitive_nest():
    grandparent = span_pattern("A", 0, 1000)
    parent = span_pattern("B", 100, 400)
    child = span_pattern("C", 150, 200)
    patterns = [grandparent, parent, child]

    assert degree_component(patterns, "C") == ["A", "B", "C"]
    assert degree_component(patterns, "A") == ["A", "B", "C"]


def test_degree_component_excludes_an_adjacent_sibling():
    parent = span_pattern("A", 0, 1000)
    child = span_pattern("B", 100, 400)
    sibling = span_pattern("C", 1000, 2000)     # starts on A's terminus

    component = degree_component([parent, child, sibling], "B")
    assert component == ["A", "B"]
    assert degree_component([parent, child, sibling], "C") == ["C"]


def test_degree_component_keeps_disjoint_trees_apart():
    patterns = [
        span_pattern("A", 0, 100), span_pattern("B", 10, 40),
        span_pattern("X", 500, 600), span_pattern("Y", 510, 540),
    ]
    assert degree_component(patterns, "A") == ["A", "B"]
    assert degree_component(patterns, "Y") == ["X", "Y"]


def test_degree_component_of_an_unknown_id_is_empty():
    assert degree_component([span_pattern("A", 0, 100)], "nope") == []
    assert degree_component("not a list", "A") == []


# --------------------------------------------------------------- shift_degree

def shift(pattern_id, delta):
    return {"type": "shift_degree", "id": pattern_id, "delta": delta}


def test_shift_degree_cascades_through_the_whole_nest():
    patterns = [
        span_pattern("A", 0, 1000, "Minor"),          # index 8
        span_pattern("B", 100, 400, "Minuette"),      # index 10
        span_pattern("C", 150, 200, "Micro"),         # index 12
    ]
    state = {"15m": patterns}

    # Ctrl and + on the middle pattern: every member moves one step senior.
    senior = apply_wave_event(state, "15m", shift("B", 1))
    assert degrees_of(senior["15m"]) == {
        "A": "Intermediate", "B": "Minute", "C": "Subminuette",
    }

    # ...and back down again, restoring the original relative offsets.
    junior = apply_wave_event(senior, "15m", shift("B", -1))
    assert degrees_of(junior["15m"]) == {
        "A": "Minor", "B": "Minuette", "C": "Micro",
    }
    assert state["15m"] is patterns                  # the input is never mutated


def test_shift_degree_leaves_unrelated_patterns_alone():
    patterns = [
        span_pattern("A", 0, 1000, "Minor"),
        span_pattern("B", 100, 400, "Minuette"),
        span_pattern("Z", 5000, 6000, "Minor"),
    ]
    state = apply_wave_event({"15m": patterns}, "15m", shift("A", 1))
    assert degrees_of(state["15m"]) == {
        "A": "Intermediate", "B": "Minute", "Z": "Minor",
    }


def test_shift_degree_is_all_or_nothing_at_the_senior_end():
    patterns = [
        span_pattern("A", 0, 1000, "Supermillennium"),   # index 0 -- already the top
        span_pattern("B", 100, 400, "Millennium"),
    ]
    before = {"15m": patterns}
    after = apply_wave_event(before, "15m", shift("B", 1))

    assert after is before
    assert degrees_of(after["15m"]) == {"A": "Supermillennium", "B": "Millennium"}


def test_shift_degree_is_all_or_nothing_at_the_junior_end():
    patterns = [
        span_pattern("A", 0, 1000, "Subnano"),
        span_pattern("B", 100, 400, "Pico"),             # index 17 -- the bottom
    ]
    before = {"15m": patterns}
    after = apply_wave_event(before, "15m", shift("A", -1))

    assert after is before
    assert degrees_of(after["15m"]) == {"A": "Subnano", "B": "Pico"}


@pytest.mark.parametrize("delta", [0, 2, -2, "1", 1.0, True, None])
def test_shift_degree_rejects_any_delta_other_than_plus_or_minus_one(delta):
    before = {"15m": [span_pattern("A", 0, 1000, "Minor")]}
    assert apply_wave_event(before, "15m", shift("A", delta)) is before


def test_shift_degree_ignores_an_unknown_id():
    before = {"15m": [span_pattern("A", 0, 1000, "Minor")]}
    assert apply_wave_event(before, "15m", shift("nope", 1)) is before
    assert apply_wave_event(before, "15m", shift(None, 1)) is before


def test_shift_degree_rejects_a_pattern_at_an_unknown_degree():
    broken = span_pattern("A", 0, 1000, "Minor")
    broken["degree"] = "Nonexistent"
    before = {"15m": [broken]}
    assert apply_wave_event(before, "15m", shift("A", 1)) is before


# ------------------------------------------------------------------ move_point

def move(pattern_id, index, time, price=999.0, kind="high"):
    return {"type": "move_point", "id": pattern_id, "point_index": index,
            "time": time, "price": price, "kind": kind}


def test_move_point_replaces_the_point():
    before = {"15m": [span_pattern("A", 100, 200)]}
    after = apply_wave_event(before, "15m", move("A", 1, 150, 123.5, "high"))

    assert after["15m"][0]["points"][1] == {"time": 150, "price": 123.5, "kind": "high"}
    assert before["15m"][0]["points"][1]["time"] == 101      # input untouched


def test_move_point_rejects_a_time_on_or_past_a_neighbour():
    before = {"15m": [span_pattern("A", 100, 200)]}          # times 100, 101, 199, 200

    assert apply_wave_event(before, "15m", move("A", 1, 100)) is before   # == previous
    assert apply_wave_event(before, "15m", move("A", 1, 199)) is before   # == next
    assert apply_wave_event(before, "15m", move("A", 1, 250)) is before   # past next
    assert apply_wave_event(before, "15m", move("A", 2, 101)) is before   # == previous


def test_move_point_accepts_the_first_and_last_points_bounded_by_one_neighbour():
    before = {"15m": [span_pattern("A", 100, 200)]}          # times 100, 101, 199, 200

    first = apply_wave_event(before, "15m", move("A", 0, 40, 5.0, "low"))
    assert [p["time"] for p in first["15m"][0]["points"]] == [40, 101, 199, 200]

    last = apply_wave_event(before, "15m", move("A", 3, 900, 5.0, "low"))
    assert [p["time"] for p in last["15m"][0]["points"]] == [100, 101, 199, 900]

    # ...but they are still bounded on their one side.
    assert apply_wave_event(before, "15m", move("A", 0, 101)) is before
    assert apply_wave_event(before, "15m", move("A", 3, 199)) is before


def test_move_point_rejects_an_out_of_range_index():
    before = {"15m": [span_pattern("A", 100, 200)]}
    for index in (-1, 4, 99):
        assert apply_wave_event(before, "15m", move("A", index, 150)) is before


def test_move_point_rejects_bad_fields_and_unknown_ids():
    before = {"15m": [span_pattern("A", 100, 200)]}

    assert apply_wave_event(before, "15m", move("nope", 1, 150)) is before
    assert apply_wave_event(before, "15m", move("A", 1, 150.5)) is before      # float time
    assert apply_wave_event(before, "15m", move("A", "1", 150)) is before      # str index
    assert apply_wave_event(before, "15m", move("A", 1, 150, "cheap")) is before
    assert apply_wave_event(before, "15m", move("A", 1, 150, 9.0, "middle")) is before
    assert apply_wave_event(before, "15m", move("A", True, 150)) is before     # bool index


def test_move_point_only_touches_the_named_timeframe():
    other = [span_pattern("A", 100, 200)]
    before = {"15m": [span_pattern("A", 100, 200)], "1h": other}
    after = apply_wave_event(before, "15m", move("A", 1, 150))

    assert after["1h"] is other
    assert after["15m"][0]["points"][1]["time"] == 150


# -------------------------------------------------------------- delete_pattern

def delete(pattern_id):
    return {"type": "delete_pattern", "id": pattern_id}


def test_delete_pattern_removes_by_id():
    before = {"15m": [span_pattern("A", 0, 100), span_pattern("B", 200, 300)]}
    after = apply_wave_event(before, "15m", delete("A"))

    assert [p["id"] for p in after["15m"]] == ["B"]
    assert len(before["15m"]) == 2                   # the input list is not mutated


def test_delete_pattern_ignores_an_unknown_id():
    before = {"15m": [span_pattern("A", 0, 100)]}
    assert apply_wave_event(before, "15m", delete("nope")) is before
    assert apply_wave_event(before, "15m", delete("")) is before
    assert apply_wave_event(before, "15m", delete(None)) is before
    assert apply_wave_event({}, "15m", delete("A")) == {}


def test_delete_pattern_leaves_other_timeframes_untouched():
    other = [span_pattern("A", 0, 100)]
    before = {"15m": [span_pattern("A", 0, 100)], "1h": other}
    after = apply_wave_event(before, "15m", delete("A"))

    assert after["15m"] == []
    assert after["1h"] is other


# ------------------------------------------------------------ validate_patterns

def test_validate_flags_a_same_degree_interior_overlap():
    patterns = [
        span_pattern("A", 0, 100),
        span_pattern("B", 50, 150),          # shares A's interior
        span_pattern("C", 300, 400),         # same degree, nowhere near
    ]
    assert colors_of(validate_patterns(patterns)) == {
        "A": "red", "B": "red", "C": "yellow",
    }


def test_validate_leaves_endpoint_chained_patterns_yellow():
    # The client's core workflow: the next pattern's point 0 sits exactly on
    # the previous pattern's terminal point.
    patterns = [span_pattern("A", 0, 100), span_pattern("B", 100, 200)]
    assert colors_of(validate_patterns(patterns)) == {"A": "yellow", "B": "yellow"}


def test_validate_flags_same_degree_containment():
    patterns = [
        span_pattern("parent", 0, 1000, "Minor"),
        span_pattern("child", 100, 400, "Minor"),
    ]
    assert colors_of(validate_patterns(patterns)) == {"parent": "red", "child": "red"}


def test_validate_never_flags_across_degrees():
    patterns = [
        span_pattern("parent", 0, 1000, "Minuette"),
        span_pattern("child", 100, 400, "Subminuette"),
        span_pattern("crossing", 500, 1500, "Micro"),
    ]
    assert set(colors_of(validate_patterns(patterns)).values()) == {"yellow"}


def test_validate_flags_every_member_of_a_mutual_overlap():
    patterns = [
        span_pattern("A", 0, 100),
        span_pattern("B", 20, 120),
        span_pattern("C", 40, 140),          # all three share 40..100
    ]
    assert set(colors_of(validate_patterns(patterns)).values()) == {"red"}


def test_validate_heals_survivors_once_the_middle_pattern_goes():
    chain = [
        span_pattern("A", 0, 100),
        span_pattern("B", 50, 150),          # overlaps A and C
        span_pattern("C", 120, 200),         # does not reach back into A
    ]
    assert set(colors_of(validate_patterns(chain)).values()) == {"red"}

    survivors = validate_patterns([p for p in chain if p["id"] != "B"])
    assert colors_of(survivors) == {"A": "yellow", "C": "yellow"}


def test_validate_recolours_a_stale_red_back_to_yellow():
    stale = span_pattern("A", 0, 100)
    stale["color"] = "red"
    assert colors_of(validate_patterns([stale])) == {"A": "yellow"}


def test_validate_does_not_mutate_its_input():
    patterns = [span_pattern("A", 0, 100), span_pattern("B", 50, 150)]
    snapshot = copy.deepcopy(patterns)

    result = validate_patterns(patterns)

    assert patterns == snapshot
    assert set(colors_of(result).values()) == {"red"}
    assert result is not patterns


def test_validate_tolerates_junk_input():
    assert validate_patterns("not a list") == []
    assert validate_patterns([]) == []
    assert validate_patterns([{"id": "A", "points": []}]) == [
        {"id": "A", "points": [], "color": "yellow"}]


# ------------------------------------------------------- batched event folding

def batch_from(last_seq, *events):
    """A component value whose events continue on from ``last_seq``."""
    numbered = [dict(event, eseq=last_seq + i + 1) for i, event in enumerate(events)]
    return {"seq": last_seq + len(numbered), "events": numbered}


def batch(*events):
    """A component value as protocol v2 posts it: the whole outbox."""
    return batch_from(0, *events)


def test_batch_applies_dependent_events_in_order():
    pattern = span_pattern("A", 0, 100)
    value = batch(completed(pattern), delete("A"))

    state, seq, changed = apply_event_batch({}, "15m", value, 0)

    assert state == {"15m": []}
    assert (seq, changed) == (2, True)


def test_batch_is_a_no_op_when_replayed_under_the_seq_guard():
    value = batch(completed(span_pattern("A", 0, 100)), delete("A"))
    state, seq, _changed = apply_event_batch({}, "15m", value, 0)

    # Streamlit re-delivers the same value on the next rerun.
    again, again_seq, again_changed = apply_event_batch(state, "15m", value, seq)
    assert again is state
    assert again_seq == seq
    assert again_changed is False


def test_batch_applies_only_the_events_above_the_stored_seq():
    first = span_pattern("A", 0, 100)
    second = span_pattern("B", 200, 300)
    value = batch(completed(first), completed(second))

    state, seq, _ = apply_event_batch({}, "15m", value, 1)      # eseq 1 already seen
    assert [p["id"] for p in state["15m"]] == ["B"]
    assert seq == 2


def test_batch_applies_out_of_order_events_in_eseq_order():
    pattern = span_pattern("A", 0, 100)
    value = {"seq": 2, "events": [
        dict(delete("A"), eseq=2),
        dict(completed(pattern), eseq=1),
    ]}
    state, seq, changed = apply_event_batch({}, "15m", value, 0)

    assert state == {"15m": []}
    assert (seq, changed) == (2, True)


def test_batch_acks_rejected_events_so_they_are_never_replayed():
    before = {"15m": []}
    value = batch(delete("nope"))

    state, seq, changed = apply_event_batch(before, "15m", value, 0)
    assert state is before
    assert seq == 1                      # acked anyway
    assert changed is False


def test_batch_ignores_values_that_carry_no_events():
    before = {"15m": []}
    for value in [None, {}, {"seq": 3}, {"events": "nope"}, "junk"]:
        assert apply_event_batch(before, "15m", value, 7) == (before, 7, False)


def test_batch_colours_a_completed_overlap_red():
    value = batch(completed(span_pattern("A", 0, 100)),
                  completed(span_pattern("B", 50, 150)))
    state, _seq, changed = apply_event_batch({}, "15m", value, 0)

    assert changed is True
    assert colors_of(state["15m"]) == {"A": "red", "B": "red"}


def test_batch_revalidates_after_a_shift_degree():
    # Neither contains the other, so the cascade touches B alone -- but the
    # shift lands B on A's degree, where their interiors already overlap.
    state = {"15m": [span_pattern("A", 0, 100, "Minor"),
                     span_pattern("B", 50, 150, "Intermediate")]}
    assert set(colors_of(state["15m"]).values()) == {"yellow"}

    collided, _seq, _changed = apply_event_batch(
        state, "15m", batch(shift("B", -1)), 0)
    assert degrees_of(collided["15m"]) == {"A": "Minor", "B": "Minor"}
    assert colors_of(collided["15m"]) == {"A": "red", "B": "red"}

    healed, _seq, _changed = apply_event_batch(
        collided, "15m", batch(shift("B", 1)), 0)
    assert degrees_of(healed["15m"]) == {"A": "Minor", "B": "Intermediate"}
    assert colors_of(healed["15m"]) == {"A": "yellow", "B": "yellow"}


def test_batch_revalidates_after_a_move_point():
    # A and B are chained on time 100 -- legal until B's point 0 is dragged back.
    state = {"15m": [span_pattern("A", 0, 100), span_pattern("B", 100, 200)]}

    collided, _seq, _changed = apply_event_batch(
        state, "15m", batch(move("B", 0, 50, 5.0, "low")), 0)
    assert colors_of(collided["15m"]) == {"A": "red", "B": "red"}

    healed, _seq, _changed = apply_event_batch(
        collided, "15m", batch(move("B", 0, 100, 5.0, "low")), 0)
    assert colors_of(healed["15m"]) == {"A": "yellow", "B": "yellow"}


def test_batch_leaves_other_timeframes_uncoloured():
    other = [span_pattern("A", 0, 100), span_pattern("B", 50, 150)]
    before = {"15m": [], "1h": other}

    state, _seq, _changed = apply_event_batch(
        before, "15m", batch(completed(span_pattern("Z", 0, 100))), 0)

    assert state["1h"] is other                  # untouched, stale colours and all
    assert colors_of(state["15m"]) == {"Z": "yellow"}


def test_batch_ignores_events_without_a_usable_eseq():
    value = {"seq": 1, "events": [
        completed(span_pattern("A", 0, 100)),            # no eseq at all
        dict(completed(span_pattern("B", 0, 100)), eseq="2"),
        dict(completed(span_pattern("C", 0, 100)), eseq=True),
        "not an event",
    ]}
    before = {"15m": []}
    assert apply_event_batch(before, "15m", value, 0) == (before, 0, False)


# ------------------------------------------------------------------------ undo

@pytest.fixture(autouse=True)
def session(monkeypatch):
    """A plain dict standing in for Streamlit's session state.

    Undo history is session state, and there is none outside a ``streamlit run``
    script thread -- other modules in this suite even replace the whole
    streamlit package with a MagicMock at import time. A dict per test keeps the
    stack real and keeps the tests independent of each other.
    """
    state = {}
    monkeypatch.setattr(wave_analysis_tab.st, "session_state", state)
    return state


def undo():
    return {"type": "undo"}


def undo_stack(session):
    return session.get("_wa_undo", [])


def test_undo_restores_a_deleted_pattern_and_revalidates_colours():
    marked, seq, _changed = apply_event_batch(
        {}, "15m", batch(completed(span_pattern("A", 0, 100)),
                         completed(span_pattern("B", 50, 150))), 0)
    assert colors_of(marked["15m"]) == {"A": "red", "B": "red"}

    # Deleting B heals A, so the snapshot's stored colours are stale the moment
    # it is restored -- the undo has to recolour, not just put the list back.
    deleted, seq, _changed = apply_event_batch(
        marked, "15m", batch_from(seq, delete("B")), seq)
    assert colors_of(deleted["15m"]) == {"A": "yellow"}

    restored, seq, changed = apply_event_batch(
        deleted, "15m", batch_from(seq, undo()), seq)

    assert changed is True
    assert [p["id"] for p in restored["15m"]] == ["A", "B"]
    assert colors_of(restored["15m"]) == {"A": "red", "B": "red"}


def test_undo_after_a_shift_degree_reverts_the_whole_cascade():
    before = {"15m": [
        span_pattern("A", 0, 1000, "Minor"),
        span_pattern("B", 100, 400, "Minuette"),
        span_pattern("C", 150, 200, "Micro"),
    ]}

    shifted, seq, _changed = apply_event_batch(before, "15m", batch(shift("B", 1)), 0)
    assert degrees_of(shifted["15m"]) == {
        "A": "Intermediate", "B": "Minute", "C": "Subminuette",
    }

    reverted, _seq, changed = apply_event_batch(
        shifted, "15m", batch_from(seq, undo()), seq)

    assert changed is True
    assert degrees_of(reverted["15m"]) == {"A": "Minor", "B": "Minuette", "C": "Micro"}


def test_undo_with_an_empty_stack_is_a_no_op(session):
    before = {"15m": [span_pattern("A", 0, 100)]}

    state, seq, changed = apply_event_batch(before, "15m", batch(undo()), 0)

    assert state is before
    assert (seq, changed) == (1, False)      # acked anyway, so it is never replayed
    assert undo_stack(session) == []


def test_undo_only_reverts_the_timeframe_on_screen(session):
    state, seq, _changed = apply_event_batch(
        {}, "15m", batch(completed(span_pattern("A", 0, 100))), 0)
    state, seq, _changed = apply_event_batch(
        state, "15m", batch_from(seq, completed(span_pattern("B", 200, 300))), seq)
    state, seq, _changed = apply_event_batch(
        state, "1h", batch_from(seq, completed(span_pattern("Z", 0, 100))), seq)
    assert len(undo_stack(session)) == 3

    reverted, seq, changed = apply_event_batch(state, "1h", batch_from(seq, undo()), seq)
    assert changed is True
    assert reverted["1h"] == []
    assert [p["id"] for p in reverted["15m"]] == ["A", "B"]

    # 1h has nothing left to undo; the two 15m snapshots are stepped over, not eaten.
    again, _seq, again_changed = apply_event_batch(
        reverted, "1h", batch_from(seq, undo()), seq)
    assert again is reverted
    assert again_changed is False
    assert [entry[0] for entry in undo_stack(session)] == ["15m", "15m"]


def test_undo_stack_drops_the_oldest_snapshot_past_fifty(session):
    state, seq = {}, 0
    for i in range(60):
        state, seq, _changed = apply_event_batch(
            state, "15m",
            batch_from(seq, completed(span_pattern("p%d" % i, i * 10, i * 10 + 5))), seq)

    stack = undo_stack(session)
    assert len(stack) == 50
    # The ten oldest went: the earliest snapshot left already holds ten patterns.
    assert len(stack[0][1]) == 10
    assert len(stack[-1][1]) == 59


def test_an_applied_undo_advances_the_seq_guard_and_records_no_history(session):
    state, seq, _changed = apply_event_batch(
        {}, "15m", batch(completed(span_pattern("A", 0, 100))), 0)

    value = batch_from(seq, undo())
    reverted, new_seq, changed = apply_event_batch(state, "15m", value, seq)
    assert reverted["15m"] == []
    assert (new_seq, changed) == (seq + 1, True)
    assert undo_stack(session) == []                 # no redo entry pushed

    # Streamlit re-delivers the same component value on the next rerun.
    again, again_seq, again_changed = apply_event_batch(reverted, "15m", value, new_seq)
    assert again is reverted
    assert again_seq == new_seq
    assert again_changed is False

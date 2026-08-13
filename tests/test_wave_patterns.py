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
    child_leg_index,
    children_by_leg,
    degree_component,
    find_children,
    find_parent,
    pattern_span,
    point_labels,
    reconcile_degrees,
    related,
    relation_component,
    render_glyph,
    settle,
    validate_patterns,
    wave_defs,
)
from strategies.wave_marking_manager import load_wave_documents, save_wave_documents
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


def zigzag(pattern_id, pivots, degree="Minor", color="yellow"):
    """A valid Zigzag from explicit (time, kind) pivots."""
    return {
        "id": pattern_id,
        "pattern_type": "Zigzag",
        "variation": "Zigzag",
        "degree": degree,
        "color": color,
        "points": [{"time": time, "price": 100.0 + i, "kind": kind}
                   for i, (time, kind) in enumerate(pivots)],
    }


def impulse(pattern_id, times, degree="Minor", color="yellow", first_kind="low"):
    """A valid Impulse (0 i ii iii iv v) at six alternating pivots."""
    assert len(times) == 6
    other = "high" if first_kind == "low" else "low"
    kinds = [first_kind if i % 2 == 0 else other for i in range(6)]
    pattern = zigzag(pattern_id, list(zip(times, kinds)), degree=degree, color=color)
    pattern.update(pattern_type="Impulse", variation="Impulse")
    return pattern


def leg_child(pattern_id, parent, k, degree="Minor", color="yellow"):
    """A Zigzag spanning exactly leg ``k`` of ``parent``, endpoint for endpoint."""
    start, end = parent["points"][k], parent["points"][k + 1]
    assert end["time"] - start["time"] >= 3, "leg too narrow for four points"
    inner = "high" if start["kind"] == "low" else "low"
    return zigzag(pattern_id,
                  [(start["time"], start["kind"]),
                   (start["time"] + 1, inner),
                   (end["time"] - 1, start["kind"]),
                   (end["time"], end["kind"])],
                  degree=degree, color=color)


def wide_pattern(pattern_id="A", degree="Minor"):
    """A Zigzag whose every leg is wide enough to hold a child of its own."""
    return zigzag(pattern_id, [(0, "low"), (100, "high"), (200, "low"), (300, "high")],
                  degree=degree)


def nest(*degrees):
    """A root plus one leg-exact descendant per extra degree, senior first.

    Each child spans leg 1 of its parent, which is the only leg ``span_pattern``
    leaves wide enough to hold another four points.
    """
    root = span_pattern("A", 0, 1000, degrees[0])
    built = [root]
    for i, degree in enumerate(degrees[1:]):
        built.append(leg_child(chr(ord("B") + i), built[-1], 1, degree))
    return built


def document(patterns, base_timeframe="15m"):
    """A v2 on-disk document: one canonical list at the base timeframe."""
    return {"schema": 2, "base_timeframe": base_timeframe, "patterns": patterns}


def documents_file(tmp_path, documents):
    """A markings file holding several datasets. Never the real one."""
    path = str(tmp_path / "saved_wave_markings.json")
    save_wave_documents(documents, path)
    return path


def apply_to_dataset(path, dataset_key, value, last_seq=0):
    """What the tab does around a batch: load, fold, write that one key back.

    The read-modify-write is the whole of dataset isolation, so a test of it has
    to go through the file rather than through the reducers.
    """
    documents = load_wave_documents(path)
    entry = documents.get(dataset_key) or document([])
    updated, seq, changed = apply_event_batch(entry["patterns"], value, last_seq)
    if changed:
        documents[dataset_key] = document(updated)
        save_wave_documents(documents, path)
    return documents, seq, changed


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

def test_apply_appends_the_completed_pattern():
    pattern = make_pattern()
    state = apply_wave_event([], completed(pattern))
    assert state == [pattern]


def test_apply_leaves_another_datasets_document_alone(tmp_path):
    # There is one canonical list per *dataset* now, not one per timeframe, so
    # the isolation that matters is between instruments: a marking snapped to
    # gold's bars means nothing on silver's.
    other = [make_pattern(id="other")]
    path = documents_file(tmp_path, {"gold.csv": document([]),
                                     "silver.csv": document(other)})

    documents, _seq, changed = apply_to_dataset(
        path, "gold.csv", batch(completed(make_pattern())))

    assert changed is True
    assert [p["id"] for p in documents["gold.csv"]["patterns"]] == ["abc123"]
    assert load_wave_documents(path)["silver.csv"]["patterns"] == other


def test_apply_ignores_a_duplicate_id():
    pattern = make_pattern()
    state = apply_wave_event([], completed(pattern))
    again = apply_wave_event(state, completed(make_pattern(price_note="ignored")))
    assert again is state
    assert len(again) == 1


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
    before = []
    snapshot = copy.deepcopy(before)
    state = apply_wave_event(before, completed(bad))
    assert state is before
    assert before == snapshot


def test_apply_rejects_non_increasing_times():
    points = make_pattern()["points"]
    points[2] = dict(points[2], time=points[1]["time"])
    state = apply_wave_event([], completed(make_pattern(points=points)))
    assert state == []

    points[2] = dict(points[2], time=points[1]["time"] - 60)
    state = apply_wave_event([], completed(make_pattern(points=points)))
    assert state == []


def test_apply_rejects_bad_point_fields():
    for broken in [
        {"time": "1719878400", "price": 2360.5, "kind": "low"},
        {"time": 1719878400, "price": "cheap", "kind": "low"},
        {"time": 1719878400, "price": 2360.5, "kind": "middle"},
        "not a point",
    ]:
        points = make_pattern()["points"]
        points[0] = broken
        assert apply_wave_event([], completed(make_pattern(points=points))) == []


def test_apply_accepts_integer_prices():
    points = make_pattern()["points"]
    points[0] = dict(points[0], price=2360)      # JSON hands back an int for round prices
    state = apply_wave_event([], completed(make_pattern(points=points)))
    assert len(state) == 1


def test_apply_ignores_unknown_and_malformed_events():
    before = []
    assert apply_wave_event(before, {"type": "something_else"}) is before
    assert apply_wave_event(before, {}) is before
    assert apply_wave_event(before, None) is before
    assert apply_wave_event(before, {"type": "pattern_completed"}) is before


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


# ------------------------------------------------------- parent/child relation

def test_child_leg_index_finds_the_leg_a_child_spans_exactly():
    parent = wide_pattern()
    for k in range(3):
        assert child_leg_index(parent, leg_child("B", parent, k)) == k


def test_child_leg_index_rejects_a_mismatched_kind_on_either_endpoint():
    parent = span_pattern("A", 0, 1000)
    child = leg_child("B", parent, 1)

    wrong_start = copy.deepcopy(child)
    wrong_start["points"][0]["kind"] = "low"          # parent's point 1 is a high
    assert child_leg_index(parent, wrong_start) is None

    wrong_end = copy.deepcopy(child)
    wrong_end["points"][-1]["kind"] = "high"          # parent's point 2 is a low
    assert child_leg_index(parent, wrong_end) is None


def test_child_leg_index_rejects_a_pattern_spanning_more_than_one_leg():
    parent = span_pattern("A", 0, 1000)
    # Points 0 -> 2 of the parent: two legs, so not a child of it.
    two_legs = zigzag("B", [(0, "low"), (1, "high"), (500, "low"), (999, "low")])
    assert child_leg_index(parent, two_legs) is None


def test_child_leg_index_rejects_the_parents_own_full_span():
    parent = span_pattern("A", 0, 1000)
    flush = span_pattern("B", 0, 1000)
    assert child_leg_index(parent, flush) is None


def test_child_leg_index_never_makes_a_pattern_its_own_child():
    parent = span_pattern("A", 0, 1000)
    assert child_leg_index(parent, parent) is None
    assert child_leg_index(parent, copy.deepcopy(parent)) is None   # same id


def test_find_parent_ignores_red_patterns_on_either_side():
    parent = span_pattern("A", 0, 1000)
    child = leg_child("B", parent, 1)
    assert find_parent([parent, child], child)[0] is parent

    red_child = dict(child, color="red")
    assert find_parent([parent, red_child], red_child) == (None, None)

    red_parent = dict(parent, color="red")
    assert find_parent([red_parent, child], child) == (None, None)


def test_find_parent_ties_resolve_to_the_smallest_span():
    # Both expose the identical leg (0, low) -> (10, high); the shorter one wins.
    short = zigzag("short", [(0, "low"), (10, "high"), (20, "low"), (30, "high")])
    long = zigzag("long", [(0, "low"), (10, "high"), (20, "low"), (40, "high")])
    child = zigzag("child", [(0, "low"), (3, "high"), (7, "low"), (10, "high")])

    for order in ([short, long, child], [long, short, child]):
        parent, leg = find_parent(order, child)
        assert (parent["id"], leg) == ("short", 0)


def test_find_parent_ties_of_equal_span_resolve_to_the_lowest_id():
    a = zigzag("a", [(0, "low"), (10, "high"), (20, "low"), (30, "high")])
    b = zigzag("b", [(0, "low"), (10, "high"), (25, "low"), (30, "high")])
    child = zigzag("child", [(0, "low"), (3, "high"), (7, "low"), (10, "high")])

    for order in ([a, b, child], [b, a, child]):
        assert find_parent(order, child)[0]["id"] == "a"


def test_find_children_lists_every_child_in_input_order():
    parent = wide_pattern()
    first = leg_child("B", parent, 0)
    second = leg_child("C", parent, 1)
    stranger = span_pattern("Z", 5000, 6000)
    patterns = [parent, second, first, stranger]

    assert [p["id"] for p in find_children(patterns, parent)] == ["C", "B"]
    assert find_children(patterns, first) == []


def test_children_by_leg_indexes_a_three_level_nest():
    patterns = nest("Minor", "Minute", "Minuette")
    index = children_by_leg(patterns)

    assert set(index) == {"A", "B", "C"}
    assert {leg: [c["id"] for c in kids] for leg, kids in index["A"].items()} == {1: ["B"]}
    assert {leg: [c["id"] for c in kids] for leg, kids in index["B"].items()} == {1: ["C"]}
    assert index["C"] == {}


def test_children_by_leg_keeps_both_children_of_one_leg():
    parent = wide_pattern()
    first = leg_child("B", parent, 0)
    second = leg_child("C", parent, 0)
    third = leg_child("D", parent, 1)
    index = children_by_leg([parent, second, first, third])

    # Input order within a leg, and the second marking is kept rather than
    # silently losing to the first.
    assert [c["id"] for c in index["A"][0]] == ["C", "B"]
    assert [c["id"] for c in index["A"][1]] == ["D"]


def test_children_by_leg_leaves_a_red_pattern_out_entirely():
    patterns = nest("Minor", "Minute", "Minuette")
    patterns[1] = dict(patterns[1], color="red")
    index = children_by_leg(patterns)

    assert "B" not in index                 # not a parent
    assert index["A"] == {}                 # and not a child either
    assert index["C"] == {}


def test_children_by_leg_gives_a_childless_pattern_an_empty_mapping():
    lonely = span_pattern("A", 0, 100)
    assert children_by_leg([lonely]) == {"A": {}}
    assert children_by_leg("not a list") == {}


def test_chained_siblings_are_not_in_a_parent_child_relation():
    # His everyday workflow: the next pattern's point 0 sits exactly on the
    # previous pattern's terminal point. Neither is a leg of the other.
    first = span_pattern("A", 0, 100, "Minor")
    last = first["points"][-1]
    second = zigzag("B", [(last["time"], last["kind"]), (101, "low"),
                          (199, "high"), (200, "low")], degree="Minor")

    assert find_parent([first, second], second) == (None, None)
    assert find_parent([first, second], first) == (None, None)
    assert relation_component([first, second], "A") == ["A"]

    settled = settle([first, second])
    assert degrees_of(settled) == {"A": "Minor", "B": "Minor"}
    assert set(colors_of(settled).values()) == {"yellow"}


def test_relation_component_covers_a_whole_nest_from_any_member():
    patterns = nest("Minor", "Minute", "Minuette")
    assert relation_component(patterns, "C") == ["A", "B", "C"]
    assert relation_component(patterns, "A") == ["A", "B", "C"]


def test_relation_component_of_a_red_pattern_is_just_itself():
    patterns = nest("Minor", "Minute", "Minuette")
    patterns[1] = dict(patterns[1], color="red")
    assert relation_component(patterns, "B") == ["B"]
    # ...and the red member breaks the nest in two.
    assert relation_component(patterns, "A") == ["A"]
    assert relation_component(patterns, "C") == ["C"]


def test_relation_component_of_an_unknown_id_is_empty():
    assert relation_component([span_pattern("A", 0, 100)], "nope") == []
    assert relation_component("not a list", "A") == []


# ------------------------------------------------------------ reconcile_degrees

def test_reconcile_lands_a_three_level_nest_on_consecutive_degrees():
    patterns = nest("Subminuette", "Subminuette", "Supermillennium")
    assert degrees_of(reconcile_degrees(patterns)) == {
        "A": "Subminuette", "B": "Micro", "C": "Submicro",
    }


def test_reconcile_leaves_a_root_alone():
    lone = [span_pattern("A", 0, 1000, "Cycle")]
    assert degrees_of(reconcile_degrees(lone)) == {"A": "Cycle"}


def test_reconcile_derives_a_child_drawn_before_its_parent():
    # The child is completed first, so it is earlier in the list than the
    # pattern that turns out to own it. Walking roots first sorts that out.
    parent = span_pattern("A", 0, 1000, "Minute")
    child = leg_child("B", parent, 1, "Cycle")

    assert degrees_of(reconcile_degrees([child, parent])) == {
        "B": "Minuette", "A": "Minute",
    }


def test_reconcile_clamps_a_child_of_a_pico_parent():
    patterns = nest("Pico", "Minor")
    assert degrees_of(reconcile_degrees(patterns)) == {"A": "Pico", "B": "Pico"}

    # The clamp leaves them sharing a degree, which is a genuine collision --
    # the honest signal that the nest is deeper than the degree scale allows.
    assert set(colors_of(settle(patterns)).values()) == {"red"}


def test_reconcile_does_not_mutate_its_input():
    patterns = nest("Minor", "Cycle")
    snapshot = copy.deepcopy(patterns)
    reconciled = reconcile_degrees(patterns)

    assert degrees_of(reconciled) == {"A": "Minor", "B": "Minute"}
    assert patterns == snapshot


def test_reconcile_tolerates_junk_input():
    assert reconcile_degrees("not a list") == []
    assert reconcile_degrees([None, {"no": "id"}]) == [None, {"no": "id"}]


# ---------------------------------------------------------------------- settle

def test_settle_derives_a_childs_degree_and_keeps_the_pair_yellow():
    patterns = nest("Subminuette", "Subminuette")
    settled = settle(patterns)

    assert degrees_of(settled) == {"A": "Subminuette", "B": "Micro"}
    assert set(colors_of(settled).values()) == {"yellow"}


@pytest.mark.parametrize("shape", [
    [],
    [span_pattern("A", 0, 1000, "Minor")],
    nest("Minor", "Minor", "Minor"),
    nest("Pico", "Minor"),
    nest("Subnano", "Minor", "Minor"),
    # A red collision sitting inside a nest: X overlaps the root's interior at
    # the root's own degree, so both go red while the nest below carries on.
    nest("Minor", "Minute") + [span_pattern("X", 500, 1500, "Minor")],
    [span_pattern("A", 0, 100, "Minor"), span_pattern("B", 50, 150, "Minor")],
])
def test_settle_is_a_fixed_point(shape):
    once = settle(shape)
    assert settle(once) == once


def test_settle_recomputes_persisted_degrees_and_colours():
    # What a file written before the relation existed can hold: a child stored
    # at its parent's degree and painted red for the overlap that caused.
    patterns = nest("Subminuette", "Subminuette")
    patterns = [dict(p, color="red") for p in patterns]

    settled = settle(patterns)
    assert degrees_of(settled) == {"A": "Subminuette", "B": "Micro"}
    assert set(colors_of(settled).values()) == {"yellow"}


def test_settle_does_not_mutate_its_input():
    patterns = nest("Minor", "Cycle")
    snapshot = copy.deepcopy(patterns)
    settle(patterns)
    assert patterns == snapshot


# --------------------------------------------------------------- shift_degree

def shift(pattern_id, delta):
    return {"type": "shift_degree", "id": pattern_id, "delta": delta}


def test_shift_degree_cascades_through_the_whole_nest():
    # A real nest, not mere containment: B spans one leg of A exactly, C one
    # leg of B. Offsets are the consecutive degrees reconciliation produces.
    patterns = nest("Minor", "Minute", "Minuette")      # indexes 8, 9, 10
    state = patterns

    # Ctrl and + on the middle pattern: every member moves one step senior.
    senior = apply_wave_event(state, shift("B", 1))
    assert degrees_of(senior) == {
        "A": "Intermediate", "B": "Minor", "C": "Minute",
    }

    # ...and back down again, restoring the original relative offsets.
    junior = apply_wave_event(senior, shift("B", -1))
    assert degrees_of(junior) == {
        "A": "Minor", "B": "Minute", "C": "Minuette",
    }
    assert state is patterns                  # the input is never mutated


def test_shift_degree_leaves_unrelated_patterns_alone():
    patterns = nest("Minor", "Minute") + [span_pattern("Z", 5000, 6000, "Minor")]
    state = apply_wave_event(patterns, shift("A", 1))
    assert degrees_of(state) == {
        "A": "Intermediate", "B": "Minor", "Z": "Minor",
    }


def test_shift_degree_leaves_a_merely_contained_pattern_alone():
    # The old cascade ran on containment, so this pair moved together. It sits
    # inside A's span without spanning any single leg of it, so under the
    # parent/child relation the two are strangers and shift independently.
    patterns = [
        span_pattern("A", 0, 1000, "Minor"),
        span_pattern("B", 100, 400, "Minuette"),
    ]
    state = apply_wave_event(patterns, shift("A", 1))
    assert degrees_of(state) == {"A": "Intermediate", "B": "Minuette"}


def test_shift_degree_is_all_or_nothing_at_the_senior_end():
    patterns = nest("Supermillennium", "Millennium")     # index 0 -- already the top
    before = patterns
    after = apply_wave_event(before, shift("B", 1))

    assert after is before
    assert degrees_of(after) == {"A": "Supermillennium", "B": "Millennium"}


def test_shift_degree_is_all_or_nothing_at_the_junior_end():
    patterns = nest("Subnano", "Pico")                   # index 17 -- the bottom
    before = patterns
    after = apply_wave_event(before, shift("A", -1))

    assert after is before
    assert degrees_of(after) == {"A": "Subnano", "B": "Pico"}


def test_shift_degree_moves_a_red_pattern_on_its_own():
    # Two same-degree patterns overlapping in their interiors are red, and a
    # red pattern is in no relation -- which is exactly what lets the client
    # walk one of them to a free degree by hand to resolve the collision.
    patterns = validate_patterns([span_pattern("A", 0, 100, "Minor"),
                                  span_pattern("B", 50, 150, "Minor")])
    assert set(colors_of(patterns).values()) == {"red"}

    state = apply_wave_event(patterns, shift("B", -1))
    assert degrees_of(state) == {"A": "Minor", "B": "Minute"}


@pytest.mark.parametrize("delta", [0, 2, -2, "1", 1.0, True, None])
def test_shift_degree_rejects_any_delta_other_than_plus_or_minus_one(delta):
    before = [span_pattern("A", 0, 1000, "Minor")]
    assert apply_wave_event(before, shift("A", delta)) is before


def test_shift_degree_ignores_an_unknown_id():
    before = [span_pattern("A", 0, 1000, "Minor")]
    assert apply_wave_event(before, shift("nope", 1)) is before
    assert apply_wave_event(before, shift(None, 1)) is before


def test_shift_degree_rejects_a_pattern_at_an_unknown_degree():
    broken = span_pattern("A", 0, 1000, "Minor")
    broken["degree"] = "Nonexistent"
    before = [broken]
    assert apply_wave_event(before, shift("A", 1)) is before


# ------------------------------------------------------------------ move_point

def move(pattern_id, index, time, price=999.0, kind="high"):
    return {"type": "move_point", "id": pattern_id, "point_index": index,
            "time": time, "price": price, "kind": kind}


def test_move_point_replaces_the_point():
    before = [span_pattern("A", 100, 200)]
    after = apply_wave_event(before, move("A", 1, 150, 123.5, "high"))

    assert after[0]["points"][1] == {"time": 150, "price": 123.5, "kind": "high"}
    assert before[0]["points"][1]["time"] == 101      # input untouched


def test_move_point_rejects_a_time_on_or_past_a_neighbour():
    before = [span_pattern("A", 100, 200)]          # times 100, 101, 199, 200

    assert apply_wave_event(before, move("A", 1, 100)) is before   # == previous
    assert apply_wave_event(before, move("A", 1, 199)) is before   # == next
    assert apply_wave_event(before, move("A", 1, 250)) is before   # past next
    assert apply_wave_event(before, move("A", 2, 101)) is before   # == previous


def test_move_point_accepts_the_first_and_last_points_bounded_by_one_neighbour():
    before = [span_pattern("A", 100, 200)]          # times 100, 101, 199, 200

    first = apply_wave_event(before, move("A", 0, 40, 5.0, "low"))
    assert [p["time"] for p in first[0]["points"]] == [40, 101, 199, 200]

    last = apply_wave_event(before, move("A", 3, 900, 5.0, "low"))
    assert [p["time"] for p in last[0]["points"]] == [100, 101, 199, 900]

    # ...but they are still bounded on their one side.
    assert apply_wave_event(before, move("A", 0, 101)) is before
    assert apply_wave_event(before, move("A", 3, 199)) is before


def test_move_point_rejects_an_out_of_range_index():
    before = [span_pattern("A", 100, 200)]
    for index in (-1, 4, 99):
        assert apply_wave_event(before, move("A", index, 150)) is before


def test_move_point_rejects_bad_fields_and_unknown_ids():
    before = [span_pattern("A", 100, 200)]

    assert apply_wave_event(before, move("nope", 1, 150)) is before
    assert apply_wave_event(before, move("A", 1, 150.5)) is before      # float time
    assert apply_wave_event(before, move("A", "1", 150)) is before      # str index
    assert apply_wave_event(before, move("A", 1, 150, "cheap")) is before
    assert apply_wave_event(before, move("A", 1, 150, 9.0, "middle")) is before
    assert apply_wave_event(before, move("A", True, 150)) is before     # bool index


def test_move_point_only_touches_the_named_pattern():
    untouched = span_pattern("B", 300, 400)
    before = [span_pattern("A", 100, 200), untouched]
    after = apply_wave_event(before, move("A", 1, 150))

    assert after[1] is untouched
    assert after[0]["points"][1]["time"] == 150


def test_move_point_does_not_clear_the_legs_stored_values():
    # A moved pivot can make a stored number stale, but silently deleting the
    # client's study data is worse than showing a stale one -- he may have typed
    # a judgement rather than a reading -- so the popup surfaces the difference
    # instead of the reducer resolving it for him.
    stored = {"0": {"CMB": 12.34, "RSI": 45.6, "timeframe": "1D"}}
    pattern = span_pattern("A", 100, 200)
    pattern["leg_values"] = copy.deepcopy(stored)

    after = apply_wave_event([pattern], move("A", 1, 150))

    assert after[0]["leg_values"] == stored


# -------------------------------------------------------------- delete_pattern

def delete(pattern_id):
    return {"type": "delete_pattern", "id": pattern_id}


def test_delete_pattern_removes_by_id():
    before = [span_pattern("A", 0, 100), span_pattern("B", 200, 300)]
    after = apply_wave_event(before, delete("A"))

    assert [p["id"] for p in after] == ["B"]
    assert len(before) == 2                   # the input list is not mutated


def test_delete_pattern_ignores_an_unknown_id():
    before = [span_pattern("A", 0, 100)]
    assert apply_wave_event(before, delete("nope")) is before
    assert apply_wave_event(before, delete("")) is before
    assert apply_wave_event(before, delete(None)) is before
    assert apply_wave_event([], delete("A")) == []


def test_delete_pattern_leaves_another_datasets_document_alone(tmp_path):
    other = [span_pattern("A", 0, 100)]
    path = documents_file(tmp_path, {"gold.csv": document([span_pattern("A", 0, 100)]),
                                     "silver.csv": document(other)})

    documents, _seq, changed = apply_to_dataset(path, "gold.csv", batch(delete("A")))

    assert changed is True
    assert documents["gold.csv"]["patterns"] == []
    # Same id on the other instrument -- deleting one must not reach the other.
    assert load_wave_documents(path)["silver.csv"]["patterns"] == other


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

    state, seq, changed = apply_event_batch([], value, 0)

    assert state == []
    assert (seq, changed) == (2, True)


def test_batch_is_a_no_op_when_replayed_under_the_seq_guard():
    value = batch(completed(span_pattern("A", 0, 100)), delete("A"))
    state, seq, _changed = apply_event_batch([], value, 0)

    # Streamlit re-delivers the same value on the next rerun.
    again, again_seq, again_changed = apply_event_batch(state, value, seq)
    assert again is state
    assert again_seq == seq
    assert again_changed is False


def test_batch_applies_only_the_events_above_the_stored_seq():
    first = span_pattern("A", 0, 100)
    second = span_pattern("B", 200, 300)
    value = batch(completed(first), completed(second))

    state, seq, _ = apply_event_batch([], value, 1)      # eseq 1 already seen
    assert [p["id"] for p in state] == ["B"]
    assert seq == 2


def test_batch_applies_out_of_order_events_in_eseq_order():
    pattern = span_pattern("A", 0, 100)
    value = {"seq": 2, "events": [
        dict(delete("A"), eseq=2),
        dict(completed(pattern), eseq=1),
    ]}
    state, seq, changed = apply_event_batch([], value, 0)

    assert state == []
    assert (seq, changed) == (2, True)


def test_batch_acks_rejected_events_so_they_are_never_replayed():
    before = []
    value = batch(delete("nope"))

    state, seq, changed = apply_event_batch(before, value, 0)
    assert state is before
    assert seq == 1                      # acked anyway
    assert changed is False


def test_batch_ignores_values_that_carry_no_events():
    before = []
    for value in [None, {}, {"seq": 3}, {"events": "nope"}, "junk"]:
        assert apply_event_batch(before, value, 7) == (before, 7, False)


def test_batch_colours_a_completed_overlap_red():
    value = batch(completed(span_pattern("A", 0, 100)),
                  completed(span_pattern("B", 50, 150)))
    state, _seq, changed = apply_event_batch([], value, 0)

    assert changed is True
    assert colors_of(state) == {"A": "red", "B": "red"}


def test_batch_revalidates_after_a_shift_degree():
    # Neither contains the other, so the cascade touches B alone -- but the
    # shift lands B on A's degree, where their interiors already overlap.
    state = [span_pattern("A", 0, 100, "Minor"),
                     span_pattern("B", 50, 150, "Intermediate")]
    assert set(colors_of(state).values()) == {"yellow"}

    collided, _seq, _changed = apply_event_batch(state, batch(shift("B", -1)), 0)
    assert degrees_of(collided) == {"A": "Minor", "B": "Minor"}
    assert colors_of(collided) == {"A": "red", "B": "red"}

    healed, _seq, _changed = apply_event_batch(collided, batch(shift("B", 1)), 0)
    assert degrees_of(healed) == {"A": "Minor", "B": "Intermediate"}
    assert colors_of(healed) == {"A": "yellow", "B": "yellow"}


def test_batch_revalidates_after_a_move_point():
    # A and B are chained on time 100 -- legal until B's point 0 is dragged back.
    state = [span_pattern("A", 0, 100), span_pattern("B", 100, 200)]

    collided, _seq, _changed = apply_event_batch(state, batch(move("B", 0, 50, 5.0, "low")), 0)
    assert colors_of(collided) == {"A": "red", "B": "red"}

    healed, _seq, _changed = apply_event_batch(collided, batch(move("B", 0, 100, 5.0, "low")), 0)
    assert colors_of(healed) == {"A": "yellow", "B": "yellow"}


def test_batch_leaves_another_datasets_document_uncoloured(tmp_path):
    # An overlapping pair stored on another instrument, never settled. Folding a
    # batch on this dataset must not so much as recolour it.
    other = [span_pattern("A", 0, 100), span_pattern("B", 50, 150)]
    path = documents_file(tmp_path, {"gold.csv": document([]),
                                     "silver.csv": document(other)})

    documents, _seq, _changed = apply_to_dataset(
        path, "gold.csv", batch(completed(span_pattern("Z", 0, 100))))

    assert colors_of(documents["gold.csv"]["patterns"]) == {"Z": "yellow"}
    assert load_wave_documents(path)["silver.csv"]["patterns"] == other


def test_batch_auto_degrees_a_child_marked_at_its_parents_degree():
    """The client's Video 4, end to end.

    He marked a Subminuette Impulse, left the toolbar on Subminuette and marked
    a second Impulse from the first's ``iv`` pivot to its ``v`` pivot. It stayed
    Subminuette and the pair went red. It must come out Micro, both yellow.
    """
    parent = impulse("P", [0, 10, 20, 30, 40, 50], "Subminuette")
    iv, v = parent["points"][4], parent["points"][5]
    child = impulse("C", [iv["time"], 42, 44, 46, 48, v["time"]], "Subminuette")
    assert (child["points"][0]["kind"], child["points"][-1]["kind"]) == (iv["kind"], v["kind"])

    state, _seq, changed = apply_event_batch([], batch(completed(parent), completed(child)), 0)

    assert changed is True
    assert degrees_of(state) == {"P": "Subminuette", "C": "Micro"}
    assert colors_of(state) == {"P": "yellow", "C": "yellow"}


def test_batch_re_derives_a_child_drawn_before_its_parent():
    parent = span_pattern("A", 0, 1000, "Minute")
    child = leg_child("B", parent, 1, "Minute")

    state, _seq, _changed = apply_event_batch([], batch(completed(child)), 0)
    assert degrees_of(state) == {"B": "Minute"}      # still a root

    state, _seq, _changed = apply_event_batch(state, batch_from(1, completed(parent)), 1)
    assert degrees_of(state) == {"B": "Minuette", "A": "Minute"}
    assert set(colors_of(state).values()) == {"yellow"}


def test_batch_move_point_off_a_parents_pivot_makes_a_root():
    parent = wide_pattern("A", "Minute")
    child = leg_child("B", parent, 1, "Cycle")
    state, seq, _changed = apply_event_batch([], batch(completed(parent), completed(child)), 0)
    assert degrees_of(state) == {"A": "Minute", "B": "Minuette"}

    # Drag the child's point 0 off its parent's pivot: the link breaks and the
    # ex-child keeps the degree it currently has -- there is no memory of the
    # "Cycle" it was drawn with.
    broken, seq, _changed = apply_event_batch(state, batch_from(seq, move("B", 0, 50, 101.0, "high")), seq)
    assert degrees_of(broken) == {"A": "Minute", "B": "Minuette"}
    assert find_parent(broken, broken[1]) == (None, None)

    # ...and dragging it back re-establishes the relation.
    remade, _seq, _changed = apply_event_batch(broken, batch_from(seq, move("B", 0, 100, 101.0, "high")), seq)
    assert find_parent(remade, remade[1])[0]["id"] == "A"


def test_batch_ignores_events_without_a_usable_eseq():
    value = {"seq": 1, "events": [
        completed(span_pattern("A", 0, 100)),            # no eseq at all
        dict(completed(span_pattern("B", 0, 100)), eseq="2"),
        dict(completed(span_pattern("C", 0, 100)), eseq=True),
        "not an event",
    ]}
    before = []
    assert apply_event_batch(before, value, 0) == (before, 0, False)


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
    marked, seq, _changed = apply_event_batch([], batch(completed(span_pattern("A", 0, 100)),
                         completed(span_pattern("B", 50, 150))), 0)
    assert colors_of(marked) == {"A": "red", "B": "red"}

    # Deleting B heals A, so the snapshot's stored colours are stale the moment
    # it is restored -- the undo has to recolour, not just put the list back.
    deleted, seq, _changed = apply_event_batch(marked, batch_from(seq, delete("B")), seq)
    assert colors_of(deleted) == {"A": "yellow"}

    restored, seq, changed = apply_event_batch(deleted, batch_from(seq, undo()), seq)

    assert changed is True
    assert [p["id"] for p in restored] == ["A", "B"]
    assert colors_of(restored) == {"A": "red", "B": "red"}


def test_undo_after_a_shift_degree_reverts_the_whole_cascade():
    before = nest("Minor", "Minute", "Minuette")

    shifted, seq, _changed = apply_event_batch(before, batch(shift("B", 1)), 0)
    assert degrees_of(shifted) == {
        "A": "Intermediate", "B": "Minor", "C": "Minute",
    }

    reverted, _seq, changed = apply_event_batch(shifted, batch_from(seq, undo()), seq)

    assert changed is True
    assert degrees_of(reverted) == {"A": "Minor", "B": "Minute", "C": "Minuette"}


def test_undo_with_an_empty_stack_is_a_no_op(session):
    before = [span_pattern("A", 0, 100)]

    state, seq, changed = apply_event_batch(before, batch(undo()), 0)

    assert state is before
    assert (seq, changed) == (1, False)      # acked anyway, so it is never replayed
    assert undo_stack(session) == []


def test_undo_walks_the_whole_history_back_one_snapshot_at_a_time(session):
    # Deliberate behaviour change from the per-timeframe stack: there is one
    # canonical list now, so every snapshot is on the same stack and Ctrl+Z
    # walks straight back through all of them whatever the client is looking at.
    state, seq, _changed = apply_event_batch([], batch(completed(span_pattern("A", 0, 100))), 0)
    state, seq, _changed = apply_event_batch(
        state, batch_from(seq, completed(span_pattern("B", 200, 300))), seq)
    state, seq, _changed = apply_event_batch(
        state, batch_from(seq, completed(span_pattern("Z", 400, 500))), seq)
    assert len(undo_stack(session)) == 3

    reverted, seq, changed = apply_event_batch(state, batch_from(seq, undo()), seq)
    assert changed is True
    assert [p["id"] for p in reverted] == ["A", "B"]

    # ...and again, one snapshot at a time, until the stack is empty.
    reverted, seq, _changed = apply_event_batch(reverted, batch_from(seq, undo()), seq)
    assert [p["id"] for p in reverted] == ["A"]
    reverted, seq, _changed = apply_event_batch(reverted, batch_from(seq, undo()), seq)
    assert reverted == []

    again, _seq, again_changed = apply_event_batch(reverted, batch_from(seq, undo()), seq)
    assert again is reverted
    assert again_changed is False
    assert undo_stack(session) == []


def test_undo_stack_drops_the_oldest_snapshot_past_fifty(session):
    state, seq = [], 0
    for i in range(60):
        state, seq, _changed = apply_event_batch(
            state,
            batch_from(seq, completed(span_pattern("p%d" % i, i * 10, i * 10 + 5))), seq)

    stack = undo_stack(session)
    assert len(stack) == 50
    # The ten oldest went: the earliest snapshot left already holds ten patterns.
    assert len(stack[0]) == 10
    assert len(stack[-1]) == 59


def test_an_applied_undo_advances_the_seq_guard_and_records_no_history(session):
    state, seq, _changed = apply_event_batch([], batch(completed(span_pattern("A", 0, 100))), 0)

    value = batch_from(seq, undo())
    reverted, new_seq, changed = apply_event_batch(state, value, seq)
    assert reverted == []
    assert (new_seq, changed) == (seq + 1, True)
    assert undo_stack(session) == []                 # no redo entry pushed

    # Streamlit re-delivers the same component value on the next rerun.
    again, again_seq, again_changed = apply_event_batch(reverted, value, new_seq)
    assert again is reverted
    assert again_seq == new_seq
    assert again_changed is False

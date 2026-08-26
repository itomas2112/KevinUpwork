"""
Tests for actionary/reactionary roles, direction, chaining and pair discovery.

Every structure here is built inline -- a parent, then children spanning its
legs endpoint for endpoint -- so each expected pair is obvious by construction
rather than by trusting the discovery code to agree with itself. The prices are
chosen so a bull impulse's waves really do run 1 3 5 up and 2 4 back, because
direction is one of the client's pairing rules and a fixture that got it wrong
would still produce pairs, just the wrong ones.

Leg indexing, as everywhere in this feature: leg ``k`` runs ``points[k] ->
points[k + 1]`` and is the wave ``point_labels(...)[k + 1]`` names. So wave "1"
of an impulse is leg 0 and wave "5" is leg 4.

The rename of ``CMB``/``RSI`` to ``Peak CMB``/``Peak RSI`` is tested at the
bottom: it ships in the same phase and, because a field name is the key a
reading is stored under, it is a migration rather than a relabelling.
"""

import json

import pytest

from config.wave_analysis import (
    ACTIONARY_LABELS,
    LEG_VALUE_FIELDS,
    PATTERN_DEFS,
    REACTIONARY_LABELS,
    chained,
    is_valid_pattern,
    leg_is_complete,
    migrate_leg_value_fields,
    pattern_direction,
    pattern_role,
    role_label,
    settle,
)
from strategies.wave_marking_manager import load_wave_documents, save_wave_documents
from strategies.wave_study import (PARENT_WAVE_LABELS, SKIP_NOT_MEASURED,
                                   adjacent_pairs, inverse_parallel_pairs,
                                   parallel_pairs, run_pair_study)


# --------------------------------------------------------------- the fixtures


def marking(pattern_id, pattern_type, variation, pivots, degree="Minor",
            color="yellow", leg_values=None):
    """A pattern from explicit ``(time, price, kind)`` pivots."""
    pattern = {
        "id": pattern_id,
        "pattern_type": pattern_type,
        "variation": variation,
        "degree": degree,
        "color": color,
        "points": [{"time": time, "price": float(price), "kind": kind}
                   for time, price, kind in pivots],
    }
    if leg_values is not None:
        pattern["leg_values"] = leg_values
    return pattern


# A bull five-wave count: up, back, further up, back, up again. Read off these
# six numbers, waves 1 3 5 are bullish and waves 2 4 bearish, which is what
# makes the actionary waves agree in direction and the reactionary ones oppose.
BULL_FIVE = [10.0, 20.0, 15.0, 35.0, 30.0, 50.0]


def six_point(pattern_id, pattern_type, variation, start=0, step=100,
              degree="Minor", prices=None):
    """A six-point count -- five legs -- starting on a low."""
    prices = BULL_FIVE if prices is None else prices
    times = [start + step * index for index in range(6)]
    kinds = ["low" if index % 2 == 0 else "high" for index in range(6)]
    return marking(pattern_id, pattern_type, variation,
                   list(zip(times, prices, kinds)), degree)


def impulse(pattern_id="i", start=0, step=100, degree="Minor", prices=None):
    return six_point(pattern_id, "Impulse", "Impulse", start, step, degree, prices)


def triple_zigzag(pattern_id="t", start=0, step=100, degree="Minor"):
    """W X Y X Z -- the one shape with two legs sharing a label."""
    return six_point(pattern_id, "Zigzag", "Triple Zigzag", start, step, degree)


def zigzag(pattern_id="z", start=0, step=100, degree="Minor",
           prices=(10.0, 20.0, 15.0, 35.0)):
    times = [start + step * index for index in range(4)]
    kinds = ["low" if index % 2 == 0 else "high" for index in range(4)]
    return marking(pattern_id, "Zigzag", "Zigzag", list(zip(times, prices, kinds)),
                   degree)


def leg_child(pattern_id, parent, k, degree="Minute", color="yellow"):
    """A Zigzag spanning exactly leg ``k`` of ``parent``, endpoint for endpoint.

    Its own first and last prices are the parent leg's, so the child runs the
    way the wave it marks runs -- which is precisely what ``pattern_direction``
    reads off it.
    """
    start, end = parent["points"][k], parent["points"][k + 1]
    assert end["time"] - start["time"] >= 3, "leg too narrow for four points"
    inner = "high" if start["kind"] == "low" else "low"
    mid = (start["price"] + end["price"]) / 2.0
    return marking(pattern_id, "Zigzag", "Zigzag",
                   [(start["time"], start["price"], start["kind"]),
                    (start["time"] + 1, mid, inner),
                    (end["time"] - 1, mid, start["kind"]),
                    (end["time"], end["price"], end["kind"])],
                   degree, color)


def fully_marked(parent, prefix, legs=None, degree="Minute"):
    """The parent followed by one child per leg, in leg order."""
    legs = range(len(parent["points"]) - 1) if legs is None else legs
    return [parent] + [leg_child(f"{prefix}{k}", parent, k, degree) for k in legs]


def correction_after(pattern_id, prev, prices=(30.0, 45.0, 20.0), step=100,
                     offset=0):
    """A Zigzag whose point 0 sits on ``prev``'s last point, ``offset`` bars off.

    With ``offset`` at zero the two counts are chained the way the client marks
    them. Any other value is how the tests below break the chain: the shared
    pivot is what adjacency means, and a bar's worth of daylight ends it.
    """
    last = prev["points"][-1]
    other = "high" if last["kind"] == "low" else "low"
    kinds = [last["kind"], other, last["kind"], other]
    times = [last["time"] + offset + step * index for index in range(4)]
    pivots = [(times[0], last["price"], kinds[0])]
    pivots += [(times[index + 1], prices[index], kinds[index + 1])
               for index in range(3)]
    return marking(pattern_id, "Zigzag", "Zigzag", pivots, prev["degree"])


def impulse_into_zigzag(offset=0, prices=(30.0, 45.0, 20.0)):
    """The client's worked example: a marked bull Impulse into its correction.

    Every leg of both counts is marked, so waves (1)..(5) and (A)(B)(C) are all
    present as patterns in their own right -- which is what lets them be paired
    at all, since a role only exists for a pattern whose parent is marked.

    Settled, because colour and degree are outputs and every rule here reads
    one of them.
    """
    parent = impulse("i")
    correction = correction_after("z", parent, prices, offset=offset)
    return settle(fully_marked(parent, "i") + fully_marked(correction, "z"))


def summary(pairs):
    """Pairs as plain tuples: (a id, b id, role a, role b, gap id)."""
    return [(pair["a"]["id"], pair["b"]["id"], pair["role_a"], pair["role_b"],
             pair["gap"]["id"] if pair["gap"] else None)
            for pair in pairs]


def roles(pairs):
    return [(pair["role_a"], pair["role_b"]) for pair in pairs]


# ------------------------------------------------------------ 1. the label sets


@pytest.mark.parametrize("label", ["1", "3", "5", "A", "C", "E", "W", "Y", "Z"])
def test_every_label_the_client_calls_actionary_is_actionary(label):
    # His lists, type by type: Impulse/Diagonal 1 3 5; Zigzag/Flat A C; Triangle
    # A C E; Double/Double-Zigzag W Y; Triple/Triple-Zigzag W Y Z.
    assert label in ACTIONARY_LABELS


@pytest.mark.parametrize("label", ["2", "4", "B", "D", "X"])
def test_every_label_the_client_calls_reactionary_is_reactionary(label):
    # Impulse/Diagonal 2 4; Zigzag/Flat B; Triangle B D; Double W-X-Y and Triple
    # W-X-Y-X-Z alike contribute X, twice in the triple's case.
    assert label in REACTIONARY_LABELS


def test_the_two_role_sets_are_disjoint():
    assert not ACTIONARY_LABELS & REACTIONARY_LABELS


def test_the_origin_click_is_neither_role():
    # "0" is where the count starts, not a wave, so it has no direction to be in
    # the same as or opposite to.
    assert "0" not in ACTIONARY_LABELS
    assert "0" not in REACTIONARY_LABELS


def test_every_wave_label_in_the_definitions_has_a_role():
    # The two sets are not merely correct about the labels he listed; they cover
    # every label the pattern table can produce, so no marked wave is left
    # unclassifiable by an oversight in a variation nobody thought about.
    labels = {label for variations in PATTERN_DEFS.values()
              for _name, label_seq in variations for label in label_seq}

    assert labels <= (ACTIONARY_LABELS | REACTIONARY_LABELS)


# ---------------------------------------------------- 2. role_label/pattern_role


def test_a_child_on_leg_zero_of_an_impulse_is_wave_one_and_actionary():
    patterns = settle(fully_marked(impulse(), "i", legs=[0]))

    assert role_label(patterns, patterns[1]) == "1"
    assert pattern_role(patterns, patterns[1]) == "actionary"


def test_a_child_on_leg_one_of_an_impulse_is_wave_two_and_reactionary():
    patterns = settle(fully_marked(impulse(), "i", legs=[1]))

    assert role_label(patterns, patterns[1]) == "2"
    assert pattern_role(patterns, patterns[1]) == "reactionary"


def test_a_root_has_no_role_at_all():
    # Actionary is defined against the larger wave, and only a marked parent
    # says what that is. A root gets None rather than a guess.
    patterns = settle(fully_marked(impulse(), "i", legs=[0]))

    assert role_label(patterns, patterns[0]) is None
    assert pattern_role(patterns, patterns[0]) is None


def test_a_red_child_has_no_role():
    # The parent/child relation is yellow-only, so a contested marking is in no
    # relation and therefore occupies no wave of anything.
    parent = impulse()
    child = leg_child("c", parent, 0, color="red")

    assert role_label([parent, child], child) is None
    assert pattern_role([parent, child], child) is None


def test_each_x_child_of_a_triple_zigzag_gets_its_own_role_from_its_leg():
    # W X Y X Z: two legs carry the label "X". The lookup is by leg, not by
    # label, so neither child is ambiguous and both come out reactionary.
    patterns = settle(fully_marked(triple_zigzag(), "t"))
    by_id = {p["id"]: p for p in patterns}

    assert [role_label(patterns, by_id[f"t{k}"]) for k in range(5)] == \
        ["W", "X", "Y", "X", "Z"]
    assert pattern_role(patterns, by_id["t1"]) == "reactionary"
    assert pattern_role(patterns, by_id["t3"]) == "reactionary"
    assert pattern_role(patterns, by_id["t0"]) == "actionary"


# ------------------------------------------------------- 3. pattern_direction


def test_a_pattern_ending_above_where_it_started_is_bullish():
    assert pattern_direction(zigzag(prices=(10.0, 20.0, 15.0, 35.0))) == "bullish"


def test_a_pattern_ending_below_where_it_started_is_bearish():
    assert pattern_direction(zigzag(prices=(35.0, 15.0, 20.0, 10.0))) == "bearish"


def test_a_pattern_ending_where_it_started_has_no_direction():
    # A flat count agrees with nothing and opposes nothing, so it is left out of
    # every analysis that pairs on direction rather than being called one.
    assert pattern_direction(zigzag(prices=(10.0, 20.0, 15.0, 10.0))) is None


def test_direction_of_something_unusable_is_none():
    assert pattern_direction(None) is None
    assert pattern_direction({}) is None
    assert pattern_direction({"points": [{"time": 0, "price": 1.0, "kind": "low"}]}) is None


# ------------------------------------------------------------------ 4. chained


def test_a_count_starting_on_the_previous_counts_last_pivot_is_chained():
    first = impulse()
    second = correction_after("z", first)

    assert chained(first, second)


def test_the_same_time_on_the_other_kind_is_not_chained():
    # A bar's high and its low are two different pivots. A count hanging off the
    # wrong one is not adjacent to anything.
    first = impulse()
    second = correction_after("z", first)
    second["points"][0]["kind"] = "low"            # the impulse ends on a high

    assert not chained(first, second)


def test_one_bar_of_daylight_breaks_the_chain():
    first = impulse()

    assert not chained(first, correction_after("z", first, offset=1))


def test_chaining_is_directional_and_survives_rubbish():
    first = impulse()
    second = correction_after("z", first)

    assert not chained(second, first)
    assert not chained(first, None)
    assert not chained({}, second)


# ----------------------------------------------------------- 5. parallel_pairs


def test_a_five_child_impulse_yields_every_same_role_sibling_pair():
    # His examples, in one structure: (1)v(3), (1)v(5), (3)v(5) and (2)v(4).
    # (1)v(5) has three waves between it, which is the whole reason discovery is
    # "same role, same parent, different legs" rather than "adjacent but one".
    patterns = settle(fully_marked(impulse(), "i"))
    pairs = parallel_pairs(patterns)

    assert summary(pairs) == [
        ("i0", "i2", "1", "3", None),
        ("i0", "i4", "1", "5", None),
        ("i1", "i3", "2", "4", None),
        ("i2", "i4", "3", "5", None),
    ]


def test_a_zigzag_parent_yields_only_a_against_c():
    # A and C act, B reacts, so there is exactly one same-role pair to find.
    patterns = settle(fully_marked(zigzag(), "z"))

    assert summary(parallel_pairs(patterns)) == [("z0", "z2", "A", "C", None)]


def test_a_triple_zigzag_parent_yields_the_x_against_x_pair():
    # "Triple Three/Triple Zigzag: W, Y, Z" act and "X, X" react -- he lists the
    # second X explicitly, so the two X waves of one count are a pair.
    patterns = settle(fully_marked(triple_zigzag(), "t"))

    assert ("X", "X") in roles(parallel_pairs(patterns))
    assert summary(parallel_pairs(patterns)) == [
        ("t0", "t2", "W", "Y", None),
        ("t0", "t4", "W", "Z", None),
        ("t1", "t3", "X", "X", None),
        ("t2", "t4", "Y", "Z", None),
    ]


def test_two_children_of_one_leg_are_a_duplicate_marking_not_a_pair():
    # Deliberately not settled. Two markings of one wave span identically, so
    # settle paints both red and they would drop out on colour long before the
    # rule was reached; the rule still has to hold on its own, because it is
    # what the relation means rather than a consequence of the colouring.
    parent = impulse()
    patterns = fully_marked(parent, "i") + [leg_child("i0b", parent, 0)]

    found = {(pair["a"]["id"], pair["b"]["id"]) for pair in parallel_pairs(patterns)}

    assert ("i0", "i0b") not in found and ("i0b", "i0") not in found
    # The duplicate still pairs with the *other* legs, exactly as its twin does.
    assert ("i0b", "i2") in found


def test_children_of_different_parents_are_never_parallel():
    # Parallel is a within-one-parent relation; the cross-pattern cases are what
    # Inverse Parallel and Adjacent are for.
    first = impulse("i", start=0)
    second = impulse("j", start=1000)
    patterns = settle(fully_marked(first, "i", legs=[0, 2])
                      + fully_marked(second, "j", legs=[0, 2]))

    assert summary(parallel_pairs(patterns)) == [
        ("i0", "i2", "1", "3", None),
        ("j0", "j2", "1", "3", None),
    ]


def test_a_flat_child_never_appears_in_a_parallel_pair():
    # No net direction, so it can neither agree nor disagree with a sibling's.
    parent = impulse()
    flat = leg_child("i0", parent, 0)
    flat["points"][-1]["price"] = flat["points"][0]["price"]
    patterns = settle([parent, flat] + [leg_child(f"i{k}", parent, k)
                                        for k in (2, 4)])

    assert summary(parallel_pairs(patterns)) == [("i2", "i4", "3", "5", None)]


# --------------------------------------------------- 6. inverse_parallel_pairs


def test_the_worked_impulse_into_zigzag_yields_exactly_two_inverse_pairs():
    # (4)->(5)->(A): reactionary against actionary, gap (5) actionary, both
    # bearish. (5)->(A)->(B): actionary against reactionary, gap (A) actionary,
    # both bullish. Nothing else in the structure clears all five rules.
    assert summary(inverse_parallel_pairs(impulse_into_zigzag())) == [
        ("i3", "z0", "4", "A", "i4"),
        ("i4", "z1", "5", "B", "z0"),
    ]


def test_three_four_five_of_one_impulse_is_not_an_inverse_pair():
    # Same parent, same role, and the gap (4) reacts -- it fails every rule that
    # separates Inverse Parallel from Parallel.
    found = {(pair["role_a"], pair["role_b"]) for pair
             in inverse_parallel_pairs(impulse_into_zigzag())}

    assert ("3", "5") not in found


def test_a_reactionary_gap_never_survives_into_an_inverse_pair():
    # The rule is checked, and in a well-formed marking it can only ever fire
    # together with the different-parent rule: every pattern type's first and
    # last waves are actionary, so a reactionary wave is interior to its parent
    # and both counts chained to it are its own siblings. The property is
    # asserted over the whole worked structure rather than forced with a
    # hand-built triple that no real marking could produce.
    for pair in inverse_parallel_pairs(impulse_into_zigzag()):
        assert pattern_role(impulse_into_zigzag(), pair["gap"]) == "actionary"


def test_one_bar_of_daylight_removes_the_inverse_pairs():
    # Adjacency is a shared pivot, nothing looser. The correction is drawn one
    # bar after the impulse ends instead of on it.
    assert inverse_parallel_pairs(impulse_into_zigzag(offset=1)) == []


def test_two_waves_running_opposite_ways_are_not_an_inverse_pair():
    # An upward correction: wave A now rises, so (4) and (A) disagree and so do
    # (5) and (B). Everything else about the structure is unchanged.
    upward = impulse_into_zigzag(prices=(60.0, 55.0, 70.0))

    assert inverse_parallel_pairs(upward) == []


def test_an_unmarked_gap_leaves_nothing_to_pair_across():
    # Wave (5) is never drawn, so nothing chains (4) to (A): the gap of an
    # Inverse Parallel pair is a marked pattern whose role is asked about, not
    # a leg of somebody's parent.
    parent = impulse("i")
    correction = correction_after("z", parent)
    patterns = settle(fully_marked(parent, "i", legs=[0, 1, 2, 3])
                      + fully_marked(correction, "z"))

    assert inverse_parallel_pairs(patterns) == []


# ----------------------------------------------------------- 7. adjacent_pairs


def test_wave_five_against_the_a_that_follows_it_is_an_adjacent_pair():
    # His canonical case: the impulse ends and the correction starts on the same
    # pivot, both waves act, and they run opposite ways.
    assert summary(adjacent_pairs(impulse_into_zigzag())) == [
        ("i4", "z0", "5", "A", None),
    ]


def test_a_against_b_is_not_adjacent_because_b_reacts():
    found = {(pair["role_a"], pair["role_b"]) for pair
             in adjacent_pairs(impulse_into_zigzag())}

    assert ("A", "B") not in found


def test_two_adjacent_actionary_waves_running_the_same_way_are_not_a_pair():
    # An upward correction makes (A) bullish like the (5) before it, and his
    # rule is "opposite directions".
    assert adjacent_pairs(impulse_into_zigzag(prices=(60.0, 55.0, 70.0))) == []


def test_siblings_of_one_parent_are_never_adjacent_pairs():
    # (1) and (2) meet on a pivot, but (2) reacts; (1) and (3) share a role and
    # never touch. One parent yields no adjacent pair at all.
    patterns = settle(fully_marked(impulse(), "i"))

    assert adjacent_pairs(patterns) == []


def test_one_bar_of_daylight_removes_the_adjacent_pair():
    assert adjacent_pairs(impulse_into_zigzag(offset=1)) == []


# ------------------------------------------------------------ 8. the rename


def measured(**overrides):
    """A fully measured leg entry under the pre-rename field names."""
    entry = {"Origin CMB": 1.0, "Origin RSI": 2.0,
             "CMB": 3.0, "RSI": 4.0,
             "Terminating CMB": 5.0, "Terminating RSI": 6.0,
             "timeframe": "1D"}
    entry.update(overrides)
    return entry


def test_the_old_two_keys_become_the_peak_keys():
    pattern = zigzag()
    pattern["leg_values"] = {"1": measured()}

    migrated = migrate_leg_value_fields(pattern)

    assert migrated["leg_values"]["1"] == {
        "Origin CMB": 1.0, "Origin RSI": 2.0,
        "Peak CMB": 3.0, "Peak RSI": 4.0,
        "Terminating CMB": 5.0, "Terminating RSI": 6.0,
        "timeframe": "1D",
    }


def test_the_prefixed_fields_and_the_timeframe_pass_through_untouched():
    # "Origin CMB" contains "CMB" and is not it. A substring rule would rewrite
    # two good fields to save one.
    pattern = zigzag()
    pattern["leg_values"] = {"0": {"Origin CMB": 1.0, "Terminating RSI": 6.0,
                                   "timeframe": "15m"}}

    migrated = migrate_leg_value_fields(pattern)

    assert migrated is pattern            # nothing to rename, nothing rebuilt
    assert migrated["leg_values"]["0"]["Origin CMB"] == 1.0
    assert migrated["leg_values"]["0"]["timeframe"] == "15m"


def test_a_pattern_already_on_the_new_names_is_left_alone():
    pattern = zigzag()
    pattern["leg_values"] = {"2": {"Peak CMB": 3.0, "Peak RSI": 4.0,
                                   "timeframe": "1D"}}

    assert migrate_leg_value_fields(pattern) is pattern


def test_a_stale_old_key_loses_to_the_new_one_beside_it():
    # Two keys for one field means two versions of the app wrote the file. The
    # new name holds what he last saw and edited.
    pattern = zigzag()
    pattern["leg_values"] = {"0": {"CMB": 3.0, "Peak CMB": 99.0,
                                   "RSI": 4.0, "Peak RSI": 88.0,
                                   "timeframe": "1D"}}

    entry = migrate_leg_value_fields(pattern)["leg_values"]["0"]

    assert entry == {"Peak CMB": 99.0, "Peak RSI": 88.0, "timeframe": "1D"}


def test_migration_never_touches_the_pattern_it_was_given():
    pattern = zigzag()
    pattern["leg_values"] = {"1": measured()}

    migrate_leg_value_fields(pattern)

    assert pattern["leg_values"]["1"]["CMB"] == 3.0


def test_migration_survives_a_pattern_with_no_values_at_all():
    assert migrate_leg_value_fields(zigzag()) is not None
    assert migrate_leg_value_fields(None) is None
    assert migrate_leg_value_fields({"leg_values": "rubbish"}) == {"leg_values": "rubbish"}


def test_a_pre_rename_file_loads_with_the_new_names_and_keeps_its_values(tmp_path):
    path = str(tmp_path / "saved_wave_markings.json")
    stale = zigzag("z1")
    stale["leg_values"] = {"1": measured()}
    # Not vacuous: the old keys are exactly what the current table rejects, so
    # without the migration the pattern would be dropped on load rather than
    # loaded with stale names.
    assert not is_valid_pattern(stale)

    save_wave_documents({"gold.csv": {"schema": 2, "base_timeframe": "15m",
                                      "patterns": [stale]}}, path)
    loaded = load_wave_documents(path)["gold.csv"]["patterns"]

    assert len(loaded) == 1
    entry = loaded[0]["leg_values"]["1"]
    assert entry["Peak CMB"] == 3.0 and entry["Peak RSI"] == 4.0
    assert entry["Origin CMB"] == 1.0 and entry["Terminating RSI"] == 6.0
    assert entry["timeframe"] == "1D"


def test_a_migrated_leg_reads_as_fully_measured():
    # All six under their current names, so the wave keeps its white symbol and
    # stays eligible for analysis instead of quietly dropping out of it.
    stale = zigzag("z1")
    stale["leg_values"] = {"1": measured()}

    assert leg_is_complete(migrate_leg_value_fields(stale), 1)


def test_a_load_then_save_round_trip_persists_only_the_new_names(tmp_path):
    path = str(tmp_path / "saved_wave_markings.json")
    stale = zigzag("z1")
    stale["leg_values"] = {"1": measured()}
    save_wave_documents({"gold.csv": {"schema": 2, "base_timeframe": "15m",
                                      "patterns": [stale]}}, path)

    documents = load_wave_documents(path)
    save_wave_documents(documents, path)

    with open(path) as handle:
        written = handle.read()
    assert '"Peak CMB"' in written and '"Peak RSI"' in written
    assert '"CMB":' not in written and '"RSI":' not in written
    assert json.loads(written)["gold.csv"]["patterns"][0]["leg_values"]["1"][
        "Peak CMB"] == 3.0


# ---------------------------------------------------------- 9. determinism


@pytest.mark.parametrize("discover",
                         [parallel_pairs, inverse_parallel_pairs, adjacent_pairs])
def test_discovery_gives_the_same_answer_in_the_same_order_every_time(discover):
    # A percentage he intends to trade on cannot depend on dict iteration luck.
    patterns = impulse_into_zigzag()

    assert summary(discover(patterns)) == summary(discover(patterns))
    assert summary(discover(list(patterns))) == summary(discover(patterns))


@pytest.mark.parametrize("discover",
                         [parallel_pairs, inverse_parallel_pairs, adjacent_pairs])
def test_discovery_survives_rubbish_input(discover):
    assert discover([]) == []
    assert discover("not a list") == []


# ------------------------------------------- 10. the pair study on his example
#
# The two analyses whose discovery only exists on a chained structure, counted
# for real: the worked Impulse-into-Zigzag above, with numbers planted on the
# waves each pair reaches.


def measure(patterns, by_id):
    """Plant a Peak CMB on one leg of the named patterns, in place.

    All six fields, because a wave is only analysable when every one of them is
    a number; only Peak CMB varies, so the arithmetic in each test is the two
    numbers it names and nothing else.
    """
    index = {pattern["id"]: pattern for pattern in patterns}
    for pattern_id, (leg, cmb) in by_id.items():
        entry = {field: 1.0 for field in LEG_VALUE_FIELDS}
        entry["Peak CMB"] = cmb
        entry["timeframe"] = "1D"
        index[pattern_id]["leg_values"] = {str(leg): entry}
    return patterns


def pair_spec(analysis, **overrides):
    """Wave A of one Zigzag against wave A of the other, pooled over every pair.

    Every wave of both counts is marked as a Zigzag in the fixture, so "wave A"
    names leg 0 of whichever child a pair reaches.
    """
    study = {"analysis": analysis,
             "type_a": "Zigzag", "variation_a": None,
             "wave_a": "A", "parent_wave_a": None,
             "type_b": "Zigzag", "variation_b": None,
             "wave_b": "A", "parent_wave_b": None,
             "relative_degree": 0,
             "field_a": "Peak CMB", "field_b": "Peak CMB",
             "operator": ">"}
    study.update(overrides)
    return study


def test_the_parent_wave_labels_are_exactly_the_two_role_sets():
    # The forms offer these as "Parent wave (1)", and a label no pattern type
    # names could only ever filter every pair away in silence.
    assert set(PARENT_WAVE_LABELS) == ACTIONARY_LABELS | REACTIONARY_LABELS
    assert len(set(PARENT_WAVE_LABELS)) == len(PARENT_WAVE_LABELS)


def test_the_two_inverse_parallel_pairs_are_counted():
    # (4)v(A) and (5)v(B), as discovery finds them: 5 > 2 and 1 > 4.
    patterns = measure(impulse_into_zigzag(),
                       {"i3": (0, 5.0), "z0": (0, 2.0),
                        "i4": (0, 1.0), "z1": (0, 4.0)})

    result = run_pair_study(patterns, pair_spec("inverse_parallel"))

    assert result["pairs"] == 2
    assert result["samples"] == 2
    assert result["true"] == 1
    assert result["false"] == 1
    assert result["pct_true"] == 50.0
    assert result["skipped"] == {}


def test_the_inverse_parallel_parent_wave_filters_pick_one_of_the_two():
    patterns = measure(impulse_into_zigzag(),
                       {"i3": (0, 5.0), "z0": (0, 2.0),
                        "i4": (0, 1.0), "z1": (0, 4.0)})

    result = run_pair_study(patterns, pair_spec("inverse_parallel",
                                                parent_wave_a="5",
                                                parent_wave_b="B"))

    assert result["pairs"] == 2
    assert result["samples"] == 1
    assert result["true"] == 0                       # 1 > 4 is false


def test_the_single_adjacent_pair_is_counted():
    # (5) against the (A) that follows it: the only pair the rules leave.
    patterns = measure(impulse_into_zigzag(), {"i4": (0, 8.0), "z0": (0, 3.0)})

    result = run_pair_study(patterns, pair_spec("adjacent"))

    assert result["pairs"] == 1
    assert result["samples"] == 1
    assert result["true"] == 1                       # 8 > 3
    assert result["skipped"] == {}


def test_an_unmeasured_side_of_an_adjacent_pair_is_skipped():
    patterns = measure(impulse_into_zigzag(), {"i4": (0, 8.0)})

    result = run_pair_study(patterns, pair_spec("adjacent"))

    assert result["pairs"] == 1
    assert result["samples"] == 0
    assert result["skipped"] == {SKIP_NOT_MEASURED: 1}

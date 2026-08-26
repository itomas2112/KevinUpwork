"""
Tests for the analysis study: one comparison counted across many marked patterns.

Every pattern here is built inline with hand-set ``leg_values``, so each expected
percentage is obvious by construction -- five impulses, three of which satisfy
the comparison, is sixty per cent and nothing in the fixture hides that.

Leg indexing is the standing risk, as everywhere in this feature: leg ``k`` runs
``points[k] -> points[k + 1]`` and is the wave ``point_labels(...)[k + 1]`` names.
So wave "2" of an impulse is leg 1 and wave "4" is leg 3, and the fixtures below
are written in terms of the wave labels the study is actually asked about.
"""

import time

import pytest

from config.wave_analysis import LEG_VALUE_FIELDS, analysable_legs, analysable_waves, settle
from strategies.wave_study import (ALL_WAVES, NOTE_ONE_PATTERN, SKIP_AMBIGUOUS,
                                   SKIP_LABEL_ABSENT, SKIP_NOT_MEASURED,
                                   SKIP_NO_COUNTERPART, SKIP_NO_DEGREE,
                                   SKIP_PAIR_FILTERED, SKIP_RED, field_family,
                                   pair_context, run_grid, run_pair_study,
                                   run_study, wave_labels)

BASE = 1719878400
DAY = 86400


# --------------------------------------------------------------- the fixtures


def values(cmb):
    """All six fields on one leg, with Peak CMB carrying the number under test.

    All six, because a wave is only analysable when every one of them is a
    number; only Peak CMB varies, because a study reads one field at a time and
    a second moving number would just make the arithmetic harder to check.
    """
    entry = {field: 1.0 for field in LEG_VALUE_FIELDS}
    entry["Peak CMB"] = cmb
    entry["timeframe"] = "1D"
    return entry


def measured(**fields):
    """All six fields at 1.0, with the named ones planted at what they say.

    For the studies that read a different field on each side: two numbers on one
    wave, so which field was compared is visible in the answer rather than
    inferred from it.
    """
    entry = {field: 1.0 for field in LEG_VALUE_FIELDS}
    entry.update(fields)
    entry["timeframe"] = "1D"
    return entry


def leg_values(by_leg):
    """``{3: 10.0}`` -> a stored ``leg_values`` measuring leg 3 with CMB 10.

    A dict in place of the number plants those fields instead, so one wave can
    carry a Peak CMB and an Origin CMB that disagree.
    """
    return {str(leg): (measured(**entry) if isinstance(entry, dict) else values(entry))
            for leg, entry in by_leg.items()}


def spaced(start, count, step=900):
    return [start + step * index for index in range(count)]


def marking(pattern_id, pattern_type, variation, times, by_leg=None,
            degree="Subminuette", kinds=None):
    pattern = {
        "id": pattern_id,
        "pattern_type": pattern_type,
        "variation": variation,
        "degree": degree,
        "color": "yellow",
        "points": [{"time": time, "price": 2360.0 + index,
                    "kind": kinds[index] if kinds else
                            ("low" if index % 2 == 0 else "high")}
                   for index, time in enumerate(times)],
    }
    if by_leg:
        pattern["leg_values"] = leg_values(by_leg)
    return pattern


def impulse(pattern_id, start, by_leg=None, variation="Impulse",
            degree="Subminuette", step=900):
    """A five-wave count. Wave 2 is leg 1, wave 4 is leg 3."""
    return marking(pattern_id, "Impulse", variation, spaced(start, 6, step),
                   by_leg, degree)


def leg_child(pattern_id, parent, leg, by_leg=None, pattern_type="Impulse",
              variation="Impulse", count=6):
    """A count spanning exactly leg ``leg`` of ``parent``, endpoint for endpoint.

    Time and kind both, on the first point and the last -- the only thing this
    codebase calls a child, and what every pair the analyses discover hangs off.
    The interior points are spread evenly through the leg, so a child is always
    strictly inside its parent and never collides with it.
    """
    start, end = parent["points"][leg], parent["points"][leg + 1]
    step = (end["time"] - start["time"]) // (count - 1)
    assert step > 0, "leg too narrow for a child of this many points"
    times = ([start["time"] + step * index for index in range(count - 1)]
             + [end["time"]])
    other = "high" if start["kind"] == "low" else "low"
    kinds = ([start["kind"]]
             + [other if index % 2 else start["kind"]
                for index in range(1, count - 1)]
             + [end["kind"]])
    return marking(pattern_id, pattern_type, variation, times, by_leg, kinds=kinds)


def wave_4_vs_2(wave_4, wave_2):
    return {3: wave_4, 1: wave_2}


def spec(**overrides):
    """The client's headline study, with anything he might change overridden."""
    study = {"pattern_type": "Impulse", "variation": None,
             "wave_a": "4", "offset_a": 0,
             "wave_b": "2", "offset_b": 0,
             "field": "Peak CMB", "operator": "<"}
    study.update(overrides)
    return study


def five_impulses():
    """Three where wave 4's CMB is below wave 2's, two where it is above.

    Started a day apart so no two spans touch: same-degree overlap is what turns
    a marking red, and a fixture that went red by accident would report zero
    samples for the wrong reason.
    """
    lower = [wave_4_vs_2(1.0, 5.0), wave_4_vs_2(2.0, 8.0), wave_4_vs_2(-3.0, 0.5)]
    higher = [wave_4_vs_2(9.0, 2.0), wave_4_vs_2(4.0, 4.0 - 1.0)]
    return [impulse(f"i{index}", BASE + DAY * index, by_leg)
            for index, by_leg in enumerate(lower + higher)]


# --------------------------------------------------------------- 1. wave_labels


def test_an_impulse_offers_waves_one_to_five():
    assert wave_labels("Impulse") == ["1", "2", "3", "4", "5"]


def test_a_named_variation_offers_just_its_own_labels():
    assert wave_labels("Zigzag", "Zigzag") == ["A", "B", "C"]
    assert wave_labels("Zigzag", "Double Zigzag") == ["W", "X", "Y"]
    assert wave_labels("Combination", "Triple Three") == ["W", "X", "Y", "Z"]


def test_a_type_with_no_variation_unions_its_variations_without_duplicates():
    # Double Three is W X Y and Triple Three is W X Y X Z; the second X is the
    # same wave name, so it appears once and Z lands after it.
    assert wave_labels("Combination") == ["W", "X", "Y", "Z"]


def test_a_type_whose_variations_disagree_offers_all_of_them():
    # Zigzag holds A B C alongside W X Y and W X Y X Z, in first-seen order.
    assert wave_labels("Zigzag") == ["A", "B", "C", "W", "X", "Y", "Z"]


def test_an_unknown_type_offers_nothing():
    assert wave_labels("Elephant") == []


# ------------------------------------------------------- 2/3. the headline case


def test_three_of_five_impulses_have_a_lower_wave_four_cmb():
    result = run_study(five_impulses(), spec(operator="<"))

    assert result["samples"] == 5
    assert result["true"] == 3
    assert result["false"] == 2
    assert result["pct_true"] == 60.0
    assert result["pct_false"] == 40.0
    assert result["ties"] == 0


def test_the_other_operator_gives_the_complement():
    result = run_study(five_impulses(), spec(operator=">"))

    assert result["samples"] == 5
    assert result["true"] == 2
    assert result["false"] == 3
    assert result["pct_true"] == 40.0
    assert result["pct_false"] == 60.0


# ------------------------------------------------------------------- 4. ties


def test_a_tie_counts_false_and_is_reported_separately():
    # He has not defined "=" -- "perfectly equal values is near impossible" --
    # so it resolves as false, and the tie count is the evidence he asked for
    # before deciding whether that was the right call.
    patterns = [impulse("tie", BASE, wave_4_vs_2(7.5, 7.5))]

    for operator in ("<", ">"):
        result = run_study(patterns, spec(operator=operator))

        assert result["samples"] == 1
        assert result["true"] == 0
        assert result["false"] == 1
        assert result["ties"] == 1
        assert result["pct_false"] == 100.0


# ------------------------------------------------------------ 5. the variation


def test_no_variation_pools_every_variation_of_the_type():
    patterns = [
        impulse("a", BASE, wave_4_vs_2(1.0, 5.0)),
        impulse("b", BASE + DAY, wave_4_vs_2(1.0, 5.0)),
        impulse("c", BASE + 2 * DAY, wave_4_vs_2(1.0, 5.0),
                variation="Extended Impulse"),
    ]

    assert run_study(patterns, spec())["samples"] == 3


def test_a_named_variation_counts_only_its_own_patterns():
    patterns = [
        impulse("a", BASE, wave_4_vs_2(1.0, 5.0)),
        impulse("b", BASE + DAY, wave_4_vs_2(1.0, 5.0)),
        impulse("c", BASE + 2 * DAY, wave_4_vs_2(1.0, 5.0),
                variation="Extended Impulse"),
    ]

    named = run_study(patterns, spec(variation="Impulse"))
    assert named["samples"] == 2
    assert named["true"] == 2

    extended = run_study(patterns, spec(variation="Extended Impulse"))
    assert extended["samples"] == 1


def test_a_pattern_of_another_type_is_not_a_candidate_at_all():
    # Not skipped, not counted: it was never in the study's population, so
    # reporting it would make the skip line meaningless.
    patterns = [
        impulse("a", BASE, wave_4_vs_2(1.0, 5.0)),
        marking("z", "Zigzag", "Zigzag", spaced(BASE + DAY, 4), {0: 1.0, 2: 2.0}),
    ]

    result = run_study(patterns, spec())
    assert result["samples"] == 1
    assert result["skipped"] == {}


# ------------------------------------------------------------- 6. eligibility


def test_a_wave_missing_one_of_the_six_fields_is_skipped_and_counted():
    patterns = [impulse("half", BASE, wave_4_vs_2(1.0, 5.0))]
    patterns[0]["leg_values"]["1"].pop("Terminating RSI")

    result = run_study(patterns, spec())

    assert result["samples"] == 0
    assert result["skipped"][SKIP_NOT_MEASURED] == 1


def test_a_wave_never_measured_at_all_is_skipped_and_counted():
    patterns = [impulse("bare", BASE, {3: 1.0})]     # wave 4 only, no wave 2

    result = run_study(patterns, spec())

    assert result["samples"] == 0
    assert result["skipped"][SKIP_NOT_MEASURED] == 1


def test_both_sides_unmeasured_counts_two_waves():
    result = run_study([impulse("bare", BASE)], spec())

    assert result["skipped"][SKIP_NOT_MEASURED] == 2


def overlapping_pair():
    """Two impulses at one degree whose interiors overlap: settle paints both red.

    Not a parent and a child -- the second one's endpoints match no single leg
    of the first -- so nothing derives a junior degree for it and the collision
    stands, which is exactly the state the client's rule calls invalid.
    """
    return [impulse("clash_a", BASE, wave_4_vs_2(1.0, 5.0)),
            impulse("clash_b", BASE + 1800, wave_4_vs_2(1.0, 5.0))]


def test_every_leg_of_a_red_pattern_is_skipped_even_when_complete():
    patterns = overlapping_pair() + [impulse("clean", BASE + DAY,
                                             wave_4_vs_2(1.0, 5.0))]

    result = run_study(patterns, spec())

    assert result["samples"] == 1                    # the clean one only
    assert result["skipped"][SKIP_RED] == 2
    # ...and the fixture really is fully measured, so the emptiness above is the
    # colour rule rather than a broken pattern.
    assert analysable_legs(patterns[0]) == [1, 3]


def test_resolving_the_conflict_brings_the_waves_back():
    patterns = overlapping_pair()
    # Move the second count clear of the first: nothing else changes.
    patterns[1]["points"] = [dict(point, time=point["time"] + DAY)
                             for point in patterns[1]["points"]]

    result = run_study(patterns, spec())

    assert result["samples"] == 2
    assert SKIP_RED not in result["skipped"]


# ------------------------------------------------------------ 7/8. cross-degree


def nested_pair():
    """A parent impulse and a child impulse spanning the parent's wave 4.

    Wave 4 is leg 3, from ``points[3]`` to ``points[4]``, so the child's first
    and last points are those two pivots exactly -- time and kind both -- which
    is the only thing this codebase calls a child.

    The two waves the study should reach carry 10.0 (parent's wave 4) and 4.0
    (child's wave 2). The other two are decoys: any other pairing the engine
    might fall into compares against 99.0 or 1.0 and gives the opposite answer.
    """
    parent = impulse("parent", BASE, {3: 10.0, 1: 99.0})
    child_times = [BASE + 2700, BASE + 2850, BASE + 3000,
                   BASE + 3150, BASE + 3300, BASE + 3600]
    child = marking("child", "Impulse", "Impulse", child_times, {1: 4.0, 3: 1.0},
                    kinds=["high", "low", "high", "low", "high", "low"])
    return [parent, child]


def test_a_plus_one_study_pairs_a_parents_wave_with_its_childs():
    result = run_study(nested_pair(),
                       spec(wave_a="4", offset_a=1, wave_b="2", operator=">"))

    assert result["samples"] == 1
    # 10.0 > 4.0. Same-pattern pairing would have compared 10.0 with 99.0, and
    # the mirrored roles 1.0 with 99.0 -- both false.
    assert result["true"] == 1
    assert result["pct_true"] == 100.0


def test_the_mirror_offset_swaps_which_side_is_the_child():
    # offset_b = +1 says wave B's pattern is the parent, so the same two waves
    # meet with their roles exchanged: 4.0 (child's wave 2) against 10.0.
    result = run_study(nested_pair(),
                       spec(wave_a="2", wave_b="4", offset_b=1, operator="<"))

    assert result["samples"] == 1
    assert result["true"] == 1


def test_a_parent_with_no_parent_of_its_own_is_counted_as_uncoupled():
    result = run_study(nested_pair(),
                       spec(wave_a="4", offset_a=1, wave_b="2", operator=">"))

    # The parent is walked too, and has nothing above it.
    assert result["skipped"][SKIP_NO_COUNTERPART] == 1


def test_unrelated_patterns_at_adjacent_degrees_never_pair():
    # The noise guard. Two counts a degree apart that are not a parent and a
    # child are not comparable structure, and pairing them would put noise into
    # a percentage the client intends to trade on.
    patterns = [impulse("senior", BASE, {3: 10.0, 1: 99.0}),
                impulse("junior", BASE + DAY, {1: 4.0, 3: 1.0}, degree="Micro")]

    result = run_study(patterns, spec(wave_a="4", offset_a=1, wave_b="2",
                                      operator=">"))

    assert result["samples"] == 0
    assert result["skipped"][SKIP_NO_COUNTERPART] == 2


def test_deleting_the_child_leaves_a_cross_degree_study_with_nothing():
    patterns = nested_pair()
    study = spec(wave_a="4", offset_a=1, wave_b="2", operator=">")

    assert run_study(patterns, study)["samples"] == 1
    assert run_study(patterns[:1], study)["samples"] == 0


def test_a_counterpart_of_another_type_does_not_pair():
    # Both sides must match the type filter, so a cross-degree study always
    # compares like with like.
    parent = marking("parent", "Zigzag", "Zigzag",
                     [BASE, BASE + 900, BASE + 1800, BASE + 2700], {0: 1.0})
    child_times = [BASE + 1800, BASE + 1950, BASE + 2100,
                   BASE + 2250, BASE + 2400, BASE + 2700]
    child = marking("child", "Impulse", "Impulse", child_times, {1: 4.0, 3: 1.0},
                    kinds=["low", "high", "low", "high", "low", "high"])

    result = run_study([parent, child],
                       spec(wave_a="4", offset_a=1, wave_b="2", operator=">"))

    assert result["samples"] == 0
    assert result["skipped"][SKIP_NO_COUNTERPART] == 1


# ------------------------------------------------------------ 9. a bad spec


def test_moving_both_sides_a_degree_is_rejected():
    with pytest.raises(ValueError):
        run_study(five_impulses(), spec(offset_a=1, offset_b=-1))


@pytest.mark.parametrize("bad", [
    {"pattern_type": "Elephant"},
    {"variation": "Zigzag"},                # not a variation of Impulse
    {"field": "ADX"},
    {"field_a": "Peak CMB", "field_b": "Peak RSI"},     # not one family
    {"field_a": "Peak CMB", "field": None},             # half a value pair
    {"operator": ">="},
    {"wave_a": None},
    {"wave_b": ""},
    {"offset_a": 2},
    {"offset_a": "+1"},
    {"offset_a": True},
])
def test_a_malformed_spec_is_rejected_rather_than_guessed_at(bad):
    with pytest.raises(ValueError):
        run_study(five_impulses(), spec(**bad))


# ------------------------------------------------------------ 10. ambiguity


def test_a_duplicated_wave_label_is_skipped_as_ambiguous():
    # A Triple Zigzag's sequence is W X Y X Z, so "X" names two legs. Taking the
    # first silently would make the study unreproducible.
    double = marking("dz", "Zigzag", "Double Zigzag", spaced(BASE, 4),
                     {1: 10.0, 2: 5.0})
    triple = marking("tz", "Zigzag", "Triple Zigzag", spaced(BASE + DAY, 6),
                     {1: 1.0, 2: 2.0, 3: 3.0})

    result = run_study([double, triple],
                       spec(pattern_type="Zigzag", wave_a="X", wave_b="Y",
                            operator=">"))

    # The unambiguous sibling still counts: 10.0 > 5.0.
    assert result["samples"] == 1
    assert result["true"] == 1
    assert result["skipped"][SKIP_AMBIGUOUS] == 1


def test_a_pattern_that_has_no_such_wave_is_counted_separately():
    # A plain Zigzag is A B C: "X" is not a wave it has at all, which is a
    # different thing from having two of them.
    plain = marking("z", "Zigzag", "Zigzag", spaced(BASE, 4), {0: 1.0, 1: 2.0})
    double = marking("dz", "Zigzag", "Double Zigzag", spaced(BASE + DAY, 4),
                     {1: 10.0, 2: 5.0})

    result = run_study([plain, double],
                       spec(pattern_type="Zigzag", wave_a="X", wave_b="Y",
                            operator=">"))

    assert result["samples"] == 1
    assert result["skipped"][SKIP_LABEL_ABSENT] == 2      # one per side
    assert SKIP_AMBIGUOUS not in result["skipped"]


# ---------------------------------------------------------- 11. no samples


def test_no_samples_gives_zero_percentages_rather_than_a_division_error():
    result = run_study([], spec())

    assert result["samples"] == 0
    assert result["pct_true"] == 0.0
    assert result["pct_false"] == 0.0
    assert result["true"] == result["false"] == result["ties"] == 0
    assert result["skipped"] == {}


def test_a_list_of_unmeasured_patterns_gives_zero_percentages():
    result = run_study([impulse("a", BASE), impulse("b", BASE + DAY)], spec())

    assert result["samples"] == 0
    assert result["pct_true"] == 0.0


# ------------------------------------------------- 12. analysable_waves settles


def test_analysable_waves_settles_before_deciding_what_is_eligible():
    patterns = overlapping_pair() + [impulse("clean", BASE + DAY,
                                             wave_4_vs_2(1.0, 5.0))]

    # Not vacuous: handed in, every one of them is yellow and fully measured.
    assert all(analysable_legs(pattern) == [1, 3] for pattern in patterns)

    rows = analysable_waves(patterns)

    assert {row["pattern_id"] for row in rows} == {"clean"}
    assert [row["label"] for row in rows] == ["2", "4"]


def test_an_analysable_wave_carries_its_identity_and_its_values():
    rows = analysable_waves([impulse("solo", BASE, {3: 10.0})])

    assert rows == [{
        "pattern_id": "solo",
        "pattern_type": "Impulse",
        "variation": "Impulse",
        "degree": "Subminuette",
        "leg": 3,
        "label": "4",
        # The six fields alone: the timeframe travels beside the numbers, not
        # inside them, so a study cannot mistake it for a value.
        "values": {field: values(10.0)[field] for field in LEG_VALUE_FIELDS},
        "timeframe": "1D",
    }]


def test_analysable_waves_reports_a_childs_reconciled_degree():
    # settle rewrites the child's degree from its parent's, so a row reports the
    # degree the marking actually has rather than the one it was drawn with.
    rows = analysable_waves(nested_pair())
    by_id = {row["pattern_id"]: row["degree"] for row in rows}

    assert by_id["parent"] == "Subminuette"
    assert by_id["child"] == "Micro"


def test_analysable_waves_of_an_empty_list_is_empty():
    assert analysable_waves([]) == []
    assert analysable_waves("not a list") == []


# ------------------------------------------------------------ 13. determinism


def test_the_same_input_twice_gives_an_identical_result():
    patterns = (five_impulses() + overlapping_pair() + nested_pair()
                + [impulse("bare", BASE + 40 * DAY)])
    study = spec(operator="<")

    assert run_study(patterns, study) == run_study(patterns, study)
    # Including the skip line, whose ordering is fixed rather than incidental.
    assert (list(run_study(patterns, study)["skipped"])
            == list(run_study(patterns, study)["skipped"]))


def test_a_study_does_not_mutate_the_markings_it_counts():
    patterns = five_impulses()
    before = [dict(pattern) for pattern in patterns]

    run_study(patterns, spec())

    assert patterns == before


# ------------------------------------------------------------ 14. value pairs


def test_the_bare_field_key_still_means_both_sides():
    # Every study written before the two boxes existed keeps asking exactly what
    # it asked -- which is what the whole section above this one relies on.
    patterns = [impulse("a", BASE, wave_4_vs_2(1.0, 5.0))]

    assert run_study(patterns, spec())["true"] == 1


def test_the_two_sides_may_read_different_fields():
    # Wave 4's Peak CMB against wave 2's Origin CMB: two fields, planted so that
    # reading either one on both sides would give a different answer.
    patterns = [
        impulse("a", BASE, {3: {"Peak CMB": 9.0}, 1: {"Origin CMB": 4.0}}),
        impulse("b", BASE + DAY, {3: {"Peak CMB": 2.0}, 1: {"Origin CMB": 4.0}}),
    ]

    result = run_study(patterns, spec(field_a="Peak CMB", field_b="Origin CMB",
                                      operator=">"))

    assert result["samples"] == 2
    assert result["true"] == 1                       # 9 > 4; 2 > 4 is not
    assert result["false"] == 1


def test_two_rsi_fields_are_a_family_and_compare():
    patterns = [impulse("a", BASE, {3: {"Peak RSI": 71.0},
                                    1: {"Origin RSI": 40.0}})]

    result = run_study(patterns, spec(field_a="Peak RSI", field_b="Origin RSI",
                                      operator=">"))

    assert result["samples"] == 1 and result["true"] == 1


def test_a_cmb_is_never_compared_with_an_rsi():
    # His own rule, and the one refusal that is not about a typo: an RSI of 62
    # against a CMB of 0.4 has no meaning to report a percentage of.
    with pytest.raises(ValueError):
        run_study(five_impulses(), spec(field_a="Peak CMB", field_b="Origin RSI"))


# ---------------------------------------------------------- 15. the = operator


def test_exact_equality_counts_true_under_equals():
    patterns = [impulse("tie", BASE, wave_4_vs_2(7.5, 7.5))]

    equal = run_study(patterns, spec(operator="="))

    assert equal["samples"] == 1
    assert equal["true"] == 1
    assert equal["false"] == 0
    assert equal["pct_true"] == 100.0
    # Reported for every operator; under "=" it simply equals the true count.
    assert equal["ties"] == 1


def test_the_same_tie_is_false_under_the_strict_operators():
    patterns = [impulse("tie", BASE, wave_4_vs_2(7.5, 7.5))]

    strict = run_study(patterns, spec(operator=">"))

    assert strict["true"] == 0 and strict["false"] == 1 and strict["ties"] == 1


def test_unequal_values_are_false_under_equals():
    result = run_study(five_impulses(), spec(operator="="))

    assert result["samples"] == 5
    assert result["true"] == 0
    assert result["ties"] == 0


# ------------------------------------------------- 16. per-side pattern filters


def impulse_with_zigzag_child():
    """An Impulse whose wave 4 is marked out as a Zigzag in its own right.

    The study Phase 19 flagged as inexpressible: the two sides of a cross-degree
    comparison are different pattern types, so one filter for both cannot name
    it. Wave 4 is leg 3, so the child's endpoints are ``points[3]`` and
    ``points[4]`` exactly.
    """
    parent = impulse("parent", BASE, {3: 10.0}, step=9000)
    child = leg_child("child", parent, 3, {0: 4.0}, "Zigzag", "Zigzag", count=4)
    return [parent, child]


def test_each_side_may_name_its_own_pattern_type_across_a_degree():
    result = run_study(impulse_with_zigzag_child(),
                       spec(pattern_type="Impulse", pattern_type_b="Zigzag",
                            wave_a="4", wave_b="A", offset_b=-1, operator=">"))

    assert result["samples"] == 1
    assert result["true"] == 1                       # 10.0 > 4.0
    assert result["skipped"] == {}


def test_one_filter_for_both_sides_still_finds_nothing_there():
    # Not vacuous: the same structure under the pre-phase spelling is exactly
    # the study that could not be asked, and it comes back empty.
    result = run_study(impulse_with_zigzag_child(),
                       spec(pattern_type="Impulse", wave_a="4", wave_b="2",
                            offset_b=-1, operator=">"))

    assert result["samples"] == 0


def test_differing_side_filters_at_the_same_degree_are_refused():
    # Both offsets 0 means the two waves come from one pattern, so a pattern
    # that is an Impulse on one side and a Zigzag on the other cannot exist.
    with pytest.raises(ValueError):
        run_study(five_impulses(), spec(pattern_type_b="Zigzag"))
    with pytest.raises(ValueError):
        run_study(five_impulses(), spec(variation_b="Extended Impulse"))


# ------------------------------------------------------- 17. the pair fixtures


def sibling_impulses(by_child):
    """A parent impulse with children marked on waves 1, 3 and 5.

    Three same-role siblings of one parent, which is every Parallel pair the
    client's own example names: (1)v(3), (1)v(5), (3)v(5). Each child carries
    its number on its *own* wave 1, so a study reading "wave 1" reads one
    planted value per child.
    """
    parent = impulse("parent", BASE, step=9000)
    return [parent] + [leg_child(name, parent, leg, {0: cmb})
                       for name, leg, cmb in zip(("c1", "c3", "c5"), (0, 2, 4),
                                                 by_child)]


def nested_siblings():
    """The siblings again, with a Zigzag marked inside wave 1 of (1) and of (5).

    The grandchildren carry 50.0 and 60.0 against their parents' 0.0 and 0.5, so
    a study that read the wrong degree gives a visibly different answer rather
    than the same one for a different reason.
    """
    patterns = sibling_impulses((0.0, 3.0, 0.5))
    by_id = {pattern["id"]: pattern for pattern in patterns}
    return patterns + [
        leg_child("g1", by_id["c1"], 0, {0: 50.0}, "Zigzag", "Zigzag", count=4),
        leg_child("g5", by_id["c5"], 0, {0: 60.0}, "Zigzag", "Zigzag", count=4),
    ]


def pair_spec(**overrides):
    """Wave 1 of one sibling against wave 1 of the next, pooled over every pair."""
    study = {"analysis": "parallel",
             "type_a": "Impulse", "variation_a": None,
             "wave_a": "1", "parent_wave_a": None,
             "type_b": "Impulse", "variation_b": None,
             "wave_b": "1", "parent_wave_b": None,
             "relative_degree": 0,
             "field_a": "Peak CMB", "field_b": "Peak CMB",
             "operator": ">"}
    study.update(overrides)
    return study


# ------------------------------------------------------ 18. the pair headline


def test_three_sibling_impulses_give_every_parallel_pair():
    result = run_pair_study(sibling_impulses((5.0, 3.0, 9.0)), pair_spec())

    assert result["pairs"] == 3
    assert result["samples"] == 3
    assert result["true"] == 1                # 5 > 3; 5 > 9 and 3 > 9 are false
    assert result["false"] == 2
    assert result["skipped"] == {}
    assert round(result["pct_true"], 1) == 33.3


def test_a_pair_study_may_read_a_different_field_on_each_side():
    parent = impulse("parent", BASE, step=9000)
    patterns = [parent] + [
        leg_child(name, parent, leg, {0: {"Peak CMB": peak, "Origin CMB": origin}})
        for name, leg, peak, origin in (("c1", 0, 9.0, 1.0), ("c3", 2, 2.0, 3.0),
                                        ("c5", 4, 5.0, 5.0))]

    result = run_pair_study(patterns, pair_spec(field_a="Peak CMB",
                                                field_b="Origin CMB"))

    assert result["samples"] == 3
    assert result["true"] == 2                # 9 > 3 and 9 > 5; 2 > 5 is not


def test_equal_values_count_true_under_equals_in_a_pair_study():
    result = run_pair_study(sibling_impulses((4.0, 4.0, 9.0)),
                            pair_spec(operator="="))

    assert result["samples"] == 3
    assert result["ties"] == 1
    assert result["true"] == 1
    assert result["false"] == 2


# -------------------------------------------------- 19. the parent-wave filters


def test_the_parent_wave_filters_narrow_to_one_pair():
    result = run_pair_study(sibling_impulses((5.0, 3.0, 9.0)),
                            pair_spec(parent_wave_a="1", parent_wave_b="3"))

    # The denominator is what discovery found, not what survived the filters:
    # one sample out of three pairs is a different claim from one out of one.
    assert result["pairs"] == 3
    assert result["samples"] == 1
    assert result["true"] == 1                       # 5 > 3
    assert result["skipped"] == {SKIP_PAIR_FILTERED: 2}


def test_a_parent_wave_nothing_occupies_filters_every_pair_away():
    result = run_pair_study(sibling_impulses((5.0, 3.0, 9.0)),
                            pair_spec(parent_wave_a="B"))

    assert result["pairs"] == 3
    assert result["samples"] == 0
    assert result["pct_true"] == 0.0
    assert result["skipped"] == {SKIP_PAIR_FILTERED: 3}


def test_a_side_pattern_filter_the_pair_member_fails_reads_as_filtered():
    # Nothing in the structure is a Zigzag, so no pair has a side to read from
    # -- and that is a filter question, not a missing-degree one.
    result = run_pair_study(sibling_impulses((5.0, 3.0, 9.0)),
                            pair_spec(type_a="Zigzag", wave_a="A"))

    assert result["samples"] == 0
    assert result["skipped"] == {SKIP_PAIR_FILTERED: 3}


# ----------------------------------------------------------- 20. relative degree


def test_a_junior_side_reads_its_wave_from_the_pair_members_children():
    result = run_pair_study(nested_siblings(),
                            pair_spec(type_a="Zigzag", wave_a="A",
                                      relative_degree=-1))

    assert result["pairs"] == 3
    # (1)v(3) and (1)v(5) read the Zigzag inside wave 1 of (1) -- 50.0, not the
    # 0.0 sitting on (1)'s own wave 1, which would make both of them false.
    assert result["samples"] == 2
    assert result["true"] == 2
    # (3) has no child at all, so that pair has no pattern at that degree.
    assert result["skipped"] == {SKIP_NO_DEGREE: 1}


def test_the_mirror_reads_wave_b_from_its_own_children():
    result = run_pair_study(nested_siblings(),
                            pair_spec(type_b="Zigzag", wave_b="A",
                                      relative_degree=1))

    # (1)v(5) and (3)v(5) read the Zigzag inside wave 1 of (5) -- 60.0. Reading
    # (5)'s own 0.5 instead would have made the second of them true.
    assert result["samples"] == 2
    assert result["true"] == 0
    assert result["skipped"] == {SKIP_NO_DEGREE: 1}


def test_a_junior_side_with_no_child_of_that_type_has_nothing_at_that_degree():
    result = run_pair_study(sibling_impulses((5.0, 3.0, 9.0)),
                            pair_spec(type_a="Zigzag", wave_a="A",
                                      relative_degree=-1))

    assert result["samples"] == 0
    assert result["skipped"] == {SKIP_NO_DEGREE: 3}


def test_a_shifted_side_filters_the_child_and_not_the_member_it_hangs_off():
    # Side A names Zigzag while every pair member is an Impulse: the filter
    # follows the wave, so the Zigzag it describes is the grandchild.
    result = run_pair_study(nested_siblings(),
                            pair_spec(type_a="Zigzag", variation_a="Zigzag",
                                      wave_a="A", relative_degree=-1))

    assert result["samples"] == 2


# --------------------------------------------------- 21. two types, two sides


def test_the_two_sides_of_a_pair_may_be_different_pattern_types():
    # His 2B example: a Flat marked as wave (2) and a Double Zigzag as wave (4)
    # of one parent -- wave C against wave Y.
    parent = impulse("parent", BASE, step=9000)
    flat = leg_child("flat", parent, 1, {2: 8.0}, "Flat", "Flat", count=4)
    double = leg_child("dz", parent, 3, {2: 2.0}, "Zigzag", "Double Zigzag",
                       count=4)

    result = run_pair_study([parent, flat, double],
                            pair_spec(type_a="Flat", wave_a="C",
                                      type_b="Zigzag", variation_b="Double Zigzag",
                                      wave_b="Y"))

    assert result["pairs"] == 1                      # (2) and (4) both react
    assert result["samples"] == 1
    assert result["true"] == 1                       # 8.0 > 2.0


# ------------------------------------------------- 22. eligibility inside pairs


def test_an_unmeasured_wave_inside_a_pair_skips_as_it_does_anywhere():
    patterns = sibling_impulses((5.0, 3.0, 9.0))
    del patterns[2]["leg_values"]                    # (3) loses its only reading

    result = run_pair_study(patterns, pair_spec())

    assert result["pairs"] == 3
    assert result["samples"] == 1                    # (1) against (5) only
    assert result["skipped"] == {SKIP_NOT_MEASURED: 2}


def test_a_pattern_that_has_no_such_wave_is_counted_inside_a_pair_too():
    patterns = sibling_impulses((5.0, 3.0, 9.0))

    result = run_pair_study(patterns, pair_spec(wave_a="A"))

    assert result["samples"] == 0
    assert result["skipped"] == {SKIP_LABEL_ABSENT: 3}


def test_a_red_marking_never_reaches_a_pair_at_all():
    # Two counts of one wave overlap outright, so settle paints both red and the
    # relation drops them -- they were never candidates, which is why they do
    # not appear as a skip. Only (3) against (5) survives.
    patterns = sibling_impulses((5.0, 3.0, 9.0))
    parent = patterns[0]

    result = run_pair_study(patterns + [leg_child("c1b", parent, 0, {0: 5.0})],
                            pair_spec())

    assert result["pairs"] == 1
    assert result["samples"] == 1
    assert result["true"] == 0                       # 3 > 9 is false


# ------------------------------------------- 23. empty, deterministic, reusable


def test_a_pair_study_with_nothing_marked_gives_zero_percentages():
    assert run_pair_study([], pair_spec()) == {
        "pairs": 0, "samples": 0, "true": 0, "false": 0, "ties": 0,
        "pct_true": 0.0, "pct_false": 0.0, "skipped": {},
    }
    assert run_pair_study("not a list", pair_spec())["pairs"] == 0


def test_a_pair_study_gives_the_same_answer_every_time():
    patterns = nested_siblings()
    study = pair_spec(type_a="Zigzag", wave_a="A", relative_degree=-1)

    assert run_pair_study(patterns, study) == run_pair_study(patterns, study)
    assert (list(run_pair_study(patterns, study)["skipped"])
            == list(run_pair_study(patterns, study)["skipped"]))


def test_a_prebuilt_context_answers_identically():
    # What Phase 23's grid leans on: settle and discover once, then run many
    # specs against the one discovery.
    patterns = sibling_impulses((5.0, 3.0, 9.0))
    context = pair_context(patterns, "parallel")

    assert (run_pair_study(patterns, pair_spec(), context=context)
            == run_pair_study(patterns, pair_spec()))
    assert (run_pair_study(patterns, pair_spec(operator="<"), context=context)
            == run_pair_study(patterns, pair_spec(operator="<")))


def test_a_pair_study_does_not_mutate_the_markings_it_counts():
    patterns = nested_siblings()
    before = [dict(pattern) for pattern in patterns]

    run_pair_study(patterns, pair_spec())

    assert patterns == before


# ------------------------------------------------------- 24. a bad pair spec


@pytest.mark.parametrize("bad", [
    {"analysis": "diagonal"},
    {"analysis": None},
    {"type_a": "Elephant"},
    {"variation_b": "Zigzag"},              # not a variation of Impulse
    {"wave_a": None},
    {"wave_b": ""},
    {"parent_wave_a": "9"},
    {"parent_wave_b": 1},
    {"relative_degree": 2},
    {"relative_degree": True},
    {"relative_degree": "+1"},
    {"field_a": "ADX"},
    {"field_b": "Peak RSI"},                # not the same family as field_a
    {"operator": ">="},
])
def test_a_malformed_pair_spec_is_rejected_rather_than_guessed_at(bad):
    with pytest.raises(ValueError):
        run_pair_study(sibling_impulses((1.0, 2.0, 3.0)), pair_spec(**bad))


def test_a_pair_spec_that_is_not_even_a_dict_is_rejected():
    with pytest.raises(ValueError):
        run_pair_study([], "wave 1 against wave 3")


# ------------------------------------------------------------- 25. the grid
#
# The manual forms above answer the question he already has; the grid is what he
# trawls with. Its promise is narrow and worth stating: every row must be the
# answer the manual form would have given for that combination, counted the same
# way -- otherwise the table is a second engine wearing the first one's numbers.
# So the cross-checks below are not spot checks. Each one walks a whole sweep
# and re-runs every row's combination through the public entry point.


CMB_FIELDS = ("Origin CMB", "Peak CMB", "Terminating CMB")
RSI_FIELDS = ("Origin RSI", "Peak RSI", "Terminating RSI")

# The client's 2A and 2B, written out rather than derived from the field table:
# a test that recomputed the eighteen from the same list the engine reads would
# pass just as happily on a broken list.
EXPECTED_VALUE_PAIRS = ({(field_a, field_b) for field_a in CMB_FIELDS
                         for field_b in CMB_FIELDS}
                        | {(field_a, field_b) for field_a in RSI_FIELDS
                           for field_b in RSI_FIELDS})


def grid_spec(**overrides):
    """The Parallel grid over one fixed wave on each side: his three factors."""
    spec = {"analysis": "parallel",
            "type_a": "Impulse", "variation_a": None, "parent_wave_a": None,
            "wave_a": "1",
            "type_b": "Impulse", "variation_b": None, "parent_wave_b": None,
            "wave_b": "1"}
    spec.update(overrides)
    return spec


def only(rows, **wanted):
    """The single row carrying these values, or an assertion naming the count."""
    found = [row for row in rows
             if all(row[key] == value for key, value in wanted.items())]
    assert len(found) == 1, f"{len(found)} rows match {wanted}"
    return found[0]


def counts_of(row, result):
    """The row's answer, restricted to exactly what a manual study reports."""
    return {key: row[key] for key in result}


def directed(pattern, rising):
    """The same pattern with endpoint prices that give it a direction.

    ``marking`` prices every point ascending, which makes every count bullish --
    harmless where nothing reads a direction, useless to the two analyses that
    pair on it. Only the endpoints matter: ``pattern_direction`` reads the first
    price against the last.
    """
    points = [dict(point) for point in pattern["points"]]
    points[0]["price"] = 100.0
    points[-1]["price"] = 200.0 if rising else 50.0
    pattern["points"] = points
    return pattern


def impulse_into_zigzag():
    """The client's worked structure: a bull Impulse chained into its correction.

    Every leg of both counts is marked as a pattern in its own right, so waves
    (1)..(5) and (A)(B)(C) all have roles -- which is what the Inverse Parallel
    and Adjacent discoveries stand on. Discovery finds (4)v(A) and (5)v(B)
    inverse-parallel, and (5)v(A) adjacent.

    Each child carries its number on its own wave 1 (or wave A), so a study
    reading one wave reads one planted value per child.
    """
    parent = directed(impulse("i", BASE, step=9000), True)
    last = parent["points"][-1]
    correction = directed(marking("z", "Zigzag", "Zigzag",
                                  spaced(last["time"], 4, 9000),
                                  kinds=["high", "low", "high", "low"]), False)
    # Chained exactly: the correction's point 0 is the impulse's last pivot.
    correction["points"][0]["price"] = last["price"]

    patterns = [parent, correction]
    for leg, rising in enumerate((True, False, True, False, True)):
        patterns.append(directed(
            leg_child(f"i{leg}", parent, leg, {0: float(leg + 1)}), rising))
    for leg, rising in enumerate((False, True, False)):
        patterns.append(directed(
            leg_child(f"z{leg}", correction, leg, {0: float(10 + leg)},
                      "Zigzag", "Zigzag", count=4), rising))
    return settle(patterns)


def two_tied_siblings():
    """One Parallel pair whose two waves carry the identical number."""
    parent = impulse("parent", BASE, step=9000)
    return [parent,
            leg_child("c1", parent, 0, {0: 4.0}),
            leg_child("c3", parent, 2, {0: 4.0})]


def many_impulses(count=8):
    """Forty marked patterns: eight parent impulses, four marked waves each."""
    patterns = []
    for index in range(count):
        parent = impulse(f"p{index}", BASE + 10 * DAY * index, step=90000)
        patterns.append(parent)
        for leg in range(4):
            patterns.append(leg_child(f"p{index}w{leg}", parent, leg,
                                      {0: float(leg), 2: float(index)}))
    return patterns


# ------------------------------------------------------- 25a. what it enumerates


def test_a_fixed_grid_is_the_clients_three_factors_and_nothing_else():
    rows = run_grid(sibling_impulses((5.0, 3.0, 9.0)), grid_spec())

    assert len(rows) == 162                          # 3 x 18 x 3
    assert {row["operator"] for row in rows} == {">", "<", "="}
    assert {row["relative_degree"] for row in rows} == {-1, 0, 1}
    assert {(row["field_a"], row["field_b"]) for row in rows} == EXPECTED_VALUE_PAIRS
    # Every row a distinct combination: 162 rows, 162 keys, nothing swept twice.
    assert len({(row["operator"], row["field_a"], row["field_b"],
                 row["relative_degree"]) for row in rows}) == 162


def test_no_combination_ever_compares_a_cmb_with_an_rsi():
    # His own rule, and the reason the value axis is eighteen rather than
    # thirty-six: an RSI of 62 against a CMB of 0.4 has no meaning to report.
    for row in run_grid(sibling_impulses((5.0, 3.0, 9.0)), grid_spec()):
        assert field_family(row["field_a"]) is not None
        assert field_family(row["field_a"]) == field_family(row["field_b"])


def test_all_waves_on_one_side_sweeps_that_sides_five_labels():
    rows = run_grid(sibling_impulses((5.0, 3.0, 9.0)), grid_spec(wave_a=ALL_WAVES))

    assert len(rows) == 810                          # 5 x 162
    assert {row["wave_a"] for row in rows} == {"1", "2", "3", "4", "5"}
    assert {row["wave_b"] for row in rows} == {"1"}


def test_all_waves_on_both_sides_squares_it():
    rows = run_grid(sibling_impulses((5.0, 3.0, 9.0)),
                    grid_spec(wave_a=ALL_WAVES, wave_b=ALL_WAVES))

    assert len(rows) == 4050                         # 25 x 162


def test_a_pair_grid_carries_the_discovery_count_and_an_inter_grid_does_not():
    # ``pairs`` is the denominator context a pair study leads with; the
    # inter-pattern study discovers no pairs, so the column does not exist.
    for row in run_grid(sibling_impulses((5.0, 3.0, 9.0)), grid_spec()):
        assert row["pairs"] == 3

    for row in run_grid(five_impulses(),
                        grid_spec(analysis="inter", wave_a="4", wave_b="2")):
        assert "pairs" not in row


# ---------------------------------------------- 25b. every row is the manual one


def test_every_parallel_grid_row_equals_the_manual_study_it_stands_for():
    patterns = nested_siblings()
    rows = run_grid(patterns, grid_spec(type_a="Zigzag", wave_a="A"))

    for row in rows:
        manual = run_pair_study(patterns, pair_spec(
            type_a="Zigzag", wave_a="A", wave_b=row["wave_b"],
            relative_degree=row["relative_degree"], field_a=row["field_a"],
            field_b=row["field_b"], operator=row["operator"]))
        assert counts_of(row, manual) == manual

    assert any(row["samples"] for row in rows)       # and not vacuously so


@pytest.mark.parametrize("analysis", ["inverse_parallel", "adjacent"])
def test_every_chained_grid_row_equals_the_manual_study_it_stands_for(analysis):
    patterns = impulse_into_zigzag()
    rows = run_grid(patterns, grid_spec(analysis=analysis, type_a="Impulse",
                                        wave_a="1", type_b="Zigzag", wave_b="A"))

    for row in rows:
        manual = run_pair_study(patterns, pair_spec(
            analysis=analysis, type_a="Impulse", wave_a="1",
            type_b="Zigzag", wave_b="A",
            relative_degree=row["relative_degree"], field_a=row["field_a"],
            field_b=row["field_b"], operator=row["operator"]))
        assert counts_of(row, manual) == manual

    assert any(row["samples"] for row in rows)


def test_every_inter_grid_row_equals_the_manual_study_it_stands_for():
    patterns = nested_pair()
    rows = run_grid(patterns, grid_spec(analysis="inter", wave_a="4", wave_b="2"))

    for row in rows:
        assert row["note"] is None
        # The grid's one degree knob in the inter engine's two-offset spelling:
        # +1 makes wave A's pattern the parent of wave B's.
        manual = run_study(patterns, spec(
            wave_a="4", wave_b="2", offset_a=row["relative_degree"], offset_b=0,
            field_a=row["field_a"], field_b=row["field_b"],
            operator=row["operator"]))
        assert counts_of(row, manual) == manual

    assert any(row["samples"] for row in rows)


# ----------------------------------------------------------- 25c. the = operator


def test_a_planted_tie_is_true_under_equals_and_false_under_greater():
    # Both rows come out of one sweep, which is the point: the operator axis is
    # counted off a single evaluation of the pair, not three of them.
    rows = run_grid(two_tied_siblings(), grid_spec())
    same = {"field_a": "Peak CMB", "field_b": "Peak CMB", "relative_degree": 0}

    equal = only(rows, operator="=", **same)
    assert equal["samples"] == 1
    assert equal["true"] == 1
    assert equal["ties"] == 1

    strict = only(rows, operator=">", **same)
    assert strict["samples"] == 1
    assert strict["false"] == 1
    assert strict["ties"] == 1


# ------------------------------------------------- 25d. combinations with no answer


def test_an_inter_grid_across_two_types_notes_the_same_degree_rows():
    # At the same degree the two waves come from one pattern, so two different
    # side filters describe a pattern that cannot exist. The manual form raises;
    # a sweep that raised would lose the rows it had already counted.
    patterns = impulse_with_zigzag_child()
    rows = run_grid(patterns, grid_spec(analysis="inter", type_a="Impulse",
                                        wave_a="4", type_b="Zigzag", wave_b="A"))

    assert len(rows) == 162                          # the sweep completed
    noted = [row for row in rows if row["note"]]
    assert len(noted) == 54                          # one degree's worth
    assert {row["relative_degree"] for row in noted} == {0}
    assert all(row["note"] == NOTE_ONE_PATTERN for row in noted)
    assert all(row["samples"] == 0 for row in noted)
    assert all(row["skipped"] == {} for row in noted)

    # ...and the combinations that do have an answer were answered.
    counted = only(rows, relative_degree=1, operator=">",
                   field_a="Peak CMB", field_b="Peak CMB")
    assert counted["samples"] == 1
    assert counted["true"] == 1                      # 10.0 > 4.0


# --------------------------------------------- 25e. deterministic, empty, quick


def test_the_same_sweep_twice_gives_an_identical_list():
    patterns = nested_siblings()
    grid = grid_spec(wave_a=ALL_WAVES)

    assert run_grid(patterns, grid) == run_grid(patterns, grid)


@pytest.mark.parametrize("analysis", ["inter", "parallel", "inverse_parallel",
                                      "adjacent"])
def test_a_sweep_over_nothing_marked_is_all_zero_rows(analysis):
    # What the tab shows on his real file today: he has marked almost nothing
    # with values yet, and near-all-zero must render rather than raise.
    rows = run_grid([], grid_spec(analysis=analysis, wave_a="4", wave_b="2"))

    assert len(rows) == 162
    assert all(row["samples"] == 0 for row in rows)
    assert all(row["pct_true"] == 0.0 and row["pct_false"] == 0.0 for row in rows)
    assert all(row["skipped"] == {} for row in rows)
    assert all(row["note"] is None for row in rows)


def test_an_all_by_all_impulse_sweep_finishes_in_seconds():
    # A tripwire, not a benchmark. It fails when the grid stops sharing work:
    # one evaluation per wave pair and degree, with the fifty-four operator and
    # value combinations counted off it, rather than 4,050 settles.
    patterns = many_impulses()
    assert len(patterns) == 40

    started = time.monotonic()
    rows = run_grid(patterns, grid_spec(wave_a=ALL_WAVES, wave_b=ALL_WAVES))
    elapsed = time.monotonic() - started

    assert len(rows) == 4050
    assert any(row["samples"] for row in rows)
    assert elapsed < 10.0, f"the sweep took {elapsed:.1f}s"


# ------------------------------------------------------- 25f. a bad grid spec


@pytest.mark.parametrize("bad", [
    {"analysis": "everything"},
    {"analysis": None},
    {"type_a": "Elephant"},
    {"variation_b": "Zigzag"},              # not a variation of Impulse
    {"wave_a": None},
    {"wave_b": ""},
    {"parent_wave_a": "9"},
])
def test_a_malformed_grid_spec_is_rejected_rather_than_swept(bad):
    with pytest.raises(ValueError):
        run_grid(sibling_impulses((1.0, 2.0, 3.0)), grid_spec(**bad))


def test_a_grid_spec_that_is_not_even_a_dict_is_rejected():
    with pytest.raises(ValueError):
        run_grid([], "every combination please")


# ------------------------------------------------- 26. the grid's table helpers
#
# The pure half of the tab's grid: the frame it draws, the sweep size it quotes
# and the denominator line under the table. Kept here beside the engine because
# they are the only place a row dict becomes something the client reads, and a
# table that hid a note or multiplied a skip count by fifty-four would be a
# worse lie than a wrong percentage -- it would look like evidence.


def grid_rows(patterns=None, **overrides):
    return run_grid(patterns if patterns is not None
                    else sibling_impulses((5.0, 3.0, 9.0)), grid_spec(**overrides))


def test_the_table_drops_the_pairs_column_for_an_inter_sweep():
    from ui.wave_analysis_tab import grid_frame

    pair_frame = grid_frame(grid_rows())
    assert "Pairs" in pair_frame.columns

    inter_frame = grid_frame(grid_rows(five_impulses(), analysis="inter",
                                       wave_a="4", wave_b="2"))
    assert "Pairs" not in inter_frame.columns
    assert list(inter_frame.columns)[:6] == ["Wave A", "Wave B", "Value A",
                                             "Value B", "Op", "Rel deg"]


def test_a_noted_row_shows_the_note_in_place_of_its_counts():
    from ui.wave_analysis_tab import grid_frame

    frame = grid_frame(grid_rows(impulse_with_zigzag_child(), analysis="inter",
                                 type_a="Impulse", wave_a="4",
                                 type_b="Zigzag", wave_b="A"))

    noted = frame[frame["Note"] == NOTE_ONE_PATTERN]
    assert len(noted) == 54
    # Blank, not zero: a zero in the Samples column reads as "nothing measured
    # yet", which is a different thing from "that question has no answer".
    for column in ("Samples", "True", "False", "Ties", "% True", "% False"):
        assert noted[column].isna().all()
    # ...and the rows that were counted keep their counts and an empty note.
    counted = frame[frame["Note"] == ""]
    assert len(counted) == 108
    assert counted["Samples"].notna().all()


def test_a_sweep_with_nothing_to_note_grows_no_note_column():
    from ui.wave_analysis_tab import grid_frame

    assert "Note" not in grid_frame(grid_rows()).columns


def test_the_table_leads_with_the_strongest_claims():
    from ui.wave_analysis_tab import grid_frame

    frame = grid_frame(grid_rows())
    percentages = list(frame["% True"])
    assert percentages == sorted(percentages, reverse=True)
    # Within one percentage, the fuller sample first.
    top = frame[frame["% True"] == percentages[0]]
    assert list(top["Samples"]) == sorted(top["Samples"], reverse=True)


def test_an_empty_sweep_still_draws_its_columns():
    from ui.wave_analysis_tab import grid_frame

    frame = grid_frame([])
    assert list(frame.columns)[0] == "Wave A"
    assert len(frame) == 0


def test_the_skip_line_counts_one_evaluation_once_not_fifty_four_times():
    from ui.wave_analysis_tab import grid_skips

    rows = grid_rows(nested_siblings(), type_a="Zigzag", wave_a="A")
    totals = grid_skips(rows)

    # Three pairs, one of which has no child to read: counted once for the
    # degree that asked, not once per operator and value pair.
    assert totals == {SKIP_NO_DEGREE: 1, SKIP_PAIR_FILTERED: 6}
    summed_per_row = sum(count for row in rows for count in row["skipped"].values())
    assert summed_per_row == 54 * sum(totals.values())


def test_the_caption_multiplies_the_sweep_out_before_it_runs():
    from ui.wave_analysis_tab import grid_caption, grid_size

    fixed = grid_spec()
    assert grid_size(fixed) == (162, 1, 1)
    assert grid_caption(fixed) == ("162 combinations — 3 operators × 18 value "
                                   "pairs × 3 relative degrees")

    swept = grid_spec(wave_a=ALL_WAVES, wave_b=ALL_WAVES)
    assert grid_size(swept) == (4050, 5, 5)
    assert grid_caption(swept).endswith("× 5 Wave A × 5 Wave B labels")


def test_the_summary_gives_the_table_its_denominators():
    from ui.wave_analysis_tab import grid_summary

    summary = grid_summary(grid_rows(), "parallel")

    assert summary.startswith("162 combinations · 54 with samples")
    # The one thing the table invites him to get backwards.
    assert "Samples can exceed Pairs" in summary
    assert "Skipped: 6 pairs with no pattern at that degree" in summary

    # No pairs to fan out from, so the warning is left off.
    assert "Samples can exceed Pairs" not in grid_summary(
        grid_rows(five_impulses(), analysis="inter", wave_a="4", wave_b="2"),
        "inter")


# ------------------------------------------- 26a. the % True threshold filter
#
# The second view filter, hand-made rather than swept: five rows chosen so that
# every edge of the threshold is one row -- clearly above, clearly below, sat
# exactly on it, never sampled, and never evaluated at all.


def filter_row(wave_a, samples, pct_true, note=None):
    return {"wave_a": wave_a, "wave_b": "2", "field_a": "Peak CMB",
            "field_b": "Peak CMB", "operator": ">", "relative_degree":
            "Same degree", "pairs": 1, "samples": samples,
            "true": samples, "false": 0, "ties": 0,
            "pct_true": pct_true, "pct_false": 0.0,
            "note": note, "skipped": {}}


def filter_frame():
    from ui.wave_analysis_tab import grid_frame

    return grid_frame([filter_row("high", 5, 80.0),
                       filter_row("low", 5, 40.0),
                       filter_row("exact", 5, 75.0),
                       filter_row("empty", 0, 0.0),
                       filter_row("noted", 0, 0.0, note=NOTE_ONE_PATTERN)])


def shown_waves(frame):
    return set(frame["Wave A"])


def test_the_threshold_at_zero_leaves_the_table_exactly_as_it_was():
    from ui.wave_analysis_tab import grid_view

    frame = filter_frame()
    # Off means off: the noted row's % True is blank, and NaN >= 0 is False, so
    # a threshold compared unconditionally would delete it here.
    assert shown_waves(grid_view(frame, False, 0)) == {"high", "low", "exact",
                                                       "empty", "noted"}
    # ...and with the hide-empty box ticked, only that box's own two rows go.
    assert shown_waves(grid_view(frame, True, 0)) == {"high", "low", "exact"}


def test_a_threshold_keeps_the_rows_that_reach_it_including_the_exact_one():
    from ui.wave_analysis_tab import grid_view

    shown = grid_view(filter_frame(), True, 75)

    # ">= 75", not "> 75": the column is drawn to one decimal, so a row reading
    # exactly 75.0 must not vanish for a reason he cannot see on screen.
    assert shown_waves(shown) == {"high", "exact"}


def test_the_threshold_drops_an_unsampled_row_with_hide_empty_unticked():
    from ui.wave_analysis_tab import grid_view

    shown = grid_view(filter_frame(), False, 75)

    # The two filters compose rather than fight: 0.0 does not clear 75, and a
    # row that was never evaluated has no percentage to clear it with.
    assert shown_waves(shown) == {"high", "exact"}


def test_the_view_filter_never_touches_the_frame_it_was_given():
    from ui.wave_analysis_tab import grid_view

    frame = filter_frame()
    before = frame.copy()

    grid_view(frame, True, 75)

    assert frame.equals(before)
    assert len(frame) == 5

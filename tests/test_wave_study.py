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

import pytest

from config.wave_analysis import LEG_VALUE_FIELDS, analysable_legs, analysable_waves
from strategies.wave_study import (SKIP_AMBIGUOUS, SKIP_LABEL_ABSENT,
                                   SKIP_NOT_MEASURED, SKIP_NO_COUNTERPART, SKIP_RED,
                                   run_study, wave_labels)

BASE = 1719878400
DAY = 86400


# --------------------------------------------------------------- the fixtures


def values(cmb):
    """All six fields on one leg, with CMB carrying the number under test.

    All six, because a wave is only analysable when every one of them is a
    number; only CMB varies, because a study reads one field at a time and a
    second moving number would just make the arithmetic harder to check.
    """
    entry = {field: 1.0 for field in LEG_VALUE_FIELDS}
    entry["CMB"] = cmb
    entry["timeframe"] = "1D"
    return entry


def leg_values(by_leg):
    """``{3: 10.0}`` -> a stored ``leg_values`` measuring leg 3 with CMB 10."""
    return {str(leg): values(cmb) for leg, cmb in by_leg.items()}


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


def wave_4_vs_2(wave_4, wave_2):
    return {3: wave_4, 1: wave_2}


def spec(**overrides):
    """The client's headline study, with anything he might change overridden."""
    study = {"pattern_type": "Impulse", "variation": None,
             "wave_a": "4", "offset_a": 0,
             "wave_b": "2", "offset_b": 0,
             "field": "CMB", "operator": "<"}
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
    {"operator": "="},
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

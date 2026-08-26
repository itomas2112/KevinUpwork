"""
The Wave Analysis study: one comparison, counted across every marked pattern.

This is what everything since Phase 16 was built to serve. The client marks
counts, measures six values on each wave he cares about, and then wants to ask
one question of the whole collection -- "how often is wave 4's CMB lower than
wave 2's?" -- and be given a percentage.

Pure: no Streamlit, no I/O, no clock. A study over the same list twice gives the
identical result, which is the least a number he intends to trade on can offer.

Alongside it live the three pair-discovery functions -- ``parallel_pairs``,
``inverse_parallel_pairs`` and ``adjacent_pairs`` -- which answer "which two
marked patterns may be compared" for the three cross-pattern analyses, and
``run_pair_study``, which counts one comparison across whichever of them the
client asked for. Three decisions run through the discoveries:

* **Chaining is exact.** Two patterns are adjacent only when they share a pivot
  outright: same time *and* same kind. A one-bar gap between two marked patterns
  means no pair. This mirrors the strictness of the parent/child relation and
  matches the client's own workflow, where the next count's point 0 is clicked
  on the previous count's last point. If his real markings turn out to carry
  hairline gaps, relaxing this is his decision, not the engine's.
* **Only classifiable patterns take part**: yellow, with a marked parent (so it
  occupies a wave and has a role), and with a direction. Red patterns, roots and
  flat patterns therefore never appear in a pair, and never silently -- they
  were never candidates.
* **Determinism.** Same input, same output, same order. Every function returns
  its pairs in the input list's order: by the position of the earlier member,
  then of the later.
"""

from config.wave_analysis import (ACTIONARY_LABELS, LEG_VALUE_FIELDS, PATTERN_DEFS,
                                  REACTIONARY_LABELS, analysable_waves, chained,
                                  children_by_leg, find_parent, leg_value_families,
                                  pattern_direction, pattern_role, pattern_span,
                                  point_labels, role_label, settle)

# ``=`` joined the other two when the client's grid-search factors arrived
# spelling it out. It was left out before on his own "perfectly equal values is
# near impossible", which is a statement about how often it fires, not about
# whether he may ask for it.
OPERATORS = (">", "<", "=")

# The offsets a side may carry. +1 means "the pattern one degree senior", which
# in this codebase is only ever the parent -- see the note on pairing below.
OFFSETS = (-1, 0, 1)

# The labels a marked pattern may occupy in its parent -- his "Parent wave (1)"
# -- in the order the analysis forms offer them. Exactly the two role sets
# together, which a test holds us to: a label no pattern type names could only
# ever filter every pair away.
PARENT_WAVE_LABELS = ("1", "2", "3", "4", "5", "A", "B", "C", "D", "E",
                      "W", "X", "Y", "Z")

# Why a candidate did not become a sample. The strings are the report: a
# percentage with no denominator context is worse than no percentage, and the
# client needs to read "63 waves not yet measured" without a lookup table.
SKIP_NOT_MEASURED = "waves not yet measured"
SKIP_RED = "patterns in conflict (red)"
SKIP_AMBIGUOUS = "ambiguous wave labels"
SKIP_LABEL_ABSENT = "patterns that have no such wave"
SKIP_NO_COUNTERPART = "patterns with no matching counterpart"
SKIP_PAIR_FILTERED = "pairs outside the filters"
SKIP_NO_DEGREE = "pairs with no pattern at that degree"

# Reported in this order however they were counted, so the same input always
# renders the same line. The two pair-level reasons lead because they are the
# funnel's first stage: a pair dropped for its roles never reached a wave, and
# reading the line in the order the engine narrows makes the denominator
# obvious.
SKIP_ORDER = (SKIP_PAIR_FILTERED, SKIP_NO_DEGREE, SKIP_NOT_MEASURED, SKIP_RED,
              SKIP_AMBIGUOUS, SKIP_LABEL_ABSENT, SKIP_NO_COUNTERPART)


def wave_labels(pattern_type, variation=None):
    """The wave labels selectable for a type, or for one variation of it.

    With no variation, the union across that type's variations in first-seen
    order -- a Zigzag offers A B C, a type whose variations disagree offers all
    of them, and a duplicate inside one sequence appears once.
    """
    variations = PATTERN_DEFS.get(pattern_type)
    if variations is None:
        return []

    labels = []
    for name, label_seq in variations:
        if variation is not None and name != variation:
            continue
        for label in label_seq:
            if label not in labels:
                labels.append(label)
    return labels


def _variation_names(pattern_type):
    return [name for name, _label_seq in PATTERN_DEFS.get(pattern_type, ())]


def field_family(field):
    """Which oscillator pane a value field is read off: "cmb", "rsi" or None.

    Read back through the field table rather than restated here, so the one
    place a field is added or renamed stays ``LEG_VALUE_FIELDS``. The studies
    need it because of the client's own rule -- an Origin CMB "can only be
    compared with other CMB values" -- and the analysis forms need it to decide
    which fields the second value box may offer at all.
    """
    return leg_value_families().get(field)


def _read_pattern_filter(spec, type_key, variation_key, default_type=None,
                         default_variation=None):
    """One side's (type, variation), checked, or a ValueError naming what is wrong."""
    pattern_type = spec.get(type_key, default_type)
    if pattern_type not in PATTERN_DEFS:
        raise ValueError(f"Unknown pattern type: {pattern_type!r}")

    variation = spec.get(variation_key, default_variation)
    if variation is not None and variation not in _variation_names(pattern_type):
        raise ValueError(f"Unknown variation {variation!r} for {pattern_type!r}")
    return pattern_type, variation


def _read_fields(spec):
    """(field_a, field_b), checked as one family, or a ValueError.

    ``field`` is the pre-phase spelling and still means both sides at once, so
    every study written before the two boxes existed keeps asking exactly what
    it asked. ``field_a``/``field_b`` win where they are given.

    The family rule is the client's: CMB is compared with CMB and RSI with RSI.
    A cross-family study is refused rather than run, because the number it would
    produce is not a smaller truth -- an RSI of 62 against a CMB of 0.4 has no
    meaning to report a percentage of.
    """
    field = spec.get("field")
    field_a = spec.get("field_a", field)
    field_b = spec.get("field_b", field)
    for field_name in (field_a, field_b):
        if field_name not in LEG_VALUE_FIELDS:
            raise ValueError(f"Not a measured value: {field_name!r}")

    family = field_family(field_a)
    if family is None or family != field_family(field_b):
        raise ValueError(f"{field_a!r} and {field_b!r} are not the same family "
                         "of value and cannot be compared")
    return field_a, field_b


def _read_operator(spec):
    operator = spec.get("operator")
    if operator not in OPERATORS:
        raise ValueError(f"Not a comparison: {operator!r}")
    return operator


def _read_wave(spec, key):
    wave = spec.get(key)
    if not isinstance(wave, str) or not wave:
        raise ValueError(f"Not a wave label: {wave!r}")
    return wave


def _read_spec(spec):
    """The spec unpacked and checked, or a ValueError naming what is wrong.

    Every field is rejected rather than defaulted. A study is a claim about the
    client's markings; guessing at half a question and answering the other half
    would hand him a percentage that is not an answer to anything he asked.
    """
    if not isinstance(spec, dict):
        raise ValueError("A study spec must be a dict")

    pattern_type, variation = _read_pattern_filter(spec, "pattern_type", "variation")
    # Side B's filter defaults to side A's, so a study that never mentions it is
    # the like-with-like study it always was. Given and different, it constrains
    # side B's own pattern -- which is what makes "wave 1 of a Zigzag child
    # inside an Impulse's wave 4" expressible at all.
    pattern_type_b, variation_b = _read_pattern_filter(
        spec, "pattern_type_b", "variation_b", pattern_type, variation)

    wave_a = _read_wave(spec, "wave_a")
    wave_b = _read_wave(spec, "wave_b")
    field_a, field_b = _read_fields(spec)
    operator = _read_operator(spec)

    offsets = []
    for name in ("offset_a", "offset_b"):
        offset = spec.get(name, 0)
        if isinstance(offset, bool) or not isinstance(offset, int) or offset not in OFFSETS:
            raise ValueError(f"{name} must be -1, 0 or +1, not {offset!r}")
        offsets.append(offset)
    offset_a, offset_b = offsets

    # The client's own rule -- "only 1 can be active" -- and the UI greys the
    # other box out to enforce it. The engine refuses rather than picking one,
    # because either reading of "both sides moved a degree" is a different study.
    if offset_a and offset_b:
        raise ValueError("Only one side of a study may be moved a degree")

    # At the same degree the two sides are one pattern, so two different side
    # filters describe a pattern that cannot exist. Refused rather than answered
    # with a truthful-looking zero, which would read as "he has not marked one
    # of those yet" rather than "that study is not a thing".
    if not offset_a and not offset_b and (pattern_type_b, variation_b) != (pattern_type,
                                                                          variation):
        raise ValueError("Both waves come from one pattern at the same degree, "
                         "so the two sides cannot carry different pattern filters")

    return {"pattern_type": pattern_type, "variation": variation,
            "pattern_type_b": pattern_type_b, "variation_b": variation_b,
            "wave_a": wave_a, "wave_b": wave_b,
            "field_a": field_a, "field_b": field_b, "operator": operator,
            "offset_a": offset_a, "offset_b": offset_b}


def _matches(pattern, pattern_type, variation):
    """True when a pattern is inside the study's type (and variation) filter."""
    if not isinstance(pattern, dict):
        return False
    if pattern.get("pattern_type") != pattern_type:
        return False
    return variation is None or pattern.get("variation") == variation


def _holds(value_a, value_b, operator):
    """True when the comparison the client asked for holds between two numbers.

    ``>`` and ``<`` stay strict, so a tie is false under both; ``=`` is exact
    equality, so a tie is the only thing that is true under it. Ties are counted
    beside the verdict either way, which is how he sees whether an ``=`` study
    found anything at all.
    """
    if operator == ">":
        return value_a > value_b
    if operator == "<":
        return value_a < value_b
    return value_a == value_b


def _legs_labelled(pattern, label):
    """Every leg of a pattern that the given wave label names.

    Usually one. A Triple Zigzag's sequence is ``W X Y X Z``, so ``X`` names
    two of them -- the caller has to decide what to do about that rather than
    being handed the first.
    """
    try:
        labels = point_labels(pattern.get("pattern_type"), pattern.get("variation"))
    except (ValueError, TypeError):
        return []
    # Leg k runs points[k] -> points[k + 1] and is the wave labels[k + 1] names.
    return [k for k in range(len(labels) - 1) if labels[k + 1] == label]


def _report(samples, true_count, false_count, ties, skipped):
    """The shape every study answers in, whatever it counted over.

    One builder for both engines, so a pair study's percentages can never come
    out rounded, ordered or zero-guarded differently from an inter-pattern
    study's -- the client reads the two lines side by side.
    """
    return {
        "samples": samples,
        "true": true_count,
        "false": false_count,
        "ties": ties,
        # No samples is a legitimate answer -- he has not measured enough yet --
        # and must read as zero rather than blow up the tab.
        "pct_true": (100.0 * true_count / samples) if samples else 0.0,
        "pct_false": (100.0 * false_count / samples) if samples else 0.0,
        "skipped": {reason: skipped[reason] for reason in SKIP_ORDER
                    if skipped.get(reason)},
    }


def _skipper():
    """A ``{reason: count}`` dict and the one function that fills it.

    Handed out together so a new engine cannot invent a second spelling of
    "not measured": the client reads the skip line as a denominator, and two
    reasons that mean the same thing would split one number into two.
    """
    skipped = {}

    def skip(reason, count=1):
        if count:
            skipped[reason] = skipped.get(reason, 0) + count

    return skipped, skip


def _wave_pair(source_a, wave_a, source_b, wave_b, eligible, skip):
    """The two waves' measured values, or None with the reason counted.

    The single place a wave *label* becomes a pair of numbers, shared by every
    engine here. Ambiguity, absence and unmeasured waves are the three ways it
    fails, and each is counted where it happens rather than inferred later.
    """
    legs_a = _legs_labelled(source_a, wave_a)
    legs_b = _legs_labelled(source_b, wave_b)
    unresolved = False
    for legs in (legs_a, legs_b):
        if len(legs) > 1:
            # A study that quietly picked one of two X waves would be
            # unreproducible, so the pattern drops out for that side.
            skip(SKIP_AMBIGUOUS)
            unresolved = True
        elif not legs:
            skip(SKIP_LABEL_ABSENT)
            unresolved = True
    if unresolved:
        return None

    key_a = (source_a.get("id"), legs_a[0])
    key_b = (source_b.get("id"), legs_b[0])
    row_a, row_b = eligible.get(key_a), eligible.get(key_b)
    if row_a is None or row_b is None:
        # Counted per wave, and the same wave asked for twice is one wave.
        missing = (1 if row_a is None else 0)
        if key_b != key_a and row_b is None:
            missing += 1
        skip(SKIP_NOT_MEASURED, missing)
        return None
    return row_a["values"], row_b["values"]


def _tally(samples, field_a, field_b, operator, skipped):
    """The report for one value pair and operator over samples already found.

    The split the grid is built on: *which* waves a study compares depends on
    the pattern filters, the wave labels and the degree, and on neither the
    value fields nor the operator. So the samples are found once and the
    combinations that share them are counted here, off the same list.
    """
    true_count = false_count = ties = 0
    for values_a, values_b in samples:
        value_a, value_b = values_a[field_a], values_b[field_b]
        if value_a == value_b:
            ties += 1
        if _holds(value_a, value_b, operator):
            true_count += 1
        else:
            false_count += 1
    return _report(len(samples), true_count, false_count, ties, skipped)


def study_context(patterns, analysis=None):
    """Everything a study reads, built once for a pattern list.

    Settling, the parent/child index, discovery and the eligibility index are
    the expensive half of any study here and none of them depends on the
    comparison being asked, so they are separated out: the grid enumerates many
    combinations and must not re-derive the same structure for every one.

    ``analysis`` names one pair analysis, ``"inter"`` (which needs no
    discovery), or None for all three.
    """
    settled = settle(patterns if isinstance(patterns, list) else [])
    names = [name for name in
             (list(PAIR_ANALYSES) if analysis is None else [analysis])
             if name in PAIR_ANALYSES]
    children = children_by_leg(settled)
    return {
        "settled": settled,
        "pairs": {name: PAIR_ANALYSES[name](settled) for name in names},
        "children": children,
        "parents": _parent_index(settled, children),
        "eligible": {(row["pattern_id"], row["leg"]): row
                     for row in analysable_waves(settled)},
    }


def _parent_index(settled, children):
    """{child id: (parent, leg)}, inverted from the downward index.

    ``find_parent`` scans the whole list, so asking it once per pattern per
    study makes the grid quadratic in the markings for no reason:
    ``children_by_leg`` already paid for exactly these answers, once.
    """
    by_id = {pattern["id"]: pattern for pattern in settled
             if isinstance(pattern, dict) and isinstance(pattern.get("id"), str)}
    parents = {}
    for parent_id, by_leg in children.items():
        parent = by_id.get(parent_id)
        if parent is None:
            continue
        for leg, kids in by_leg.items():
            for child in kids:
                parents[child["id"]] = (parent, leg)
    return parents


def _study_samples(read, context):
    """Every pair of values an inter-pattern spec compares, and what it dropped.

    ``run_study`` minus the comparison: the walk down the pattern list, the
    parent lookup and the eligibility check. Returns ``(samples, skipped)``.
    """
    settled, eligible = context["settled"], context["eligible"]
    parents = context["parents"]
    pattern_type, variation = read["pattern_type"], read["variation"]
    pattern_type_b, variation_b = read["pattern_type_b"], read["variation_b"]
    wave_a, wave_b = read["wave_a"], read["wave_b"]
    offset_a, offset_b = read["offset_a"], read["offset_b"]

    skipped, skip = _skipper()

    # Which side of the pair the pattern being walked plays. In a cross-degree
    # study the walk is always driven from the *child*, because find_parent
    # answers "who is above me" and gives exactly one answer -- a parent may
    # expose several legs and so have several children.
    a_is_parent = offset_a == 1 or offset_b == -1

    # Which side's filter applies to the pattern being walked, and which to the
    # one found above it. With both offsets 0 the two filters are the same
    # object -- _read_spec refuses a spec where they differ -- so the walked
    # side's filter is the whole story.
    walked_type, walked_variation = ((pattern_type_b, variation_b) if a_is_parent
                                     else (pattern_type, variation))
    parent_type, parent_variation = ((pattern_type, variation) if a_is_parent
                                     else (pattern_type_b, variation_b))

    samples = []
    for pattern in settled:
        if not _matches(pattern, walked_type, walked_variation):
            continue
        if pattern.get("color") == "red":
            # The client's rule: only white counts, and red is the marking
            # itself being contested. Counted per pattern -- the whole marking
            # drops out, not one wave of it.
            skip(SKIP_RED)
            continue

        if offset_a == 0 and offset_b == 0:
            pattern_a = pattern_b = pattern
        else:
            pattern_id = pattern.get("id")
            if isinstance(pattern_id, str) and pattern_id:
                parent, _leg = parents.get(pattern_id, (None, None))
            else:
                # No id means no row in the index, and no sample either -- the
                # eligibility index is keyed by id. Asked outright so the skip
                # reason stays the one this study has always given.
                parent, _leg = find_parent(settled, pattern)
            if parent is None or not _matches(parent, parent_type, parent_variation):
                skip(SKIP_NO_COUNTERPART)
                continue
            pattern_a, pattern_b = ((parent, pattern) if a_is_parent
                                    else (pattern, parent))

        found = _wave_pair(pattern_a, wave_a, pattern_b, wave_b, eligible, skip)
        if found is not None:
            samples.append(found)

    return samples, skipped


def run_study(patterns, spec, context=None):
    """Returns counts and percentages, plus why samples were dropped.

    {"samples": int, "true": int, "false": int, "ties": int,
     "pct_true": float, "pct_false": float, "skipped": {reason: count}}

    Which pairs are compared:

    * **Both offsets 0** -- the two waves come from the *same* pattern. The
      headline case: wave 4 and wave 2 of one impulse.
    * **One offset non-zero** -- the two waves come from a parent and its child,
      in the direction the offset names. Parent and child mean the relation
      ``find_parent`` already models: a child spans exactly one leg of its
      parent, endpoint for endpoint. Two unrelated patterns that happen to sit
      at adjacent degrees are *not* comparable structure, and pairing them would
      put noise into a percentage the client intends to trade on.

    Each side carries its own type filter (``pattern_type_b`` / ``variation_b``,
    defaulting to side A's), so a cross-degree study compares like with like
    unless he says otherwise -- and when he does, "wave 1 of a Zigzag child
    inside an Impulse's wave 4" is finally a study he can ask for.

    ``>`` and ``<`` are strict, so ties count false under both; ``=`` counts
    them true. The tie count is reported for every operator.

    ``context`` is what ``study_context`` built, for a caller running many
    specs against one settled list; None builds it here for this one spec.
    """
    read = _read_spec(spec)
    if context is None:
        # Settled once, here, and everything downstream reads that list:
        # colours and degrees are outputs, and a study run on an unsettled list
        # could count a wave that the very same markings would paint red.
        context = study_context(patterns, "inter")
    samples, skipped = _study_samples(read, context)
    return _tally(samples, read["field_a"], read["field_b"], read["operator"],
                  skipped)


# ------------------------------------------------------------- pair discovery
#
# The three cross-pattern analyses -- Parallel, Inverse Parallel, Adjacent --
# each compare the wave counts of two *different* marked patterns, and each is
# defined by the client purely in terms of actionary/reactionary roles, pattern
# direction and adjacency. What differs between them is only which pairs are
# eligible, so that is all that lives here: discovery, not comparison.
#
# Every function takes a **settled** list. Colour and degree are outputs of
# ``settle``, and both gate pairing -- red patterns are out and the degree
# groups are what the "same degree" rules mean -- so an unsettled list would
# pair on stale answers. The engine that calls these settles once and hands the
# same list to all three.
#
# Each pair is reported as:
#
#     {"a": pattern, "b": pattern,        # a is always the earlier in time
#      "role_a": label, "role_b": label,  # the wave labels: "1", "3", "A", ...
#      "gap": pattern or None}            # the wave between them, where one exists


def _positions(patterns):
    """{pattern id: its index in the input list}, for ordering the results."""
    positions = {}
    if not isinstance(patterns, list):
        return positions
    for index, pattern in enumerate(patterns):
        if isinstance(pattern, dict) and isinstance(pattern.get("id"), str):
            positions.setdefault(pattern["id"], index)
    return positions


def _ordered(pairs, positions):
    """Pairs in the input list's order: earlier member first, then the later.

    ``sorted`` is stable, so two pairs on the same two patterns -- which happens
    in Inverse Parallel when two different gaps chain the same endpoints -- keep
    the order they were discovered in rather than swapping between runs.
    """
    return sorted(pairs, key=lambda pair: (positions.get(pair["a"].get("id"), -1),
                                           positions.get(pair["b"].get("id"), -1)))


def _pairable(patterns):
    """Every pattern that may take part in a pair, in input order.

    Yellow, with a marked parent, with a role that parent's labels name, and
    with a direction. Each row carries what the pairing rules ask about, so the
    rules read as the client stated them rather than as a chain of lookups:

        {"pattern", "parent_id", "label", "role", "direction", "degree", "start"}
    """
    if not isinstance(patterns, list):
        return []

    rows = []
    for pattern in patterns:
        if not isinstance(pattern, dict) or pattern.get("color") != "yellow":
            continue
        parent, _leg = find_parent(patterns, pattern)
        if parent is None:
            continue                    # a root occupies no wave of anything
        label = role_label(patterns, pattern)
        role = pattern_role(patterns, pattern)
        direction = pattern_direction(pattern)
        span = pattern_span(pattern)
        if label is None or role is None or direction is None or span is None:
            continue
        rows.append({
            "pattern": pattern,
            "parent_id": parent.get("id"),
            "label": label,
            "role": role,
            "direction": direction,
            "degree": pattern.get("degree"),
            "start": span[0],
        })
    return rows


def _degree_groups(rows):
    """{degree: [row, ...]} with each group in time order.

    The client's "all of one degree" rule, made structural: a pair can only be
    found inside a group, so it can never be forgotten in one of the rules.
    Groups appear in the order their first member does, and time order inside a
    group is what makes a chained triple read left to right.
    """
    groups = {}
    for row in rows:
        groups.setdefault(row["degree"], []).append(row)
    for group in groups.values():
        group.sort(key=lambda row: row["start"])
    return groups


def parallel_pairs(patterns):
    """Same-role sibling waves of one marked parent.

    The client's four rules are: both waves Actionary or both Reactionary; the
    wave between them the opposite; the two in the same direction; all of one
    degree. His own summary of what that adds up to is "the 2 parent patterns
    are within the same even larger degree pattern", and his examples --
    (1)v(3), (1)v(5), (3)v(5), (W)v(Y), (A)v(C), (2)v(4) -- are exactly *every*
    pair of same-role children of one common parent, non-adjacent ones included:
    (1)v(5) has three waves between it, not one.

    So discovery is that, and only that. Rules 2 and 3 need no checking because
    they cannot fail: an Elliott label sequence alternates actionary and
    reactionary, so any two same-role waves of one parent are separated by an
    odd run whose ends are the opposite role, and two waves acting in the same
    parent act in the same direction by the definition of actionary. Re-deriving
    them from the marked children would be worse than redundant -- the wave
    between two children need not itself be marked, so a check would fail on
    incomplete markings rather than on wrong ones. ``gap`` is None for the same
    reason: the wave between them is a leg of the parent, marked or not.

    Rule 4 is automatic after ``settle``: every child sits exactly one degree
    below its parent, so siblings share a degree.

    Two children spanning the *same* leg are two markings of one wave -- a
    duplicate, not a pair -- and are never paired with each other.
    """
    positions = _positions(patterns)
    by_id = {p["id"]: p for p in patterns
             if isinstance(p, dict) and isinstance(p.get("id"), str)} \
        if isinstance(patterns, list) else {}
    index = children_by_leg(patterns)

    pairs = []
    for parent_id, by_leg in index.items():
        parent = by_id.get(parent_id)
        if parent is None:
            continue
        try:
            labels = point_labels(parent.get("pattern_type"), parent.get("variation"))
        except (ValueError, TypeError):
            continue

        # Walked in ascending leg order, so the earlier member of every pair is
        # found first: leg k ends where leg k + 1 begins. A Triple Zigzag's two
        # X legs each yield their own entry, which is why "X, X" -- a pair the
        # client names explicitly -- comes out of the same loop as the rest.
        members = []
        for leg in sorted(by_leg):
            if not 0 <= leg + 1 < len(labels):
                continue
            for child in by_leg[leg]:
                role = pattern_role(patterns, child)
                if role is None or pattern_direction(child) is None:
                    continue
                members.append((leg, labels[leg + 1], role, child))

        for position, (leg_a, label_a, role_a, child_a) in enumerate(members):
            for leg_b, label_b, role_b, child_b in members[position + 1:]:
                if leg_a == leg_b or role_a != role_b:
                    continue
                pairs.append({"a": child_a, "b": child_b,
                              "role_a": label_a, "role_b": label_b,
                              "gap": None})

    return _ordered(pairs, positions)


def inverse_parallel_pairs(patterns):
    """Opposite-role waves across two parents, separated by an actionary wave.

    The client's rules: the two are *not* within the same larger pattern; one is
    Actionary and the other Reactionary; the wave between them is always
    Actionary; both run in the same direction; all three are of one degree.

    Unlike Parallel, none of that follows from one parent's label sequence, so
    every rule is checked. Discovery walks chained triples P1 -> G -> P2 inside
    a degree group: the gap here is a marked pattern in its own right, which is
    what lets its role be asked about at all.

    On a marked bull Impulse chained into its Zigzag correction, with every
    child marked: (4)->(5)->(A) qualifies (reactionary against actionary, gap
    (5) actionary, both legs bearish) and so does (5)->(A)->(B) (actionary
    against reactionary, gap (A) actionary, both bullish), while (3)->(4)->(5)
    does not -- one parent, and (3) and (5) share a role.
    """
    positions = _positions(patterns)

    pairs = []
    for group in _degree_groups(_pairable(patterns)).values():
        for first in group:
            for gap in group:
                if gap is first or gap["role"] != "actionary":
                    continue
                if not chained(first["pattern"], gap["pattern"]):
                    continue
                for second in group:
                    if second is first or second is gap:
                        continue
                    if not chained(gap["pattern"], second["pattern"]):
                        continue
                    # "Not within the same larger pattern": one common parent
                    # would make this a Parallel or an Adjacent question.
                    if first["parent_id"] == second["parent_id"]:
                        continue
                    # Opposite roles as actionary/reactionary, not as labels --
                    # (1) against (3) is two labels but one role.
                    if first["role"] == second["role"]:
                        continue
                    if first["direction"] != second["direction"]:
                        continue
                    pairs.append({"a": first["pattern"], "b": second["pattern"],
                                  "role_a": first["label"], "role_b": second["label"],
                                  "gap": gap["pattern"]})

    return _ordered(pairs, positions)


def adjacent_pairs(patterns):
    """Two actionary waves that meet, running opposite ways.

    The client's rules: both Actionary; no gap, directly adjacent; opposite
    directions; same degree; not within the same larger pattern. The canonical
    case is wave (5) of an impulse against wave (A) of the correction that
    follows it -- the pattern ends and the next one starts on the same pivot.

    The different-parent check is belt and braces: two actionary waves are never
    adjacent inside one parent, because label sequences alternate roles. It is
    kept because it is his stated rule, and a rule that happens to be implied is
    still the rule that must survive a change to the label sets.
    """
    positions = _positions(patterns)

    pairs = []
    for group in _degree_groups(_pairable(patterns)).values():
        for first in group:
            if first["role"] != "actionary":
                continue
            for second in group:
                if second is first or second["role"] != "actionary":
                    continue
                if not chained(first["pattern"], second["pattern"]):
                    continue
                if first["direction"] == second["direction"]:
                    continue
                if first["parent_id"] == second["parent_id"]:
                    continue
                pairs.append({"a": first["pattern"], "b": second["pattern"],
                              "role_a": first["label"], "role_b": second["label"],
                              "gap": None})

    return _ordered(pairs, positions)


# ------------------------------------------------------------- the pair study
#
# What the three discoveries above are for: one comparison counted across every
# pair of parent patterns they find. The client's own examples -- "Wave 1-5 of
# Parent wave (1) will be compared to Wave 1-5 of Parent Wave (3)", "Wave A-C of
# parent wave (2) will be compared to Wave W-Y of Parent wave (4)" -- say two
# things that shape the whole engine: each side carries its own filters (the
# two sides can be different pattern types), and a side names the wave he wants
# out of a *parent* pattern the discovery found.
#
# Three decisions worth stating, because none of them is forced:
#
# * **Side A is always the earlier pattern**, as discovery reports it. Asking
#   "(3) against (1)" is expressed with the parent-wave filters and swapped
#   waves, not by reordering a pair, so a pair is never counted twice under two
#   spellings of one question.
# * **Parent-wave filters are optional and pool by default.** Every discovered
#   pair counts unless he narrows it, so the headline number answers "across
#   everything I have marked" -- which is the question the feature exists for --
#   and the narrow question is one dropdown away.
# * **Nothing climbs up.** The junior side reads from the pair member's
#   children; there is no +1 that walks to a parent, because upward from a pair
#   member is the shared parent the pair was found under, and comparing a wave
#   with its own container is Inter-Pattern's territory.

PAIR_ANALYSES = {
    "parallel": parallel_pairs,
    "inverse_parallel": inverse_parallel_pairs,
    "adjacent": adjacent_pairs,
}

RELATIVE_DEGREES = (-1, 0, 1)


def pair_context(patterns, analysis=None):
    """Everything a pair study reads, built once for a pattern list.

    The name Phase 22 gave ``study_context`` while only the pair studies needed
    one; the inter-pattern study and the grid read the same structure, so there
    is one builder and this is its pair-shaped spelling. ``analysis`` builds one
    discovery; None builds all three.
    """
    return study_context(patterns, analysis)


def _read_parent_wave(spec, key):
    """One side's parent-wave filter: a label, or None for "any"."""
    label = spec.get(key)
    if label is None:
        return None
    if label not in PARENT_WAVE_LABELS:
        raise ValueError(f"Not a parent wave: {label!r}")
    return label


def _read_pair_spec(spec):
    """The pair spec unpacked and checked, or a ValueError naming what is wrong.

    Same refusal to guess as ``_read_spec``: a half-understood question answered
    with a percentage is worse than no percentage.
    """
    if not isinstance(spec, dict):
        raise ValueError("A study spec must be a dict")

    analysis = spec.get("analysis")
    if analysis not in PAIR_ANALYSES:
        raise ValueError(f"Unknown analysis: {analysis!r}")

    type_a, variation_a = _read_pattern_filter(spec, "type_a", "variation_a")
    type_b, variation_b = _read_pattern_filter(spec, "type_b", "variation_b")

    relative_degree = spec.get("relative_degree", 0)
    if (isinstance(relative_degree, bool) or not isinstance(relative_degree, int)
            or relative_degree not in RELATIVE_DEGREES):
        raise ValueError("relative_degree must be -1, 0 or +1, not "
                         f"{relative_degree!r}")

    field_a, field_b = _read_fields(spec)
    return {"analysis": analysis,
            "type_a": type_a, "variation_a": variation_a,
            "wave_a": _read_wave(spec, "wave_a"),
            "parent_wave_a": _read_parent_wave(spec, "parent_wave_a"),
            "type_b": type_b, "variation_b": variation_b,
            "wave_b": _read_wave(spec, "wave_b"),
            "parent_wave_b": _read_parent_wave(spec, "parent_wave_b"),
            "relative_degree": relative_degree,
            "field_a": field_a, "field_b": field_b,
            "operator": _read_operator(spec)}


def _sources(pattern, shifted, children, pattern_type, variation):
    """The pattern(s) one side's wave is actually read from.

    Unshifted, that is the pair member itself -- when it passes the side's own
    type filter. Shifted a degree junior it is every child of the pair member
    that passes, in leg order: a fan-out, exactly as one parent with several
    children fans out in ``run_study``. The filter follows the wave, so a
    shifted side's filter describes the child and never the member it hangs
    off, which is the only reading under which "wave 1 of the Zigzag inside
    parent wave (2)" means what it says.
    """
    if not shifted:
        return [pattern] if _matches(pattern, pattern_type, variation) else []
    by_leg = children.get(pattern.get("id")) or {}
    return [child for leg in sorted(by_leg) for child in by_leg[leg]
            if _matches(child, pattern_type, variation)]


def run_pair_study(patterns, spec, context=None):
    """One comparison counted across every discovered pair of parent patterns.

    {"pairs": int, "samples": int, "true": int, "false": int, "ties": int,
     "pct_true": float, "pct_false": float, "skipped": {reason: count}}

    ``pairs`` is the raw discovery count, before a single filter: a percentage
    counted over two of eighteen pairs is a different claim from one counted
    over all eighteen, and the denominator has to be visible for either to be
    read honestly.

    ``context`` is what ``pair_context`` built, for a caller running many specs
    against one discovery; None builds it here for this one spec.
    """
    read = _read_pair_spec(spec)
    if context is None:
        context = pair_context(patterns, read["analysis"])
    pair_count, samples, skipped = _pair_samples(read, context)

    result = {"pairs": pair_count}
    result.update(_tally(samples, read["field_a"], read["field_b"],
                         read["operator"], skipped))
    return result


def _pair_samples(read, context):
    """Every pair of values a pair spec compares, and what it dropped.

    ``run_pair_study`` minus the comparison: the pair filters, the degree shift
    and the eligibility check, none of which depends on the value fields or the
    operator. Returns ``(pairs discovered, samples, skipped)``.
    """
    pairs = context["pairs"][read["analysis"]]
    children = context["children"]
    eligible = context["eligible"]
    relative_degree = read["relative_degree"]

    skipped, skip = _skipper()
    samples = []

    for pair in pairs:
        # The parent-wave filters first: they are about the pair as discovered,
        # so a pair he excluded never reaches a wave lookup and never inflates
        # the wave-level skip counts with waves he did not ask about.
        if read["parent_wave_a"] is not None and pair["role_a"] != read["parent_wave_a"]:
            skip(SKIP_PAIR_FILTERED)
            continue
        if read["parent_wave_b"] is not None and pair["role_b"] != read["parent_wave_b"]:
            skip(SKIP_PAIR_FILTERED)
            continue

        sources_a = _sources(pair["a"], relative_degree == -1, children,
                             read["type_a"], read["variation_a"])
        sources_b = _sources(pair["b"], relative_degree == 1, children,
                             read["type_b"], read["variation_b"])
        if not sources_a:
            # An unshifted side found nothing means the pair member is not the
            # type he asked for; a shifted one means there is no marking at that
            # degree to read from. Two different things to fix, so two reasons.
            skip(SKIP_NO_DEGREE if relative_degree == -1 else SKIP_PAIR_FILTERED)
            continue
        if not sources_b:
            skip(SKIP_NO_DEGREE if relative_degree == 1 else SKIP_PAIR_FILTERED)
            continue

        for source_a in sources_a:
            for source_b in sources_b:
                found = _wave_pair(source_a, read["wave_a"], source_b,
                                   read["wave_b"], eligible, skip)
                if found is not None:
                    samples.append(found)

    return len(pairs), samples, skipped


# ----------------------------------------------------------------- the grid
#
# What the manual forms above are for one question, the grid is for the whole
# space of them. The client named three factors -- the operator, the value pair
# and the relative degree -- and asked for every combination of them, per
# analysis, four grids rather than one: "They serve 4 different purposes so I do
# not want a single Grid Search combining all 4."
#
# Two decisions shape it:
#
# * **The wave labels are a fourth axis, but only when he asks.** A grid over
#   fixed waves is his three factors and nothing else, 162 rows; ``ALL_WAVES``
#   on a side sweeps that side's labels too, which is 25x as many rows on an
#   impulse against an impulse. Both are one call, and which he wants is a
#   dropdown rather than a guess.
# * **Nothing dies mid-sweep.** The manual forms refuse a spec they cannot
#   answer, which is right for one question asked deliberately; a sweep that
#   raised on combination 40 of 162 would lose the 39 answers it already had. So
#   an impossible combination comes back as a zero row carrying a ``note``, and
#   the sweep finishes.

INTER = "inter"

# The four analyses a grid may sweep, in the order the tab draws them.
GRID_ANALYSES = (INTER,) + tuple(PAIR_ANALYSES)

# "Sweep this side's wave labels too", in place of one of them. Not a label any
# pattern type names, so it can never collide with a real selection.
ALL_WAVES = "*"

# His own order: "+1, same, and -1", wave A relative to wave B.
GRID_DEGREES = (1, 0, -1)

# Why a combination could not be evaluated at all. One case exists today: at the
# same degree the two waves come from one pattern, so two different side filters
# describe a pattern that cannot exist -- the manual form refuses it outright.
NOTE_ONE_PATTERN = "same degree: both sides are one pattern, so the two "\
                   "pattern filters cannot differ"


def value_pairs():
    """The ordered value pairs a grid sweeps: 3x3 CMB, then 3x3 RSI. Eighteen.

    Ordered, so both "Origin CMB > Peak CMB" and "Peak CMB > Origin CMB" are
    swept: under one operator those are two different claims, and the client
    named the operator and the value pair as two separate factors. Never across
    families, which is his own rule -- a CMB is only compared with a CMB.

    A field the family table cannot place is left out rather than paired with
    itself: a percentage over a value with no pane to read it off is not a
    smaller truth.
    """
    families = {}
    for field in LEG_VALUE_FIELDS:
        family = field_family(field)
        if family is not None:
            families.setdefault(family, []).append(field)
    return [(field_a, field_b) for fields in families.values()
            for field_a in fields for field_b in fields]


GRID_VALUE_PAIRS = tuple(value_pairs())

# 3 x 18 x 3. The number the tab quotes back before he presses Run, and the
# multiplier on every wave label a side sweeps.
GRID_COMBINATIONS = len(OPERATORS) * len(GRID_VALUE_PAIRS) * len(GRID_DEGREES)


def _read_grid_waves(grid, key, pattern_type, variation):
    """One side's wave labels: the whole selectable set, or the one he named."""
    if grid.get(key) == ALL_WAVES:
        return wave_labels(pattern_type, variation)
    return [_read_wave(grid, key)]


def _read_grid(grid):
    """The grid spec unpacked and checked, or a ValueError naming what is wrong.

    The same refusal to guess as the manual specs, applied to what the grid does
    *not* enumerate. What it does enumerate is fixed and needs no reading: the
    operators, the eighteen value pairs and the three relative degrees are the
    grid, not settings on it.
    """
    if not isinstance(grid, dict):
        raise ValueError("A grid spec must be a dict")

    analysis = grid.get("analysis")
    if analysis not in GRID_ANALYSES:
        raise ValueError(f"Unknown analysis: {analysis!r}")

    type_a, variation_a = _read_pattern_filter(grid, "type_a", "variation_a")
    # Side B defaults to side A, as it does everywhere here: a grid that never
    # mentions it is the like-with-like sweep it reads as.
    type_b, variation_b = _read_pattern_filter(grid, "type_b", "variation_b",
                                               type_a, variation_a)

    return {"analysis": analysis,
            "type_a": type_a, "variation_a": variation_a,
            "type_b": type_b, "variation_b": variation_b,
            "parent_wave_a": _read_parent_wave(grid, "parent_wave_a"),
            "parent_wave_b": _read_parent_wave(grid, "parent_wave_b"),
            "waves_a": _read_grid_waves(grid, "wave_a", type_a, variation_a),
            "waves_b": _read_grid_waves(grid, "wave_b", type_b, variation_b)}


def _grid_samples(read, wave_a, wave_b, relative_degree, context):
    """(note, pairs, samples, skipped) for one wave/wave/degree of the sweep.

    Where the grid's three factors are *not*. The samples a combination compares
    depend on the filters, the two wave labels and the degree alone, so the
    fifty-four operator and value-pair combinations that share them are counted
    off this one evaluation rather than fifty-four repeats of it.
    """
    if read["analysis"] != INTER:
        spec = {"analysis": read["analysis"],
                "type_a": read["type_a"], "variation_a": read["variation_a"],
                "parent_wave_a": read["parent_wave_a"], "wave_a": wave_a,
                "type_b": read["type_b"], "variation_b": read["variation_b"],
                "parent_wave_b": read["parent_wave_b"], "wave_b": wave_b,
                "relative_degree": relative_degree}
        pair_count, samples, skipped = _pair_samples(spec, context)
        return None, pair_count, samples, skipped

    if relative_degree == 0 and ((read["type_b"], read["variation_b"])
                                 != (read["type_a"], read["variation_a"])):
        # The manual form raises here, and it is right to: he asked one question
        # and it has no answer. A sweep says so on the row and carries on.
        return NOTE_ONE_PATTERN, None, [], {}

    # The client's one knob, wave A relative to wave B, in this engine's older
    # two-offset spelling: +1 makes wave A's pattern the parent of wave B's.
    spec = {"pattern_type": read["type_a"], "variation": read["variation_a"],
            "pattern_type_b": read["type_b"], "variation_b": read["variation_b"],
            "wave_a": wave_a, "offset_a": relative_degree,
            "wave_b": wave_b, "offset_b": 0}
    samples, skipped = _study_samples(spec, context)
    return None, None, samples, skipped


def run_grid(patterns, grid):
    """Every factor combination of one analysis, as a list of row dicts.

    Each row is one manual study's answer -- the same counts, from the same
    evaluation core -- plus the combination that produced it:

        {"wave_a", "wave_b", "field_a", "field_b", "operator",
         "relative_degree", "pairs" (pair analyses only), "samples", "true",
         "false", "ties", "pct_true", "pct_false", "skipped", "note"}

    ``note`` is None for a row that was counted and a short reason for one that
    could not be. Rows come back in enumeration order -- waves, then degree,
    then value pair, then operator -- and zero-sample rows come back with the
    rest: which of them are worth looking at is the tab's question, not this
    one's.

    On the pair analyses ``samples`` may legitimately exceed ``pairs``: a side
    shifted a degree fans out to every matching child, so one pair can yield
    several samples.
    """
    read = _read_grid(grid)
    context = study_context(patterns, read["analysis"])

    rows = []
    for wave_a in read["waves_a"]:
        for wave_b in read["waves_b"]:
            for relative_degree in GRID_DEGREES:
                note, pair_count, samples, skipped = _grid_samples(
                    read, wave_a, wave_b, relative_degree, context)
                for field_a, field_b in GRID_VALUE_PAIRS:
                    for operator in OPERATORS:
                        row = {"wave_a": wave_a, "wave_b": wave_b,
                               "field_a": field_a, "field_b": field_b,
                               "operator": operator,
                               "relative_degree": relative_degree}
                        if pair_count is not None:
                            row["pairs"] = pair_count
                        row.update(_tally(samples, field_a, field_b, operator,
                                          skipped))
                        row["note"] = note
                        rows.append(row)
    return rows

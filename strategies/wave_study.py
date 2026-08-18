"""
The Wave Analysis study: one comparison, counted across every marked pattern.

This is what everything since Phase 16 was built to serve. The client marks
counts, measures six values on each wave he cares about, and then wants to ask
one question of the whole collection -- "how often is wave 4's CMB lower than
wave 2's?" -- and be given a percentage.

Pure: no Streamlit, no I/O, no clock. A study over the same list twice gives the
identical result, which is the least a number he intends to trade on can offer.
"""

from config.wave_analysis import (LEG_VALUE_FIELDS, PATTERN_DEFS, analysable_waves,
                                  find_parent, point_labels, settle)

OPERATORS = (">", "<")

# The offsets a side may carry. +1 means "the pattern one degree senior", which
# in this codebase is only ever the parent -- see the note on pairing below.
OFFSETS = (-1, 0, 1)

# Why a candidate did not become a sample. The strings are the report: a
# percentage with no denominator context is worse than no percentage, and the
# client needs to read "63 waves not yet measured" without a lookup table.
SKIP_NOT_MEASURED = "waves not yet measured"
SKIP_RED = "patterns in conflict (red)"
SKIP_AMBIGUOUS = "ambiguous wave labels"
SKIP_LABEL_ABSENT = "patterns that have no such wave"
SKIP_NO_COUNTERPART = "patterns with no matching counterpart"

# Reported in this order however they were counted, so the same input always
# renders the same line.
SKIP_ORDER = (SKIP_NOT_MEASURED, SKIP_RED, SKIP_AMBIGUOUS, SKIP_LABEL_ABSENT,
              SKIP_NO_COUNTERPART)


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


def _read_spec(spec):
    """The spec unpacked and checked, or a ValueError naming what is wrong.

    Every field is rejected rather than defaulted. A study is a claim about the
    client's markings; guessing at half a question and answering the other half
    would hand him a percentage that is not an answer to anything he asked.
    """
    if not isinstance(spec, dict):
        raise ValueError("A study spec must be a dict")

    pattern_type = spec.get("pattern_type")
    if pattern_type not in PATTERN_DEFS:
        raise ValueError(f"Unknown pattern type: {pattern_type!r}")

    variation = spec.get("variation")
    if variation is not None and variation not in _variation_names(pattern_type):
        raise ValueError(f"Unknown variation {variation!r} for {pattern_type!r}")

    wave_a, wave_b = spec.get("wave_a"), spec.get("wave_b")
    for wave in (wave_a, wave_b):
        if not isinstance(wave, str) or not wave:
            raise ValueError(f"Not a wave label: {wave!r}")

    field = spec.get("field")
    if field not in LEG_VALUE_FIELDS:
        raise ValueError(f"Not a measured value: {field!r}")

    operator = spec.get("operator")
    if operator not in OPERATORS:
        raise ValueError(f"Not a comparison: {operator!r}")

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

    return (pattern_type, variation, wave_a, wave_b, field, operator,
            offset_a, offset_b)


def _matches(pattern, pattern_type, variation):
    """True when a pattern is inside the study's type (and variation) filter."""
    if not isinstance(pattern, dict):
        return False
    if pattern.get("pattern_type") != pattern_type:
        return False
    return variation is None or pattern.get("variation") == variation


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


def run_study(patterns, spec):
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

    Both patterns of a pair must match the type filter (and the variation filter
    when one is given), so a cross-degree study always compares like with like.

    ``>`` and ``<`` are strict. The client has not defined ``=`` -- "perfectly
    equal values is near impossible" -- so ties count as false, and are reported
    separately under ``ties`` so he can see whether they exist at all before
    deciding what ``=`` should mean.
    """
    (pattern_type, variation, wave_a, wave_b, field, operator,
     offset_a, offset_b) = _read_spec(spec)

    # Settled once, here, and everything downstream reads this list: colours and
    # degrees are outputs, and a study run on an unsettled list could count a
    # wave that the very same markings would paint red.
    settled = settle(patterns if isinstance(patterns, list) else [])
    eligible = {(row["pattern_id"], row["leg"]): row
                for row in analysable_waves(settled)}

    skipped = {}

    def skip(reason, count=1):
        if count:
            skipped[reason] = skipped.get(reason, 0) + count

    # Which side of the pair the pattern being walked plays. In a cross-degree
    # study the walk is always driven from the *child*, because find_parent
    # answers "who is above me" and gives exactly one answer -- a parent may
    # expose several legs and so have several children.
    a_is_parent = offset_a == 1 or offset_b == -1

    samples = true_count = false_count = ties = 0

    for pattern in settled:
        if not _matches(pattern, pattern_type, variation):
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
            parent, _leg = find_parent(settled, pattern)
            if parent is None or not _matches(parent, pattern_type, variation):
                skip(SKIP_NO_COUNTERPART)
                continue
            pattern_a, pattern_b = ((parent, pattern) if a_is_parent
                                    else (pattern, parent))

        legs_a = _legs_labelled(pattern_a, wave_a)
        legs_b = _legs_labelled(pattern_b, wave_b)
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
            continue

        key_a = (pattern_a.get("id"), legs_a[0])
        key_b = (pattern_b.get("id"), legs_b[0])
        row_a, row_b = eligible.get(key_a), eligible.get(key_b)
        if row_a is None or row_b is None:
            # Counted per wave, and the same wave asked for twice is one wave.
            missing = (1 if row_a is None else 0)
            if key_b != key_a and row_b is None:
                missing += 1
            skip(SKIP_NOT_MEASURED, missing)
            continue

        value_a = row_a["values"][field]
        value_b = row_b["values"][field]
        samples += 1
        if value_a == value_b:
            ties += 1
            false_count += 1
        elif (value_a > value_b) if operator == ">" else (value_a < value_b):
            true_count += 1
        else:
            false_count += 1

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

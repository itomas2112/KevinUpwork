"""
Elliott Wave pattern and degree definitions for the Wave Analysis tab.

Single source of truth: consumed by the Python tab and handed to the chart
frontend verbatim through the component args (tuples arrive as JSON arrays).
"""

# Elliott Wave pattern hierarchy. label_seq excludes point 0 (always the first click).
PATTERN_DEFS = {
    "Impulse":     [("Impulse", ["1", "2", "3", "4", "5"]), ("Extended Impulse", ["1", "2", "3", "4", "5"])],
    "Diagonal":    [("Leading Diagonal", ["1", "2", "3", "4", "5"]), ("Ending Diagonal", ["1", "2", "3", "4", "5"])],
    "Zigzag":      [("Zigzag", ["A", "B", "C"]), ("Double Zigzag", ["W", "X", "Y"]), ("Triple Zigzag", ["W", "X", "Y", "X", "Z"])],
    "Flat":        [("Flat", ["A", "B", "C"]), ("Expanded Flat", ["A", "B", "C"]), ("Running Flat", ["A", "B", "C"])],
    "Triangle":    [("Contracting Triangle", ["A", "B", "C", "D", "E"]), ("Running Triangle", ["A", "B", "C", "D", "E"]), ("Expanding Triangle", ["A", "B", "C", "D", "E"])],
    "Combination": [("Double Three", ["W", "X", "Y"]), ("Triple Three", ["W", "X", "Y", "X", "Z"])],
}

# 18 degrees, most senior first. Index = seniority rank; Ctrl+/- (Phase 3) moves through this list.
# (name, letter_style, numeral_style, decoration, font_px)
# letter_style: "upper_sans" | "lower_serif"; numeral_style: "arabic" | "roman_upper" | "roman_lower"
# decoration: "circle" | "parens" | "plain"
DEGREES = [
    ("Supermillennium",  "upper_sans",  "arabic",      "circle", 22),
    ("Millennium",       "upper_sans",  "arabic",      "parens", 22),
    ("Submillennium",    "upper_sans",  "arabic",      "plain",  22),
    ("Grand Supercycle", "lower_serif", "roman_upper", "circle", 19),
    ("Supercycle",       "lower_serif", "roman_upper", "parens", 19),
    ("Cycle",            "lower_serif", "roman_upper", "plain",  19),
    ("Primary",          "upper_sans",  "arabic",      "circle", 17),
    ("Intermediate",     "upper_sans",  "arabic",      "parens", 17),
    ("Minor",            "upper_sans",  "arabic",      "plain",  17),
    ("Minute",           "lower_serif", "roman_lower", "circle", 13),
    ("Minuette",         "lower_serif", "roman_lower", "parens", 13),
    ("Subminuette",      "lower_serif", "roman_lower", "plain",  13),
    ("Micro",            "upper_sans",  "arabic",      "circle", 11),
    ("Submicro",         "upper_sans",  "arabic",      "parens", 11),
    ("Miniscule",        "upper_sans",  "arabic",      "plain",  11),
    ("Nano",             "lower_serif", "roman_lower", "circle", 9),
    ("Subnano",          "lower_serif", "roman_lower", "parens", 9),
    ("Pico",             "lower_serif", "roman_lower", "plain",  9),
]
DEFAULT_DEGREE = "Subminuette"

DEGREE_NAMES = [d[0] for d in DEGREES]
DEGREE_BY_NAME = {d[0]: d for d in DEGREES}

# Pattern colours. Phase 4 uses "red" for same-degree overlap violations.
PATTERN_COLORS = ("yellow", "red")

POINT_KINDS = ("high", "low")

_ROMAN = {"1": "I", "2": "II", "3": "III", "4": "IV", "5": "V"}
_LETTERS = set("ABCDEWXYZ")


def point_labels(pattern_type, variation):
    """Labels for every click of a pattern: point 0 (origin) followed by label_seq."""
    variations = PATTERN_DEFS.get(pattern_type)
    if variations is None:
        raise ValueError(f"Unknown pattern type: {pattern_type!r}")
    for name, label_seq in variations:
        if name == variation:
            return ["0"] + list(label_seq)
    raise ValueError(f"Unknown variation {variation!r} for pattern type {pattern_type!r}")


def render_glyph(label, letter_style, numeral_style):
    """The glyph a point label displays at a given degree's typography.

    Decoration (parens / circle) is applied at draw time, not here.
    """
    if label == "0":
        return "0"

    if label in _ROMAN:
        if numeral_style == "arabic":
            return label
        if numeral_style == "roman_upper":
            return _ROMAN[label]
        if numeral_style == "roman_lower":
            return _ROMAN[label].lower()
        raise ValueError(f"Unknown numeral style: {numeral_style!r}")

    if label.upper() in _LETTERS:
        if letter_style == "upper_sans":
            return label.upper()
        if letter_style == "lower_serif":
            return label.lower()
        raise ValueError(f"Unknown letter style: {letter_style!r}")

    raise ValueError(f"Unknown point label: {label!r}")


def degree_index(name):
    """Seniority rank of a degree (0 = most senior). -1 when unknown."""
    try:
        return DEGREE_NAMES.index(name)
    except ValueError:
        return -1


def wave_defs():
    """Definitions in the plain-list shape the frontend consumes."""
    return {
        "pattern_defs": {
            ptype: [[name, list(label_seq)] for name, label_seq in variations]
            for ptype, variations in PATTERN_DEFS.items()
        },
        "degrees": [list(d) for d in DEGREES],
        "default_degree": DEFAULT_DEGREE,
    }


def _is_number(value):
    return isinstance(value, (int, float)) and not isinstance(value, bool)


def _is_int(value):
    return isinstance(value, int) and not isinstance(value, bool)


def is_valid_pattern(pattern):
    """True when a pattern dict from the frontend is structurally sound."""
    if not isinstance(pattern, dict):
        return False
    if not isinstance(pattern.get("id"), str) or not pattern["id"]:
        return False
    try:
        labels = point_labels(pattern.get("pattern_type"), pattern.get("variation"))
    except (ValueError, TypeError):
        return False
    if pattern.get("degree") not in DEGREE_BY_NAME:
        return False
    if pattern.get("color") not in PATTERN_COLORS:
        return False

    points = pattern.get("points")
    if not isinstance(points, list) or len(points) != len(labels):
        return False

    previous_time = None
    for point in points:
        if not isinstance(point, dict):
            return False
        # JSON hands back an int for a round price, so any number is accepted.
        if not _is_int(point.get("time")) or not _is_number(point.get("price")):
            return False
        if point.get("kind") not in POINT_KINDS:
            return False
        if previous_time is not None and point["time"] <= previous_time:
            return False
        previous_time = point["time"]

    return True


# ------------------------------------------------------------------ containment
#
# Two patterns belong to the same "degree component" when one is nested inside
# the other. A degree change made on any member cascades through the whole
# component so a parent can never end up junior to its own child.


def pattern_span(pattern):
    """(earliest time, latest time) of a pattern's points, or None if unusable."""
    if not isinstance(pattern, dict):
        return None
    points = pattern.get("points")
    if not isinstance(points, list) or not points:
        return None
    times = []
    for point in points:
        if not isinstance(point, dict) or not _is_int(point.get("time")):
            return None
        times.append(point["time"])
    return (min(times), max(times))


def related(pattern_a, pattern_b):
    """True when one pattern's span contains the other's (either direction).

    Containment is inclusive, so a child that starts and ends exactly on its
    parent's termini still counts. Merely *sharing* an endpoint does not:
    adjacent siblings (the next pattern's point 0 sitting on the previous
    pattern's last point) stay independent because neither span contains
    the other.
    """
    span_a = pattern_span(pattern_a)
    span_b = pattern_span(pattern_b)
    if span_a is None or span_b is None:
        return False
    contains_ab = span_a[0] <= span_b[0] and span_b[1] <= span_a[1]
    contains_ba = span_b[0] <= span_a[0] and span_a[1] <= span_b[1]
    return contains_ab or contains_ba


def degree_component(patterns, pattern_id):
    """Ids of every pattern transitively nested with ``pattern_id``.

    The containment relation is not transitive on its own (A ⊃ B and B ⊃ C
    happen to imply A ⊃ C, but only because spans are intervals), so the
    component is grown by BFS rather than by a single containment pass. The
    result includes ``pattern_id`` itself and follows the input list's order;
    an unknown id yields an empty list.
    """
    if not isinstance(patterns, list):
        return []
    usable = [p for p in patterns
              if isinstance(p, dict) and isinstance(p.get("id"), str) and p["id"]]

    start = next((p for p in usable if p["id"] == pattern_id), None)
    if start is None:
        return []

    seen = {start["id"]}
    queue = [start]
    while queue:
        current = queue.pop()
        for candidate in usable:
            if candidate["id"] in seen:
                continue
            if related(current, candidate):
                seen.add(candidate["id"])
                queue.append(candidate)

    return [p["id"] for p in usable if p["id"] in seen]


# ------------------------------------------------------------------ validation


def validate_patterns(patterns):
    """Recolour every pattern: "red" when it overlaps a same-degree neighbour.

    The client's rule is "two notations of the same degree in the same location
    is invalid". Chaining -- the next pattern's point 0 sitting exactly on the
    previous pattern's terminal point -- is his core workflow and must stay
    legal, so a violation needs a strictly non-empty *interior* intersection:
    ``max(a0, b0) < min(a1, b1)``. Containment counts (the inner span's whole
    interior is shared); a bare shared endpoint does not. Different degrees
    never conflict.

    Violations are pairwise but the colour is per pattern: anything caught in
    at least one violating pair goes red, everything else goes back to yellow,
    so resolving an overlap heals the survivors on the next pass.

    Pure: returns a new list and never touches the patterns it was given.
    """
    if not isinstance(patterns, list):
        return []

    by_degree = {}
    for index, pattern in enumerate(patterns):
        span = pattern_span(pattern)
        if span is None:
            continue                    # unusable span -- cannot violate anything
        by_degree.setdefault(pattern.get("degree"), []).append((span, index))

    violating = set()
    for entries in by_degree.values():
        entries.sort(key=lambda entry: entry[0])
        for position, (span_a, index_a) in enumerate(entries):
            for span_b, index_b in entries[position + 1:]:
                # Sorted by start, so once a span begins on or after a's end,
                # no later span can reach back into a either.
                if span_b[0] >= span_a[1]:
                    break
                if max(span_a[0], span_b[0]) < min(span_a[1], span_b[1]):
                    violating.add(index_a)
                    violating.add(index_b)

    recoloured = []
    for index, pattern in enumerate(patterns):
        if not isinstance(pattern, dict):
            recoloured.append(pattern)
            continue
        color = "red" if index in violating else "yellow"
        recoloured.append(pattern if pattern.get("color") == color
                          else dict(pattern, color=color))
    return recoloured


# ------------------------------------------------------------------- reducers


def _timeframe_list(patterns_by_tf, timeframe):
    existing = patterns_by_tf.get(timeframe)
    return existing if isinstance(existing, list) else []


def _with_list(patterns_by_tf, timeframe, new_list):
    updated = dict(patterns_by_tf)
    updated[timeframe] = new_list
    return updated


def _find(pattern_list, pattern_id):
    for index, pattern in enumerate(pattern_list):
        if isinstance(pattern, dict) and pattern.get("id") == pattern_id:
            return index, pattern
    return -1, None


def _apply_pattern_completed(patterns_by_tf, timeframe, event):
    pattern = event.get("pattern")
    if not is_valid_pattern(pattern):
        return patterns_by_tf

    existing = _timeframe_list(patterns_by_tf, timeframe)
    if any(isinstance(p, dict) and p.get("id") == pattern["id"] for p in existing):
        return patterns_by_tf

    return _with_list(patterns_by_tf, timeframe, list(existing) + [pattern])


def _apply_delete_pattern(patterns_by_tf, timeframe, event):
    pattern_id = event.get("id")
    if not isinstance(pattern_id, str) or not pattern_id:
        return patterns_by_tf

    existing = _timeframe_list(patterns_by_tf, timeframe)
    remaining = [p for p in existing
                 if not (isinstance(p, dict) and p.get("id") == pattern_id)]
    if len(remaining) == len(existing):
        return patterns_by_tf                       # unknown id -- no-op

    return _with_list(patterns_by_tf, timeframe, remaining)


def _apply_move_point(patterns_by_tf, timeframe, event):
    pattern_id = event.get("id")
    index = event.get("point_index")
    time = event.get("time")
    price = event.get("price")
    kind = event.get("kind")

    if not isinstance(pattern_id, str) or not _is_int(index) or not _is_int(time):
        return patterns_by_tf
    if not _is_number(price) or kind not in POINT_KINDS:
        return patterns_by_tf

    existing = _timeframe_list(patterns_by_tf, timeframe)
    position, pattern = _find(existing, pattern_id)
    if pattern is None:
        return patterns_by_tf

    points = pattern.get("points")
    if not isinstance(points, list) or not (0 <= index < len(points)):
        return patterns_by_tf

    # The moved point must stay strictly between its neighbours; the first and
    # last points are bounded on one side only.
    if index > 0 and not _is_int(points[index - 1].get("time")):
        return patterns_by_tf
    if index > 0 and time <= points[index - 1]["time"]:
        return patterns_by_tf
    if index < len(points) - 1 and not _is_int(points[index + 1].get("time")):
        return patterns_by_tf
    if index < len(points) - 1 and time >= points[index + 1]["time"]:
        return patterns_by_tf

    moved_points = list(points)
    moved_points[index] = {"time": time, "price": price, "kind": kind}
    moved = dict(pattern, points=moved_points)
    if not is_valid_pattern(moved):
        return patterns_by_tf

    updated_list = list(existing)
    updated_list[position] = moved
    return _with_list(patterns_by_tf, timeframe, updated_list)


def _apply_shift_degree(patterns_by_tf, timeframe, event):
    pattern_id = event.get("id")
    delta = event.get("delta")
    if not isinstance(pattern_id, str) or not _is_int(delta) or delta not in (1, -1):
        return patterns_by_tf

    existing = _timeframe_list(patterns_by_tf, timeframe)
    component = set(degree_component(existing, pattern_id))
    if not component:
        return patterns_by_tf

    # All-or-nothing: one member hitting the end of the degree list cancels the
    # whole cascade, so relative seniority inside a nest is never flattened.
    # A positive delta means "more senior", which is a *lower* index.
    shifted = {}
    for pattern in existing:
        if not isinstance(pattern, dict) or pattern.get("id") not in component:
            continue
        target = degree_index(pattern.get("degree")) - delta
        if not (0 <= target < len(DEGREES)):
            return patterns_by_tf
        shifted[pattern["id"]] = DEGREES[target][0]

    if len(shifted) != len(component):
        return patterns_by_tf

    updated_list = [dict(p, degree=shifted[p["id"]])
                    if isinstance(p, dict) and p.get("id") in shifted else p
                    for p in existing]
    return _with_list(patterns_by_tf, timeframe, updated_list)


_EVENT_REDUCERS = {
    "pattern_completed": _apply_pattern_completed,
    "delete_pattern": _apply_delete_pattern,
    "move_point": _apply_move_point,
    "shift_degree": _apply_shift_degree,
}


def apply_wave_event(patterns_by_tf, timeframe, event):
    """Fold a frontend event into the {timeframe: [pattern, ...]} state.

    Pure: returns a new mapping on success and the mapping it was given
    (unchanged) on any malformed input -- a bad event must never break the tab.
    """
    if not isinstance(patterns_by_tf, dict) or not isinstance(event, dict):
        return patterns_by_tf

    reducer = _EVENT_REDUCERS.get(event.get("type"))
    if reducer is None:
        return patterns_by_tf
    return reducer(patterns_by_tf, timeframe, event)

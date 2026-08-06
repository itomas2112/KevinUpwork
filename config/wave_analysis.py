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
# the other. This is the honest name for plain containment and is kept as such,
# but it is no longer what drives degrees: the cascade now runs on the strict
# parent/child relation further down, because merely overlapping another
# pattern's span never made something a child of it.


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


# -------------------------------------------------------- parent/child relation
#
# The client's definition of a child is strict: a child spans exactly one leg of
# its parent -- the origin of wave v until the end of wave v -- endpoint for
# endpoint. Merely sitting *inside* another pattern (which is all ``related``
# above knows about) is not enough, because that also catches a pattern drawn
# across the middle of another one, which is not a wave count at all.
#
# A child never owns its degree: it is always one step junior to its parent, so
# a nest always reads as consecutive degrees. Only yellow patterns take part --
# a red pattern is already flagged as invalid, and deriving degrees from it (or
# for it) would spread the mistake instead of letting the client fix it.


def _points_match(point_a, point_b):
    """Same pivot: same time *and* same kind.

    A bar's high and its low are two different pivots and must never be
    conflated -- a child hanging off the wrong one is not a child at all.
    """
    if not isinstance(point_a, dict) or not isinstance(point_b, dict):
        return False
    if not _is_int(point_a.get("time")) or not _is_int(point_b.get("time")):
        return False
    if point_a.get("kind") not in POINT_KINDS or point_b.get("kind") not in POINT_KINDS:
        return False
    return point_a["time"] == point_b["time"] and point_a["kind"] == point_b["kind"]


def child_leg_index(parent, child):
    """The index k of the parent leg the child spans exactly, or None.

    Endpoints match on both time and kind. A pattern is never its own child.

    Mirrored by ``childLegIndex`` in frontend/main.js, which hides a child's
    origin glyph -- the two must stay in sync.
    """
    if not isinstance(parent, dict) or not isinstance(child, dict):
        return None
    if parent is child:
        return None
    parent_id, child_id = parent.get("id"), child.get("id")
    if isinstance(parent_id, str) and parent_id and parent_id == child_id:
        return None

    parent_points = parent.get("points")
    child_points = child.get("points")
    if not isinstance(parent_points, list) or len(parent_points) < 2:
        return None
    if not isinstance(child_points, list) or len(child_points) < 2:
        return None

    first, last = child_points[0], child_points[-1]
    for k in range(len(parent_points) - 1):
        if (_points_match(first, parent_points[k])
                and _points_match(last, parent_points[k + 1])):
            return k
    return None


def _usable(patterns):
    """The entries of a pattern list that carry a usable id, in input order."""
    if not isinstance(patterns, list):
        return []
    return [p for p in patterns
            if isinstance(p, dict) and isinstance(p.get("id"), str) and p["id"]]


def _span_length(pattern):
    span = pattern_span(pattern)
    return None if span is None else span[1] - span[0]


def find_parent(patterns, child):
    """(parent, leg_index) for the child, or (None, None).

    Only yellow patterns on either side -- a red pattern is in no relation.
    Ties (two patterns exposing the identical leg) resolve to the one with the
    SMALLEST span, then the lowest id. Deliberately not "the most junior degree":
    degrees are being rewritten by reconcile_degrees, so the tie-break must not
    depend on them.
    """
    if not isinstance(child, dict) or child.get("color") != "yellow":
        return (None, None)

    best = None                     # (span length, id, parent, leg index)
    for candidate in _usable(patterns):
        if candidate.get("color") != "yellow":
            continue
        leg = child_leg_index(candidate, child)
        if leg is None:
            continue
        length = _span_length(candidate)
        if length is None:
            continue
        key = (length, candidate["id"])
        if best is None or key < (best[0], best[1]):
            best = (length, candidate["id"], candidate, leg)

    return (None, None) if best is None else (best[2], best[3])


def find_children(patterns, parent):
    """Every pattern whose find_parent resolves to this one. Input order."""
    if not isinstance(parent, dict):
        return []
    parent_id = parent.get("id")
    if not isinstance(parent_id, str) or not parent_id:
        return []

    children = []
    for candidate in _usable(patterns):
        if candidate["id"] == parent_id:
            continue
        found, _leg = find_parent(patterns, candidate)
        if found is not None and found.get("id") == parent_id:
            children.append(candidate)
    return children


def children_by_leg(patterns):
    """{parent id: {leg index: [child, ...]}} for the whole list, in one pass.

    ``find_children`` re-runs ``find_parent`` for every candidate, so calling it
    per pattern per level makes a three-level walk cubic. This builds the whole
    downward index with one ``find_parent`` per pattern instead, exactly as
    ``reconcile_degrees`` already does internally.

    Yellow-only, like the rest of the relation. Children keep the input list's
    order within a leg.

    Every yellow pattern gets a key, so a childless one reads as an empty
    mapping rather than a missing key -- a caller walking a nest downward would
    otherwise have to guard every lookup.
    """
    usable = _usable(patterns)
    index = {p["id"]: {} for p in usable if p.get("color") == "yellow"}
    for candidate in usable:
        parent, leg = find_parent(usable, candidate)
        if parent is None:
            continue
        index.setdefault(parent["id"], {}).setdefault(leg, []).append(candidate)
    return index


def relation_component(patterns, pattern_id):
    """Ids transitively connected to pattern_id through the parent/child
    relation, treated as undirected. Includes pattern_id itself. Yellow-only:
    a red pattern's component is just itself. Follows the input list's order.
    """
    usable = _usable(patterns)
    start = next((p for p in usable if p["id"] == pattern_id), None)
    if start is None:
        return []

    # One pass over the list is enough to build the whole undirected graph, and
    # it keeps find_parent -- which is itself a scan -- off the BFS inner loop.
    edges = {p["id"]: set() for p in usable}
    for candidate in usable:
        parent, _leg = find_parent(usable, candidate)
        if parent is None:
            continue
        edges[candidate["id"]].add(parent["id"])
        edges[parent["id"]].add(candidate["id"])

    seen = {start["id"]}
    queue = [start["id"]]
    while queue:
        current = queue.pop()
        for neighbour in edges.get(current, ()):
            if neighbour not in seen:
                seen.add(neighbour)
                queue.append(neighbour)

    return [p["id"] for p in usable if p["id"] in seen]


def reconcile_degrees(patterns):
    """Every child's degree := its parent's degree + 1. Roots keep theirs.

    Walks roots first so a parent is settled before its children. Pure: returns
    a new list, never mutates the input.

    Clamp: a child of a Pico parent has no junior degree to move to, so it stays
    at Pico. The same-degree overlap check then turns both red, which is the
    honest signal -- the nest is deeper than the degree scale allows.
    """
    if not isinstance(patterns, list):
        return []

    usable = _usable(patterns)
    parent_of = {}
    children_of = {p["id"]: [] for p in usable}
    for candidate in usable:
        parent, _leg = find_parent(usable, candidate)
        if parent is None:
            continue
        parent_of[candidate["id"]] = parent["id"]
        children_of[parent["id"]].append(candidate["id"])

    # A child's span is strictly inside its parent's (every pattern has at least
    # three legs), so spans shrink along every parent->child edge and the
    # relation cannot contain a cycle. ``settled`` is still tracked, so
    # pathological input costs one wasted visit rather than an endless walk.
    degrees = {p["id"]: p.get("degree") for p in usable}
    settled = set()
    queue = [p["id"] for p in usable if p["id"] not in parent_of]
    for pattern_id in queue:
        settled.add(pattern_id)
    while queue:
        current = queue.pop(0)
        parent_rank = degree_index(degrees.get(current))
        for child_id in children_of.get(current, ()):
            if child_id in settled:
                continue
            settled.add(child_id)
            # An unknown parent degree has no rank to step down from; the child
            # keeps whatever it has rather than being flung to the senior end.
            if parent_rank >= 0:
                degrees[child_id] = DEGREES[min(parent_rank + 1, len(DEGREES) - 1)][0]
            queue.append(child_id)

    reconciled = []
    for pattern in patterns:
        if not isinstance(pattern, dict) or pattern.get("id") not in degrees:
            reconciled.append(pattern)
            continue
        degree = degrees[pattern["id"]]
        reconciled.append(pattern if pattern.get("degree") == degree
                          else dict(pattern, degree=degree))
    return reconciled


def settle(patterns):
    """Colours and degrees brought to their fixed point: reconcile, validate.

    Colour is an output, never an input. Every pattern is handed to the
    reconciliation as yellow -- valid until an overlap proves otherwise -- and
    validate_patterns then has the final say on what is actually red.

    Reconciling *before* colours are decided is the whole fix for the client's
    core defect. A child drawn at its parent's own degree overlaps its parent's
    interior, so validating first paints the pair red; the relation is
    yellow-only, so a red pair reconciles to nothing, and the collision that
    the junior degree would have resolved is now the very thing preventing the
    junior degree from being derived. Red is exactly where that pair got stuck
    on his recording.

    Two passes are enough, and the result is a fixed point:
    settle(settle(x)) == settle(x). Reconciling against the whole relation --
    rather than the yellow subgraph -- already gives every child parent+1, so a
    second reconciliation over any subgraph of it has nothing left to change,
    and validate_patterns is idempotent on fixed degrees.
    """
    if not isinstance(patterns, list):
        return []
    presumed = [dict(pattern, color="yellow") if isinstance(pattern, dict) else pattern
                for pattern in patterns]
    return validate_patterns(reconcile_degrees(presumed))


# ------------------------------------------------------------------- reducers


def _find(pattern_list, pattern_id):
    for index, pattern in enumerate(pattern_list):
        if isinstance(pattern, dict) and pattern.get("id") == pattern_id:
            return index, pattern
    return -1, None


def _apply_pattern_completed(patterns, event):
    pattern = event.get("pattern")
    if not is_valid_pattern(pattern):
        return patterns

    if any(isinstance(p, dict) and p.get("id") == pattern["id"] for p in patterns):
        return patterns

    return list(patterns) + [pattern]


def _apply_delete_pattern(patterns, event):
    pattern_id = event.get("id")
    if not isinstance(pattern_id, str) or not pattern_id:
        return patterns

    remaining = [p for p in patterns
                 if not (isinstance(p, dict) and p.get("id") == pattern_id)]
    if len(remaining) == len(patterns):
        return patterns                             # unknown id -- no-op

    return remaining


def _apply_move_point(patterns, event):
    pattern_id = event.get("id")
    index = event.get("point_index")
    time = event.get("time")
    price = event.get("price")
    kind = event.get("kind")

    if not isinstance(pattern_id, str) or not _is_int(index) or not _is_int(time):
        return patterns
    if not _is_number(price) or kind not in POINT_KINDS:
        return patterns

    position, pattern = _find(patterns, pattern_id)
    if pattern is None:
        return patterns

    points = pattern.get("points")
    if not isinstance(points, list) or not (0 <= index < len(points)):
        return patterns

    # The moved point must stay strictly between its neighbours; the first and
    # last points are bounded on one side only.
    if index > 0 and not _is_int(points[index - 1].get("time")):
        return patterns
    if index > 0 and time <= points[index - 1]["time"]:
        return patterns
    if index < len(points) - 1 and not _is_int(points[index + 1].get("time")):
        return patterns
    if index < len(points) - 1 and time >= points[index + 1]["time"]:
        return patterns

    moved_points = list(points)
    moved_points[index] = {"time": time, "price": price, "kind": kind}
    moved = dict(pattern, points=moved_points)
    if not is_valid_pattern(moved):
        return patterns

    updated_list = list(patterns)
    updated_list[position] = moved
    return updated_list


def _apply_shift_degree(patterns, event):
    pattern_id = event.get("id")
    delta = event.get("delta")
    if not isinstance(pattern_id, str) or not _is_int(delta) or delta not in (1, -1):
        return patterns

    # The nest, not everything that happens to overlap: a red pattern's
    # component is itself, which is what lets the client walk it to a free
    # degree by hand to resolve a collision.
    component = set(relation_component(patterns, pattern_id))
    if not component:
        return patterns

    # All-or-nothing: one member hitting the end of the degree list cancels the
    # whole cascade, so relative seniority inside a nest is never flattened.
    # A positive delta means "more senior", which is a *lower* index.
    shifted = {}
    for pattern in patterns:
        if not isinstance(pattern, dict) or pattern.get("id") not in component:
            continue
        target = degree_index(pattern.get("degree")) - delta
        if not (0 <= target < len(DEGREES)):
            return patterns
        shifted[pattern["id"]] = DEGREES[target][0]

    if len(shifted) != len(component):
        return patterns

    return [dict(p, degree=shifted[p["id"]])
            if isinstance(p, dict) and p.get("id") in shifted else p
            for p in patterns]


_EVENT_REDUCERS = {
    "pattern_completed": _apply_pattern_completed,
    "delete_pattern": _apply_delete_pattern,
    "move_point": _apply_move_point,
    "shift_degree": _apply_shift_degree,
}


def apply_wave_event(patterns, event):
    """Fold a frontend event into the canonical [pattern, ...] state.

    One list, not one per aggregation: markings are stored at the base
    timeframe and every aggregation on screen is a projection of that list, so
    there is no timeframe left for a reducer to have an opinion about. Whatever
    the client was looking at when the event fired, ``config.wave_projection``
    has already refined its coordinates down to canonical space by the time it
    gets here.

    Pure: returns a new list on success and the list it was given (unchanged)
    on any malformed input -- a bad event must never break the tab.
    """
    if not isinstance(patterns, list) or not isinstance(event, dict):
        return patterns

    reducer = _EVENT_REDUCERS.get(event.get("type"))
    if reducer is None:
        return patterns
    return reducer(patterns, event)

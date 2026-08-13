"""
Turning Wave Analysis markings into the client's Date Range Manager workbook.

The client hand-maintains the DRM today and wants it generated from the wave
counts he now marks on the chart. A DRM row is ``(sheet, primary, secondary,
period)`` and it takes *three* marked levels to name one:

* a grandparent ``P``, a child ``C`` spanning exactly leg ``k`` of it, and a
  grandchild ``G`` spanning exactly leg ``j`` of ``C``;
* the **primary** is ``P``'s leg ``k`` in parenthesised notation -- ``W.(3)``;
* the **secondary** is ``C``'s leg ``j`` plus the *pattern type of ``G``* --
  ``W.1 Impulse``, ``W.A/W Zigzag``;
* the **period** is the time range of that leg of ``C``;
* the **sheet** is decided by that leg's own price direction, never by the
  primary's name.

Leg indices name waves off by one: leg ``k`` runs from ``points[k]`` to
``points[k + 1]`` and is therefore the wave labelled ``point_labels(...)[k + 1]``.

Pure: no Streamlit, no file I/O, no clock. The only thing that knows about
timeframes is the ``time_format`` callable the tab hands in, which is what keeps
the projection engine out of here.
"""
import io
import math

import pandas as pd

from config.wave_analysis import children_by_leg, point_labels
from data.helpers import PRIMARY_SECONDARY_MAP

# The order every sheet is written in: the enumeration of PRIMARY_SECONDARY_MAP,
# 43 rows, whether or not a row ends up with periods. The client's own file looks
# exactly like that and a shrunken sheet would read as lost work.
DRM_SKELETON = [(primary, secondary)
                for primary, secondaries in PRIMARY_SECONDARY_MAP.items()
                for secondary in secondaries]

SHEET_NAMES = ("Bullish", "Bearish")

# The format both ends of ``data.loader.parse_drm_periods`` speak.
PERIOD_FORMAT = "%d.%m.%Y_%H:%M"

# Only these two of the six pattern types have a word in the DRM's vocabulary.
_SECONDARY_TYPES = ("Impulse", "Zigzag")

# The DRM's type word is binary: motive (Impulse) or corrective (Zigzag). A
# Diagonal is a motive wave carrying the same 1-2-3-4-5 labels as an impulse,
# and the motive primaries (W.(1)/(3)/(5)) offer no secondary but Impulse -- so
# a diagonal in a motive position has exactly one name the format can express.
# Flat, Triangle and Combination get no alias: the format names one corrective
# form, Zigzag, and calling a flat a zigzag would be inventing data.
#
# One line to reverse if the client ever wants his diagonals kept apart.
SECONDARY_TYPE_ALIASES = {"Diagonal": "Impulse"}

# A zigzag's wave A and a double zigzag's wave W are the same structural
# position, and the client's file writes them under one merged name.
_ZIGZAG_ALIASES = {"A": "A/W", "W": "A/W"}


# --------------------------------------------------- what the format tracks
#
# Most unrepresentable candidates are not losses. The DRM never records an
# even impulse leg, a flat's B, or a triangle's D and E -- those are
# counter-trend legs the client deliberately does not log, and reporting them
# as drops makes a correct export read as lossy. Both sets are *derived* from
# PRIMARY_SECONDARY_MAP rather than written out, so a row added to the client's
# format is followed automatically instead of needing a second edit here.


def _primary_positions(mapping):
    """The wave labels the map's primaries name: 'W.(3)' -> '3'."""
    positions = set()
    for primary in mapping:
        if isinstance(primary, str) and primary.startswith("W.(") and primary.endswith(")"):
            positions.add(primary[3:-1])
    return positions


def _secondary_positions(mapping):
    """The wave labels the map's secondaries name: 'W.A/W Zigzag' -> 'A' and 'W'.

    The merged 'A/W' name covers two structural positions and both of them are
    tracked, so it splits rather than being kept as one opaque label.
    """
    positions = set()
    for secondaries in mapping.values():
        for secondary in secondaries:
            if not isinstance(secondary, str) or not secondary.startswith("W."):
                continue
            label, _, pattern_type = secondary[2:].rpartition(" ")
            if not label or not pattern_type:
                continue
            positions.update(part for part in label.split("/") if part)
    return positions


TRACKED_PRIMARY_POSITIONS = _primary_positions(PRIMARY_SECONDARY_MAP)
TRACKED_SECONDARY_POSITIONS = _secondary_positions(PRIMARY_SECONDARY_MAP)


# ------------------------------------------------------------------- naming


def primary_name(label):
    """'W.(3)' for leg label '3', or None when the DRM has no such primary."""
    if not isinstance(label, str) or not label:
        return None
    name = f"W.({label})"
    return name if name in PRIMARY_SECONDARY_MAP else None


def secondary_name(label, pattern_type):
    """The DRM's name for a leg and the pattern forming it, or None.

    Only ``Impulse`` and ``Zigzag`` have DRM names, plus whatever
    ``SECONDARY_TYPE_ALIASES`` folds into them -- a diagonal is logged as an
    impulse. A zigzag on leg 'A' or 'W' is written 'W.A/W Zigzag' -- the
    client's file merges the zigzag's A and the double zigzag's W, which are the
    same structural position.

    A name that comes out of here is still only a *candidate*: 'W.2 Impulse' is
    perfectly well formed and belongs to no primary, so the pair check below has
    the final say.
    """
    pattern_type = SECONDARY_TYPE_ALIASES.get(pattern_type, pattern_type)
    if pattern_type not in _SECONDARY_TYPES:
        return None
    if not isinstance(label, str) or not label:
        return None
    if pattern_type == "Zigzag":
        label = _ZIGZAG_ALIASES.get(label, label)
    return f"W.{label} {pattern_type}"


def _leg_label(pattern, leg):
    """The wave label of ``pattern``'s leg ``leg``, or None.

    The off-by-one lives here and nowhere else: leg ``k`` ends on point ``k+1``,
    and that point's label is the wave's name.
    """
    if not isinstance(pattern, dict) or not isinstance(leg, int):
        return None
    try:
        labels = point_labels(pattern.get("pattern_type"), pattern.get("variation"))
    except (ValueError, TypeError):
        return None
    if not 0 <= leg + 1 < len(labels):
        return None
    return labels[leg + 1]


def _price(point):
    value = point.get("price") if isinstance(point, dict) else None
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return None
    return None if isinstance(value, float) and not math.isfinite(value) else float(value)


# -------------------------------------------------------------- building rows


def _empty_summary():
    return {
        "rows": 0,
        "candidates": 0,
        "red_patterns": 0,
        # A position the DRM never records. Informational, not a loss: the
        # client does not log wave 2, wave 4, a flat's B or a triangle's D/E,
        # and calling those "dropped" makes a correct export read as lossy.
        "not_applicable": {
            "untracked_primary": 0,
            "untracked_secondary": 0,
        },
        # Something the DRM *could* have recorded and did not.
        "dropped": {
            "no_primary": 0,
            "no_secondary": {},
            "unknown_pair": 0,
            "flat_leg": 0,
            "unmappable_time": 0,
            "degenerate_period": 0,
            "ambiguous_leg": 0,
            "duplicate": 0,
        },
    }


def _bucket_total(group):
    total = 0
    for value in (group or {}).values():
        total += sum(value.values()) if isinstance(value, dict) else value
    return total


def not_applicable_total(summary):
    """Candidate rows in a position the DRM's own vocabulary never records."""
    return _bucket_total(summary.get("not_applicable"))


def dropped_total(summary):
    """Every candidate row the DRM could have held and did not.

    ``candidates == rows + not_applicable_total + dropped_total`` is the
    arithmetic the summary line rests on: nothing may vanish without landing in
    exactly one bucket of exactly one group.
    """
    return _bucket_total(summary.get("dropped"))


def build_drm_rows(patterns, time_format):
    """Every DRM row the canonical pattern list implies.

    ``time_format`` turns a canonical point time into the string a period cell
    carries -- the tab passes one that projects to the export timeframe first,
    so this stays free of pandas and of the projection engine. It returns None
    for a time it cannot place, which is counted rather than guessed at.

    Returns (rows, summary). A row is
    (sheet, primary, secondary, start_string, end_string).
    """
    summary = _empty_summary()
    dropped = summary["dropped"]
    not_applicable = summary["not_applicable"]

    if not isinstance(patterns, list):
        return [], summary

    ordered = [p for p in patterns
               if isinstance(p, dict) and isinstance(p.get("id"), str) and p["id"]]
    summary["red_patterns"] = sum(1 for p in ordered if p.get("color") == "red")

    index = children_by_leg(patterns)
    by_id = {p["id"]: p for p in ordered}

    # ``find_parent`` resolves to exactly one parent per child, so inverting the
    # downward index is exact and saves a second O(n^2) pass over the list.
    parent_of = {}
    for parent_id, legs in index.items():
        for leg, children in legs.items():
            for child in children:
                parent_of[child["id"]] = (parent_id, leg)

    rows = []
    seen = set()
    for child in ordered:
        entry = parent_of.get(child["id"])
        if entry is None:
            continue                    # a root has no primary to be named by
        parent_id, parent_leg = entry
        primary_label = _leg_label(by_id.get(parent_id), parent_leg)
        primary = primary_name(primary_label)

        points = child.get("points")
        if not isinstance(points, list):
            continue

        legs = index.get(child["id"]) or {}
        for leg in sorted(legs):
            grandchildren = legs[leg]
            summary["candidates"] += len(grandchildren)
            if len(grandchildren) > 1:
                # Two patterns marked over the same span. Keep the first and say
                # so, rather than silently picking one of several readings.
                dropped["ambiguous_leg"] += len(grandchildren) - 1

            if primary is None:
                if isinstance(primary_label, str) and primary_label \
                        and primary_label not in TRACKED_PRIMARY_POSITIONS:
                    # A triangle's D/E leg, or a triple zigzag's Z: structurally
                    # real, and the DRM simply does not record that position.
                    not_applicable["untracked_primary"] += 1
                else:
                    # No label at all -- the parent's own shape could not name
                    # this leg. That is a real loss, not a position the DRM skips.
                    dropped["no_primary"] += 1
                continue

            grandchild = grandchildren[0]
            secondary_label = _leg_label(child, leg)
            if isinstance(secondary_label, str) and secondary_label \
                    and secondary_label not in TRACKED_SECONDARY_POSITIONS:
                # Wave 2, 4, B or X: counter-trend legs the client does not log,
                # whatever pattern happens to form them. Checked before the type
                # so a flat on a wave 4 reads as "never recorded" rather than as
                # "the DRM has no word for a flat" -- the position settles it.
                not_applicable["untracked_secondary"] += 1
                continue

            secondary = secondary_name(secondary_label,
                                       grandchild.get("pattern_type"))
            if secondary is None:
                key = str(grandchild.get("pattern_type"))
                dropped["no_secondary"][key] = dropped["no_secondary"].get(key, 0) + 1
                continue

            if secondary not in PRIMARY_SECONDARY_MAP.get(primary, ()):
                # Well formed but not a cell of this DRM -- 'W.2 Impulse', or an
                # impulse under a primary whose sub-waves are corrective.
                dropped["unknown_pair"] += 1
                continue

            start_point = points[leg] if 0 <= leg < len(points) else None
            end_point = points[leg + 1] if 0 <= leg + 1 < len(points) else None
            start_price, end_price = _price(start_point), _price(end_point)
            if start_price is None or end_price is None or start_price == end_price:
                # A flat leg has no direction to put it on a sheet with, and the
                # client's file has no such case. A leg with no usable pair of
                # prices at all lands here too -- it has just as little direction.
                dropped["flat_leg"] += 1
                continue
            sheet = "Bullish" if end_price > start_price else "Bearish"

            start = time_format(start_point.get("time"))
            end = time_format(end_point.get("time"))
            if start is None or end is None:
                # The bar a pivot was snapped to is not in the frame any more.
                dropped["unmappable_time"] += 1
                continue
            if start == end:
                dropped["degenerate_period"] += 1
                continue

            row = (sheet, primary, secondary, start, end)
            if row in seen:
                # Two markings over the same span reach the same cell; the same
                # period must not be written twice.
                dropped["duplicate"] += 1
                continue
            seen.add(row)
            rows.append(row)

    summary["rows"] = len(rows)
    return rows, summary


# ---------------------------------------------------------------- the sheets


def _start_sort_key(start):
    """Sort key putting a row's periods in chronological order.

    A row carries formatted strings, and the DRM's format is day-first, so plain
    string order would put 03.10 before 28.09. Parsing is therefore not optional.
    Anything that does not parse (a test's synthetic formatter, a hand-edited
    cell) sorts after everything that does, by its own text, so the sort stays
    total instead of raising.
    """
    try:
        return (0, pd.to_datetime(start, format=PERIOD_FORMAT).value, "")
    except (ValueError, TypeError):
        return (1, 0, str(start))


def _sheet_frame(rows, sheet):
    """One sheet: the full skeleton, its periods spread across columns 2+."""
    periods = {}
    for row in rows:
        row_sheet, primary, secondary, start, end = row
        if row_sheet != sheet:
            continue
        periods.setdefault((primary, secondary), []).append((start, end))

    widest = max((len(cells) for cells in periods.values()), default=0)

    data = []
    for pair in DRM_SKELETON:
        cells = sorted(periods.get(pair, []), key=lambda cell: _start_sort_key(cell[0]))
        # The primary goes on *every* row, not only the first: load_drm ffills
        # this column, so a full column is a no-op there and removes a whole
        # class of misalignment.
        values = [pair[0], pair[1]] + [f"{start}, {end}" for start, end in cells]
        values += [None] * (widest - len(cells))
        data.append(values)

    # Column 0's header is the sheet name because ``load_drm`` addresses it as
    # ``df[sheet_name]``; column 1's is blank, which pandas reads back as
    # 'Unnamed: 1' and ``parse_drm_periods`` addresses positionally anyway.
    columns = [sheet, ""] + list(range(1, widest + 1))
    return pd.DataFrame(data, columns=columns)


def build_drm_sheets(rows):
    """{'Bullish': DataFrame, 'Bearish': DataFrame} in load_drm's own shape."""
    return {sheet: _sheet_frame(rows, sheet) for sheet in SHEET_NAMES}


def build_drm_workbook(sheets):
    """The two sheets as .xlsx bytes, ready for st.download_button."""
    buffer = io.BytesIO()
    with pd.ExcelWriter(buffer, engine="openpyxl") as writer:
        for sheet in SHEET_NAMES:
            frame = sheets.get(sheet)
            if frame is None:
                frame = _sheet_frame([], sheet)
            frame.to_excel(writer, sheet_name=sheet, index=False)
    return buffer.getvalue()


# ------------------------------------------------------------------ the JSON


def _json_safe(value):
    """The same structure with anything json.dumps(allow_nan=False) refuses gone.

    Prices ride in from the projection as plain floats, but a NaN is a legal
    ``float`` and would abort the whole dump -- one unusable pivot must not cost
    the client the backup of every other marking.
    """
    if isinstance(value, dict):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(item) for item in value]
    if isinstance(value, float) and not math.isfinite(value):
        return None
    return value


def build_wave_json(patterns, dataset_key, base_timeframe):
    """The canonical wave counts as a JSON-serialisable dict.

    A backup and an interchange format, not the DRM: it carries the full
    patterns, so re-importing it later could rebuild the markings exactly.
    Include the dataset key and base timeframe -- points are meaningless
    without knowing which instrument and resolution they were snapped to.

    Whole patterns means the per-leg study values come along with them, which is
    what makes this file -- rather than the ``.xlsx`` -- the thing the client
    counts his "wave 4 versus wave 2" statistics out of.
    """
    return {
        # The same schema number the on-disk markings file carries, so an
        # importer has one thing to check rather than two.
        "schema": 2,
        "dataset_key": dataset_key,
        "base_timeframe": base_timeframe,
        "patterns": _json_safe(patterns if isinstance(patterns, list) else []),
    }


# --------------------------------------------------------------- the summary


# Two groups, kept visually apart on purpose. "Not applicable" is the DRM's
# vocabulary declining to record a position; "dropped" is work that could have
# been written and was not. Reporting the first as the second is what made a
# correct export read as lossy, and the client should be able to tell which
# number he is looking at at a glance.
_NOT_APPLICABLE_REASONS = [
    ("untracked_primary", "parent leg(s) in a position the DRM has no primary "
                          "for (a triangle's D/E, a triple zigzag's Z)"),
    ("untracked_secondary", "counter-trend leg(s) the DRM never records "
                            "(wave 2, 4, B or X)"),
]

_DROPPED_REASONS = [
    ("no_primary", "leg(s) whose own pattern gives them no wave label"),
    ("unknown_pair", "primary/secondary pair(s) the DRM has no cell for"),
    ("flat_leg", "flat leg(s), with no direction to choose a sheet by"),
    ("unmappable_time", "pivot(s) on a bar this data no longer has"),
    ("degenerate_period", "leg(s) with both pivots in the same export bar"),
    ("ambiguous_leg", "extra pattern(s) marked over an already-used leg"),
    ("duplicate", "period(s) already written to the same cell"),
]


def summary_lines(summary):
    """What the export wrote and, line by line, everything it did not.

    A silent drop of a third of his markings would read as the export losing his
    work, so every reason is named with its count and the ones that fired zero
    times are left out. The two groups carry their own headings: a number under
    "not applicable" is the format declining to track a position, and nothing
    was lost by it.

    Streamlit renders these as captions, so the headings use markdown emphasis
    rather than punctuation to separate the groups.
    """
    dropped = summary.get("dropped") or {}
    not_applicable = summary.get("not_applicable") or {}
    lines = [f"{summary.get('rows', 0)} period(s) written "
             f"from {summary.get('candidates', 0)} candidate row(s)."]

    def reasons(group, table):
        return [(group.get(key) or 0, text) for key, text in table
                if group.get(key)]

    listed = reasons(not_applicable, _NOT_APPLICABLE_REASONS)
    if listed:
        lines.append("**Not applicable** (the DRM does not track these "
                     "positions, so nothing was lost):")
        lines += [f"· {count} {text}." for count, text in listed]

    listed = reasons(dropped, _DROPPED_REASONS)
    listed += [(count, f"leg(s) formed by a {pattern_type}, which the DRM has no "
                       "secondary name for")
               for pattern_type, count
               in sorted((dropped.get("no_secondary") or {}).items())]
    if listed or summary.get("red_patterns"):
        lines.append("**Dropped** (the DRM could have recorded these and did not):")
        if summary.get("red_patterns"):
            lines.append(f"· {summary['red_patterns']} red pattern(s) took no part: "
                         "the parent/child relation is yellow-only, so they name "
                         "nothing.")
        lines += [f"· {count} {text}." for count, text in listed]

    return lines

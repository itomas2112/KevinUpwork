"""
Wave marking persistence - save, load

Markings live in a single JSON file at the repo root, keyed by dataset. The
dataset key is the uploaded OHLC file's name: a marking snapped to one
instrument's bars means nothing on another's.

Two on-disk schemas exist.

* **v1**, ``{dataset_key: {timeframe: [pattern, ...]}}`` -- a separate list per
  aggregation, which is why a count drawn on 15m was invisible on 1D.
* **v2**, ``{dataset_key: {"schema": 2, "base_timeframe": ..., "patterns":
  [...]}}`` -- one canonical list at the uploaded data's own resolution, of
  which every aggregation on screen is a projection.

``load_wave_markings`` / ``save_wave_markings`` are the v1 pair and still serve
the tab unchanged. ``load_wave_documents`` / ``save_wave_documents`` read and
write either schema; ``migrate_document`` lifts a v1 entry to v2 once a caller
can supply the bar data the projection needs.
"""
import json
import os

from config.wave_analysis import is_valid_pattern, migrate_leg_value_fields

WAVE_MARKINGS_FILE = "saved_wave_markings.json"


def load_wave_markings(path=WAVE_MARKINGS_FILE):
    """Read every dataset's markings from disk.

    Never raises: a missing, unreadable, corrupt or wrongly shaped file yields
    an empty mapping, because a bad file must not stop the tab from opening.
    Patterns that fail ``is_valid_pattern`` are dropped rather than handed to
    the frontend.
    """
    try:
        with open(path, 'r') as f:
            data = json.load(f)
    except (OSError, ValueError):
        return {}

    if not isinstance(data, dict):
        return {}

    markings = {}
    for dataset_key, by_timeframe in data.items():
        if not isinstance(dataset_key, str) or not isinstance(by_timeframe, dict):
            continue
        cleaned = {}
        for timeframe, patterns in by_timeframe.items():
            if not isinstance(timeframe, str) or not isinstance(patterns, list):
                continue
            cleaned[timeframe] = [p for p in patterns if is_valid_pattern(p)]
        markings[dataset_key] = cleaned
    return markings


def save_wave_markings(markings, path=WAVE_MARKINGS_FILE):
    """Write every dataset's markings to disk atomically.

    The file is built next to its target and moved into place with
    ``os.replace``, so a machine that dies mid-write leaves the previous
    markings intact instead of a truncated file.
    """
    tmp_path = path + ".tmp"
    with open(tmp_path, 'w') as f:
        json.dump(markings, f, indent=2)
    os.replace(tmp_path, path)


# ----------------------------------------------------------------- documents


def _clean_patterns(patterns):
    """The structurally sound entries of a stored pattern list.

    Field renames are applied *before* validation, not after. A leg-value field
    name is the storage key the reading lives under, so a pattern written before
    a rename carries keys the current table does not know -- and
    ``is_valid_pattern`` rejects exactly that, which would drop the client's
    measured numbers instead of updating them. Migrating first is what lets a
    pre-rename file load with no warnings and persist the new names on its next
    save; the file itself is never rewritten in place.
    """
    if not isinstance(patterns, list):
        return []
    return [migrated for migrated in (migrate_leg_value_fields(p) for p in patterns)
            if is_valid_pattern(migrated)]


def load_wave_documents(path=WAVE_MARKINGS_FILE):
    """{dataset_key: document} for every dataset, in either schema.

    A v1 entry is returned as ``{"schema": 1, "by_timeframe": {tf: [pattern,
    ...]}}`` -- unmigrated, because migrating needs the bar data and this
    module has none. ``migrate_document`` does that once the caller can supply
    it. A legacy entry is the one with no ``"schema"`` key at all.

    That unmigrated form is also accepted *back*, and it has to be. Only the
    dataset currently on screen has bars loaded, so the caller can migrate that
    one and no other; every write of the file therefore carries the other
    datasets' documents back to disk exactly as they came out, still at schema
    1. Reading only the legacy shape would take that written-back entry for a
    v2 document with no ``patterns`` key and quietly empty it -- which is the
    one failure this whole pair exists to avoid.

    Never raises: a missing, unreadable, corrupt or wrongly shaped file yields
    an empty mapping, because the tab must open on a bad file. Patterns that
    fail ``is_valid_pattern`` are dropped in either schema, exactly as
    ``load_wave_markings`` does.
    """
    try:
        with open(path, 'r') as f:
            data = json.load(f)
    except (OSError, ValueError):
        return {}

    if not isinstance(data, dict):
        return {}

    documents = {}
    for dataset_key, entry in data.items():
        if not isinstance(dataset_key, str) or not isinstance(entry, dict):
            continue

        if "schema" not in entry:
            legacy = entry                          # the whole entry is the map
        elif entry.get("schema") == 1:
            legacy = entry.get("by_timeframe")      # written back unmigrated
        else:
            base_timeframe = entry.get("base_timeframe")
            documents[dataset_key] = {
                "schema": 2,
                # A document with no readable base timeframe is not a reason to
                # lose its patterns; the app's own default covers it.
                "base_timeframe": base_timeframe if isinstance(base_timeframe, str) else "15m",
                "patterns": _clean_patterns(entry.get("patterns")),
            }
            continue

        by_timeframe = {}
        if isinstance(legacy, dict):
            for timeframe, patterns in legacy.items():
                if not isinstance(timeframe, str) or not isinstance(patterns, list):
                    continue
                by_timeframe[timeframe] = _clean_patterns(patterns)
        documents[dataset_key] = {"schema": 1, "by_timeframe": by_timeframe}

    return documents


def save_wave_documents(documents, path=WAVE_MARKINGS_FILE):
    """Atomic write, same tmp+os.replace discipline as save_wave_markings."""
    tmp_path = path + ".tmp"
    with open(tmp_path, 'w') as f:
        json.dump(documents, f, indent=2)
    os.replace(tmp_path, path)


def migrate_document(document, base_timeframe, projectors):
    """A v1 entry to v2 canonical form.

    ``projectors`` maps a legacy timeframe name to something that turns that
    timeframe's patterns into canonical ones -- a callable taking a pattern
    list and returning a pattern list. The caller builds them from
    ``config.wave_projection``; this module stays free of pandas.

    Patterns already stored under ``base_timeframe`` are canonical and pass
    through untouched. A timeframe with no projector is dropped.

    Returns ``(document_v2, dropped_count)``. The count is everything the v1
    entry held that is not in the returned list, whatever the reason -- no
    projector, a projection that collapsed, a pattern that fails validation, or
    an id already claimed by an earlier timeframe. The client has markings on
    three datasets and silently losing some of them would be worse than
    refusing to migrate, so the caller gets one honest number to show.
    """
    if not isinstance(document, dict):
        return ({"schema": 2, "base_timeframe": base_timeframe, "patterns": []}, 0)

    # Already canonical: nothing to project, nothing to lose.
    if document.get("schema") == 2:
        stored = document.get("base_timeframe")
        return ({
            "schema": 2,
            "base_timeframe": stored if isinstance(stored, str) else base_timeframe,
            "patterns": _clean_patterns(document.get("patterns")),
        }, 0)

    by_timeframe = document.get("by_timeframe")
    if not isinstance(by_timeframe, dict):
        by_timeframe = {}

    if not isinstance(projectors, dict):
        projectors = {}

    total = 0
    canonical = []
    seen = set()
    for timeframe, patterns in by_timeframe.items():
        if not isinstance(patterns, list):
            continue
        total += len(patterns)

        if timeframe == base_timeframe:
            projected = patterns
        else:
            projector = projectors.get(timeframe)
            if projector is None:
                continue
            projected = projector(patterns)
            if not isinstance(projected, list):
                continue

        for pattern in projected:
            if not is_valid_pattern(pattern):
                continue
            # One id, one pattern. The same count drawn once and then seen at
            # three aggregations was written to all three under a single id, so
            # after projection the later copies are the same marking again --
            # the first occurrence wins and the rest are folded away.
            if pattern["id"] in seen:
                continue
            seen.add(pattern["id"])
            canonical.append(pattern)

    migrated = {
        "schema": 2,
        "base_timeframe": base_timeframe,
        "patterns": canonical,
    }
    return (migrated, total - len(canonical))

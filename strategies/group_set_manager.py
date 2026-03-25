"""
Group Set management — save, load, delete, import/export for Grid Search group sets.
Universal group sets: no type field, candidates store only element pairs.
"""
import streamlit as st
import json
import os

GROUP_SETS_FILE = "saved_group_sets.json"


def _deduplicate_candidates(group_set):
    """Remove duplicate candidates from a group set in-place. Returns count of removed duplicates."""
    candidates = group_set.get("candidates", [])
    seen = set()
    unique = []
    for c in candidates:
        key = json.dumps(c, sort_keys=True)
        if key not in seen:
            seen.add(key)
            unique.append(c)
    removed = len(candidates) - len(unique)
    group_set["candidates"] = unique
    return removed


def _strip_legacy_fields(data):
    """Strip old type-specific fields from imported/legacy group sets."""
    data.pop("type", None)
    for cand in data.get("candidates", []):
        cand.pop("event", None)
        cand.pop("operator", None)


def load_group_sets():
    """Load group sets from disk. Returns list of dicts."""
    if os.path.exists(GROUP_SETS_FILE):
        try:
            with open(GROUP_SETS_FILE, "r") as f:
                return json.load(f)
        except (json.JSONDecodeError, ValueError):
            return []
    return []


def save_group_sets_to_file():
    """Write session state group sets to disk."""
    with open(GROUP_SETS_FILE, "w") as f:
        json.dump(st.session_state.get("saved_group_sets", []), f, indent=4)


def save_group_set(group_set):
    """Append a group set (after deduplication) and persist. Returns number of duplicates removed."""
    removed = _deduplicate_candidates(group_set)
    st.session_state.setdefault("saved_group_sets", []).append(group_set)
    save_group_sets_to_file()
    return removed


def update_group_set(idx, group_set):
    """Replace a group set at index (after deduplication) and persist. Returns number of duplicates removed."""
    removed = _deduplicate_candidates(group_set)
    sets = st.session_state.get("saved_group_sets", [])
    if 0 <= idx < len(sets):
        sets[idx] = group_set
        save_group_sets_to_file()
    return removed


def delete_group_set(idx):
    """Delete a group set by index and persist."""
    sets = st.session_state.get("saved_group_sets", [])
    if 0 <= idx < len(sets):
        sets.pop(idx)
        save_group_sets_to_file()


def export_group_set(group_set):
    """Return JSON string for download."""
    return json.dumps(group_set, indent=2)


def import_group_set(json_str):
    """Parse and validate a group set from JSON string. Returns dict or raises ValueError.

    Accepts both old format (with type/event/operator fields) and new universal format.
    Legacy fields are stripped automatically.
    """
    data = json.loads(json_str)
    if not isinstance(data, dict):
        raise ValueError("Expected a JSON object.")
    if "name" not in data:
        raise ValueError("Missing required field: 'name'")
    if "candidates" not in data:
        raise ValueError("Missing required field: 'candidates'")
    if not isinstance(data["candidates"], list):
        raise ValueError("'candidates' must be a list.")
    # Strip legacy type-specific fields
    _strip_legacy_fields(data)
    removed = _deduplicate_candidates(data)
    if removed:
        data["_duplicates_removed"] = removed
    return data

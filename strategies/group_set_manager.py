"""
Group Set management — save, load, delete, import/export for Grid Search group sets.
"""
import streamlit as st
import json
import os

GROUP_SETS_FILE = "saved_group_sets.json"


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
    """Append a group set and persist."""
    st.session_state.setdefault("saved_group_sets", []).append(group_set)
    save_group_sets_to_file()


def update_group_set(idx, group_set):
    """Replace a group set at index and persist."""
    sets = st.session_state.get("saved_group_sets", [])
    if 0 <= idx < len(sets):
        sets[idx] = group_set
        save_group_sets_to_file()


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
    """Parse and validate a group set from JSON string. Returns dict or raises ValueError."""
    data = json.loads(json_str)
    if not isinstance(data, dict):
        raise ValueError("Expected a JSON object.")
    for field in ("name", "type", "candidates"):
        if field not in data:
            raise ValueError(f"Missing required field: '{field}'")
    valid_types = ("trigger", "condition", "static_stop", "dynamic_stop", "target")
    if data["type"] not in valid_types:
        raise ValueError(f"Invalid type '{data['type']}'. Must be one of: {valid_types}")
    if not isinstance(data["candidates"], list):
        raise ValueError("'candidates' must be a list.")
    return data


def get_group_sets_by_type(gs_type):
    """Return list of (index, group_set) for a given type."""
    sets = st.session_state.get("saved_group_sets", [])
    return [(i, gs) for i, gs in enumerate(sets) if gs.get("type") == gs_type]

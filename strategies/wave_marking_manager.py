"""
Wave marking persistence - save, load

Markings live in a single JSON file at the repo root, shaped
``{dataset_key: {timeframe: [pattern, ...]}}``. The dataset key is the
uploaded OHLC file's name: a marking snapped to one instrument's bars means
nothing on another's.
"""
import json
import os

from config.wave_analysis import is_valid_pattern

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

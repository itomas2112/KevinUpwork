"""
Tests for wave marking persistence (save / load of saved_wave_markings.json).

Every test writes into ``tmp_path`` through the ``path`` parameter, so the real
file at the repo root is never touched.
"""

import json

from strategies.wave_marking_manager import load_wave_markings, save_wave_markings


def make_pattern(pattern_id="abc123", degree="Subminuette", color="yellow"):
    """A structurally valid pattern as the tab would persist it."""
    return {
        "id": pattern_id,
        "pattern_type": "Zigzag",
        "variation": "Zigzag",
        "degree": degree,
        "color": color,
        "points": [
            {"time": 1719878400, "price": 2360.5, "kind": "low"},
            {"time": 1719879300, "price": 2371.0, "kind": "high"},
            {"time": 1719880200, "price": 2355.25, "kind": "low"},
            {"time": 1719881100, "price": 2380.75, "kind": "high"},
        ],
    }


def markings_file(tmp_path):
    return str(tmp_path / "saved_wave_markings.json")


# ------------------------------------------------------------------ round trip

def test_save_then_load_returns_the_same_data(tmp_path):
    path = markings_file(tmp_path)
    markings = {
        "gold_15m.csv": {
            "15m": [make_pattern("a"), make_pattern("b", degree="Minute")],
            "1h": [make_pattern("c", color="red")],
        },
        "silver.csv": {"15m": []},
    }

    save_wave_markings(markings, path)

    assert load_wave_markings(path) == markings


def test_save_writes_human_readable_json(tmp_path):
    path = markings_file(tmp_path)
    save_wave_markings({"gold.csv": {"15m": [make_pattern()]}}, path)

    text = open(path).read()
    assert "\n" in text and '"id": "abc123"' in text


# ------------------------------------------------------------ unreadable files

def test_missing_file_loads_as_empty(tmp_path):
    assert load_wave_markings(markings_file(tmp_path)) == {}


def test_corrupt_file_loads_as_empty(tmp_path):
    path = markings_file(tmp_path)
    with open(path, 'w') as f:
        f.write("{not json")

    assert load_wave_markings(path) == {}


def test_a_top_level_list_loads_as_empty(tmp_path):
    path = markings_file(tmp_path)
    with open(path, 'w') as f:
        json.dump([make_pattern()], f)

    assert load_wave_markings(path) == {}


def test_a_truncated_file_loads_as_empty(tmp_path):
    path = markings_file(tmp_path)
    save_wave_markings({"gold.csv": {"15m": [make_pattern()]}}, path)

    text = open(path).read()
    with open(path, 'w') as f:
        f.write(text[:len(text) // 2])

    assert load_wave_markings(path) == {}


# --------------------------------------------------------------------- hygiene

def test_an_invalid_pattern_is_dropped_and_the_rest_survive(tmp_path):
    path = markings_file(tmp_path)
    broken = make_pattern("broken")
    broken["degree"] = "Nonexistent"
    with open(path, 'w') as f:
        json.dump({"gold.csv": {"15m": [make_pattern("good"), broken,
                                        make_pattern("also_good")]}}, f)

    loaded = load_wave_markings(path)

    assert [p["id"] for p in loaded["gold.csv"]["15m"]] == ["good", "also_good"]


def test_non_list_timeframe_entries_are_dropped(tmp_path):
    path = markings_file(tmp_path)
    with open(path, 'w') as f:
        json.dump({"gold.csv": {"15m": [make_pattern("good")],
                                "1h": "not a list",
                                "4h": {"nope": 1}},
                   "junk_dataset.csv": "not a mapping"}, f)

    assert load_wave_markings(path) == {"gold.csv": {"15m": [make_pattern("good")]}}


# -------------------------------------------------------------------- atomicity

def test_save_leaves_no_temp_file_behind(tmp_path):
    path = markings_file(tmp_path)
    save_wave_markings({"gold.csv": {"15m": [make_pattern()]}}, path)

    assert not (tmp_path / "saved_wave_markings.json.tmp").exists()
    assert [p.name for p in tmp_path.iterdir()] == ["saved_wave_markings.json"]
    with open(path) as f:
        assert isinstance(json.load(f), dict)


def test_rewriting_an_existing_file_replaces_it_cleanly(tmp_path):
    path = markings_file(tmp_path)
    save_wave_markings({"gold.csv": {"15m": [make_pattern("first")]}}, path)
    save_wave_markings({"gold.csv": {"15m": [make_pattern("second")]}}, path)

    loaded = load_wave_markings(path)
    assert [p["id"] for p in loaded["gold.csv"]["15m"]] == ["second"]
    assert not (tmp_path / "saved_wave_markings.json.tmp").exists()


# ------------------------------------------------------- read-modify-write flow

def test_read_modify_write_keeps_other_datasets_intact(tmp_path):
    path = markings_file(tmp_path)
    save_wave_markings({"gold.csv": {"15m": [make_pattern("gold_one")]}}, path)

    # Exactly what the tab does after an applied event batch.
    data = load_wave_markings(path)
    data["silver.csv"] = {"1h": [make_pattern("silver_one")]}
    save_wave_markings(data, path)

    loaded = load_wave_markings(path)
    assert set(loaded) == {"gold.csv", "silver.csv"}
    assert [p["id"] for p in loaded["gold.csv"]["15m"]] == ["gold_one"]
    assert [p["id"] for p in loaded["silver.csv"]["1h"]] == ["silver_one"]

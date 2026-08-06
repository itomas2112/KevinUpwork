"""
Tests for wave marking persistence (save / load of saved_wave_markings.json).

Every test writes into ``tmp_path`` through the ``path`` parameter, so the real
file at the repo root is never touched.
"""

import copy
import json

from strategies.wave_marking_manager import (
    load_wave_documents,
    load_wave_markings,
    migrate_document,
    save_wave_documents,
    save_wave_markings,
)


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


# =========================================================== canonical schema
#
# Schema 2 keeps one canonical pattern list per dataset, at the uploaded data's
# own resolution, instead of a separate list per aggregation. Everything below
# exercises the new pair; the tests above are the regression guard that the v1
# pair the tab still calls has not moved.


def v2_document(patterns, base_timeframe="15m"):
    return {"schema": 2, "base_timeframe": base_timeframe, "patterns": patterns}


# ------------------------------------------------------------ document round trip

def test_a_v2_document_round_trips(tmp_path):
    path = markings_file(tmp_path)
    documents = {
        "gold_15m.csv": v2_document([make_pattern("a"),
                                     make_pattern("b", degree="Minute")]),
        "silver.csv": v2_document([], base_timeframe="1H"),
    }

    save_wave_documents(documents, path)

    assert load_wave_documents(path) == documents


def test_save_wave_documents_leaves_no_temp_file_behind(tmp_path):
    path = markings_file(tmp_path)
    save_wave_documents({"gold.csv": v2_document([make_pattern()])}, path)

    assert not (tmp_path / "saved_wave_markings.json.tmp").exists()
    assert [p.name for p in tmp_path.iterdir()] == ["saved_wave_markings.json"]


# -------------------------------------------------------------- schema detection

def test_a_legacy_file_loads_unmigrated_as_schema_1(tmp_path):
    path = markings_file(tmp_path)
    save_wave_markings({"gold.csv": {"15m": [make_pattern("a")],
                                     "1D": [make_pattern("b")]}}, path)

    assert load_wave_documents(path) == {
        "gold.csv": {"schema": 1,
                     "by_timeframe": {"15m": [make_pattern("a")],
                                      "1D": [make_pattern("b")]}},
    }


def test_a_file_holding_one_of_each_schema_loads_both(tmp_path):
    path = markings_file(tmp_path)
    with open(path, 'w') as f:
        json.dump({"legacy.csv": {"1H": [make_pattern("old")]},
                   "modern.csv": v2_document([make_pattern("new")])}, f)

    loaded = load_wave_documents(path)

    assert loaded["legacy.csv"] == {"schema": 1,
                                    "by_timeframe": {"1H": [make_pattern("old")]}}
    assert loaded["modern.csv"] == v2_document([make_pattern("new")])


def test_an_unmigrated_document_survives_being_written_back(tmp_path):
    # Only the dataset on screen has bars loaded, so every write carries the
    # others back to disk still at schema 1. Reading that back as a v2 document
    # with no "patterns" key would empty them without a word.
    path = markings_file(tmp_path)
    save_wave_markings({"legacy.csv": {"1D": [make_pattern("keepme")]}}, path)

    documents = load_wave_documents(path)
    save_wave_documents(documents, path)

    assert load_wave_documents(path) == documents
    assert [p["id"] for p in load_wave_documents(path)["legacy.csv"]
            ["by_timeframe"]["1D"]] == ["keepme"]


def test_a_migrated_and_an_unmigrated_dataset_coexist_in_one_file(tmp_path):
    path = markings_file(tmp_path)
    documents = {
        "on_screen.csv": v2_document([make_pattern("canonical")]),
        "other.csv": {"schema": 1, "by_timeframe": {"1H": [make_pattern("legacy")]}},
    }

    save_wave_documents(documents, path)

    assert load_wave_documents(path) == documents


def test_a_v2_document_with_no_base_timeframe_falls_back_to_the_default(tmp_path):
    path = markings_file(tmp_path)
    with open(path, 'w') as f:
        json.dump({"gold.csv": {"schema": 2, "patterns": [make_pattern("a")]}}, f)

    assert load_wave_documents(path)["gold.csv"]["base_timeframe"] == "15m"


# ------------------------------------------------------- documents: unreadable

def test_documents_from_a_missing_file_are_empty(tmp_path):
    assert load_wave_documents(markings_file(tmp_path)) == {}


def test_documents_from_a_corrupt_file_are_empty(tmp_path):
    path = markings_file(tmp_path)
    with open(path, 'w') as f:
        f.write("{not json")

    assert load_wave_documents(path) == {}


def test_documents_from_a_top_level_list_are_empty(tmp_path):
    path = markings_file(tmp_path)
    with open(path, 'w') as f:
        json.dump([v2_document([make_pattern()])], f)

    assert load_wave_documents(path) == {}


def test_wrongly_shaped_document_entries_are_dropped(tmp_path):
    path = markings_file(tmp_path)
    with open(path, 'w') as f:
        json.dump({"junk.csv": "not a mapping",
                   "broken.csv": {"schema": 2, "patterns": "not a list"},
                   "gold.csv": v2_document([make_pattern("a")])}, f)

    loaded = load_wave_documents(path)

    assert set(loaded) == {"broken.csv", "gold.csv"}
    assert loaded["broken.csv"]["patterns"] == []


# ------------------------------------------------------------ documents: hygiene

def test_an_invalid_pattern_is_dropped_from_a_v2_document(tmp_path):
    path = markings_file(tmp_path)
    broken = make_pattern("broken")
    broken["degree"] = "Nonexistent"
    with open(path, 'w') as f:
        json.dump({"gold.csv": v2_document([make_pattern("good"), broken,
                                            make_pattern("also_good")])}, f)

    loaded = load_wave_documents(path)

    assert [p["id"] for p in loaded["gold.csv"]["patterns"]] == ["good", "also_good"]


def test_an_invalid_pattern_is_dropped_from_a_v1_document(tmp_path):
    path = markings_file(tmp_path)
    broken = make_pattern("broken")
    broken["color"] = "chartreuse"
    with open(path, 'w') as f:
        json.dump({"gold.csv": {"15m": [make_pattern("good"), broken],
                                "1H": "not a list"}}, f)

    loaded = load_wave_documents(path)

    assert loaded["gold.csv"]["by_timeframe"] == {"15m": [make_pattern("good")]}


# ------------------------------------------------------------------- migration

def shift_times(pattern, delta):
    """A pattern moved wholesale in time -- a stand-in for a real projection."""
    return dict(pattern, points=[dict(p, time=p["time"] + delta)
                                 for p in pattern["points"]])


def shifting_projector(delta):
    return lambda patterns: [shift_times(p, delta) for p in patterns]


def test_base_timeframe_patterns_migrate_untouched():
    document = {"schema": 1, "by_timeframe": {"15m": [make_pattern("a"),
                                                      make_pattern("b")]}}

    migrated, dropped = migrate_document(document, "15m", {})

    assert migrated == {"schema": 2, "base_timeframe": "15m",
                        "patterns": [make_pattern("a"), make_pattern("b")]}
    assert dropped == 0


def test_a_non_base_timeframe_is_run_through_its_projector():
    document = {"schema": 1, "by_timeframe": {"1D": [make_pattern("a")]}}

    migrated, dropped = migrate_document(document, "15m",
                                         {"1D": shifting_projector(900)})

    assert migrated["patterns"] == [shift_times(make_pattern("a"), 900)]
    assert dropped == 0


def test_a_timeframe_with_no_projector_is_dropped_and_counted():
    document = {"schema": 1, "by_timeframe": {"15m": [make_pattern("a")],
                                              "4H": [make_pattern("b"),
                                                     make_pattern("c")]}}

    migrated, dropped = migrate_document(document, "15m", {})

    assert [p["id"] for p in migrated["patterns"]] == ["a"]
    assert dropped == 2


def test_a_pattern_the_projector_drops_is_counted():
    document = {"schema": 1, "by_timeframe": {"1D": [make_pattern("a"),
                                                     make_pattern("b")]}}

    migrated, dropped = migrate_document(document, "15m",
                                         {"1D": lambda patterns: patterns[:1]})

    assert [p["id"] for p in migrated["patterns"]] == ["a"]
    assert dropped == 1


def test_duplicate_ids_across_timeframes_collapse_to_the_first():
    # The same count drawn once and seen at two aggregations was written to
    # both under one id; after projection they are the same marking again.
    document = {"schema": 1, "by_timeframe": {"15m": [make_pattern("shared")],
                                              "1D": [make_pattern("shared")]}}

    migrated, dropped = migrate_document(document, "15m",
                                         {"1D": shifting_projector(900)})

    assert migrated["patterns"] == [make_pattern("shared")]
    assert dropped == 1


def test_invalid_patterns_are_dropped_during_migration():
    broken = make_pattern("broken")
    broken["points"] = broken["points"][:2]         # wrong point count for a Zigzag
    document = {"schema": 1, "by_timeframe": {"15m": [make_pattern("a"), broken]}}

    migrated, dropped = migrate_document(document, "15m", {})

    assert [p["id"] for p in migrated["patterns"]] == ["a"]
    assert dropped == 1


def test_migrating_an_already_canonical_document_changes_nothing():
    document = v2_document([make_pattern("a")])

    migrated, dropped = migrate_document(document, "15m", {})

    assert migrated == document
    assert dropped == 0


def test_migration_does_not_mutate_the_document_it_was_given():
    document = {"schema": 1, "by_timeframe": {"15m": [make_pattern("a")],
                                              "1D": [make_pattern("b")]}}
    before = copy.deepcopy(document)

    migrate_document(document, "15m", {"1D": shifting_projector(900)})

    assert document == before


def test_a_migrated_document_survives_a_save_and_load(tmp_path):
    path = markings_file(tmp_path)
    documents = load_wave_documents(path)
    documents["gold.csv"] = migrate_document(
        {"schema": 1, "by_timeframe": {"15m": [make_pattern("a")],
                                       "1D": [make_pattern("b")]}},
        "15m", {"1D": shifting_projector(900)})[0]
    save_wave_documents(documents, path)

    loaded = load_wave_documents(path)

    assert [p["id"] for p in loaded["gold.csv"]["patterns"]] == ["a", "b"]
    assert loaded["gold.csv"]["schema"] == 2

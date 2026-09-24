"""
Tests for the fixed pattern columns shared by the Performance, Grid Search
and Test Set tables:
  All Patterns | All Bullish | All Bearish | Global | <user rows...>

  - data.helpers.all_combos / fixed_combo_map / build_pattern_columns
  - session-state defaults for perf_selections / gs_selections
"""

from collections import OrderedDict

import pytest

from data.helpers import (
    FIXED_COLUMNS, FIXED_SELECTIONS, USER_SELECTION_MODES,
    PRIMARY_SECONDARY_MAP, all_combos, fixed_combo_map, build_pattern_columns,
    filter_combos_by_strategy, selection_label, expand_selection,
)


def _count_agg(stats):
    return {"n": len(stats)}


def _empty():
    return {"n": 0}


def _sel(mode, pattern_type="Bullish", primary=None, secondary=None):
    return {"mode": mode, "pattern_type": pattern_type,
            "primary": primary, "secondary": secondary}


def _one_stat_per_combo():
    return {combo: ["s"] for combo in all_combos()}


# ======================================================================
# Combo universe
# ======================================================================

class TestComboUniverse:

    def test_all_combos_has_86_unique_tuples(self):
        combos = all_combos()
        assert len(combos) == 86
        assert len(set(combos)) == 86
        assert all(len(c) == 3 for c in combos)
        assert {c[0] for c in combos} == {"Bullish", "Bearish"}

    def test_fixed_combo_map_sizes(self):
        sizes = {label: len(combos) for label, combos in fixed_combo_map().items()}
        assert sizes == {"All Patterns": 86, "All Bullish": 43, "All Bearish": 43}
        assert list(fixed_combo_map().keys()) == list(FIXED_SELECTIONS.keys())

    def test_fixed_columns_and_user_modes(self):
        assert FIXED_COLUMNS == ["All Patterns", "All Bullish", "All Bearish", "Global"]
        assert USER_SELECTION_MODES == [
            "Specified Primary", "Specified Secondary", "Secondary Across Primaries"]
        for mode in FIXED_SELECTIONS:
            assert mode not in USER_SELECTION_MODES

    def test_filter_combos_by_strategy(self):
        combos = all_combos()
        assert filter_combos_by_strategy(combos, []) == combos
        kept = filter_combos_by_strategy(combos, ["W.(1) → W.1 Impulse"])
        assert kept == [("Bullish", "W.(1)", "W.1 Impulse"), ("Bearish", "W.(1)", "W.1 Impulse")]


# ======================================================================
# build_pattern_columns
# ======================================================================

class TestBuildPatternColumns:

    def test_key_order_is_fixed_columns_then_user_labels(self):
        rows = [_sel("Specified Primary", primary="W.(1)"),
                _sel("Specified Primary", pattern_type="Bearish", primary="W.(3)")]
        cols = build_pattern_columns(rows, [], _one_stat_per_combo(), _count_agg, _empty)
        assert isinstance(cols, OrderedDict)
        assert list(cols.keys()) == FIXED_COLUMNS + [selection_label(r) for r in rows]

    def test_fixed_column_counts(self):
        cols = build_pattern_columns([], [], _one_stat_per_combo(), _count_agg, _empty)
        assert cols["All Patterns"] == {"n": 86}
        assert cols["All Bullish"] == {"n": 43}
        assert cols["All Bearish"] == {"n": 43}

    def test_global_is_union_of_user_rows_counted_once(self):
        w1 = _sel("Specified Primary", primary="W.(1)")
        w3 = _sel("Specified Primary", primary="W.(3)")
        k = len(PRIMARY_SECONDARY_MAP["W.(3)"])
        cols = build_pattern_columns([w1, w3], [], _one_stat_per_combo(), _count_agg, _empty)
        assert cols["W.(1) Bullish"] == {"n": 3}
        assert cols["W.(3) Bullish"] == {"n": k}
        assert cols["Global"] == {"n": 3 + k}

    def test_same_primary_twice_dedups_label_and_global(self):
        w1 = _sel("Specified Primary", primary="W.(1)")
        cols = build_pattern_columns([w1, w1], [], _one_stat_per_combo(), _count_agg, _empty)
        assert list(cols.keys()) == FIXED_COLUMNS + ["W.(1) Bullish", "W.(1) Bullish (2)"]
        assert cols["W.(1) Bullish"] == {"n": 3}
        assert cols["W.(1) Bullish (2)"] == {"n": 3}
        assert cols["Global"] == {"n": 3}

    def test_no_user_rows_gives_empty_global(self):
        sentinel = {"n": 0, "empty": True}
        cols = build_pattern_columns([], [], _one_stat_per_combo(), _count_agg, lambda: sentinel)
        assert list(cols.keys()) == FIXED_COLUMNS
        assert cols["Global"] is sentinel

    def test_none_selections_treated_as_no_rows(self):
        cols = build_pattern_columns(None, [], _one_stat_per_combo(), _count_agg, _empty)
        assert list(cols.keys()) == FIXED_COLUMNS
        assert cols["Global"] == {"n": 0}

    def test_strategy_patterns_restrict_fixed_columns_too(self):
        strategy_patterns = ["W.(1) → W.1 Impulse"]
        cols = build_pattern_columns(
            [_sel("Specified Primary", primary="W.(1)")], strategy_patterns,
            _one_stat_per_combo(), _count_agg, _empty)
        assert cols["All Patterns"] == {"n": 2}   # Bullish + Bearish W.(1) → W.1 Impulse
        assert cols["All Bullish"] == {"n": 1}
        assert cols["All Bearish"] == {"n": 1}
        assert cols["W.(1) Bullish"] == {"n": 1}
        assert cols["Global"] == {"n": 1}

    def test_row_filtered_to_nothing_gets_empty(self):
        strategy_patterns = ["W.(1) → W.1 Impulse"]
        cols = build_pattern_columns(
            [_sel("Specified Primary", primary="W.(2)")], strategy_patterns,
            _one_stat_per_combo(), _count_agg, lambda: {"n": 0, "empty": True})
        assert cols["W.(2) Bullish"] == {"n": 0, "empty": True}
        assert cols["Global"] == {"n": 0, "empty": True}

    def test_missing_combos_contribute_nothing(self):
        stats = {("Bullish", "W.(1)", "W.1 Impulse"): ["a", "b"]}
        cols = build_pattern_columns(
            [_sel("Specified Primary", primary="W.(1)")], [], stats, _count_agg, _empty)
        assert cols["All Patterns"] == {"n": 2}
        assert cols["All Bullish"] == {"n": 2}
        assert cols["All Bearish"] == {"n": 0}
        assert cols["Global"] == {"n": 2}
        assert cols["W.(1) Bullish"] == {"n": 2}

    def test_legacy_all_mode_row_dedups_against_fixed_label(self):
        # The Test Set picker still offers "All Bullish": such a row becomes a
        # second, identical column rather than colliding with the fixed one.
        cols = build_pattern_columns([_sel("All Bullish")], [], _one_stat_per_combo(),
                                     _count_agg, _empty)
        assert list(cols.keys()) == FIXED_COLUMNS + ["All Bullish (2)"]
        assert cols["All Bullish (2)"] == cols["All Bullish"] == {"n": 43}
        assert cols["Global"] == {"n": 43}

    def test_all_patterns_column_equals_every_combo(self):
        # "All Patterns" is what "Global" used to mean: every combo once.
        cols = build_pattern_columns([], [], _one_stat_per_combo(), _count_agg, _empty)
        assert cols["All Patterns"]["n"] == len(expand_selection({"mode": "All Patterns"}))


# ======================================================================
# Session-state defaults
# ======================================================================

class TestPatternSelectionDefaults:

    def test_perf_and_gs_selections_default_to_empty(self):
        from utils.session_state import init_pattern_selections, PATTERN_COLUMN_SELECTION_KEYS
        assert PATTERN_COLUMN_SELECTION_KEYS == ('perf_selections', 'gs_selections')
        state = {}
        init_pattern_selections(state)
        assert state == {'perf_selections': [], 'gs_selections': []}
        assert state['perf_selections'] is not state['gs_selections']

    def test_existing_rows_are_kept(self):
        from utils.session_state import init_pattern_selections
        rows = [_sel("Specified Primary", primary="W.(2)")]
        state = {'gs_selections': rows}
        init_pattern_selections(state)
        assert state['gs_selections'] is rows
        assert state['perf_selections'] == []

    def test_performance_tab_offers_only_user_modes(self):
        pytest.importorskip("streamlit")
        from ui.performance_tab import SELECTION_MODES, _new_user_selection
        assert SELECTION_MODES == USER_SELECTION_MODES
        row = _new_user_selection()
        assert row["mode"] == "Specified Primary"
        assert row["pattern_type"] == "Bullish"
        assert row["primary"] == list(PRIMARY_SECONDARY_MAP.keys())[0]
        assert row["secondary"] is None

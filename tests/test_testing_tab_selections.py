"""
Tests for per-section pattern selection in the Strategy Testing tab:
  - build_selection_results (thin wrapper over data.helpers.build_pattern_columns:
    Global + per-selection Test Set aggregation, fixed columns stripped)
  - _expand_and_filter strategy pattern filtering
  - legacy 'testing_selections' seeding into the three section keys
"""

import pytest


def _import_ui(module_path, name):
    """Import from a ui module, skipping if streamlit is corrupted by test mocks."""
    try:
        import importlib
        mod = importlib.import_module(module_path)
        return getattr(mod, name)
    except (ModuleNotFoundError, ImportError, AttributeError) as exc:
        if "streamlit" in str(exc).lower():
            pytest.skip(f"Streamlit unavailable in test session: {exc}")
        raise


def _count_agg(stats):
    return {"n": len(stats)}


def _empty():
    return {"n": 0}


def _sel(mode, pattern_type="Bullish", primary=None, secondary=None):
    return {"mode": mode, "pattern_type": pattern_type,
            "primary": primary, "secondary": secondary}


W1_IMPULSE = ("Bullish", "W.(1)", "W.1 Impulse")
W1_W3 = ("Bullish", "W.(1)", "W.3 Impulse")


# ======================================================================
# build_selection_results
# ======================================================================

class TestBuildSelectionResults:

    def test_shared_combo_counted_once_globally_and_in_both_selections(self):
        build = _import_ui("ui.strategy_testing_tab", "build_selection_results")
        sel_a = _sel("Specified Secondary", primary="W.(1)", secondary="W.1 Impulse")
        sel_b = _sel("Specified Primary", primary="W.(1)")  # W.1, W.3, W.5 Impulse
        stats_by_combo = {
            W1_IMPULSE: ["s1", "s2"],
            W1_W3: ["s3"],
        }
        global_agg, sel_results = build(
            [sel_a, sel_b], [], stats_by_combo, _count_agg, _empty,
        )
        # W.1 Impulse (2) counted once + W.3 Impulse (1) = 3
        assert global_agg == {"n": 3}
        assert list(sel_results.values()) == [{"n": 2}, {"n": 3}]

    def test_duplicate_labels_are_deduplicated(self):
        build = _import_ui("ui.strategy_testing_tab", "build_selection_results")
        w1 = _sel("Specified Primary", primary="W.(1)")
        _, sel_results = build([w1, w1, w1], [], {}, _count_agg, _empty)
        assert list(sel_results.keys()) == [
            "W.(1) Bullish", "W.(1) Bullish (2)", "W.(1) Bullish (3)",
        ]

    def test_all_mode_rows_dedup_against_fixed_columns(self):
        # "All Bullish" is a fixed column now, so a row in that mode gets
        # "(2)", "(3)", ... and the fixed columns are not returned here.
        build = _import_ui("ui.strategy_testing_tab", "build_selection_results")
        sels = [_sel("All Bullish"), _sel("All Bullish"), _sel("All Bullish")]
        _, sel_results = build(sels, [], {}, _count_agg, _empty)
        assert list(sel_results.keys()) == [
            "All Bullish (2)", "All Bullish (3)", "All Bullish (4)",
        ]
        for fixed in ("All Patterns", "All Bullish", "All Bearish", "Global"):
            assert fixed not in sel_results

    def test_wrapper_matches_build_pattern_columns(self):
        build = _import_ui("ui.strategy_testing_tab", "build_selection_results")
        from data.helpers import build_pattern_columns, FIXED_COLUMNS
        sels = [_sel("Specified Primary", primary="W.(1)"),
                _sel("Specified Secondary", primary="W.(1)", secondary="W.1 Impulse")]
        stats_by_combo = {W1_IMPULSE: ["s1", "s2"], W1_W3: ["s3"]}
        global_agg, sel_results = build(sels, [], stats_by_combo, _count_agg, _empty)
        cols = build_pattern_columns(sels, [], stats_by_combo, _count_agg, _empty)
        assert global_agg == cols["Global"] == {"n": 3}
        assert list(sel_results.keys()) == [k for k in cols if k not in FIXED_COLUMNS]
        assert list(sel_results.values()) == [{"n": 3}, {"n": 2}]

    def test_selection_filtered_to_nothing_gets_empty(self):
        build = _import_ui("ui.strategy_testing_tab", "build_selection_results")
        sel_bull = _sel("Specified Secondary", primary="W.(1)", secondary="W.1 Impulse")
        sel_w2 = _sel("Specified Primary", primary="W.(2)")
        strategy_patterns = ["W.(1) \u2192 W.1 Impulse"]
        # W.(2) combos have stats, but the strategy filter excludes every
        # combo except W.(1) → W.1 Impulse, so the W.(2) selection is empty.
        stats_by_combo = {
            W1_IMPULSE: ["s1"],
            ("Bullish", "W.(2)", "W.A Impulse"): ["x1", "x2"],
        }
        global_agg, sel_results = build(
            [sel_bull, sel_w2], strategy_patterns, stats_by_combo,
            _count_agg, lambda: {"n": 0, "empty": True},
        )
        labels = list(sel_results.keys())
        assert sel_results[labels[0]] == {"n": 1}
        assert sel_results[labels[1]] == {"n": 0, "empty": True}
        assert global_agg == {"n": 1}

    def test_order_follows_selections(self):
        build = _import_ui("ui.strategy_testing_tab", "build_selection_results")
        selection_label = _import_ui("data.helpers", "selection_label")
        sels = [
            _sel("Specified Primary", primary="W.(3)"),
            _sel("Secondary Across Primaries", pattern_type="Bearish", secondary="W.A Impulse"),
            _sel("Specified Primary", primary="W.(1)"),
        ]
        _, sel_results = build(sels, [], {}, _count_agg, _empty)
        assert list(sel_results.keys()) == [selection_label(s) for s in sels]

    def test_no_stats_gives_empty_global(self):
        build = _import_ui("ui.strategy_testing_tab", "build_selection_results")
        global_agg, sel_results = build(
            [_sel("All Patterns")], [], {}, _count_agg, _empty,
        )
        assert global_agg == {"n": 0}
        assert list(sel_results.keys()) == ["All Patterns (2)"]
        assert list(sel_results.values()) == [{"n": 0}]

    def test_no_rows_gives_empty_global_and_no_columns(self):
        build = _import_ui("ui.strategy_testing_tab", "build_selection_results")
        global_agg, sel_results = build([], [], {W1_IMPULSE: ["s1"]}, _count_agg, _empty)
        assert global_agg == {"n": 0}
        assert sel_results == {}


# ======================================================================
# _expand_and_filter
# ======================================================================

class TestExpandAndFilter:

    def test_strategy_filter_drops_non_matching(self):
        expand_and_filter = _import_ui("ui.strategy_testing_tab", "_expand_and_filter")
        combos = expand_and_filter(
            [_sel("Specified Primary", primary="W.(1)")],
            ["W.(1) → W.3 Impulse"],
        )
        assert combos == [W1_W3]

    def test_empty_strategy_patterns_passes_all(self):
        expand_and_filter = _import_ui("ui.strategy_testing_tab", "_expand_and_filter")
        expand_selection = _import_ui("data.helpers", "expand_selection")
        sel = _sel("Specified Primary", primary="W.(1)")
        assert expand_and_filter([sel], []) == expand_selection(sel)

    def test_union_is_deduplicated_in_first_seen_order(self):
        expand_and_filter = _import_ui("ui.strategy_testing_tab", "_expand_and_filter")
        sel_a = _sel("Specified Secondary", primary="W.(1)", secondary="W.3 Impulse")
        sel_b = _sel("Specified Primary", primary="W.(1)")
        combos = expand_and_filter([sel_a, sel_b], [])
        assert combos[0] == W1_W3
        assert len(combos) == len(set(combos))
        assert W1_IMPULSE in combos


# ======================================================================
# Session-state seeding
# ======================================================================

class TestInitTestingSelections:

    def test_legacy_key_seeds_all_three_and_is_removed(self):
        from utils.session_state import init_testing_selections, TESTING_SELECTION_KEYS
        legacy = [_sel("Specified Primary", primary="W.(2)")]
        state = {"testing_selections": legacy}
        init_testing_selections(state)
        assert "testing_selections" not in state
        for key in TESTING_SELECTION_KEYS:
            assert state[key] == legacy
            assert state[key] is not legacy
        # Deep copies: editing one section doesn't touch the others
        state["wfo_selections"][0]["primary"] = "W.(3)"
        assert state["sopt_selections"][0]["primary"] == "W.(2)"
        assert state["test_selections"][0]["primary"] == "W.(2)"

    def test_defaults_without_legacy(self):
        from utils.session_state import init_testing_selections, TESTING_SELECTION_KEYS
        state = {}
        init_testing_selections(state)
        for key in TESTING_SELECTION_KEYS:
            assert state[key] == [_sel("All Patterns")]
        state["wfo_selections"].append(_sel("All Bullish"))
        assert len(state["test_selections"]) == 1

    def test_existing_keys_are_kept(self):
        from utils.session_state import init_testing_selections
        existing = [_sel("All Bearish", pattern_type="Bearish")]
        state = {"test_selections": existing}
        init_testing_selections(state)
        assert state["test_selections"] is existing

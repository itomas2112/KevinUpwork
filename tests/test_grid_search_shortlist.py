"""Grid Search Shortlist — pure helpers, result-tuple plumbing, session state."""
import copy
from collections import OrderedDict
from datetime import datetime

import pytest

from tests.conftest import make_strategy
from ui.grid_search_helpers import (
    make_shortlist_entry, shortlist_has, shortlist_add, shortlist_remove,
    shortlist_strategy_for_save, result_strategy_for_variant,
)


def _columns():
    ap = {"num_trades": 10, "expected_value": 0.5, "trade_pnls_r": [1.0, -1.0]}
    return ap, OrderedDict([
        ("All Patterns", ap),
        ("All Bullish", {"num_trades": 6, "expected_value": 0.4}),
        ("All Bearish", {"num_trades": 4, "expected_value": 0.6}),
        ("Global", {"num_trades": 0, "expected_value": 0.0}),
    ])


def _entry(label="cand", base="Base", group="static_stop", group_set="Set A", strategy=None):
    ap, cols = _columns()
    return make_shortlist_entry(
        label, ap, cols, strategy if strategy is not None else make_strategy(strategy_name=base),
        base_strategy=base, search_group=group, group_set=group_set,
        scope="All Patterns",
        mc_settings={"instrument": "ES", "balance": 100000.0, "trades_per_sim": 100})


class TestMakeShortlistEntry:

    def test_has_id_and_added_at(self):
        e = make_shortlist_entry(
            "c", *_columns(), make_strategy(), base_strategy="B", search_group="trigger",
            group_set="S", scope="All Patterns", mc_settings={},
            now=datetime(2026, 9, 24, 10, 30, 0))
        assert isinstance(e["id"], str) and len(e["id"]) == 32
        assert e["added_at"] == "2026-09-24T10:30:00"
        assert _entry()["id"] != _entry()["id"]

    def test_deep_copies_columns_and_strategy(self):
        ap, cols = _columns()
        strategy = make_strategy()
        e = make_shortlist_entry(
            "c", ap, cols, strategy, base_strategy="B", search_group="trigger",
            group_set="S", scope="All Patterns", mc_settings={"balance": 1.0})
        expected_cols = copy.deepcopy(cols)
        expected_strategy = copy.deepcopy(strategy)

        # Mutate the live originals as MC re-enrichment / edits would
        ap["mc_avg_profit"] = 999.0
        ap["trade_pnls_r"].append(5.0)
        cols["All Bullish"]["num_trades"] = 0
        cols["New"] = {}
        strategy["strategy_name"] = "changed"
        strategy["indicator_settings"]["rsi_window"] = 99

        assert e["columns"] == expected_cols
        assert list(e["columns"].keys()) == list(expected_cols.keys())
        assert "mc_avg_profit" not in e["all_patterns_agg"]
        assert e["all_patterns_agg"]["trade_pnls_r"] == [1.0, -1.0]
        assert e["strategy"] == expected_strategy


class TestShortlistAddRemove:

    def test_add_newest_first_and_input_unmodified(self):
        a, b = _entry("a"), _entry("b")
        s0 = []
        s1 = shortlist_add(s0, a)
        s2 = shortlist_add(s1, b)
        assert s0 == [] and s1 == [a]
        assert [e["label"] for e in s2] == ["b", "a"]

    def test_add_rejects_duplicates(self):
        a = _entry("a")
        dup = _entry("a")                      # same identity, different id
        s = shortlist_add([], a)
        assert shortlist_has(s, dup)
        assert shortlist_add(s, dup) == [a]
        # Differs in any identity field -> not a duplicate
        for other in (_entry("a", base="Other"), _entry("a", group="trigger"),
                      _entry("a", group_set="Set B")):
            assert not shortlist_has(s, other)
            assert len(shortlist_add(s, other)) == 2

    def test_remove_by_id(self):
        a, b = _entry("a"), _entry("b")
        s = shortlist_add(shortlist_add([], a), b)
        assert shortlist_remove(s, a["id"]) == [b]
        assert len(s) == 2

    def test_remove_unknown_id_is_noop(self):
        a = _entry("a")
        assert shortlist_remove([a], "nope") == [a]


class TestShortlistStrategyForSave:

    def test_valid_strategy_gets_new_name(self):
        e = _entry("a")
        strategy, errors = shortlist_strategy_for_save(e, "My pick", ["Base"])
        assert errors == []
        assert strategy["strategy_name"] == "My pick"
        assert e["strategy"]["strategy_name"] == "Base"   # entry untouched

    def test_rejects_existing_name(self):
        strategy, errors = shortlist_strategy_for_save(_entry("a"), "Base", ["Base"])
        assert strategy is None
        assert errors and "already exists" in errors[0]

    def test_returns_validator_errors_for_broken_strategy(self):
        broken = make_strategy()
        broken["direction"] = "Sideways"
        broken.pop("exit_groups", None)
        strategy, errors = shortlist_strategy_for_save(_entry("a", strategy=broken), "X", [])
        assert strategy is None
        assert errors


class TestResultStrategyForVariant:

    def test_applies_variant_overrides_to_copy(self):
        base = make_strategy()
        base["indicator_settings"]["rsi_window"] = 14
        variant_groups = {"rsi": [(-1, {"rsi_window": 13}), (0, {"rsi_window": 14}),
                                  (1, {"rsi_window": 15})]}
        out = result_strategy_for_variant(base, (("rsi", 1),), variant_groups)
        assert out["indicator_settings"]["rsi_window"] == 15
        assert base["indicator_settings"]["rsi_window"] == 14

    def test_default_variant_is_plain_deep_copy(self):
        base = make_strategy()
        out = result_strategy_for_variant(base, None, {"rsi": [(0, {"rsi_window": 3})]})
        assert out == base and out is not base


class TestAssembleResultsCarriesStrategy:

    def test_four_tuple(self):
        try:
            from ui.grid_search_tab import _assemble_results
        except ImportError as exc:
            pytest.skip(f"Streamlit unavailable: {exc}")
        from data.helpers import all_combos
        strat = make_strategy()
        raw = [("c", {ck: [] for ck in all_combos()}, strat)]
        results, combo_list = _assemble_results(raw, [], all_combos())
        label, ap, cols, out_strategy = results[0]
        assert label == "c" and cols["All Patterns"] is ap
        assert out_strategy is strat


class TestShortlistSessionState:

    def test_initialises_to_empty_list(self):
        from utils.session_state import init_gs_shortlist
        state = {}
        init_gs_shortlist(state)
        assert state["gs_shortlist"] == []
        state["gs_shortlist"].append("x")
        init_gs_shortlist(state)
        assert state["gs_shortlist"] == ["x"]

    def test_survives_sidebar_cache_sweep(self):
        from utils.session_state import init_gs_shortlist
        state = {"_gs_cached_results": {"results": []}, "_gs_detail_page": 2}
        init_gs_shortlist(state)
        state["gs_shortlist"].append(_entry("a"))
        # Replicates ui/sidebar.py's timeframe-change sweep prefix check
        for k in list(state.keys()):
            if (k.startswith("_bt_cache_") or k.startswith("_perf_") or k.startswith("_gs_")
                    or k.startswith("_cv_") or k.startswith("_test_") or k.startswith("_corr_")):
                del state[k]
        assert "_gs_cached_results" not in state and "_gs_detail_page" not in state
        assert len(state["gs_shortlist"]) == 1

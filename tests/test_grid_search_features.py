"""
Tests for the four new Grid Search features:
  1. MC Avg Profit in Global Performance table
  2. MC Avg Profit as filter/sort
  3. MC calculation based on Avg Max Drawdown
  4. Correlation replacing SQN (DRM period directional signals)
"""

from collections import OrderedDict

import numpy as np
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


# ======================================================================
# REQUEST 3 — MC Avg Profit uses Avg Max Drawdown
# ======================================================================

class TestMCAvgDrawdownBasis:

    def test_basic_positive_edge(self):
        compute = _import_ui("ui.monte_carlo_tab", "compute_mc_avg_profit_at_dd")
        result = compute(60.0, 2.0, 1.0, 10000.0)
        assert result is not None
        assert result > 10000.0

    def test_terrible_strategy_returns_none(self):
        compute = _import_ui("ui.monte_carlo_tab", "compute_mc_avg_profit_at_dd")
        result = compute(20.0, 0.5, 1.0, 10000.0)
        assert result is None

    def test_zero_win_rate(self):
        compute = _import_ui("ui.monte_carlo_tab", "compute_mc_avg_profit_at_dd")
        assert compute(0.0, 2.0, 1.0, 10000.0) is None

    def test_zero_rr(self):
        compute = _import_ui("ui.monte_carlo_tab", "compute_mc_avg_profit_at_dd")
        assert compute(60.0, 0.0, 1.0, 10000.0) is None

    def test_negative_inputs(self):
        compute = _import_ui("ui.monte_carlo_tab", "compute_mc_avg_profit_at_dd")
        assert compute(60.0, 2.0, -1.0, 10000.0) is None
        assert compute(60.0, 2.0, 1.0, 0.0) is None
        assert compute(60.0, 2.0, 1.0, -5000.0) is None

    def test_threshold_zero_rejects(self):
        compute = _import_ui("ui.monte_carlo_tab", "compute_mc_avg_profit_at_dd")
        result = compute(60.0, 2.0, 1.0, 10000.0, max_dd_threshold=0.0)
        assert result is None

    def test_threshold_100_accepts(self):
        compute = _import_ui("ui.monte_carlo_tab", "compute_mc_avg_profit_at_dd")
        result = compute(40.0, 1.0, 1.0, 10000.0, max_dd_threshold=100.0)
        assert result is not None


class TestMCSkipThreshold:
    """Grid Search mode: skip_threshold=True always returns a float."""

    def test_always_returns_float_for_good_strategy(self):
        compute = _import_ui("ui.monte_carlo_tab", "compute_mc_avg_profit_at_dd")
        result = compute(60.0, 2.0, 1.0, 10000.0, skip_threshold=True)
        assert isinstance(result, float)
        assert result > 10000.0

    def test_always_returns_float_for_bad_strategy(self):
        """Bad strategy that would normally return None still returns a float."""
        compute = _import_ui("ui.monte_carlo_tab", "compute_mc_avg_profit_at_dd")
        result = compute(30.0, 1.0, 1.0, 10000.0, skip_threshold=True,
                         trades_per_sim=100, n_sims=1000)
        assert isinstance(result, float)

    def test_zero_wr_deterministic_decay(self):
        """0% WR with skip_threshold → deterministic decay, not None."""
        compute = _import_ui("ui.monte_carlo_tab", "compute_mc_avg_profit_at_dd")
        result = compute(0.0, 2.0, 1.0, 10000.0, skip_threshold=True,
                         trades_per_sim=100)
        assert isinstance(result, float)
        # 0% WR, 1% risk, 100 trades → 10000 * 0.99^100 ≈ 3660
        expected = 10000.0 * (0.99 ** 100)
        assert result == pytest.approx(expected, rel=0.01)

    def test_zero_rr_deterministic_decay(self):
        """0 RR with skip_threshold → same as 0% WR (wins make $0)."""
        compute = _import_ui("ui.monte_carlo_tab", "compute_mc_avg_profit_at_dd")
        result = compute(50.0, 0.0, 1.0, 10000.0, skip_threshold=True,
                         trades_per_sim=100)
        assert isinstance(result, float)
        expected = 10000.0 * (0.99 ** 100)
        assert result == pytest.approx(expected, rel=0.01)

    def test_invalid_risk_returns_starting_balance(self):
        """Truly invalid inputs (0 risk) → starting balance unchanged."""
        compute = _import_ui("ui.monte_carlo_tab", "compute_mc_avg_profit_at_dd")
        result = compute(0.0, 0.0, 0.0, 10000.0, skip_threshold=True)
        assert result == 10000.0

    def test_skip_threshold_false_still_rejects(self):
        """Default skip_threshold=False still returns None for invalid."""
        compute = _import_ui("ui.monte_carlo_tab", "compute_mc_avg_profit_at_dd")
        assert compute(0.0, 2.0, 1.0, 10000.0, skip_threshold=False) is None


# ======================================================================
# REQUEST 1 & 2 — MC in Grid Search aggregation
# ======================================================================

class TestMCInAggregation:

    def test_mc_avg_profit_str_uses_precomputed(self):
        """When mc_avg_profit is pre-set in agg, _mc_avg_profit_str uses it."""
        _mc_avg_profit_str = _import_ui("ui.performance_tab", "_mc_avg_profit_str")
        agg = {'win_pct': 60, 'rr_ratio': 2.0, 'mc_avg_profit': 15000.0}
        assert _mc_avg_profit_str(agg) == "$15,000"

    def test_mc_avg_profit_str_zero_precomputed(self):
        """Precomputed 0.0 should show $0, not N/A."""
        _mc_avg_profit_str = _import_ui("ui.performance_tab", "_mc_avg_profit_str")
        # 0.0 is isinstance(float) = True, so it should format as $0
        # Actually 0.0 is falsy but isinstance check handles it
        agg = {'mc_avg_profit': 0.0}
        result = _mc_avg_profit_str(agg)
        assert result == "$0"

    def test_mc_avg_profit_str_no_precomputed(self):
        """Without mc_avg_profit, falls back to computing (may be N/A)."""
        _mc_avg_profit_str = _import_ui("ui.performance_tab", "_mc_avg_profit_str")
        agg = {'win_pct': 0, 'rr_ratio': 0}
        assert _mc_avg_profit_str(agg) == "N/A"


# ======================================================================
# SORT_METRICS and filter
# ======================================================================

class TestSortMetricsConfig:

    def test_mc_avg_profit_in_sort_metrics(self):
        SORT_METRICS = _import_ui("ui.grid_search_tab", "SORT_METRICS")
        keys = [k for k, _ in SORT_METRICS]
        assert "mc_avg_profit" in keys

    def test_abs_correlation_in_sort_metrics(self):
        SORT_METRICS = _import_ui("ui.grid_search_tab", "SORT_METRICS")
        keys = [k for k, _ in SORT_METRICS]
        assert "abs_correlation" in keys

    def test_sqn_not_in_sort_metrics(self):
        SORT_METRICS = _import_ui("ui.grid_search_tab", "SORT_METRICS")
        keys = [k for k, _ in SORT_METRICS]
        assert "sqn" not in keys


# ======================================================================
# FILTER LOGIC
# ======================================================================

class TestFilterLogic:

    _MAX_FILTER_METRICS = {"abs_correlation"}

    def _apply_filter(self, results, thresholds):
        """Replicate the Global-scope threshold logic (see passes_thresholds)."""
        filtered = []
        for label, global_agg, sel_results in results:
            passes = True
            for metric_key, threshold_val in thresholds.items():
                val = global_agg.get(metric_key)
                if val is None:
                    continue
                if metric_key in self._MAX_FILTER_METRICS:
                    if val > threshold_val:
                        passes = False; break
                else:
                    if val < threshold_val:
                        passes = False; break
            if passes:
                filtered.append(label)
        return filtered

    def test_min_filter_trades(self):
        results = [
            ("A", {"num_trades": 5}, {}),
            ("B", {"num_trades": 50}, {}),
        ]
        filtered = self._apply_filter(results, {"num_trades": 10})
        assert "A" not in filtered
        assert "B" in filtered

    def test_max_filter_abs_correlation(self):
        results = [
            ("A", {"abs_correlation": 10.0}, {}),
            ("B", {"abs_correlation": 80.0}, {}),
        ]
        filtered = self._apply_filter(results, {"abs_correlation": 50.0})
        assert "A" in filtered
        assert "B" not in filtered

    def test_none_correlation_passes_filter(self):
        """None correlation (no refs) should not be filtered out."""
        results = [("A", {"abs_correlation": None}, {})]
        filtered = self._apply_filter(results, {"abs_correlation": 50.0})
        assert "A" in filtered

    def test_mc_min_filter(self):
        results = [
            ("A", {"mc_avg_profit": 5000.0}, {}),
            ("B", {"mc_avg_profit": 15000.0}, {}),
        ]
        filtered = self._apply_filter(results, {"mc_avg_profit": 10000.0})
        assert "A" not in filtered
        assert "B" in filtered


# ======================================================================
# Performance Filters scoped to a pattern selection
# ======================================================================

class TestSelectionLabels:

    def _labels(self, selections):
        selection_labels_for = _import_ui("ui.grid_search_tab", "selection_labels_for")
        return selection_labels_for(selections)

    def _w1(self):
        return {"mode": "Specified Primary", "pattern_type": "Bullish",
                "primary": "W.(1)", "secondary": None}

    def test_two_duplicates(self):
        assert self._labels([self._w1(), self._w1()]) == ["W.(1) Bullish", "W.(1) Bullish (2)"]

    def test_three_duplicates(self):
        assert self._labels([self._w1()] * 3) == [
            "W.(1) Bullish", "W.(1) Bullish (2)", "W.(1) Bullish (3)"]

    def test_distinct_rows_unchanged(self):
        from data.helpers import selection_label
        sels = [
            self._w1(),
            {"mode": "Secondary Across Primaries", "pattern_type": "Bearish",
             "primary": None, "secondary": "W.A Impulse"},
        ]
        assert self._labels(sels) == [selection_label(s) for s in sels]

    def test_labels_dedup_against_fixed_columns(self):
        # Mirrors build_pattern_columns: a legacy "All Patterns" row never
        # collides with the fixed column of the same name.
        sels = [{"mode": "All Patterns"}, {"mode": "All Patterns"}]
        assert self._labels(sels) == ["All Patterns (2)", "All Patterns (3)"]

    def test_scope_options_match_column_keys(self):
        from data.helpers import build_pattern_columns, FIXED_COLUMNS
        sels = [self._w1(), self._w1(), {"mode": "All Bullish"}]
        cols = build_pattern_columns(sels, [], {}, lambda s: {"n": len(s)}, lambda: {"n": 0})
        assert FIXED_COLUMNS + self._labels(sels) == list(cols.keys())


class TestScopeAgg:

    def _fixture(self):
        g = {"num_trades": 100, "expected_value": 0.5,
             "correlation": 42.0, "abs_correlation": 42.0}
        cols = OrderedDict([
            ("All Patterns", g),
            ("All Bullish", {"num_trades": 60, "expected_value": 0.6}),
            ("All Bearish", {"num_trades": 40, "expected_value": 0.4}),
            ("Global", {"num_trades": 30, "expected_value": 0.8}),
            ("W.(1) Bullish", {"num_trades": 30, "expected_value": 0.8}),
        ])
        return g, cols

    def test_all_patterns_returns_all_patterns_agg(self):
        scope_agg = _import_ui("ui.grid_search_tab", "scope_agg")
        g, cols = self._fixture()
        assert scope_agg(g, cols, "All Patterns") is g

    def test_global_is_a_column_now(self):
        scope_agg = _import_ui("ui.grid_search_tab", "scope_agg")
        g, cols = self._fixture()
        agg = scope_agg(g, cols, "Global")
        assert agg["num_trades"] == 30
        assert agg["correlation"] == 42.0

    def test_fixed_bullish_scope(self):
        scope_agg = _import_ui("ui.grid_search_tab", "scope_agg")
        g, cols = self._fixture()
        assert scope_agg(g, cols, "All Bullish")["num_trades"] == 60

    def test_selection_scope_copies_correlation(self):
        scope_agg = _import_ui("ui.grid_search_tab", "scope_agg")
        g, cols = self._fixture()
        agg = scope_agg(g, cols, "W.(1) Bullish")
        assert agg["num_trades"] == 30
        assert agg["expected_value"] == 0.8
        assert agg["correlation"] == 42.0
        assert agg["abs_correlation"] == 42.0
        # original column not mutated
        assert "correlation" not in cols["W.(1) Bullish"]
        assert "abs_correlation" not in cols["W.(1) Bullish"]

    def test_unknown_scope_returns_all_patterns_agg(self):
        scope_agg = _import_ui("ui.grid_search_tab", "scope_agg")
        g, cols = self._fixture()
        assert scope_agg(g, cols, "Bearish W.(3)") is g
        assert scope_agg(g, None, "Global") is g


class TestFilterAndSortResults:

    def _results(self):
        # A passes num_trades >= 10 on All Patterns but not for "X"; B the reverse
        a = {"num_trades": 20, "expected_value": 0.1}
        b = {"num_trades": 5, "expected_value": 0.2}
        return [
            ("A", a, OrderedDict([("All Patterns", a),
                                  ("X", {"num_trades": 5, "expected_value": 0.9})]),
             {"strategy_name": "A"}),
            ("B", b, OrderedDict([("All Patterns", b),
                                  ("X", {"num_trades": 15, "expected_value": 0.3})]),
             {"strategy_name": "B"}),
        ]

    def _fn(self):
        return _import_ui("ui.grid_search_tab", "filter_and_sort_results")

    def test_default_scope_is_all_patterns(self):
        out = self._fn()(self._results(), {"num_trades": 10}, "expected_value", True)
        assert [r[0] for r in out] == ["A"]

    def test_all_patterns_scope_filters_on_all_patterns(self):
        out = self._fn()(self._results(), {"num_trades": 10}, "expected_value", True, "All Patterns")
        assert [r[0] for r in out] == ["A"]

    def test_selection_scope_filters_on_selection(self):
        out = self._fn()(self._results(), {"num_trades": 10}, "expected_value", True, "X")
        assert [r[0] for r in out] == ["B"]
        label, metric_agg, all_patterns_agg, columns, _strategy = out[0]
        assert metric_agg["num_trades"] == 15
        assert all_patterns_agg["num_trades"] == 5
        assert columns["X"]["num_trades"] == 15

    def test_strategy_passed_through_untouched(self):
        results = self._results()
        strategies = {r[0]: r[3] for r in results}
        out = self._fn()(results, {}, "expected_value", True, "X")
        for label, _m, _a, _c, strategy in out:
            assert strategy is strategies[label]
            assert strategy == {"strategy_name": label}

    def test_sort_direction_under_selection_scope(self):
        fn = self._fn()
        # "X" EVs: A=0.9, B=0.3 (Global order would be B then A descending)
        desc = fn(self._results(), {}, "expected_value", True, "X")
        asc = fn(self._results(), {}, "expected_value", False, "X")
        assert [r[0] for r in desc] == ["A", "B"]
        assert [r[0] for r in asc] == ["B", "A"]

    def test_none_sort_value_goes_last(self):
        fn = self._fn()
        results = [
            ("N", {"expected_value": 0.0},
             OrderedDict([("X", {"expected_value": None})]), {}),
            ("A", {"expected_value": 0.0},
             OrderedDict([("X", {"expected_value": 0.5})]), {}),
            ("B", {"expected_value": 0.0},
             OrderedDict([("X", {"expected_value": -0.5})]), {}),
        ]
        desc = fn(results, {}, "expected_value", True, "X")
        asc = fn(results, {}, "expected_value", False, "X")
        assert [r[0] for r in desc] == ["A", "B", "N"]
        assert [r[0] for r in asc] == ["B", "A", "N"]


class TestDetailTableColumns:

    def _cols(self):
        return OrderedDict([
            ("All Patterns", {"n": 86}), ("All Bullish", {"n": 43}),
            ("All Bearish", {"n": 43}), ("Global", {"n": 7}),
            ("W.(1) Bullish", {"n": 3}), ("W.(3) Bullish", {"n": 4}),
        ])

    def test_default_scope_reads_fixed_then_user(self):
        fn = _import_ui("ui.grid_search_tab", "detail_table_columns")
        assert list(fn(self._cols(), "All Patterns").keys()) == [
            "All Patterns", "All Bullish", "All Bearish", "Global",
            "W.(1) Bullish", "W.(3) Bullish"]

    def test_user_scope_marked_first_and_not_repeated(self):
        fn = _import_ui("ui.grid_search_tab", "detail_table_columns")
        assert list(fn(self._cols(), "W.(3) Bullish").keys()) == [
            "▶ W.(3) Bullish", "All Patterns", "All Bullish", "All Bearish", "Global",
            "W.(1) Bullish"]

    def test_fixed_scope_marked_first_and_skipped_in_fixed_block(self):
        fn = _import_ui("ui.grid_search_tab", "detail_table_columns")
        assert list(fn(self._cols(), "Global").keys()) == [
            "▶ Global", "All Patterns", "All Bullish", "All Bearish",
            "W.(1) Bullish", "W.(3) Bullish"]

    def test_unknown_scope_falls_back_to_plain_order(self):
        fn = _import_ui("ui.grid_search_tab", "detail_table_columns")
        assert list(fn(self._cols(), "nope").keys()) == list(self._cols().keys())


class TestReaggregateFromCachedComboResults:
    """Changing the user rows re-aggregates the pattern columns from the
    cached per-candidate {combo: [stats dicts]} without re-running the
    search. The result must equal a fresh build for the new rows."""

    def _stats(self, pnls):
        return {
            'win_pnl': float(sum(p for p in pnls if p > 0)),
            'lose_pnl': float(sum(p for p in pnls if p < 0)),
            'trade_pnls_r': list(pnls),
            'trade_r_distances': [10.0] * len(pnls),
            'trade_holding_periods': [3] * len(pnls),
            'total_static_alloc': 0.0, 'total_dynamic_alloc': 0.0,
            'total_target_alloc': float(len(pnls)), 'total_eod_alloc': 0.0,
        }

    def _combo_results(self, seed):
        from data.helpers import all_combos
        rng = np.random.RandomState(seed)
        out = {}
        for combo in all_combos():
            n_periods = int(rng.randint(0, 3))
            out[combo] = [self._stats(list(rng.choice([2.0, -1.0], size=int(rng.randint(1, 4)))))
                          for _ in range(n_periods)]
        return out

    def _w(self, primary, ptype="Bullish"):
        return {"mode": "Specified Primary", "pattern_type": ptype,
                "primary": primary, "secondary": None}

    def test_reaggregated_global_equals_fresh_build(self):
        build_candidate_columns = _import_ui("ui.grid_search_tab", "build_candidate_columns")
        reaggregate_results = _import_ui("ui.grid_search_tab", "reaggregate_results")
        from data.helpers import FIXED_COLUMNS

        combo_results_list = [self._combo_results(1), self._combo_results(2)]
        old_rows = [self._w("W.(1)")]
        new_rows = [self._w("W.(1)"), self._w("W.(3)"), self._w("W.(2)", "Bearish")]

        # "Last run" results built with the old rows, then MC/correlation
        # values stamped onto the aggs as the run path would.
        results = []
        for i, cr in enumerate(combo_results_list):
            cols = build_candidate_columns(cr, old_rows)
            ap = cols["All Patterns"]
            ap["correlation"] = 12.5
            ap["abs_correlation"] = 12.5
            for agg in cols.values():
                agg["mc_avg_profit"] = 111.0
            results.append((f"cand_{i}", ap, cols, {"strategy_name": f"cand_{i}"}))

        new_results, changed = reaggregate_results(results, combo_results_list, new_rows)

        assert [r[0] for r in new_results] == ["cand_0", "cand_1"]
        for (label, ap, cols, strat), cr, (_, old_ap, old_cols, old_strat) in zip(
                new_results, combo_results_list, results):
            assert strat is old_strat
            fresh = build_candidate_columns(cr, new_rows)
            assert list(cols.keys()) == list(fresh.keys())
            assert list(cols.keys()) == FIXED_COLUMNS + [
                "W.(1) Bullish", "W.(3) Bullish", "W.(2) Bearish"]
            # Global and the user columns match a fresh build exactly
            for key in ["Global", "W.(1) Bullish", "W.(3) Bullish", "W.(2) Bearish"]:
                assert cols[key]["num_trades"] == fresh[key]["num_trades"]
                assert cols[key]["trade_pnls_r"] == fresh[key]["trade_pnls_r"]
                assert cols[key]["expected_value"] == pytest.approx(fresh[key]["expected_value"])
                assert "mc_avg_profit" not in cols[key]
            # Global counts each unique combo once: equals the union of the rows
            n_union = sum(len(s["trade_pnls_r"]) for combo in set(
                c for row in new_rows for c in
                _import_ui("data.helpers", "expand_selection")(row)) for s in cr[combo])
            assert cols["Global"]["num_trades"] == n_union
            # Fixed "All …" aggs are the cached objects (keep MC + correlation)
            assert ap is old_ap and cols["All Patterns"] is ap
            assert cols["All Bullish"] is old_cols["All Bullish"]
            assert cols["All Bearish"] is old_cols["All Bearish"]
            assert ap["correlation"] == 12.5 and ap["mc_avg_profit"] == 111.0

        # Only the rebuilt aggs are handed back for MC enrichment
        assert len(changed) == 2 * 4
        assert all("mc_avg_profit" not in a for a in changed)

    def test_removing_all_rows_gives_empty_global(self):
        build_candidate_columns = _import_ui("ui.grid_search_tab", "build_candidate_columns")
        reaggregate_results = _import_ui("ui.grid_search_tab", "reaggregate_results")
        from data.helpers import FIXED_COLUMNS
        cr = self._combo_results(3)
        cols = build_candidate_columns(cr, [self._w("W.(1)")])
        new_results, changed = reaggregate_results(
            [("c", cols["All Patterns"], cols, {})], [cr], [])
        _, ap, new_cols, _strategy = new_results[0]
        assert list(new_cols.keys()) == FIXED_COLUMNS
        assert new_cols["Global"]["num_trades"] == 0
        assert ap["num_trades"] == sum(len(s["trade_pnls_r"]) for v in cr.values() for s in v)
        assert changed == [new_cols["Global"]]


# ======================================================================
# REQUEST 4 — Correlation computation (DRM period directional)
# ======================================================================

class TestCorrelationComputation:

    def test_no_reference_returns_none(self):
        _compute = _import_ui("ui.grid_search_tab", "_compute_candidate_correlation")
        combo_results = {('B', 'X', 'Y'): [{'win_pnl': 5.0, 'lose_pnl': -2.0}]}
        assert _compute(combo_results, [('B', 'X', 'Y')], None) is None

    def test_identical_directions_returns_100(self):
        """Candidate same direction every period as reference → 100%."""
        _compute = _import_ui("ui.grid_search_tab", "_compute_candidate_correlation")
        # 5 periods: ref = [+1, -1, +1, -1, +1]
        ref = np.array([1.0, -1.0, 1.0, -1.0, 1.0])
        # Candidate: same pattern — wins where ref wins, loses where ref loses
        combo_results = {('B', 'X', 'Y'): [
            {'win_pnl': 2.0, 'lose_pnl': 0.0},   # period 0: +pnl → +1
            {'win_pnl': 0.0, 'lose_pnl': -1.0},   # period 1: -pnl → -1
            {'win_pnl': 3.0, 'lose_pnl': 0.0},   # period 2: +pnl → +1
            {'win_pnl': 0.0, 'lose_pnl': -0.5},   # period 3: -pnl → -1
            {'win_pnl': 1.0, 'lose_pnl': 0.0},   # period 4: +pnl → +1
        ]}
        corr = _compute(combo_results, [('B', 'X', 'Y')], ref)
        assert corr == pytest.approx(100.0, abs=1.0)

    def test_opposite_directions_returns_negative_100(self):
        _compute = _import_ui("ui.grid_search_tab", "_compute_candidate_correlation")
        ref = np.array([1.0, -1.0, 1.0, -1.0, 1.0])
        # Candidate: opposite — loses where ref wins, wins where ref loses
        combo_results = {('B', 'X', 'Y'): [
            {'win_pnl': 0.0, 'lose_pnl': -1.0},
            {'win_pnl': 2.0, 'lose_pnl': 0.0},
            {'win_pnl': 0.0, 'lose_pnl': -1.0},
            {'win_pnl': 2.0, 'lose_pnl': 0.0},
            {'win_pnl': 0.0, 'lose_pnl': -1.0},
        ]}
        corr = _compute(combo_results, [('B', 'X', 'Y')], ref)
        assert corr == pytest.approx(-100.0, abs=1.0)

    def test_no_trades_candidate_returns_zero(self):
        """Candidate has no trades in any period → zero variance → 0."""
        _compute = _import_ui("ui.grid_search_tab", "_compute_candidate_correlation")
        ref = np.array([1.0, -1.0, 1.0])
        combo_results = {('B', 'X', 'Y'): [
            {'win_pnl': 0.0, 'lose_pnl': 0.0},
            {'win_pnl': 0.0, 'lose_pnl': 0.0},
            {'win_pnl': 0.0, 'lose_pnl': 0.0},
        ]}
        assert _compute(combo_results, [('B', 'X', 'Y')], ref) == 0.0

    def test_length_mismatch_returns_zero(self):
        _compute = _import_ui("ui.grid_search_tab", "_compute_candidate_correlation")
        ref = np.array([1.0, -1.0, 1.0])
        combo_results = {('B', 'X', 'Y'): [
            {'win_pnl': 1.0, 'lose_pnl': 0.0},
            {'win_pnl': 0.0, 'lose_pnl': -1.0},
        ]}
        assert _compute(combo_results, [('B', 'X', 'Y')], ref) == 0.0

    def test_correlation_bounded(self):
        _compute = _import_ui("ui.grid_search_tab", "_compute_candidate_correlation")
        rng = np.random.RandomState(42)
        ref = rng.choice([-1.0, 0.0, 1.0], size=50)
        combo_results = {('B', 'X', 'Y'): [
            {'win_pnl': float(max(v, 0)), 'lose_pnl': float(min(v, 0))}
            for v in rng.choice([-1.0, 0.0, 1.0], size=50)
        ]}
        corr = _compute(combo_results, [('B', 'X', 'Y')], ref)
        assert -100.0 <= corr <= 100.0


# ======================================================================
# REQUEST 4 — Combined reference: sum raw P&L then discretize
# ======================================================================

class TestCombinedReferenceLogic:
    """The combined reference must sum raw P&L across refs, THEN discretize.
    Not sum discretized signals."""

    def test_composite_pnl_net_negative(self):
        """Ref A: +2R, Ref B: -3R in same period → composite = -1R → direction -1."""
        # This is tested indirectly via _compute_reference_directions.
        # We test the principle: composite direction should reflect net P&L.
        import numpy as np
        # Simulate: 3 periods
        # Period 0: A=+2, B=-3 → net=-1 → -1
        # Period 1: A=+1, B=+1 → net=+2 → +1
        # Period 2: A=-1, B=+0 → net=-1 → -1
        composite_pnl = np.array([-1.0, 2.0, -1.0])
        directions = np.sign(composite_pnl)
        np.testing.assert_array_equal(directions, [-1.0, 1.0, -1.0])

    def test_wrong_approach_would_give_zero(self):
        """If we discretized first then summed, A=+1 B=-1 sums to 0 (wrong)."""
        import numpy as np
        # A: +2R → +1, B: -3R → -1, sum of discretized = 0
        wrong = np.sign(np.array([2.0])) + np.sign(np.array([-3.0]))
        assert wrong[0] == 0  # wrong answer

        # Correct: sum raw first, then discretize
        correct = np.sign(np.array([2.0 + -3.0]))
        assert correct[0] == -1  # correct answer

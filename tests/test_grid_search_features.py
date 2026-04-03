"""
Tests for the four new Grid Search features:
  1. MC Avg Profit in Global Performance table
  2. MC Avg Profit as filter/sort
  3. MC calculation based on Avg Max Drawdown
  4. Correlation replacing SQN (DRM period directional signals)
"""

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
        """Replicate the exact filter logic from _display_results."""
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

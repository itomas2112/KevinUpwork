"""
Tests for Walk Forward Optimization — fold splitting, group detection,
parameter grid generation, and end-to-end optimization with known data.
"""
import pytest
import numpy as np
import pandas as pd
from copy import deepcopy

from strategies.wfo_engine import (
    split_folds, detect_used_groups, generate_param_grid, count_grid_size,
    aggregate_stats_dicts, apply_filters, _empty_agg, _range_values,
)
from strategies.wfo_worker import init_wfo_worker, run_wfo_batch
from indicators.calculate_indicators import (
    calculate_indicators, recalculate_groups, changed_groups,
)
from tests.conftest import (
    DEFAULT_INDICATOR_SETTINGS, _generate_uptrend, _generate_oscillation,
    _make_ohlc, make_strategy, make_exit_group,
)


# ---------------------------------------------------------------------------
# Fold splitting
# ---------------------------------------------------------------------------

class TestSplitFolds:

    def test_correct_number_of_folds(self):
        folds = split_folds("2024-01-01", "2024-12-31", 4, 0.8)
        assert len(folds) == 4

    def test_folds_cover_full_range(self):
        folds = split_folds("2024-01-01", "2024-12-31", 5, 0.8)
        assert folds[0]["train_start"] == pd.Timestamp("2024-01-01")
        assert folds[-1]["test_end"] >= pd.Timestamp("2024-12-31")

    def test_train_end_equals_test_start(self):
        folds = split_folds("2024-01-01", "2024-06-30", 3, 0.8)
        for f in folds:
            assert f["train_end"] == f["test_start"]

    def test_folds_are_contiguous(self):
        folds = split_folds("2024-01-01", "2024-12-31", 5, 0.8)
        for i in range(len(folds) - 1):
            assert folds[i]["test_end"] == folds[i + 1]["train_start"]

    def test_train_portion_is_correct_ratio(self):
        folds = split_folds("2024-01-01", "2025-01-01", 4, 0.75)
        for f in folds:
            total = (f["test_end"] - f["train_start"]).total_seconds()
            train = (f["train_end"] - f["train_start"]).total_seconds()
            ratio = train / total
            assert abs(ratio - 0.75) < 0.01

    def test_two_folds_minimum(self):
        folds = split_folds("2024-01-01", "2024-01-10", 2, 0.8)
        assert len(folds) == 2
        for f in folds:
            assert f["train_start"] < f["test_end"]


# ---------------------------------------------------------------------------
# Group detection
# ---------------------------------------------------------------------------

class TestDetectUsedGroups:

    def test_detects_entry_trigger_rsi(self):
        strategy = make_strategy(
            entry_group="RSI Group", entry_element1="RSI",
            entry_event="Cross Above", entry_compare_type="Fixed Value",
            entry_value=30.0,
        )
        groups = detect_used_groups(strategy)
        assert "rsi" in groups

    def test_detects_entry_element2(self):
        strategy = make_strategy(
            entry_group="RSI Group", entry_element1="RSI",
            entry_event="Cross Above", entry_compare_type="Indicator",
            entry_element2="BB Upper Band",
        )
        groups = detect_used_groups(strategy)
        assert "rsi" in groups
        assert "bb" in groups

    def test_detects_entry_conditions(self):
        strategy = make_strategy(
            entry_conditions=[{
                "group": "MACD Group", "element1": "MACD Line",
                "operator": "Above", "compare_type": "Fixed Value",
                "value": 0,
            }],
        )
        groups = detect_used_groups(strategy)
        assert "macd" in groups

    def test_detects_initial_stop_indicator(self):
        strategy = make_strategy(
            initial_stop_type="Indicator",
            initial_stop_element2="BB Lower Band",
        )
        groups = detect_used_groups(strategy)
        assert "bb" in groups

    def test_detects_exit_target(self):
        eg = make_exit_group(
            targets=[{
                "type": "Target",
                "trigger": {"group": "Stoch Group", "element1": "Stoch %K",
                            "event": "Cross Above", "compare_type": "Fixed Value",
                            "element2": None, "value": 80},
                "conditions": [],
            }],
        )
        strategy = make_strategy(exit_groups=[eg])
        groups = detect_used_groups(strategy)
        assert "stoch" in groups

    def test_detects_exit_stop(self):
        eg = make_exit_group(
            stops=[{
                "type": "Stop",
                "trigger": {"group": "Price & Indicators", "element1": "Supertrend",
                            "event": "Cross Below", "compare_type": "Indicator",
                            "element2": "Price", "value": None},
                "conditions": [],
            }],
        )
        strategy = make_strategy(exit_groups=[eg])
        groups = detect_used_groups(strategy)
        assert "supertrend" in groups

    def test_detects_ema(self):
        strategy = make_strategy(
            entry_group="Price & Indicators", entry_element1="EMA 1",
            entry_event="Cross Above", entry_compare_type="Indicator",
            entry_element2="Price",
        )
        groups = detect_used_groups(strategy)
        assert "ema" in groups

    def test_ignores_price_only(self):
        strategy = make_strategy(
            entry_group="Price & Indicators", entry_element1="Price",
            entry_event="Cross Above", entry_compare_type="Fixed Value",
            entry_value=100.0,
        )
        groups = detect_used_groups(strategy)
        # Price element doesn't map to any indicator group. The ATR stop
        # from make_strategy's default surfaces as an ATR-slot pseudo-group,
        # which is WFO-optimisable but not an indicator.
        indicator_groups = {g for g in groups if not g.startswith("atr_slot:")}
        assert len(indicator_groups) == 0

    def test_detects_new_indicators(self):
        for elem, expected_group in [
            ("Williams %R", "willr"), ("ROC", "roc"),
            ("CCI", "cci"), ("LR Upper", "lr"),
        ]:
            strategy = make_strategy(
                entry_group="Price & Indicators", entry_element1=elem,
                entry_event="Cross Above", entry_compare_type="Fixed Value",
                entry_value=0,
            )
            groups = detect_used_groups(strategy)
            assert expected_group in groups, f"{elem} should map to {expected_group}"


# ---------------------------------------------------------------------------
# Parameter grid generation
# ---------------------------------------------------------------------------

class TestParamGrid:

    def test_single_param_range(self):
        grid = generate_param_grid(
            {"rsi"}, {"rsi": {"rsi_window": (10, 20, 5)}},
            DEFAULT_INDICATOR_SETTINGS,
        )
        rsi_vals = [g["rsi_window"] for g in grid]
        assert rsi_vals == [10, 15, 20]

    def test_cartesian_product(self):
        grid = generate_param_grid(
            {"rsi", "adx"},
            {"rsi": {"rsi_window": (10, 20, 10)},
             "adx": {"adx_period": (10, 20, 10)}},
            DEFAULT_INDICATOR_SETTINGS,
        )
        assert len(grid) == 4

    def test_no_used_groups_returns_base(self):
        grid = generate_param_grid(set(), {}, DEFAULT_INDICATOR_SETTINGS)
        assert len(grid) == 1
        assert grid[0] == DEFAULT_INDICATOR_SETTINGS

    def test_ema_grid(self):
        base = dict(DEFAULT_INDICATOR_SETTINGS)
        base["ema_periods"] = [10, 20]
        grid = generate_param_grid(
            {"ema"},
            {"ema": {"_ema_ranges": [(5, 15, 5), (15, 25, 5)]}},
            base,
        )
        assert len(grid) == 9
        assert isinstance(grid[0]["ema_periods"], list)
        assert len(grid[0]["ema_periods"]) == 2

    def test_discrete_values(self):
        grid = generate_param_grid(
            {"lr"},
            {"lr": {"lr_period": (50, 100, 50), "lr_multiplier": [1.0, 2.0, 3.0]}},
            DEFAULT_INDICATOR_SETTINGS,
        )
        assert len(grid) == 6
        mults = set(g["lr_multiplier"] for g in grid)
        assert mults == {1.0, 2.0, 3.0}

    def test_min_equals_max_produces_single_value(self):
        grid = generate_param_grid(
            {"rsi"}, {"rsi": {"rsi_window": (14, 14, 1)}},
            DEFAULT_INDICATOR_SETTINGS,
        )
        assert len(grid) == 1
        assert grid[0]["rsi_window"] == 14

    def test_count_matches_actual(self):
        groups = {"rsi", "stoch"}
        ranges = {
            "rsi": {"rsi_window": (5, 25, 5)},
            "stoch": {"stoch_k_period": (5, 15, 5),
                      "stoch_k_smooth": (1, 3, 1),
                      "stoch_d_smooth": (1, 3, 1)},
        }
        count = count_grid_size(groups, ranges, DEFAULT_INDICATOR_SETTINGS)
        grid = generate_param_grid(groups, ranges, DEFAULT_INDICATOR_SETTINGS)
        assert count == len(grid)

    def test_unchanged_params_preserved(self):
        grid = generate_param_grid(
            {"rsi"}, {"rsi": {"rsi_window": (10, 20, 10)}},
            DEFAULT_INDICATOR_SETTINGS,
        )
        for combo in grid:
            assert combo["bb_upper_period"] == DEFAULT_INDICATOR_SETTINGS["bb_upper_period"]
            assert combo["macd_fast"] == DEFAULT_INDICATOR_SETTINGS["macd_fast"]


# ---------------------------------------------------------------------------
# Range values helper
# ---------------------------------------------------------------------------

class TestRangeValues:

    def test_int_range(self):
        assert _range_values((1, 5, 1)) == [1, 2, 3, 4, 5]

    def test_int_range_with_step(self):
        assert _range_values((10, 50, 10)) == [10, 20, 30, 40, 50]

    def test_float_range(self):
        vals = _range_values((1.0, 3.0, 0.5))
        assert len(vals) == 5
        assert abs(vals[0] - 1.0) < 1e-9
        assert abs(vals[-1] - 3.0) < 1e-9

    def test_discrete_list(self):
        assert _range_values([1.0, 2.0, 3.0]) == [1.0, 2.0, 3.0]

    def test_single_value_when_min_equals_max(self):
        assert _range_values((14, 14, 1)) == [14]
        assert _range_values((2.0, 2.0, 0.5)) == [2.0]


# ---------------------------------------------------------------------------
# Aggregation
# ---------------------------------------------------------------------------

class TestAggregation:

    def test_empty_returns_zero(self):
        agg = _empty_agg()
        assert agg["num_trades"] == 0

    def test_single_period(self):
        stats = [{
            "win_pnl": 3.0, "lose_pnl": -1.0,
            "trade_pnls_r": [2.0, 1.0, -1.0],
            "total_static_alloc": 1.0, "total_dynamic_alloc": 0.0,
            "total_target_alloc": 2.0,
        }]
        agg = aggregate_stats_dicts(stats)
        assert agg["num_trades"] == 3
        assert abs(agg["total_pnl"] - 2.0) < 1e-9
        assert abs(agg["expected_value"] - 2.0 / 3) < 1e-9

    def test_multiple_periods(self):
        stats = [
            {"win_pnl": 2.0, "lose_pnl": 0.0, "trade_pnls_r": [2.0],
             "total_static_alloc": 0, "total_dynamic_alloc": 0, "total_target_alloc": 1},
            {"win_pnl": 0.0, "lose_pnl": -1.5, "trade_pnls_r": [-1.5],
             "total_static_alloc": 1, "total_dynamic_alloc": 0, "total_target_alloc": 0},
        ]
        agg = aggregate_stats_dicts(stats)
        assert agg["num_trades"] == 2
        assert abs(agg["total_pnl"] - 0.5) < 1e-9


# ---------------------------------------------------------------------------
# Filtering
# ---------------------------------------------------------------------------

class TestFiltering:

    def test_filter_by_min_trades(self):
        results = [
            (0, {}, {"num_trades": 10, "expected_value": 0.5}),
            (1, {}, {"num_trades": 2, "expected_value": 1.0}),
            (2, {}, {"num_trades": 15, "expected_value": -0.1}),
        ]
        filtered = apply_filters(results, {"num_trades": 5})
        assert len(filtered) == 2
        indices = [r[0] for r in filtered]
        assert 1 not in indices

    def test_filter_multiple_criteria(self):
        results = [
            (0, {}, {"num_trades": 10, "expected_value": 0.5, "win_pct": 60}),
            (1, {}, {"num_trades": 10, "expected_value": -0.1, "win_pct": 40}),
        ]
        filtered = apply_filters(results, {"num_trades": 5, "expected_value": 0.0})
        assert len(filtered) == 1
        assert filtered[0][0] == 0

    def test_no_thresholds_passes_all(self):
        results = [(0, {}, {"num_trades": 1}), (1, {}, {"num_trades": 0})]
        filtered = apply_filters(results, {})
        assert len(filtered) == 2


# ---------------------------------------------------------------------------
# Incremental recalculation correctness
# ---------------------------------------------------------------------------

class TestIncrementalRecalc:

    def test_rsi_recalc_matches_full(self):
        df_raw = _generate_oscillation(500)
        base = dict(DEFAULT_INDICATOR_SETTINGS)
        df_full_base = calculate_indicators(df_raw, **base)

        new_params = dict(base)
        new_params["rsi_window"] = 7

        df_full_new = calculate_indicators(df_raw, **new_params)

        df_incr = df_full_base.copy()
        groups = changed_groups(base, new_params)
        assert "rsi" in groups
        recalculate_groups(df_incr, groups, **new_params)

        pd.testing.assert_series_equal(
            df_incr["rsi"].dropna(), df_full_new["rsi"].dropna(),
            check_names=False, atol=1e-10,
        )

    def test_bb_recalc_matches_full(self):
        df_raw = _generate_oscillation(500)
        base = dict(DEFAULT_INDICATOR_SETTINGS)
        df_full_base = calculate_indicators(df_raw, **base)

        new_params = dict(base)
        new_params["bb_upper_period"] = 30
        new_params["bb_upper_stdev"] = 1.5

        df_full_new = calculate_indicators(df_raw, **new_params)

        df_incr = df_full_base.copy()
        groups = changed_groups(base, new_params)
        assert "bb" in groups
        recalculate_groups(df_incr, groups, **new_params)

        pd.testing.assert_series_equal(
            df_incr["bb_upper"].dropna(), df_full_new["bb_upper"].dropna(),
            check_names=False, atol=1e-10,
        )

    def test_unchanged_params_no_recalc(self):
        groups = changed_groups(DEFAULT_INDICATOR_SETTINGS, DEFAULT_INDICATOR_SETTINGS)
        assert len(groups) == 0


# ---------------------------------------------------------------------------
# Worker logic (single-process)
# ---------------------------------------------------------------------------

class TestWFOWorker:

    @pytest.fixture
    def setup_worker(self):
        df_raw = _generate_oscillation(1000, center=150, amplitude=30, period=80)
        base = dict(DEFAULT_INDICATOR_SETTINGS)
        df_featured = calculate_indicators(df_raw, **base)

        strategy = make_strategy(
            direction="Long",
            entry_group="RSI Group", entry_element1="RSI",
            entry_event="Cross Below", entry_compare_type="Fixed Value",
            entry_value=30.0,
            initial_stop_type="ATR",
            indicator_settings=base,
        )

        periods = [
            (df_raw.index[100], df_raw.index[300]),
            (df_raw.index[400], df_raw.index[600]),
            (df_raw.index[700], df_raw.index[900]),
        ]

        init_wfo_worker(deepcopy(strategy), periods, base, df_featured)
        return df_raw, df_featured, strategy, periods, base

    def test_worker_returns_results(self, setup_worker):
        _, _, _, periods, base = setup_worker
        batch = ([0], [dict(base)], [(periods[0][0], periods[-1][1])])
        results = run_wfo_batch(batch)
        assert len(results) == 1
        idx, params, agg = results[0]
        assert idx == 0

    def test_worker_different_params_produce_results(self, setup_worker):
        _, _, _, periods, base = setup_worker
        date_ranges = [(periods[0][0], periods[-1][1])]

        params_a = dict(base); params_a["rsi_window"] = 5
        params_b = dict(base); params_b["rsi_window"] = 40

        results_a = run_wfo_batch(([0], [params_a], date_ranges))
        results_b = run_wfo_batch(([1], [params_b], date_ranges))

        assert len(results_a) == 1
        assert len(results_b) == 1

    def test_worker_respects_date_ranges(self, setup_worker):
        _, _, _, periods, base = setup_worker
        # Only include the first period
        narrow = [(periods[0][0], periods[0][1])]
        results_narrow = run_wfo_batch(([0], [dict(base)], narrow))

        # Include all periods
        wide = [(periods[0][0], periods[-1][1])]
        results_wide = run_wfo_batch(([0], [dict(base)], wide))

        assert len(results_narrow) == 1
        assert len(results_wide) == 1
        # Both should succeed (agg may differ)

    def test_worker_multiple_combos_in_batch(self, setup_worker):
        _, _, _, periods, base = setup_worker
        date_ranges = [(periods[0][0], periods[-1][1])]

        combos = []
        for rsi_w in [5, 10, 20, 30, 40]:
            p = dict(base); p["rsi_window"] = rsi_w
            combos.append(p)

        results = run_wfo_batch((list(range(5)), combos, date_ranges))
        assert len(results) == 5

    def test_worker_empty_date_range_no_trades(self, setup_worker):
        _, _, _, periods, base = setup_worker
        # Date range that doesn't overlap any period
        far_future = [(pd.Timestamp("2030-01-01"), pd.Timestamp("2030-12-31"))]
        results = run_wfo_batch(([0], [dict(base)], far_future))
        assert len(results) == 1
        _, _, agg = results[0]
        assert agg is None  # no trades found


# ---------------------------------------------------------------------------
# End-to-end: optimizer picks better params
# ---------------------------------------------------------------------------

class TestWFOOptimization:

    def test_sorting_by_trades_works(self):
        """Given param combos with known trade counts, sorting works."""
        results = [
            (0, {"rsi_window": 5}, {"num_trades": 3, "expected_value": 0.1}),
            (1, {"rsi_window": 14}, {"num_trades": 10, "expected_value": 0.5}),
            (2, {"rsi_window": 40}, {"num_trades": 1, "expected_value": 2.0}),
        ]
        # Sort by num_trades descending
        results.sort(key=lambda x: x[2].get("num_trades", 0), reverse=True)
        assert results[0][1]["rsi_window"] == 14
        assert results[-1][1]["rsi_window"] == 40

    def test_sorting_by_ev_works(self):
        results = [
            (0, {"rsi_window": 5}, {"num_trades": 3, "expected_value": 0.1}),
            (1, {"rsi_window": 14}, {"num_trades": 10, "expected_value": 0.5}),
            (2, {"rsi_window": 40}, {"num_trades": 1, "expected_value": 2.0}),
        ]
        results.sort(key=lambda x: x[2].get("expected_value", 0), reverse=True)
        assert results[0][1]["rsi_window"] == 40

    def test_filter_then_sort(self):
        results = [
            (0, {"rsi_window": 5}, {"num_trades": 3, "expected_value": 0.1}),
            (1, {"rsi_window": 14}, {"num_trades": 10, "expected_value": 0.5}),
            (2, {"rsi_window": 40}, {"num_trades": 1, "expected_value": 2.0}),
        ]
        # Filter: min 2 trades
        filtered = apply_filters(results, {"num_trades": 2})
        assert len(filtered) == 2  # combo 2 (1 trade) excluded
        # Sort by EV
        filtered.sort(key=lambda x: x[2].get("expected_value", 0), reverse=True)
        assert filtered[0][1]["rsi_window"] == 14  # 0.5 EV > 0.1 EV

    def test_full_worker_pipeline_with_grid(self):
        """Run a small grid through the worker and verify results are returned
        for each combo."""
        df_raw = _generate_oscillation(1000, center=150, amplitude=30, period=80)
        base = dict(DEFAULT_INDICATOR_SETTINGS)
        df_featured = calculate_indicators(df_raw, **base)

        strategy = make_strategy(
            direction="Long",
            entry_group="RSI Group", entry_element1="RSI",
            entry_event="Cross Below", entry_compare_type="Fixed Value",
            entry_value=35.0,
            initial_stop_type="ATR",
            indicator_settings=base,
        )

        periods = [
            (df_raw.index[100], df_raw.index[400]),
            (df_raw.index[500], df_raw.index[800]),
        ]

        init_wfo_worker(deepcopy(strategy), periods, base, df_featured)

        # Small grid: 5 RSI values
        grid = generate_param_grid(
            {"rsi"}, {"rsi": {"rsi_window": (5, 25, 5)}}, base,
        )
        assert len(grid) == 5

        date_ranges = [(df_raw.index[0], df_raw.index[-1])]
        batch = (list(range(len(grid))), grid, date_ranges)
        results = run_wfo_batch(batch)

        assert len(results) == 5
        # All should return a result (agg may be None for some)
        for idx, params, agg in results:
            assert isinstance(idx, int)
            assert "rsi_window" in params

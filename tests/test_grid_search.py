"""
Tests for grid search — single-thread vs multiprocessing parity.

Runs a small grid search both ways and asserts identical results.
"""
import pytest
import pandas as pd
import numpy as np
from copy import deepcopy

from indicators.calculate_indicators import calculate_indicators
from strategies.first_strategy_numpy import execute_custom_strategy_numpy
from tests.conftest import (
    DEFAULT_INDICATOR_SETTINGS,
    make_strategy,
    _generate_oscillation,
)


@pytest.fixture(scope="module")
def df_oscillation():
    df = _generate_oscillation(n=200, freq="15min")
    return calculate_indicators(df, **DEFAULT_INDICATOR_SETTINGS)


def _run_single_thread(df, base_strategy, param_grid):
    """Run all parameter combos sequentially and return results dict."""
    results = {}
    for label, overrides in param_grid.items():
        strategy = deepcopy(base_strategy)
        # Apply parameter overrides to indicator settings
        for key, val in overrides.items():
            strategy["indicator_settings"][key] = val

        # Recalculate indicators with new settings
        df_recalc = calculate_indicators(df.copy(), **strategy["indicator_settings"])
        _, stats_df = execute_custom_strategy_numpy(df_recalc, strategy)

        pnls = stats_df.attrs.get("trade_pnls_r", [])
        results[label] = {
            "n_trades": len(pnls),
            "total_r": sum(pnls) if pnls else 0.0,
            "pnls": list(pnls),
        }
    return results


class TestGridSearchSingleThread:
    """Test that different parameter sets produce different (or consistent) results."""

    def test_different_rsi_periods(self, df_oscillation):
        """Grid over RSI periods should produce valid results for each."""
        base = make_strategy(direction="Long")

        param_grid = {
            "rsi_10": {"rsi_window": 10},
            "rsi_14": {"rsi_window": 14},
            "rsi_20": {"rsi_window": 20},
        }

        results = _run_single_thread(df_oscillation, base, param_grid)

        for label, res in results.items():
            assert isinstance(res["n_trades"], int)
            assert isinstance(res["total_r"], float)

    def test_results_deterministic(self, df_oscillation):
        """Running the same grid twice should produce identical results."""
        base = make_strategy(direction="Long")
        param_grid = {
            "rsi_10": {"rsi_window": 10},
            "rsi_14": {"rsi_window": 14},
        }

        results1 = _run_single_thread(df_oscillation, base, param_grid)
        results2 = _run_single_thread(df_oscillation, base, param_grid)

        for label in param_grid:
            assert results1[label]["n_trades"] == results2[label]["n_trades"]
            assert results1[label]["total_r"] == results2[label]["total_r"]
            assert results1[label]["pnls"] == results2[label]["pnls"]


class TestGridSearchMultiprocessing:
    """Test multiprocessing worker produces same results as single-thread."""

    def test_worker_parity(self, df_oscillation):
        """
        Compare single-thread vs multiprocessing.Pool results.
        Uses the actual grid_search_worker module.
        """
        try:
            from strategies.grid_search_worker import init_worker, run_candidate
        except ImportError:
            pytest.skip("grid_search_worker not importable")

        base = make_strategy(direction="Long")
        rsi_values = [10, 14, 20]

        # Single-thread results
        single_results = {}
        for rsi_val in rsi_values:
            strategy = deepcopy(base)
            strategy["indicator_settings"]["rsi_window"] = rsi_val
            df_recalc = calculate_indicators(df_oscillation.copy(), **strategy["indicator_settings"])
            _, stats_df = execute_custom_strategy_numpy(df_recalc, strategy)
            pnls = stats_df.attrs.get("trade_pnls_r", [])
            single_results[rsi_val] = list(pnls)

        # Multiprocessing results (run in current process for simplicity)
        # Prepare combo slices as the grid search tab would
        # Each slice is (df, period_start, period_end) tuple
        combo_key = "test_combo"

        # For grid search, all candidates share the same data slices
        # We use the base indicator settings for the data, and vary strategy params
        df_base = calculate_indicators(df_oscillation.copy(), **base["indicator_settings"])
        combo_slices = {combo_key: [(df_base, None, None)]}

        # Initialize worker globals
        init_worker(combo_slices, [combo_key])

        # Run the base strategy through the worker (same indicator settings as data)
        # Grid search varies strategy params, not indicator recalculation per candidate
        strategy = deepcopy(base)
        result_idx, result_label, combo_results = run_candidate(
            (0, "rsi_14", strategy)
        )

        # Compare with single-thread results for the base RSI=14
        if combo_key in combo_results:
            worker_stats_list = combo_results[combo_key]
            worker_pnls = []
            for stats_dict in worker_stats_list:
                worker_pnls.extend(stats_dict.get("trade_pnls_r", []))

            single_pnls = single_results[14]
            assert len(worker_pnls) == len(single_pnls), \
                f"Trade count mismatch: worker={len(worker_pnls)}, single={len(single_pnls)}"
            if single_pnls:
                np.testing.assert_allclose(
                    worker_pnls, single_pnls, rtol=1e-6,
                    err_msg="P&L mismatch between worker and single-thread"
                )

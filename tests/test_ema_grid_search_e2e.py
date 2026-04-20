"""
End-to-end regression test for the EMA grid search bug.

Simulates the exact scenario from the client's issue: a grid search where
all candidates reference EMA elements ("EMA 1", "EMA 2", ...). Before the
ema_count fix in grid_search_helpers.py, every candidate was rejected and
the grid search returned no results.
"""
import sys
from unittest.mock import MagicMock
from copy import deepcopy

if "streamlit" not in sys.modules:
    sys.modules["streamlit"] = MagicMock()

import pandas as pd
from indicators.calculate_indicators import calculate_indicators
from strategies.first_strategy import execute_custom_strategy
from ui.grid_search_helpers import generate_run_configs
from tests.conftest import (
    DEFAULT_INDICATOR_SETTINGS,
    make_strategy,
    _generate_oscillation,
)


def _make_base_strategy():
    """Long strategy with EMA-based entry trigger — mirrors real client usage."""
    base = make_strategy(
        direction="Long",
        entry_group="Price & Indicators",
        entry_element1="Price",
        entry_event="Cross Above",
        entry_compare_type="Indicator",
        entry_element2="EMA 1",
        initial_stop_type="ATR",
        initial_stop_atr_period=14,
        initial_stop_atr_multiplier=1.5,
    )
    # DEFAULT_INDICATOR_SETTINGS has ema_periods=[10, 20, 50, 200]
    assert len(base["indicator_settings"]["ema_periods"]) == 4
    return base


def test_generate_run_configs_with_ema_candidates_produces_configs():
    """Fix verification: EMA-referencing candidates must NOT be rejected."""
    base = _make_base_strategy()
    candidates = [
        {"group": "Price & Indicators", "element1": "Price",
         "compare_type": "Indicator", "element2": "EMA 1"},
        {"group": "Price & Indicators", "element1": "Price",
         "compare_type": "Indicator", "element2": "EMA 2"},
        {"group": "Price & Indicators", "element1": "EMA 1",
         "compare_type": "Indicator", "element2": "EMA 3"},
        {"group": "Price & Indicators", "element1": "EMA 2",
         "compare_type": "Indicator", "element2": "EMA 4"},
    ]

    runs = generate_run_configs(base, "trigger", candidates, ["Cross Above", "Cross Below"])

    # 4 candidates × 2 events = 8 runs, all must survive validation
    assert len(runs) == 8, (
        f"Expected 8 runs (4 EMA candidates × 2 events), got {len(runs)}. "
        "If this is 0, the ema_count bug has regressed."
    )

    # Every run must be a properly structured strategy
    for label, strat in runs:
        assert strat["entry"]["trigger"]["element1"] in ("Price", "EMA 1", "EMA 2")
        assert strat["entry"]["trigger"]["element2"].startswith("EMA ")
        assert strat["indicator_settings"]["ema_periods"] == [10, 20, 50, 200]


def test_end_to_end_grid_search_with_ema_produces_results():
    """Drive the full pipeline: build configs → execute each → collect stats.

    This is the exact code path the client hits when they click Calculate
    in the Grid Search tab. Before the fix: 0 results. After the fix: real stats.
    """
    # Build OHLC data + indicators (mirrors what the app does on data load)
    df = _generate_oscillation(n=400, freq="15min")
    df = calculate_indicators(df, **DEFAULT_INDICATOR_SETTINGS)

    base = _make_base_strategy()
    candidates = [
        {"group": "Price & Indicators", "element1": "Price",
         "compare_type": "Indicator", "element2": "EMA 1"},
        {"group": "Price & Indicators", "element1": "Price",
         "compare_type": "Indicator", "element2": "EMA 2"},
        {"group": "Price & Indicators", "element1": "Price",
         "compare_type": "Indicator", "element2": "EMA 3"},
    ]

    runs = generate_run_configs(base, "trigger", candidates, ["Cross Above"])
    assert len(runs) == 3, f"All 3 EMA candidates should validate, got {len(runs)}"

    # Execute each candidate strategy against the data (what the engine does)
    period_start = df.index[0]
    period_end = df.index[-1]

    results = []
    for label, strat in runs:
        df_copy = df.copy()
        _, stats_df = execute_custom_strategy(df_copy, strat, period_start, period_end)
        assert stats_df is not None, f"Engine returned None for {label}"

        pnls = stats_df.attrs.get("trade_pnls_r", [])
        results.append({
            "label": label,
            "n_trades": len(pnls),
            "total_r": sum(pnls) if pnls else 0.0,
        })

    # The point of the test: results actually came back.
    # (Number of trades can be zero for an oscillation — that's fine;
    # what matters is the pipeline didn't drop everything at validation.)
    assert len(results) == 3
    print("\n=== Grid search results (simulating Calculate button) ===")
    for r in results:
        print(f"  {r['label']:50s} trades={r['n_trades']:3d}  total_r={r['total_r']:.3f}")

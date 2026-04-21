"""
End-to-end regression test for the EMA grid search bug.

Simulates the exact scenario from the client's issue: a grid search where
all candidates reference EMA elements ("EMA 1", "EMA 2", ...). Before the
ema_count fix in grid_search_helpers.py, every candidate was rejected and
the grid search returned no results.
"""
import sys
import time
from collections import OrderedDict
from unittest.mock import MagicMock
from copy import deepcopy

if "streamlit" not in sys.modules:
    sys.modules["streamlit"] = MagicMock()
if "streamlit.components.v1" not in sys.modules:
    sys.modules["streamlit.components.v1"] = MagicMock()

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


def test_mc_enrichment_parallel_end_to_end():
    """Exercise _enrich_mc_parallel end-to-end: build realistic `results`
    structure (mirroring what _run_grid_search_multiprocessing produces),
    feed it through the parallel MC enricher, and verify every agg ends up
    with a populated mc_avg_profit field.
    """
    from ui.grid_search_tab import _enrich_mc_parallel

    # Build 12 candidates, each with a global agg + 2 per-selection aggs.
    # Vary win_pct and rr_ratio so binary-search inside MC has real work.
    def _agg(num_trades, win_pct, rr_ratio):
        return {
            'num_trades': num_trades,
            'win_pct': win_pct,
            'lose_pct': 100 - win_pct,
            'avg_win_pnl': rr_ratio,
            'avg_lose_pnl': -1.0,
            'total_pnl': num_trades * ((win_pct / 100) * rr_ratio - (1 - win_pct / 100)),
            'expected_value': 0.0,
            'target_exit_pct': 50.0,
            'static_exit_pct': 30.0,
            'dynamic_exit_pct': 20.0,
            'eod_exit_pct': 0.0,
            'rr_ratio': rr_ratio,
            'max_drawdown': 10.0,
            'sqn': 1.5,
            'correlation': None,
            'abs_correlation': None,
        }

    results = []
    for i in range(12):
        wr = 40 + i * 2            # 40, 42, 44, ...
        rr = 1.5 + i * 0.1         # 1.5, 1.6, 1.7, ...
        global_agg = _agg(num_trades=100, win_pct=wr, rr_ratio=rr)
        sel_results = OrderedDict([
            ("Selection A", _agg(num_trades=50, win_pct=wr - 2, rr_ratio=rr)),
            ("Selection B", _agg(num_trades=50, win_pct=wr + 2, rr_ratio=rr)),
        ])
        results.append((f"candidate_{i}", global_agg, sel_results))

    # Include a zero-trade candidate (fast-path: should get 0.0, skip compute)
    zero_global = _agg(num_trades=0, win_pct=0, rr_ratio=0)
    zero_sel = OrderedDict([("Selection A", _agg(num_trades=0, win_pct=0, rr_ratio=0))])
    results.append(("zero_candidate", zero_global, zero_sel))

    # Count expected aggs: 12 candidates × (1 global + 2 sel) + 1 zero × (1 + 1) = 38
    expected_aggs = 12 * 3 + 1 * 2
    total_aggs = sum(1 + len(sr) for _, _, sr in results)
    assert total_aggs == expected_aggs, f"Setup error: {total_aggs} != {expected_aggs}"

    # Record original state — nothing should have mc_avg_profit yet
    for _, g, sr in results:
        assert 'mc_avg_profit' not in g
        for s in sr.values():
            assert 'mc_avg_profit' not in s

    # Run the parallel enricher with MC-tab-default-like params (small n_sims
    # to keep the test fast; behavior is identical at higher values).
    # Note: trades_per_sim is NOT passed — each agg uses its own num_trades.
    t0 = time.time()
    _enrich_mc_parallel(results, balance=10000.0, n_sims=1000, target_dd=5.0)
    dt = time.time() - t0

    # Every agg must now have mc_avg_profit set
    for label, g, sr in results:
        assert 'mc_avg_profit' in g, f"Missing mc_avg_profit on {label} global"
        assert isinstance(g['mc_avg_profit'], float), \
            f"{label} global mc_avg_profit is {type(g['mc_avg_profit'])}"
        for sel_name, s in sr.items():
            assert 'mc_avg_profit' in s, f"Missing mc_avg_profit on {label}/{sel_name}"
            assert isinstance(s['mc_avg_profit'], float)

    # Zero-trade candidates must short-circuit to exactly 0.0
    _, zg, zsr = results[-1]
    assert zg['mc_avg_profit'] == 0.0
    assert list(zsr.values())[0]['mc_avg_profit'] == 0.0

    # Non-zero candidates with positive edge (win_pct > 50 and rr_ratio > 1)
    # should produce mc_avg_profit > starting balance (10000)
    positive_edge_found = False
    for label, g, _ in results[:-1]:  # skip zero
        if g['win_pct'] > 50 and g['rr_ratio'] > 1:
            assert g['mc_avg_profit'] > 10000, (
                f"{label} has edge (wr={g['win_pct']}, rr={g['rr_ratio']}) "
                f"but mc_avg_profit={g['mc_avg_profit']:.2f} <= 10000"
            )
            positive_edge_found = True
    assert positive_edge_found, "Test setup should include at least one positive-edge candidate"

    print(f"\n=== MC enrichment parallel pipeline ===")
    print(f"  Enriched {expected_aggs} aggs in {dt:.2f}s")
    print(f"  Sample candidate results (first 5):")
    for label, g, sr in results[:5]:
        print(f"    {label}: wr={g['win_pct']:.0f}%, rr={g['rr_ratio']:.1f}, "
              f"mc_avg_profit=${g['mc_avg_profit']:,.0f}")


def test_mc_uses_each_candidates_trade_count():
    """Regression test: per-candidate trade count must be used as trades_per_sim.

    Two candidates with identical win_pct/rr_ratio but very different trade
    counts should produce different mc_avg_profit values — specifically, more
    trades with a positive edge compounds to a larger final balance.
    """
    from ui.grid_search_tab import _enrich_mc_parallel

    # Both candidates: same win rate (60%), same RR (2.0). Only num_trades differs.
    # With a +EV edge, more trades → more compounding → higher avg profit.
    short_agg = {'num_trades': 11, 'win_pct': 60.0, 'rr_ratio': 2.0}
    long_agg = {'num_trades': 296, 'win_pct': 60.0, 'rr_ratio': 2.0}

    results = [
        ('short_candidate', short_agg, OrderedDict()),
        ('long_candidate', long_agg, OrderedDict()),
    ]

    _enrich_mc_parallel(results, balance=10000.0, n_sims=2000, target_dd=5.0)

    short_profit = short_agg['mc_avg_profit']
    long_profit = long_agg['mc_avg_profit']

    # Both should be positive (above $10k starting balance)
    assert short_profit > 10000, f"Short candidate should have edge, got ${short_profit:.0f}"
    assert long_profit > 10000, f"Long candidate should have edge, got ${long_profit:.0f}"

    # The 296-trade path must compound to substantially more than the 11-trade path
    # (If the bug regressed and both used trades_per_sim=100, they'd be similar.)
    assert long_profit > short_profit * 3, (
        f"Per-candidate trade count not being used: "
        f"11-trade=${short_profit:,.0f} vs 296-trade=${long_profit:,.0f}. "
        f"With same WR/RR but 27x more trades, compounding should produce a much larger gap."
    )

    print(f"\n=== Per-candidate trade count ===")
    print(f"  11-trade candidate  (WR=60%, RR=2.0): ${short_profit:>12,.0f}")
    print(f"  296-trade candidate (WR=60%, RR=2.0): ${long_profit:>12,.0f}")
    print(f"  Ratio: {long_profit / short_profit:.1f}x (would be ~1.0 if bug regressed)")

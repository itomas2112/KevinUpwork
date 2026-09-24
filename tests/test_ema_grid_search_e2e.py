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
    for label, strat, _ in runs:
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
    for label, strat, _ in runs:
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


def _trade_pool(num_trades, win_pct, rr_ratio, r_dist=10.0):
    """Deterministic trade list with the given win rate / RR (pnl_r, r_dist)."""
    n_wins = round(num_trades * win_pct / 100)
    pnls = [rr_ratio] * n_wins + [-1.0] * (num_trades - n_wins)
    return pnls, [r_dist] * num_trades


def test_mc_enrichment_parallel_end_to_end():
    """Exercise _enrich_mc_parallel end-to-end: build realistic `results`
    structure (mirroring what _run_grid_search_multiprocessing produces),
    feed it through the parallel MC enricher, and verify every agg ends up
    with populated mc_avg_profit / mc_margin_capped fields.
    """
    from ui.grid_search_tab import _enrich_mc_parallel

    # Build 12 candidates, each with a global agg + 2 per-selection aggs.
    # Vary win_pct and rr_ratio so binary-search inside MC has real work.
    def _agg(num_trades, win_pct, rr_ratio):
        pnls, dists = _trade_pool(num_trades, win_pct, rr_ratio)
        return {
            'num_trades': num_trades,
            'win_pct': win_pct,
            'lose_pct': 100 - win_pct,
            'avg_win_pnl': rr_ratio,
            'avg_lose_pnl': -1.0,
            'total_pnl': sum(pnls),
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
            'trade_pnls_r': pnls,
            'trade_r_distances': dists,
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
        results.append((f"candidate_{i}", global_agg, sel_results, {}))

    # Include a zero-trade candidate (fast-path: should get 0.0, skip compute)
    zero_global = _agg(num_trades=0, win_pct=0, rr_ratio=0)
    zero_sel = OrderedDict([("Selection A", _agg(num_trades=0, win_pct=0, rr_ratio=0))])
    results.append(("zero_candidate", zero_global, zero_sel, {}))

    expected_aggs = 12 * 3 + 1 * 2
    total_aggs = sum(1 + len(sr) for _, _, sr, _ in results)
    assert total_aggs == expected_aggs, f"Setup error: {total_aggs} != {expected_aggs}"

    for _, g, sr, _ in results:
        assert 'mc_avg_profit' not in g
        for s in sr.values():
            assert 'mc_avg_profit' not in s

    balance = 100_000.0
    t0 = time.time()
    _enrich_mc_parallel(results, balance=balance, n_sims=500, target_dd=5.0,
                        point_value=50.0, margin=28000.0, trades_per_sim=100)
    dt = time.time() - t0

    for label, g, sr, _ in results:
        assert isinstance(g['mc_avg_profit'], float), \
            f"{label} global mc_avg_profit is {type(g.get('mc_avg_profit'))}"
        assert isinstance(g['mc_margin_capped'], bool)
        for sel_name, s in sr.items():
            assert isinstance(s['mc_avg_profit'], float), f"{label}/{sel_name}"
            assert isinstance(s['mc_margin_capped'], bool)

    # Zero-trade candidates must short-circuit to exactly 0.0 / not capped
    _, zg, zsr, _ = results[-1]
    assert zg['mc_avg_profit'] == 0.0 and zg['mc_margin_capped'] is False
    assert list(zsr.values())[0]['mc_avg_profit'] == 0.0

    # Positive-edge candidates end above the starting balance
    positive_edge_found = False
    for label, g, _, _ in results[:-1]:
        if g['win_pct'] > 50 and g['rr_ratio'] > 1:
            assert g['mc_avg_profit'] > balance, (
                f"{label} has edge (wr={g['win_pct']}, rr={g['rr_ratio']}) "
                f"but mc_avg_profit={g['mc_avg_profit']:.2f} <= {balance}")
            positive_edge_found = True
    assert positive_edge_found

    print(f"\n=== MC enrichment parallel pipeline ===")
    print(f"  Enriched {expected_aggs} aggs in {dt:.2f}s")
    for label, g, sr, _ in results[:5]:
        print(f"    {label}: wr={g['win_pct']:.0f}%, rr={g['rr_ratio']:.1f}, "
              f"mc_avg_profit=${g['mc_avg_profit']:,.0f} cap={g['mc_margin_capped']}")


def test_mc_uses_fixed_trades_per_sim_not_trade_count():
    """The MC horizon is the configured trades_per_sim, not the candidate's
    trade count: two candidates with the same trade distribution but 11 vs
    296 trades bootstrap to similar values, while a longer horizon compounds
    to more."""
    from ui.grid_search_tab import _enrich_mc_parallel

    def _agg(n):
        pnls, dists = _trade_pool(n, 60.0, 2.0)
        return {'num_trades': n, 'trade_pnls_r': pnls, 'trade_r_distances': dists}

    # 10 / 250 trades so both pools are exactly 60% winners
    short_agg, long_agg = _agg(10), _agg(250)
    results = [('short', short_agg, OrderedDict(), {}), ('long', long_agg, OrderedDict(), {})]
    _enrich_mc_parallel(results, balance=100_000.0, n_sims=2000, target_dd=5.0,
                        point_value=50.0, margin=28000.0, trades_per_sim=100)
    short_profit, long_profit = short_agg['mc_avg_profit'], long_agg['mc_avg_profit']
    assert short_profit > 100_000 and long_profit > 100_000
    assert abs(long_profit - short_profit) / short_profit < 0.15, (short_profit, long_profit)

    longer = _agg(250)
    _enrich_mc_parallel([('longer', longer, OrderedDict(), {})], balance=100_000.0,
                        n_sims=2000, target_dd=5.0, point_value=50.0,
                        margin=28000.0, trades_per_sim=300)
    assert longer['mc_avg_profit'] > long_profit

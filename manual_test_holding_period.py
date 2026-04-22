"""
Manual verification of the new Average Holding Period metric.

Runs end-to-end against both strategy engines, the aggregator, and the
grid-search worker, then checks that holding periods match expected values
computed independently from the trade timeline.
"""
import sys
import numpy as np
import pandas as pd

sys.path.insert(0, '.')
sys.path.insert(0, 'tests')

from tests.conftest import (
    _generate_v_shape, _generate_oscillation, _generate_uptrend,
    DEFAULT_INDICATOR_SETTINGS, make_strategy,
)
from indicators.calculate_indicators import calculate_indicators
from strategies.first_strategy import execute_custom_strategy
from strategies.first_strategy_numpy import execute_custom_strategy_numpy
from strategies.grid_search_worker import _extract_stats as _gs_extract


def banner(msg):
    print(f"\n{'='*70}\n{msg}\n{'='*70}")


def _hold_from_attrs(stats_df):
    return list(stats_df.attrs.get("trade_holding_periods", []))


# ---------------------------------------------------------------------------
# Test 1 — short-target strategy on V-shape: many quick exits
# ---------------------------------------------------------------------------
banner("TEST 1 — Pandas vs NumPy engine: same holding periods?")

df = calculate_indicators(_generate_v_shape(n=400), **DEFAULT_INDICATOR_SETTINGS)

strategy = make_strategy(
    direction="Long",
    entry_group="RSI Group", entry_element1="RSI",
    entry_event="Cross Above", entry_compare_type="Fixed Value", entry_value=30.0,
    initial_stop_type="ATR", initial_stop_event="Cross Below",
    initial_stop_atr_period=14, initial_stop_atr_multiplier=1.5,
)

_, stats_pd = execute_custom_strategy(df.copy(), strategy, None, None)
_, stats_np = execute_custom_strategy_numpy(df.copy(), strategy, None, None)

hp_pd = _hold_from_attrs(stats_pd)
hp_np = _hold_from_attrs(stats_np)

print(f"Pandas engine  trades={len(hp_pd)}  holding periods={hp_pd}")
print(f"NumPy  engine  trades={len(hp_np)}  holding periods={hp_np}")

assert hp_pd == hp_np, f"Engines disagree!\n  pd={hp_pd}\n  np={hp_np}"
assert all(h >= 1 for h in hp_pd), "Holding period must be >=1 (entry bar excluded from exits)"
print(f"OK — engines agree, all holding periods >=1, mean={np.mean(hp_pd):.2f} bars")


# ---------------------------------------------------------------------------
# Test 2 — verify the holding period mean by recomputing from the trade table
# ---------------------------------------------------------------------------
banner("TEST 2 — _aggregate_stats avg_holding_period vs hand calc")

from ui.charting_tab import _aggregate_stats

agg = _aggregate_stats([stats_pd])
manual_avg = float(np.mean(hp_pd)) if hp_pd else 0.0

print(f"_aggregate_stats avg_holding_period = {agg['avg_holding_period']:.4f}")
print(f"Hand-computed mean of {len(hp_pd)} trades = {manual_avg:.4f}")

assert abs(agg['avg_holding_period'] - manual_avg) < 1e-9, "Aggregator mean diverges from hand calc"
print("OK")


# ---------------------------------------------------------------------------
# Test 3 — aggregation across multiple periods (simulating multiple DRM slices)
# ---------------------------------------------------------------------------
banner("TEST 3 — aggregator combines holding periods across periods")

df2 = calculate_indicators(_generate_oscillation(n=300), **DEFAULT_INDICATOR_SETTINGS)
_, stats2 = execute_custom_strategy(df2.copy(), strategy, None, None)
hp2 = _hold_from_attrs(stats2)
print(f"Period A: {len(hp_pd)} trades, hp={hp_pd}")
print(f"Period B: {len(hp2)} trades, hp={hp2}")

agg_combined = _aggregate_stats([stats_pd, stats2])
combined_manual = (sum(hp_pd) + sum(hp2)) / (len(hp_pd) + len(hp2)) if (hp_pd or hp2) else 0.0
print(f"Combined avg_holding_period = {agg_combined['avg_holding_period']:.4f}")
print(f"Hand-combined mean         = {combined_manual:.4f}")
assert abs(agg_combined['avg_holding_period'] - combined_manual) < 1e-9
print("OK")


# ---------------------------------------------------------------------------
# Test 4 — grid_search_worker._extract_stats carries holding periods
# ---------------------------------------------------------------------------
banner("TEST 4 — grid search worker preserves trade_holding_periods")

extracted = _gs_extract(stats_pd)
print("Extracted dict keys:", sorted(extracted.keys()))
assert "trade_holding_periods" in extracted, "Worker dropped trade_holding_periods!"
assert extracted["trade_holding_periods"] == hp_pd, "Worker corrupted the list"
print(f"OK — worker dict contains {len(extracted['trade_holding_periods'])} holding periods")


# ---------------------------------------------------------------------------
# Test 5 — _aggregate_stats_dicts (multiprocess path) produces same average
# ---------------------------------------------------------------------------
banner("TEST 5 — multiprocess aggregator matches single-process")

from ui.grid_search_tab import _aggregate_stats_dicts

extracted_a = _gs_extract(stats_pd)
extracted_b = _gs_extract(stats2)
agg_dicts = _aggregate_stats_dicts([extracted_a, extracted_b])

print(f"_aggregate_stats_dicts.avg_holding_period = {agg_dicts['avg_holding_period']:.4f}")
print(f"_aggregate_stats.avg_holding_period       = {agg_combined['avg_holding_period']:.4f}")
assert abs(agg_dicts['avg_holding_period'] - agg_combined['avg_holding_period']) < 1e-9
print("OK — both aggregators agree")


# ---------------------------------------------------------------------------
# Test 6 — End-of-Data trades use last bar index
# ---------------------------------------------------------------------------
banner("TEST 6 — End-of-Data trades cap holding period at last bar")

# Reuse the V-shape data from Test 1 (we know the entry triggers there).
# Override the exit groups so the target is unreachable and there's no dynamic stop;
# the wide ATR initial stop also won't fire — trade should run to End-of-Data.
strat_eod = make_strategy(
    direction="Long",
    entry_group="RSI Group", entry_element1="RSI",
    entry_event="Cross Above", entry_compare_type="Fixed Value", entry_value=30.0,
    initial_stop_type="ATR", initial_stop_event="Cross Below",
    initial_stop_atr_period=14, initial_stop_atr_multiplier=50.0,  # very wide
    exit_groups=[{
        "group_id": 1, "allocation_pct": 100.0,
        "targets": [{
            "type": "Target",
            "trigger": {
                "group": "R Profit / R Loss", "element1": "R Profit",
                "event": "Cross Above", "compare_type": "Fixed Value",
                "element2": None, "value": 999.0,  # unreachable
            },
            "conditions": [],
        }],
        "stops": [],
    }],
)
_, stats_eod_pd = execute_custom_strategy(df.copy(), strat_eod, None, None)
_, stats_eod_np = execute_custom_strategy_numpy(df.copy(), strat_eod, None, None)
hp_eod_pd = _hold_from_attrs(stats_eod_pd)
hp_eod_np = _hold_from_attrs(stats_eod_np)
print(f"EOD trades (pd): {hp_eod_pd}")
print(f"EOD trades (np): {hp_eod_np}")
print(f"Exit-type pct (pd): target={stats_eod_pd.loc['Target exit (%)', 'value']:.0f}  "
      f"EOD={stats_eod_pd.loc['EOD exit (%)', 'value']:.0f}")
last_bar = len(df) - 1
assert hp_eod_pd, "Expected at least one EOD trade — strategy may not be entering"
assert hp_eod_pd == hp_eod_np, "Engines disagree on EOD holding periods"
assert max(hp_eod_pd) <= last_bar, f"max hp {max(hp_eod_pd)} > last bar {last_bar}"
# Compute the expected EOD holding period for the LAST entered trade
df_eod_out, _ = execute_custom_strategy(df.copy(), strat_eod, None, None)
entry_idx_eod = np.where(df_eod_out["entry_signal"].values)[0]
if len(entry_idx_eod):
    expected_eod = last_bar - int(entry_idx_eod[-1])
    print(f"Last entry bar={int(entry_idx_eod[-1])}, last bar={last_bar}, expected hp={expected_eod}")
    assert hp_eod_pd[-1] == expected_eod, f"EOD hp mismatch: got {hp_eod_pd[-1]}, expected {expected_eod}"
    print(f"OK — EOD hp ({hp_eod_pd[-1]}) = last_bar - entry_bar")


# ---------------------------------------------------------------------------
# Test 7 — empty result safe path
# ---------------------------------------------------------------------------
banner("TEST 7 — zero-trade strategies produce avg_holding_period=0.0")

strat_no_entry = make_strategy(
    entry_group="RSI Group", entry_element1="RSI",
    entry_event="Cross Above", entry_compare_type="Fixed Value", entry_value=999.0,  # impossible
)
_, stats_empty = execute_custom_strategy(df.copy(), strat_no_entry, None, None)
agg_empty = _aggregate_stats([stats_empty])
print(f"trades={agg_empty['num_trades']}  avg_holding_period={agg_empty['avg_holding_period']}")
assert agg_empty['num_trades'] == 0
assert agg_empty['avg_holding_period'] == 0.0
print("OK")


# ---------------------------------------------------------------------------
# Test 8 — manual spot-check: pick the first trade and verify hp by index
# ---------------------------------------------------------------------------
banner("TEST 8 — manual spot-check of one trade's bar indices")

# Re-run pandas engine and inspect one entry/exit
entry_bars = np.where(stats_pd is not None and df["entry_signal"] if "entry_signal" in df.columns else [])[0]
# After execute_custom_strategy returns (df_out, stats), df has signals attached:
df_out, _ = execute_custom_strategy(df.copy(), strategy, None, None)
entry_idx_arr = np.where(df_out["entry_signal"].values)[0]
exit_idx_arr = np.where(df_out["exit_signal"].values)[0]
print(f"Entry bar indices: {entry_idx_arr.tolist()[:8]}{'...' if len(entry_idx_arr) > 8 else ''}")
print(f"Exit  bar indices: {exit_idx_arr.tolist()[:8]}{'...' if len(exit_idx_arr) > 8 else ''}")
print(f"trade_holding_periods (first 8): {hp_pd[:8]}")

if len(entry_idx_arr) and len(exit_idx_arr):
    # First trade: from first entry bar to its first exit
    first_entry = int(entry_idx_arr[0])
    first_exit_after_entry = next((int(x) for x in exit_idx_arr if x > first_entry), None)
    if first_exit_after_entry is not None and hp_pd:
        expected_first_hp = first_exit_after_entry - first_entry
        print(f"First trade: entry@bar{first_entry}, exit@bar{first_exit_after_entry}, expected hp={expected_first_hp}, recorded hp={hp_pd[0]}")
        assert hp_pd[0] == expected_first_hp, "First trade hp mismatch!"
        print("OK — first-trade holding period matches bar-index arithmetic")


banner("ALL MANUAL TESTS PASSED")

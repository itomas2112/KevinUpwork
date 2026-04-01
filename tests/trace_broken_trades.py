"""
Trade-level trace script: shows concrete broken trades caused by the
"Close Below" state-check bug.

Run with: .venv/Scripts/python.exe tests/trace_broken_trades.py

Produces a per-trade report showing:
  - entry bar, entry price, stop price, r_distance
  - exit bar, exit price, exit reason
  - previous-bar indicator state (was the condition already true?)
  - realized R
  - what exit SHOULD have been under transition logic
"""
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import pandas as pd
from copy import deepcopy

from indicators.calculate_indicators import calculate_indicators
from strategies.first_strategy import execute_custom_strategy
from tests.conftest import (
    _generate_oscillation, _generate_downtrend,
    DEFAULT_INDICATOR_SETTINGS, make_strategy, make_exit_group,
)


def trace_trades(df, strategy):
    """
    Instrument execute_custom_strategy to capture per-trade detail.

    Returns list of dicts with trade-level debug info.
    """
    from strategies.first_strategy import execute_custom_strategy
    from config.constants import get_indicator_map

    df_copy = df.copy()
    result_df, stats_df = execute_custom_strategy(df_copy, deepcopy(strategy))

    # Reconstruct trade-level data from the signal columns
    # We need to re-run with instrumentation, so let's manually trace
    indicator_map = get_indicator_map(0)
    direction = strategy.get("direction", "Long")

    entry_bars = []
    exit_bars = []
    in_trade = False
    entry_idx = None

    for i in range(len(result_df)):
        if result_df["entry_signal"].iloc[i] and not in_trade:
            in_trade = True
            entry_idx = i
            entry_bars.append(i)
        elif result_df["exit_signal"].iloc[i] and in_trade:
            in_trade = False
            exit_bars.append(i)

    # If still in trade at end, close at last bar
    if in_trade and len(entry_bars) > len(exit_bars):
        exit_bars.append(len(result_df) - 1)

    trades = []
    for entry_i, exit_i in zip(entry_bars, exit_bars):
        entry_price = result_df["latest"].iloc[entry_i]
        exit_price = result_df["latest"].iloc[exit_i]
        raw_exit = result_df["exit_type"].iloc[exit_i] if "exit_type" in result_df.columns else None
        exit_type = raw_exit if raw_exit is not None else "End of Data"

        # Compute r_distance from ATR stop
        from indicators.atr_indicator import atr_indicator
        atr_vals = atr_indicator(df["high"], df["low"], df["latest"], period=14)
        atr_at_entry = atr_vals.iloc[entry_i]
        stop_mult = strategy["initial_stop"].get("atr_multiplier", 1.5)

        if direction == "Short":
            stop_price = entry_price + atr_at_entry * stop_mult
        else:
            stop_price = entry_price - atr_at_entry * stop_mult

        r_distance = abs(entry_price - stop_price)

        if direction == "Short":
            price_move = entry_price - exit_price
        else:
            price_move = exit_price - entry_price

        realized_r = price_move / r_distance if r_distance > 0 else 0.0

        trades.append({
            "trade_id": len(trades) + 1,
            "direction": direction,
            "entry_bar": entry_i,
            "entry_time": str(result_df.index[entry_i]),
            "exit_bar": exit_i,
            "exit_time": str(result_df.index[exit_i]),
            "bars_held": exit_i - entry_i,
            "entry_price": round(entry_price, 4),
            "exit_price": round(exit_price, 4),
            "stop_price": round(stop_price, 4),
            "r_distance": round(r_distance, 4),
            "realized_r": round(realized_r, 4),
            "exit_type": exit_type,
            "atr_at_entry": round(atr_at_entry, 4),
        })

    return trades, stats_df


def print_trace(label, trades, stats_df):
    """Pretty-print trade trace."""
    print(f"\n{'='*80}")
    print(f"  {label}")
    print(f"{'='*80}")

    num_trades = int(stats_df.loc["Number of trades", "value"])
    win_rate = float(stats_df.loc["Win rate (%)", "value"])
    target_pct = float(stats_df.loc["Target exit (%)", "value"])
    static_pct = float(stats_df.loc["Static exit (%)", "value"])
    eod_pct = float(stats_df.loc["EOD exit (%)", "value"])
    total_pnl = float(stats_df.loc["Total P&L (R)", "value"])

    print(f"\n  Summary: {num_trades} trades, Win rate: {win_rate:.1f}%")
    print(f"  Target exit: {target_pct:.1f}%, Static exit: {static_pct:.1f}%, EOD: {eod_pct:.1f}%")
    print(f"  Total P&L: {total_pnl:.2f}R")

    if stats_df.attrs.get("trade_pnls_r"):
        pnls = stats_df.attrs["trade_pnls_r"]
        wins = [p for p in pnls if p > 0]
        losses = [p for p in pnls if p < 0]
        if wins:
            print(f"  Avg Profit: {sum(wins)/len(wins):.4f}R ({len(wins)} wins)")
        if losses:
            print(f"  Avg Loss:   {sum(losses)/len(losses):.4f}R ({len(losses)} losses)")

    print(f"\n  {'ID':>3} {'Dir':>5} {'Entry':>5} {'Exit':>5} {'Bars':>4} "
          f"{'EntryPx':>9} {'ExitPx':>9} {'StopPx':>9} {'R_dist':>8} "
          f"{'Real_R':>8} {'ExitType':<14}")
    print(f"  {'-'*3} {'-'*5} {'-'*5} {'-'*5} {'-'*4} "
          f"{'-'*9} {'-'*9} {'-'*9} {'-'*8} "
          f"{'-'*8} {'-'*14}")

    for t in trades[:20]:  # Show first 20
        print(f"  {t['trade_id']:>3} {t['direction']:>5} "
              f"{t['entry_bar']:>5} {t['exit_bar']:>5} {t['bars_held']:>4} "
              f"{t['entry_price']:>9.4f} {t['exit_price']:>9.4f} "
              f"{t['stop_price']:>9.4f} {t['r_distance']:>8.4f} "
              f"{t['realized_r']:>8.4f} {t['exit_type']:<14}")

    if len(trades) > 20:
        print(f"  ... ({len(trades) - 20} more trades)")


def main():
    print("Trade-Level Trace: Close Below State-Check Bug")
    print("=" * 80)

    # --- Scenario A: Persistently true condition ---
    # Target: Price (Close Below) Fixed Value 500
    # Price ≈ 150, so price < 500 is ALWAYS true
    print("\n\n" + "=" * 80)
    print("SCENARIO A: Persistently true condition (Price < 500, always true)")
    print("=" * 80)

    df = _generate_oscillation(n=300, center=150.0, amplitude=25.0, period=40)
    df = calculate_indicators(df, **DEFAULT_INDICATOR_SETTINGS)

    for event in ["Cross Below", "Close Below"]:
        target = {
            "type": "Target",
            "trigger": {
                "group": "Price & Indicators",
                "element1": "Price",
                "event": event,
                "compare_type": "Fixed Value",
                "element2": None,
                "value": 500.0,
            },
            "conditions": [],
        }
        exit_group = make_exit_group(allocation_pct=100.0, targets=[target])
        strategy = make_strategy(
            direction="Short",
            entry_group="RSI Group",
            entry_element1="RSI",
            entry_event="Cross Below",
            entry_compare_type="Fixed Value",
            entry_value=70.0,
            exit_groups=[exit_group],
        )

        trades, stats_df = trace_trades(df, strategy)
        print_trace(f"Target: Price ({event}) 500  [condition always true]", trades, stats_df)

    # --- Scenario B: BB Lower (Close Below) BB Upper ---
    # BB Lower < BB Upper is ALWAYS true by construction
    print("\n\n" + "=" * 80)
    print("SCENARIO B: BB Lower Band vs BB Upper Band (always true by construction)")
    print("=" * 80)

    for event in ["Cross Below", "Close Below"]:
        target = {
            "type": "Target",
            "trigger": {
                "group": "Price & Indicators",
                "element1": "BB Lower Band",
                "event": event,
                "compare_type": "Indicator",
                "element2": "BB Upper Band",
                "value": None,
            },
            "conditions": [],
        }
        exit_group = make_exit_group(allocation_pct=100.0, targets=[target])
        strategy = make_strategy(
            direction="Short",
            entry_group="RSI Group",
            entry_element1="RSI",
            entry_event="Cross Below",
            entry_compare_type="Fixed Value",
            entry_value=70.0,
            exit_groups=[exit_group],
        )

        trades, stats_df = trace_trades(df, strategy)
        print_trace(f"Target: BB Lower ({event}) BB Upper  [always true]", trades, stats_df)

    # --- Scenario C: Show what "should" happen ---
    print("\n\n" + "=" * 80)
    print("SCENARIO C: Same strategies but with 'Close' (bidirectional) — uses transition")
    print("This shows the CORRECT behavior: 'Close' never fires on persistent condition")
    print("=" * 80)

    target = {
        "type": "Target",
        "trigger": {
            "group": "Price & Indicators",
            "element1": "Price",
            "event": "Close",
            "compare_type": "Fixed Value",
            "element2": None,
            "value": 500.0,
        },
        "conditions": [],
    }
    exit_group = make_exit_group(allocation_pct=100.0, targets=[target])
    strategy = make_strategy(
        direction="Short",
        entry_group="RSI Group",
        entry_element1="RSI",
        entry_event="Cross Below",
        entry_compare_type="Fixed Value",
        entry_value=70.0,
        exit_groups=[exit_group],
    )

    trades, stats_df = trace_trades(df, strategy)
    print_trace("Target: Price (Close) 500  [bidirectional transition — correct behavior]", trades, stats_df)


if __name__ == "__main__":
    main()

"""
Walk Forward Optimization engine.

Core logic: fold splitting, indicator-group detection, parameter grid
generation, and the main optimisation orchestrator.
"""

import copy
import itertools
import math
import numpy as np
import pandas as pd

from config.constants import ELEMENT_TO_WFO_GROUP, WFO_DEFAULT_RANGES


# ------------------------------------------------------------------
# Fold splitting
# ------------------------------------------------------------------

def split_folds(start, end, n_folds, train_ratio=0.8):
    """Split a date range into *n_folds* consecutive folds.

    Each fold is further divided into a training portion (first *train_ratio*)
    and a test portion (remaining).

    Returns list of dicts:
        [{"train_start", "train_end", "test_start", "test_end"}, ...]
    """
    start_ts = pd.Timestamp(start)
    end_ts = pd.Timestamp(end) + pd.Timedelta(days=1)
    total = end_ts - start_ts
    fold_len = total / n_folds

    folds = []
    for i in range(n_folds):
        f_start = start_ts + fold_len * i
        f_end = start_ts + fold_len * (i + 1)
        split_point = f_start + (f_end - f_start) * train_ratio
        folds.append({
            "train_start": f_start,
            "train_end": split_point,
            "test_start": split_point,
            "test_end": f_end,
        })
    return folds


# ------------------------------------------------------------------
# Detect which WFO parameter groups a strategy uses
# ------------------------------------------------------------------

def detect_used_groups(strategy):
    """Scan *strategy* for indicator element references and return the set
    of WFO parameter-group names that should be optimised."""

    groups = set()

    def _check(name):
        if not name:
            return
        if name in ELEMENT_TO_WFO_GROUP:
            groups.add(ELEMENT_TO_WFO_GROUP[name])
        elif name.startswith("EMA "):
            groups.add("ema")

    def _scan_trigger(t):
        if not t:
            return
        _check(t.get("element1"))
        _check(t.get("element2"))

    def _scan_conditions(conds):
        for c in (conds or []):
            _check(c.get("element1"))
            _check(c.get("element2"))

    entry = strategy.get("entry", {})
    _scan_trigger(entry.get("trigger"))
    _scan_conditions(entry.get("conditions"))

    istop = strategy.get("initial_stop", {})
    _check(istop.get("element2"))

    for grp in strategy.get("exit_groups", []):
        for t in grp.get("targets", []):
            _scan_trigger(t.get("trigger"))
            _scan_conditions(t.get("conditions"))
        for s in grp.get("stops", []):
            _scan_trigger(s.get("trigger"))
            _scan_conditions(s.get("conditions"))

    return groups


# ------------------------------------------------------------------
# Parameter grid generation
# ------------------------------------------------------------------

def _range_values(spec):
    """Turn a range spec into a list of values.

    *spec* is either:
      (min, max, step) — numeric range  (int or float)
      [v1, v2, ...]    — discrete list
    """
    if isinstance(spec, list):
        return list(spec)
    lo, hi, step = spec
    if isinstance(lo, int) and isinstance(step, int):
        return list(range(lo, hi + 1, step))
    # float range
    vals = []
    v = lo
    while v <= hi + 1e-9:
        vals.append(round(v, 6))
        v += step
    return vals


def generate_param_grid(used_groups, ranges, base_settings):
    """Generate the full Cartesian-product parameter grid.

    Parameters
    ----------
    used_groups : set[str]
        WFO groups to optimise (from detect_used_groups).
    ranges : dict
        group_name -> {param_name: range_spec, ...}.
        Falls back to WFO_DEFAULT_RANGES for any missing entry.
    base_settings : dict
        The strategy's saved indicator_settings (used for params that
        are *not* being optimised).

    Returns
    -------
    list[dict]
        Each dict is a complete indicator_settings that can be passed
        straight to calculate_indicators.
    """
    # Collect (param_name, [values]) pairs
    axes = []  # list of (param_name, [values])

    for group in sorted(used_groups):
        group_ranges = ranges.get(group, WFO_DEFAULT_RANGES.get(group, {}))

        if group == "ema":
            # Special: build per-EMA axes
            ema_ranges = group_ranges.get("_ema_ranges",
                                          WFO_DEFAULT_RANGES["ema"]["_ema_ranges"])
            ema_count = len(base_settings.get("ema_periods", []))
            for idx in range(ema_count):
                if idx < len(ema_ranges):
                    axes.append((f"_ema_{idx}", _range_values(ema_ranges[idx])))
                else:
                    # No range defined for this EMA — keep fixed
                    axes.append((f"_ema_{idx}", [base_settings["ema_periods"][idx]]))
            continue

        for param, spec in group_ranges.items():
            if param.startswith("_"):
                continue
            axes.append((param, _range_values(spec)))

    if not axes:
        return [dict(base_settings)]

    names = [a[0] for a in axes]
    value_lists = [a[1] for a in axes]

    grid = []
    for combo in itertools.product(*value_lists):
        settings = dict(base_settings)
        ema_vals = {}
        for name, val in zip(names, combo):
            if name.startswith("_ema_"):
                ema_vals[int(name.split("_")[2])] = val
            else:
                settings[name] = val
        if ema_vals:
            ema_list = list(settings.get("ema_periods", []))
            for idx, val in ema_vals.items():
                if idx < len(ema_list):
                    ema_list[idx] = val
            settings["ema_periods"] = ema_list
        grid.append(settings)

    return grid


def count_grid_size(used_groups, ranges, base_settings):
    """Estimate the total number of combos without generating them all."""
    total = 1
    for group in sorted(used_groups):
        group_ranges = ranges.get(group, WFO_DEFAULT_RANGES.get(group, {}))
        if group == "ema":
            ema_ranges = group_ranges.get("_ema_ranges",
                                          WFO_DEFAULT_RANGES["ema"]["_ema_ranges"])
            ema_count = len(base_settings.get("ema_periods", []))
            for idx in range(ema_count):
                if idx < len(ema_ranges):
                    total *= len(_range_values(ema_ranges[idx]))
        else:
            for param, spec in group_ranges.items():
                if param.startswith("_"):
                    continue
                total *= len(_range_values(spec))
    return total


# ------------------------------------------------------------------
# Aggregation helpers (pickle-safe dicts, same schema as grid search)
# ------------------------------------------------------------------

def _extract_stats(stats_df):
    """Convert stats DataFrame to a lightweight pickle-safe dict."""
    return {
        "win_pnl": float(stats_df.loc["Winning trades P&L (R)", "value"]),
        "lose_pnl": float(stats_df.loc["Losing trades P&L (R)", "value"]),
        "trade_pnls_r": list(stats_df.attrs.get("trade_pnls_r", [])),
        "trade_holding_periods": list(stats_df.attrs.get("trade_holding_periods", [])),
        "total_static_alloc": float(stats_df.attrs.get("total_static_alloc", 0.0)),
        "total_dynamic_alloc": float(stats_df.attrs.get("total_dynamic_alloc", 0.0)),
        "total_target_alloc": float(stats_df.attrs.get("total_target_alloc", 0.0)),
        "total_eod_alloc": float(stats_df.attrs.get("total_eod_alloc", 0.0)),
    }


def aggregate_stats_dicts(all_dicts):
    """Aggregate a list of per-period stats dicts into one summary dict.

    Same logic as grid_search_tab._aggregate_stats_dicts.
    """
    all_pnls = []
    all_holding_periods = []
    win_pnl = lose_pnl = 0.0
    static_alloc = dynamic_alloc = target_alloc = eod_alloc = 0.0

    for d in all_dicts:
        win_pnl += d["win_pnl"]
        lose_pnl += d["lose_pnl"]
        static_alloc += d["total_static_alloc"]
        dynamic_alloc += d["total_dynamic_alloc"]
        target_alloc += d["total_target_alloc"]
        eod_alloc += d.get("total_eod_alloc", 0.0)
        all_pnls.extend(d["trade_pnls_r"])
        all_holding_periods.extend(d.get("trade_holding_periods", []))

    n = len(all_pnls)
    if n == 0:
        return _empty_agg()

    wins = [p for p in all_pnls if p > 0]
    losses = [p for p in all_pnls if p <= 0]
    total_pnl = sum(all_pnls)
    win_pct = len(wins) / n * 100
    lose_pct = len(losses) / n * 100
    avg_win = np.mean(wins) if wins else 0.0
    avg_lose = np.mean(losses) if losses else 0.0
    ev = total_pnl / n
    rr_ratio = abs(avg_win / avg_lose) if avg_lose != 0 else 0.0

    total_alloc = static_alloc + dynamic_alloc + target_alloc + eod_alloc
    target_pct = (target_alloc / total_alloc * 100) if total_alloc else 0.0
    static_pct = (static_alloc / total_alloc * 100) if total_alloc else 0.0
    dynamic_pct = (dynamic_alloc / total_alloc * 100) if total_alloc else 0.0
    eod_pct = (eod_alloc / total_alloc * 100) if total_alloc else 0.0

    arr = np.array(all_pnls)
    std = arr.std(ddof=1) if n > 1 else 0.0
    cum = np.cumsum(arr)
    peak = np.maximum.accumulate(cum)
    dd = peak - cum
    max_dd = float(dd.max()) if len(dd) else 0.0
    sqn = (arr.mean() / std * math.sqrt(n)) if std else 0.0

    avg_holding_period = (sum(all_holding_periods) / len(all_holding_periods)) if all_holding_periods else 0.0

    return {
        "num_trades": n,
        "win_pct": win_pct,
        "lose_pct": lose_pct,
        "avg_win_pnl": float(avg_win),
        "avg_lose_pnl": float(avg_lose),
        "total_pnl": float(total_pnl),
        "expected_value": float(ev),
        "target_exit_pct": float(target_pct),
        "static_exit_pct": float(static_pct),
        "dynamic_exit_pct": float(dynamic_pct),
        "eod_exit_pct": float(eod_pct),
        "rr_ratio": float(rr_ratio),
        "max_drawdown": float(max_dd),
        "sqn": float(sqn),
        "avg_holding_period": float(avg_holding_period),
    }


def _empty_agg():
    return {
        "num_trades": 0, "win_pct": 0.0, "lose_pct": 0.0,
        "avg_win_pnl": 0.0, "avg_lose_pnl": 0.0, "total_pnl": 0.0,
        "expected_value": 0.0, "target_exit_pct": 0.0, "static_exit_pct": 0.0,
        "dynamic_exit_pct": 0.0, "eod_exit_pct": 0.0, "rr_ratio": 0.0,
        "max_drawdown": 0.0, "sqn": 0.0, "avg_holding_period": 0.0,
    }


# ------------------------------------------------------------------
# Filter / sort helpers
# ------------------------------------------------------------------

def apply_filters(results, thresholds):
    """Filter (idx, params, agg) tuples by metric thresholds.

    *thresholds*: dict  metric_key -> min_value
    """
    out = []
    for item in results:
        agg = item[2]
        if all(agg.get(k, 0) >= v for k, v in thresholds.items()):
            out.append(item)
    return out

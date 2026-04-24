"""
Multiprocessing worker for Walk Forward Optimization.
Separate module to avoid Streamlit serialisation issues on Windows (spawn).

Performance strategy: receive a pre-computed featured DataFrame (all indicators
already calculated with base params).  For each parameter combo, copy the df
and use recalculate_groups() to update ONLY the changed indicator groups
in-place — much faster than a full calculate_indicators() call.
"""

import pandas as pd
from indicators.calculate_indicators import recalculate_groups, changed_groups
from strategies.first_strategy_numpy import execute_custom_strategy_numpy
from strategies.wfo_engine import (
    _extract_stats, aggregate_stats_dicts,
    split_atr_overlay, apply_atr_overlay,
    split_value_overlay, apply_value_overlay,
)


# Shared state set once per worker via initializer
_worker_strategy = None
_worker_periods = None       # list of (start_dt, end_dt)
_worker_base_params = None   # indicator_settings the base df was calculated with
_worker_df_featured = None   # pre-computed DataFrame with all indicators

# Context bars prepended to each slice so indicators have lookback data
_CONTEXT_BARS = 50


def init_wfo_worker(strategy, periods, base_params, df_featured):
    """Called once per worker process."""
    global _worker_strategy, _worker_periods, _worker_base_params, _worker_df_featured
    _worker_strategy = strategy
    _worker_periods = periods
    _worker_base_params = base_params
    _worker_df_featured = df_featured


def run_wfo_batch(args):
    """Evaluate a batch of parameter combos.

    Args:
        args: (batch_indices, param_combos, date_ranges)
              - batch_indices: list of ints (original grid indices)
              - param_combos: list of indicator_settings dicts
              - date_ranges: list of (range_start, range_end) tuples

    Returns:
        list of (grid_idx, param_combo, stats_dict_or_None)
    """
    batch_indices, param_combos, date_ranges = args

    results = []
    for grid_idx, params in zip(batch_indices, param_combos):
        try:
            agg = _evaluate_single(params, date_ranges)
            results.append((grid_idx, params, agg))
        except Exception:
            results.append((grid_idx, params, None))

    return results


def _evaluate_single(params, date_ranges):
    """Run strategy with *params* on all DRM periods within *date_ranges*.

    Uses incremental recalculation: only recomputes indicator groups whose
    parameters differ from the base. ATR slot overrides (keys namespaced with
    the ATR overlay prefix) are split off and applied to a deepcopy of the
    strategy before execution.
    """
    # Strip out namespaced overlay keys before they reach indicator recalc.
    rest, atr_overlay = split_atr_overlay(params)
    indicator_params, value_overlay = split_value_overlay(rest)

    # Determine which groups changed (indicator-only view)
    groups = changed_groups(_worker_base_params, indicator_params)

    if groups:
        df = _worker_df_featured.copy()
        recalculate_groups(df, groups, **indicator_params)
    else:
        df = _worker_df_featured

    # Apply both overlays — value overlay first since it doesn't conflict with ATR.
    strategy = apply_value_overlay(_worker_strategy, value_overlay)
    strategy = apply_atr_overlay(strategy, atr_overlay)
    full_index = df.index
    all_stats = []

    for start_dt, end_dt in _worker_periods:
        # Check if period start falls within any training date range
        in_range = False
        for r_start, r_end in date_ranges:
            if start_dt >= r_start and start_dt < r_end:
                in_range = True
                break
        if not in_range:
            continue

        try:
            # Slice with context bars
            start_pos = full_index.searchsorted(start_dt, side="left")
            end_pos = full_index.searchsorted(end_dt, side="right") - 1
            if end_pos < start_pos:
                continue
            ext_start = max(0, start_pos - _CONTEXT_BARS)
            df_slice = df.iloc[ext_start: end_pos + 1]
            if df_slice.empty:
                continue

            # ps/pe = actual DRM period boundaries
            ps = df_slice.index[df_slice.index.searchsorted(start_dt, side="left")]
            pe_pos = df_slice.index.searchsorted(end_dt, side="right") - 1
            pe = df_slice.index[min(pe_pos, len(df_slice.index) - 1)]

            _, stats = execute_custom_strategy_numpy(
                df_slice.copy(), strategy, ps, pe
            )
            if stats is not None:
                all_stats.append(_extract_stats(stats))
        except Exception:
            continue

    if not all_stats:
        return None

    return aggregate_stats_dicts(all_stats)

"""
Worker module for multiprocessing grid search.
Separate from Streamlit code to avoid serialization issues on Windows (spawn).
"""

from strategies.first_strategy_numpy import execute_custom_strategy_numpy

# Shared data set once per worker process via initializer
_worker_combo_slices = None
_worker_combo_keys = None


def init_worker(combo_slices, combo_keys):
    """Called once per worker process. Receives shared data (pickled once per worker)."""
    global _worker_combo_slices, _worker_combo_keys
    _worker_combo_slices = combo_slices
    _worker_combo_keys = combo_keys


def run_candidate(args):
    """Run a single candidate strategy across all combo/period slices.

    Args:
        args: (idx, label, strategy_dict)

    Returns:
        (idx, label, {combo_key: [stats_dict, ...]})
        stats_dict is a lightweight pickle-safe dict (no DataFrame).
    """
    idx, label, strategy = args
    combo_results = {}

    for combo_key in _worker_combo_keys:
        slices = _worker_combo_slices.get(combo_key, [])
        stats_list = []
        for df_slice, ps, pe in slices:
            try:
                _, stats_df = execute_custom_strategy_numpy(df_slice.copy(), strategy, ps, pe)
                if stats_df is not None:
                    stats_list.append(_extract_stats(stats_df))
            except Exception:
                pass
        combo_results[combo_key] = stats_list

    return (idx, label, combo_results)


def _extract_stats(stats_df):
    """Extract minimal data needed for aggregation — pickle-safe, no DataFrame."""
    return {
        'win_pnl': float(stats_df.loc['Winning trades P&L (R)', 'value']),
        'lose_pnl': float(stats_df.loc['Losing trades P&L (R)', 'value']),
        'trade_pnls_r': list(stats_df.attrs.get('trade_pnls_r', [])),
        'trade_holding_periods': list(stats_df.attrs.get('trade_holding_periods', [])),
        'total_static_alloc': float(stats_df.attrs.get('total_static_alloc', 0.0)),
        'total_dynamic_alloc': float(stats_df.attrs.get('total_dynamic_alloc', 0.0)),
        'total_target_alloc': float(stats_df.attrs.get('total_target_alloc', 0.0)),
        'total_eod_alloc': float(stats_df.attrs.get('total_eod_alloc', 0.0)),
    }

"""
Worker module for parallel Monte Carlo enrichment in Grid Search.

Imports only from monte_carlo_core (no streamlit) so Windows spawn workers
don't re-import the entire UI layer. init_worker sets the constants once per
process; enrich_one bootstraps each agg's own trades (pnl_r + stop distance)
into a dollar-based, margin-capped, phased-risk simulation (mc_phased_profit:
1% for trades 1–30, then risk re-assessed every 200 trades to target 5%
avg max DD).
"""
from strategies.monte_carlo_core import mc_phased_profit

_balance = None
_n_sims = None
_target_dd = None
_point_value = None
_margin = None
_trades_per_sim = None


def init_worker(balance, n_sims, target_dd, point_value, margin, trades_per_sim):
    """Called once per worker process. Stores MC constants in module globals."""
    global _balance, _n_sims, _target_dd, _point_value, _margin, _trades_per_sim
    _balance = balance
    _n_sims = n_sims
    _target_dd = target_dd
    _point_value = point_value
    _margin = margin
    _trades_per_sim = trades_per_sim


def enrich_one(args):
    """Compute mc_avg_profit for a single agg from its own trade lists.

    Args: (key, pnls_r, r_dists)
    Returns (key, avg_profit, margin_capped, initial_skipped_fraction).
    The target DD is MC_TARGET_DD inside mc_phased_profit; _target_dd is kept
    so the init_worker signature is unchanged.
    """
    key, pnls_r, r_dists = args
    if not pnls_r:
        return (key, 0.0, False, 0.0)
    res = mc_phased_profit(
        pnls_r, r_dists, _balance, _point_value, _margin,
        trades_per_sim=_trades_per_sim,
        n_sims=_n_sims)
    return (key, res["avg_profit"], res["margin_capped"],
            res["initial_skipped_fraction"])

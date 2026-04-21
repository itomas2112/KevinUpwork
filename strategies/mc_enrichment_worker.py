"""
Worker module for parallel Monte Carlo enrichment in Grid Search.

Imports only from monte_carlo_core (no streamlit) so Windows spawn workers
don't re-import the entire UI layer. init_worker sets the constants once per
process; enrich_one is called per agg and uses that agg's own trade count as
the simulation length (trades_per_sim).
"""
from strategies.monte_carlo_core import compute_mc_avg_profit_at_target_dd

_balance = None
_n_sims = None
_target_dd = None


def init_worker(balance, n_sims, target_dd):
    """Called once per worker process. Stores MC constants in module globals.

    trades_per_sim is NOT stored here — it's per-task (= candidate trade count).
    """
    global _balance, _n_sims, _target_dd
    _balance = balance
    _n_sims = n_sims
    _target_dd = target_dd


def enrich_one(args):
    """Compute mc_avg_profit for a single agg using its own trade count.

    Args: (key, win_pct, rr_ratio, num_trades)
      - trades_per_sim is set to num_trades, so each candidate simulates
        exactly as many trades as it actually produced.
    Returns (key, mc_avg_profit).
    """
    key, win_pct, rr_ratio, num_trades = args
    if num_trades == 0:
        return (key, 0.0)
    value = compute_mc_avg_profit_at_target_dd(
        win_pct, rr_ratio, _balance,
        target_dd=_target_dd,
        trades_per_sim=num_trades,
        n_sims=_n_sims)
    return (key, value)

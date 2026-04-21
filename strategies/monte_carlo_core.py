"""
Monte Carlo simulation math — pure numpy, no streamlit or plotly.

Extracted from ui/monte_carlo_tab.py so that:
 - Multiprocessing workers (e.g. grid search MC enrichment) can import this
   without pulling in streamlit, which is expensive under spawn mode.
 - UI code (monte_carlo_tab, performance_tab, grid_search_tab) still
   imports from the original location via re-export in monte_carlo_tab.
"""
import numpy as np


def compute_mc_avg_profit_at_dd(win_rate, reward_risk, risk_pct,
                                starting_balance, max_dd_threshold=5.0,
                                trades_per_sim=100, n_sims=5000,
                                skip_threshold=False):
    """Quick Monte Carlo: return avg final balance based on avg max drawdown.

    Uses the average of all per-simulation max drawdowns (not per-sim filtering).
    When avg max DD <= threshold: returns mean of all final balances.
    When avg max DD > threshold: returns None.

    If skip_threshold=True (used by Grid Search for comparability):
      - Always returns a float, never None.
      - For invalid inputs (0% WR, 0 RR), returns deterministic result:
        balance * (1 - risk_pct/100)^trades_per_sim (pure losing account).
    """
    if risk_pct <= 0 or starting_balance <= 0:
        if skip_threshold:
            return float(starting_balance)
        return None
    if win_rate <= 0 or reward_risk <= 0:
        if skip_threshold:
            decay = (1.0 - risk_pct / 100.0) ** trades_per_sim
            return float(starting_balance * decay)
        return None
    results = _run_simulation(starting_balance, trades_per_sim, n_sims,
                              win_rate, reward_risk, risk_pct)
    if skip_threshold:
        return float(np.mean(results["final_balances"]))
    avg_max_dd = float(np.mean(results["max_drawdowns"]))
    if avg_max_dd > max_dd_threshold:
        return None
    return float(np.mean(results["final_balances"]))


def compute_mc_avg_profit_at_target_dd(win_rate, reward_risk,
                                       starting_balance, target_dd=5.0,
                                       trades_per_sim=100, n_sims=5000):
    """Find the risk % that produces target avg max DD, return avg profit there.

    Uses binary search to find the risk_pct where avg max drawdown ≈ target_dd%.
    Returns the mean final balance at that risk level, or 0.0 if no valid risk
    can be found (e.g. 0% win rate or 0 RR).
    """
    if win_rate <= 0 or reward_risk <= 0:
        return 0.0

    lo, hi = 0.01, 100.0
    tolerance = 0.05
    max_iterations = 30
    best_profit = 0.0

    for _ in range(max_iterations):
        mid = (lo + hi) / 2.0
        results = _run_simulation(starting_balance, trades_per_sim, n_sims,
                                  win_rate, reward_risk, mid)
        avg_dd = float(np.mean(results["max_drawdowns"]))
        avg_profit = float(np.mean(results["final_balances"]))

        if abs(avg_dd - target_dd) <= tolerance:
            return avg_profit
        if avg_dd < target_dd:
            best_profit = avg_profit
            lo = mid
        else:
            hi = mid

    mid = (lo + hi) / 2.0
    results = _run_simulation(starting_balance, trades_per_sim, n_sims,
                              win_rate, reward_risk, mid)
    return float(np.mean(results["final_balances"]))


def _run_simulation(starting_balance, trades_per_sim, n_simulations,
                    win_rate, reward_risk, risk_pct):
    """Run Monte Carlo simulation using vectorized NumPy operations.

    Returns dict with final_balances, max_drawdowns, and sampled equity curves.
    """
    rng = np.random.default_rng()

    outcomes = rng.random((n_simulations, trades_per_sim)) < (win_rate / 100.0)

    equity = np.empty((n_simulations, trades_per_sim + 1))
    equity[:, 0] = starting_balance

    for t in range(trades_per_sim):
        risk_amount = equity[:, t] * (risk_pct / 100.0)
        pnl = np.where(outcomes[:, t],
                       risk_amount * reward_risk,
                       -risk_amount)
        equity[:, t + 1] = equity[:, t] + pnl

    final_balances = equity[:, -1]

    running_peak = np.maximum.accumulate(equity, axis=1)
    drawdowns = (running_peak - equity) / running_peak * 100.0
    max_drawdowns = np.max(drawdowns, axis=1)

    max_curves = 500
    if n_simulations <= max_curves:
        sampled_equity = equity
    else:
        indices = rng.choice(n_simulations, max_curves, replace=False)
        sampled_equity = equity[indices]

    return {
        "final_balances": final_balances,
        "max_drawdowns": max_drawdowns,
        "sampled_equity": sampled_equity,
        "median_curve": np.median(equity, axis=0),
    }

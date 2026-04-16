"""
Monte Carlo Simulation Tab
Standalone simulator — no dependency on other tabs, indicators, or strategies.
"""
import streamlit as st
import numpy as np
import plotly.graph_objects as go


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
            # Deterministic: every trade is a full loss of risk_pct
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

    # Binary search for risk_pct that yields avg max DD ≈ target_dd
    lo, hi = 0.01, 100.0
    tolerance = 0.05  # accept within 0.05% of target DD
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
            # DD too low — can take more risk
            best_profit = avg_profit
            lo = mid
        else:
            # DD too high — reduce risk
            hi = mid

    # Return best approximation found
    # Run one final sim at the converged midpoint
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

    # Generate win/loss matrix: True = win, False = loss
    outcomes = rng.random((n_simulations, trades_per_sim)) < (win_rate / 100.0)

    # Build equity curves iteratively (balance-dependent risk requires loop over trades)
    equity = np.empty((n_simulations, trades_per_sim + 1))
    equity[:, 0] = starting_balance

    for t in range(trades_per_sim):
        risk_amount = equity[:, t] * (risk_pct / 100.0)
        pnl = np.where(outcomes[:, t],
                       risk_amount * reward_risk,   # win
                       -risk_amount)                 # loss
        equity[:, t + 1] = equity[:, t] + pnl

    # Final balances
    final_balances = equity[:, -1]

    # Max drawdown per simulation (measured from highest balance)
    running_peak = np.maximum.accumulate(equity, axis=1)
    drawdowns = (running_peak - equity) / running_peak * 100.0
    max_drawdowns = np.max(drawdowns, axis=1)

    # Sample equity curves for plotting (cap at 500 for performance)
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


def _build_equity_chart(results, trades_per_sim, starting_balance):
    """Build Plotly figure with all sampled equity curves + median highlight."""
    fig = go.Figure()
    x = np.arange(trades_per_sim + 1)
    sampled = results["sampled_equity"]

    # Individual simulation paths
    for i in range(sampled.shape[0]):
        fig.add_trace(go.Scatter(
            x=x, y=sampled[i],
            mode="lines",
            line=dict(width=0.5, color="rgba(100,149,237,0.08)"),
            hoverinfo="skip",
            showlegend=False,
        ))

    # Median curve
    fig.add_trace(go.Scatter(
        x=x, y=results["median_curve"],
        mode="lines",
        line=dict(width=2.5, color="#FFD700"),
        name="Median",
    ))

    # Starting balance reference line
    fig.add_hline(y=starting_balance, line_dash="dot",
                  line_color="rgba(255,255,255,0.3)")

    fig.update_layout(
        template="plotly_dark",
        height=520,
        margin=dict(l=60, r=20, t=30, b=40),
        xaxis_title="Trade #",
        yaxis_title="Balance ($)",
        legend=dict(x=0.01, y=0.99),
    )
    return fig


def render_monte_carlo_tab():
    """Render the Monte Carlo Simulation tab."""

    st.subheader("Monte Carlo Simulation")

    # ── Inputs ──────────────────────────────────────
    col1, col2, col3 = st.columns(3)
    with col1:
        starting_balance = st.number_input(
            "Starting Balance ($)", min_value=1.0, value=10000.0,
            step=1000.0, format="%.2f", key="mc_starting_balance")
        win_rate = st.number_input(
            "Win Rate (%)", min_value=0.0, max_value=100.0, value=50.0,
            step=1.0, format="%.1f", key="mc_win_rate")
    with col2:
        trades_per_sim = st.number_input(
            "Trades per Simulation", min_value=1, value=100,
            step=10, key="mc_trades_per_sim")
        reward_risk = st.number_input(
            "Reward:Risk Ratio", min_value=0.01, value=2.0,
            step=0.1, format="%.2f", key="mc_reward_risk")
    with col3:
        n_simulations = st.number_input(
            "Number of Simulations", min_value=1, value=20000,
            step=1000, key="mc_n_simulations")
        risk_pct = st.number_input(
            "Risk per Trade (%)", min_value=0.01, max_value=100.0,
            value=1.0, step=0.25, format="%.2f", key="mc_risk_pct")

    st.markdown("---")

    # ── Run button ──────────────────────────────────
    if st.button("Run Simulation", key="mc_run_btn", type="primary"):
        with st.spinner("Running simulations..."):
            results = _run_simulation(
                starting_balance, trades_per_sim, n_simulations,
                win_rate, reward_risk, risk_pct)
            st.session_state["mc_results"] = results
            st.session_state["mc_last_params"] = (
                starting_balance, trades_per_sim, n_simulations,
                win_rate, reward_risk, risk_pct)

    # ── Results ─────────────────────────────────────
    results = st.session_state.get("mc_results")
    if results is None:
        st.info("Configure inputs above and click **Run Simulation**.")
        return

    fb = results["final_balances"]
    md = results["max_drawdowns"]
    params = st.session_state.get("mc_last_params", (starting_balance,
                                  trades_per_sim, n_simulations,
                                  win_rate, reward_risk, risk_pct))

    # Metrics row
    m1, m2, m3, m4, m5, m6 = st.columns(6)
    m1.metric("Median Final Balance", f"${np.median(fb):,.2f}")
    m2.metric("Best Final Balance", f"${np.max(fb):,.2f}")
    m3.metric("Worst Final Balance", f"${np.min(fb):,.2f}")
    m4.metric("Avg Max Drawdown", f"{np.mean(md):.2f}%")
    m5.metric("Worst Drawdown Seen", f"{np.max(md):.2f}%")
    m6.metric("95th Pctl Max DD", f"{np.percentile(md, 95):.2f}%")

    # Equity curves chart
    fig = _build_equity_chart(results, params[1], params[0])
    st.plotly_chart(fig, use_container_width=True)

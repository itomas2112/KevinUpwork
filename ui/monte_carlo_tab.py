"""
Monte Carlo Simulation Tab
Standalone simulator — no dependency on other tabs, indicators, or strategies.
"""
import streamlit as st
import numpy as np
import plotly.graph_objects as go


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
            "Number of Simulations", min_value=1, value=1000,
            step=100, key="mc_n_simulations")
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
    m1, m2, m3, m4, m5 = st.columns(5)
    m1.metric("Median Final Balance", f"${np.median(fb):,.2f}")
    m2.metric("Best Final Balance", f"${np.max(fb):,.2f}")
    m3.metric("Worst Final Balance", f"${np.min(fb):,.2f}")
    m4.metric("Avg Max Drawdown", f"{np.mean(md):.2f}%")
    m5.metric("Worst Drawdown Seen", f"{np.max(md):.2f}%")

    # Equity curves chart
    fig = _build_equity_chart(results, params[1], params[0])
    st.plotly_chart(fig, use_container_width=True)

"""
Monte Carlo Simulation Tab
Standalone simulator — no dependency on other tabs, indicators, or strategies.
"""
import streamlit as st
import numpy as np
import plotly.graph_objects as go

from config.constants import (DEFAULT_INSTRUMENT, INSTRUMENTS, MC_DEFAULT_BALANCE,
                              MC_DEFAULT_TRADES_PER_SIM, MC_INITIAL_RISK_PCT,
                              MC_INITIAL_RISK_BAND, MC_INITIAL_TRADES,
                              MC_REASSESS_EVERY,
                              MC_REASSESS_SIMS, instrument_min_tick,
                              instrument_point_value, instrument_margin)

# MC math lives in strategies/monte_carlo_core.py (streamlit-free) so workers
# can import it without dragging streamlit along. Re-exported here so existing
# callers using `from ui.monte_carlo_tab import ...` keep working.
from strategies.monte_carlo_core import (
    _run_simulation,
    _size_contracts,
    compute_mc_avg_profit_at_dd,
    compute_mc_avg_profit_at_target_dd,
)


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

    symbol = st.session_state.get("instrument", DEFAULT_INSTRUMENT)
    if symbol not in INSTRUMENTS:
        symbol = DEFAULT_INSTRUMENT
    point_value = instrument_point_value(symbol)
    margin = instrument_margin(symbol)
    min_tick = instrument_min_tick(symbol)

    # ── Inputs ──────────────────────────────────────
    # Widget keys are mc_tab_* — the sidebar Instrument section owns
    # mc_starting_balance / mc_trades_per_sim; these default from them.
    phased = st.checkbox(
        f"Phased risk (trades 1–{MC_INITIAL_TRADES} at {MC_INITIAL_RISK_PCT:g}%, "
        f"re-assess every {MC_REASSESS_EVERY} trades, {MC_REASSESS_SIMS}-sim MC)",
        value=True, key="mc_phased",
        help=f"Trades 1–{MC_INITIAL_TRADES} risk {MC_INITIAL_RISK_PCT:g}% "
             f"({MC_INITIAL_RISK_BAND[0]:g}–{MC_INITIAL_RISK_BAND[1]:g}% after "
             f"whole-contract rounding, trade skipped if no size fits); every "
             f"{MC_REASSESS_EVERY} trades thereafter each path's risk % is "
             f"re-assessed from its own outcomes so far ({MC_REASSESS_SIMS}-sim MC "
             f"of the next block) to target 5% avg max DD.")
    col1, col2, col3 = st.columns(3)
    with col1:
        starting_balance = st.number_input(
            "Starting Balance ($)", min_value=1.0,
            value=float(st.session_state.get("mc_starting_balance", MC_DEFAULT_BALANCE)),
            step=1000.0, format="%.2f", key="mc_tab_starting_balance")
        win_rate = st.number_input(
            "Win Rate (%)", min_value=0.0, max_value=100.0, value=50.0,
            step=1.0, format="%.1f", key="mc_win_rate")
    with col2:
        trades_per_sim = st.number_input(
            "Trades per Simulation", min_value=1,
            value=int(st.session_state.get("mc_trades_per_sim", MC_DEFAULT_TRADES_PER_SIM)),
            step=10, key="mc_tab_trades_per_sim")
        reward_risk = st.number_input(
            "Reward:Risk Ratio", min_value=0.01, value=2.0,
            step=0.1, format="%.2f", key="mc_reward_risk")
    with col3:
        n_simulations = st.number_input(
            "Number of Simulations", min_value=1, value=20000,
            step=1000, key="mc_n_simulations")
        risk_pct = st.number_input(
            "Risk per Trade (%)", min_value=0.01, max_value=100.0,
            value=1.0, step=0.25, format="%.2f", key="mc_risk_pct",
            disabled=phased)

    stop_pts = st.number_input(
        "Stop distance (points)", min_value=float(min_tick), value=max(10.0, float(min_tick)),
        step=max(float(min_tick), 0.25), format="%g", key=f"mc_stop_pts_{symbol}")
    start_contracts = int(_size_contracts(
        starting_balance, MC_INITIAL_RISK_PCT if phased else risk_pct,
        stop_pts * point_value, margin,
        band=MC_INITIAL_RISK_BAND if phased else None))
    st.caption(f"Instrument {symbol}: ${point_value:,.2f}/pt, margin ${margin:,.0f} "
               f"→ {max(start_contracts, 0)} contracts at start")

    st.markdown("---")

    # ── Run button ──────────────────────────────────
    if st.button("Run Simulation", key="mc_run_btn", type="primary"):
        with st.spinner("Running simulations..."):
            results = _run_simulation(
                starting_balance, trades_per_sim, n_simulations,
                win_rate, reward_risk, risk_pct,
                point_value=point_value, margin_per_contract=margin,
                stop_pts=stop_pts, phased=phased)
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
    m1, m2, m3, m4, m5, m6, m7 = st.columns(7)
    m1.metric("Median Final Balance", f"${np.median(fb):,.2f}")
    m2.metric("Best Final Balance", f"${np.max(fb):,.2f}")
    m3.metric("Worst Final Balance", f"${np.min(fb):,.2f}")
    m4.metric("Avg Max Drawdown", f"{np.mean(md):.2f}%")
    m5.metric("Worst Drawdown Seen", f"{np.max(md):.2f}%")
    m6.metric("95th Pctl Max DD", f"{np.percentile(md, 95):.2f}%")
    block_risks = results.get("block_risks")
    if block_risks is not None and block_risks.shape[1] > 1:
        m7.metric("Avg risk % after re-assessment",
                  f"{np.mean(block_risks[:, 1:]):.2f}%")
    else:
        m7.metric("Avg risk % after re-assessment", "—")

    # Equity curves chart
    fig = _build_equity_chart(results, params[1], params[0])
    st.plotly_chart(fig, use_container_width=True)

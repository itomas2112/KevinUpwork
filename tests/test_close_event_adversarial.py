"""
Adversarial tests for Close/Cross event semantics.

Close Above/Below = STATE (fires every bar condition holds)
Cross Above/Below = TRANSITION (fires once per crossing)

Also tests R Profit/Loss Close Above/Below state fix and pandas/numpy parity.
"""
import pytest
import numpy as np
import pandas as pd
from copy import deepcopy

from indicators.calculate_indicators import calculate_indicators
from strategies.first_strategy import execute_custom_strategy
from strategies.first_strategy_numpy import execute_custom_strategy_numpy
from tests.conftest import (
    _make_ohlc, _generate_downtrend, _generate_oscillation, _generate_uptrend,
    DEFAULT_INDICATOR_SETTINGS, make_strategy, make_exit_group,
)


def _run_both(df, strategy):
    result_df, stats_pd = execute_custom_strategy(df.copy(), strategy)
    _, stats_np = execute_custom_strategy_numpy(df.copy(), strategy)
    return result_df, stats_pd, stats_np


def _make_target(element1, event, compare_type, element2=None, value=None,
                 atr_period=None, atr_multiplier=None):
    group = "Price & Indicators"
    if element1 in ("R Profit", "R Loss"):
        group = "R Profit / R Loss"
    elif element1 == "ATR Target":
        group = "ATR Target"
    trigger = {
        "group": group,
        "element1": element1,
        "event": event,
        "compare_type": compare_type,
        "element2": element2 if compare_type == "Indicator" else None,
        "value": value if compare_type == "Fixed Value" else None,
    }
    if element1 == "ATR Target":
        trigger["atr_period"] = atr_period or 14
        trigger["atr_multiplier"] = atr_multiplier or 2.0
        trigger["compare_type"] = "ATR"
    return {"type": "Target", "trigger": trigger, "conditions": []}


def _stats_match(stats_pd, stats_np, fields=None):
    if fields is None:
        fields = ["Number of trades", "Target exit (%)", "Static exit (%)",
                  "EOD exit (%)", "Win rate (%)"]
    for f in fields:
        v_pd = float(stats_pd.loc[f, "value"])
        v_np = float(stats_np.loc[f, "value"])
        if abs(v_pd - v_np) > 0.5:
            return False, f"'{f}': pandas={v_pd:.2f}, numpy={v_np:.2f}"
    return True, ""


# ---------------------------------------------------------------------------
# Close Below vs Cross Below: state fires, transition does not
# ---------------------------------------------------------------------------

class TestCloseVsCrossSemantics:
    """Close Below fires as state on persistent conditions;
    Cross Below requires a transition and does not."""

    @pytest.fixture(scope="class")
    def df_osc(self):
        df = _generate_oscillation(n=400, center=150.0, amplitude=25.0, period=40)
        return calculate_indicators(df, **DEFAULT_INDICATOR_SETTINGS)

    def test_bb_lower_always_below_bb_upper(self, df_osc):
        """BB Lower < BB Upper is always true.
        Close Below (state) fires → 100% target exits.
        Cross Below (transition) does not → 0% target exits."""
        for event, expected_target_pct in [("Close Below", 100.0), ("Cross Below", 0.0)]:
            target = _make_target("BB Lower Band", event, "Indicator",
                                  element2="BB Upper Band")
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
            _, stats_df = execute_custom_strategy(df_osc.copy(), strategy)
            n = int(stats_df.loc["Number of trades", "value"])
            if n == 0:
                continue
            actual = float(stats_df.loc["Target exit (%)", "value"])
            assert actual == expected_target_pct, (
                f"{event}: expected {expected_target_pct}% target exits, got {actual}%"
            )


# ---------------------------------------------------------------------------
# R Profit/Loss: Close Above is now state, Cross Above is transition
# ---------------------------------------------------------------------------

class TestRProfitCloseVsCross:
    """R Profit 'Close Above' (state) and 'Cross Above' (transition) should
    both fire on the first profitable bar for a fresh trade, but differ if
    R oscillates around the target level."""

    @pytest.fixture(scope="class")
    def df_ready(self):
        df = _generate_oscillation(n=300, center=150.0, amplitude=30.0, period=40)
        return calculate_indicators(df, **DEFAULT_INDICATOR_SETTINGS)

    def test_r_profit_close_above_is_state(self, df_ready):
        """R Profit (Close Above) 0.01 — tiny target, should fire as soon as
        R > 0.01 (state). This should produce target exits."""
        target = _make_target("R Profit", "Close Above", "Fixed Value", value=0.01)
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
        _, stats_pd, stats_np = _run_both(df_ready, strategy)
        n = int(stats_pd.loc["Number of trades", "value"])
        if n == 0:
            pytest.skip("No trades")
        tgt = float(stats_pd.loc["Target exit (%)", "value"])
        assert tgt > 0.0, "R Profit Close Above should fire as state"
        # Parity
        assert int(stats_np.loc["Number of trades", "value"]) == n

    def test_r_profit_cross_above_is_transition(self, df_ready):
        """R Profit (Cross Above) 0.01 — same tiny target. As a transition,
        it also fires on the first bar R exceeds 0.01 (prev was <= 0.01).
        So for a fresh trade, both state and transition produce the same result."""
        target = _make_target("R Profit", "Cross Above", "Fixed Value", value=0.01)
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
        _, stats_df = execute_custom_strategy(df_ready.copy(), strategy)
        n = int(stats_df.loc["Number of trades", "value"])
        if n == 0:
            pytest.skip("No trades")
        tgt = float(stats_df.loc["Target exit (%)", "value"])
        assert tgt > 0.0, "R Profit Cross Above should also fire for fresh trades"


# ---------------------------------------------------------------------------
# Full parity: Close Above/Below across data shapes
# ---------------------------------------------------------------------------

class TestFullParity:
    """Exhaustive parity between pandas and numpy for Close events."""

    @pytest.fixture(
        params=["Close Above", "Close Below"],
        ids=lambda e: e.replace(" ", "_"),
    )
    def event(self, request):
        return request.param

    @pytest.fixture(
        params=["uptrend", "downtrend", "oscillation"],
        ids=lambda s: s,
    )
    def df_shape(self, request):
        generators = {
            "uptrend": _generate_uptrend,
            "downtrend": _generate_downtrend,
            "oscillation": _generate_oscillation,
        }
        df = generators[request.param](n=200)
        return calculate_indicators(df, **DEFAULT_INDICATOR_SETTINGS)

    def test_parity_fixed_value(self, df_shape, event):
        value = 140.0 if "Below" in event else 160.0
        direction = "Short" if "Below" in event else "Long"
        entry_event = "Cross Below" if direction == "Short" else "Cross Above"
        entry_value = 70.0 if direction == "Short" else 30.0

        target = _make_target("Price", event, "Fixed Value", value=value)
        exit_group = make_exit_group(allocation_pct=100.0, targets=[target])
        strategy = make_strategy(
            direction=direction,
            entry_group="RSI Group",
            entry_element1="RSI",
            entry_event=entry_event,
            entry_compare_type="Fixed Value",
            entry_value=entry_value,
            exit_groups=[exit_group],
        )

        _, stats_pd, stats_np = _run_both(df_shape, strategy)
        n_pd = int(stats_pd.loc["Number of trades", "value"])
        n_np = int(stats_np.loc["Number of trades", "value"])
        assert n_pd == n_np, f"{event}: pandas={n_pd}, numpy={n_np}"
        if n_pd > 0:
            match, msg = _stats_match(stats_pd, stats_np)
            assert match, f"{event}: {msg}"

    def test_parity_indicator(self, df_shape, event):
        direction = "Short" if "Below" in event else "Long"
        entry_event = "Cross Below" if direction == "Short" else "Cross Above"
        entry_value = 70.0 if direction == "Short" else 30.0

        target = _make_target("BB Lower Band", event, "Indicator",
                              element2="BB Upper Band")
        exit_group = make_exit_group(allocation_pct=100.0, targets=[target])
        strategy = make_strategy(
            direction=direction,
            entry_group="RSI Group",
            entry_element1="RSI",
            entry_event=entry_event,
            entry_compare_type="Fixed Value",
            entry_value=entry_value,
            exit_groups=[exit_group],
        )

        _, stats_pd, stats_np = _run_both(df_shape, strategy)
        n_pd = int(stats_pd.loc["Number of trades", "value"])
        n_np = int(stats_np.loc["Number of trades", "value"])
        assert n_pd == n_np, f"{event} indicator: pandas={n_pd}, numpy={n_np}"
        if n_pd > 0:
            match, msg = _stats_match(stats_pd, stats_np)
            assert match, f"{event} indicator: {msg}"


# ---------------------------------------------------------------------------
# Dynamic stop: Close Above is state too
# ---------------------------------------------------------------------------

class TestDynamicStopCloseAboveState:
    """Dynamic stop with Close Above is state — fires when condition holds."""

    @pytest.fixture(scope="class")
    def df_ready(self):
        df = _generate_oscillation(n=300, center=150.0, amplitude=25.0, period=40)
        return calculate_indicators(df, **DEFAULT_INDICATOR_SETTINGS)

    def test_dynamic_stop_close_above_state(self, df_ready):
        """Dynamic stop: Price (Close Above) 1.0 — always true for any trade.
        Close Above (state) should fire → high dynamic exit %.
        Cross Above (transition) should NOT fire → 0% dynamic exit %."""
        for event, expect_fire in [("Close Above", True), ("Cross Above", False)]:
            stop = {
                "type": "Stop",
                "trigger": {
                    "group": "Price & Indicators",
                    "element1": "Price",
                    "event": event,
                    "compare_type": "Fixed Value",
                    "element2": None,
                    "value": 1.0,
                },
                "conditions": [],
            }
            target = _make_target("R Profit", "Cross Above", "Fixed Value", value=50.0)
            exit_group = make_exit_group(
                allocation_pct=100.0, targets=[target], stops=[stop])
            strategy = make_strategy(
                direction="Short",
                entry_group="RSI Group",
                entry_element1="RSI",
                entry_event="Cross Below",
                entry_compare_type="Fixed Value",
                entry_value=70.0,
                exit_groups=[exit_group],
            )
            _, stats_df = execute_custom_strategy(df_ready.copy(), strategy)
            n = int(stats_df.loc["Number of trades", "value"])
            if n == 0:
                continue
            dynamic_pct = float(stats_df.loc["Dynamic exit (%)", "value"])
            if expect_fire:
                assert dynamic_pct > 0.0, (
                    f"Close Above (state) dynamic stop should fire, got 0%"
                )
            else:
                assert dynamic_pct == 0.0, (
                    f"Cross Above (transition) dynamic stop should NOT fire "
                    f"on persistent condition, got {dynamic_pct:.1f}%"
                )

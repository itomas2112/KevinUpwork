"""
Tests verifying "Close Below" / "Close Above" state semantics and
"Cross Below" / "Cross Above" transition semantics.

Definitions:
  - "Close Above/Below" = STATE: the bar closed above/below the level.
    Fires every bar the condition holds.
  - "Cross Above/Below" = TRANSITION: previous bar was on the other side,
    current bar crossed. One-time event per crossing.

Also tests that R Profit/Loss "Close Above/Below" uses state semantics
(was previously incorrectly using transition logic).
"""
import pytest
import numpy as np
import pandas as pd
from copy import deepcopy

from indicators.calculate_indicators import calculate_indicators
from strategies.first_strategy import execute_custom_strategy
from strategies.first_strategy_numpy import execute_custom_strategy_numpy
from tests.conftest import (
    _make_ohlc, _generate_downtrend, _generate_oscillation,
    DEFAULT_INDICATOR_SETTINGS, make_strategy, make_exit_group,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _run_both(df, strategy):
    result_df, stats_pd = execute_custom_strategy(df.copy(), strategy)
    _, stats_np = execute_custom_strategy_numpy(df.copy(), strategy)
    return result_df, stats_pd, stats_np


def _make_target(element1, event, compare_type, element2=None, value=None):
    return {
        "type": "Target",
        "trigger": {
            "group": "Price & Indicators" if element1 != "R Profit" else "R Profit / R Loss",
            "element1": element1,
            "event": event,
            "compare_type": compare_type,
            "element2": element2 if compare_type == "Indicator" else None,
            "value": value if compare_type == "Fixed Value" else None,
        },
        "conditions": [],
    }


# ---------------------------------------------------------------------------
# TEST 1: Close Below IS a state check — fires on persistently true condition
# ---------------------------------------------------------------------------

class TestCloseBelow_IsState:
    """Close Below = state. Should fire when condition holds, even persistently."""

    @pytest.fixture(scope="class")
    def df_ready(self):
        df = _generate_oscillation(n=300, center=150.0, amplitude=30.0, period=40)
        return calculate_indicators(df, **DEFAULT_INDICATOR_SETTINGS)

    def _make_strategy(self, event):
        target = _make_target("Price", event, "Fixed Value", value=500.0)
        exit_group = make_exit_group(allocation_pct=100.0, targets=[target])
        return make_strategy(
            direction="Short",
            entry_group="RSI Group",
            entry_element1="RSI",
            entry_event="Cross Below",
            entry_compare_type="Fixed Value",
            entry_value=70.0,
            exit_groups=[exit_group],
        )

    def test_cross_below_does_not_fire_on_persistent(self, df_ready):
        """Cross Below is a transition — should NOT fire when price < 500 perpetually."""
        strategy = self._make_strategy("Cross Below")
        _, stats_df = execute_custom_strategy(df_ready.copy(), strategy)
        num = int(stats_df.loc["Number of trades", "value"])
        if num == 0:
            pytest.skip("No trades")
        assert float(stats_df.loc["Target exit (%)", "value"]) == 0.0

    def test_close_below_fires_on_persistent(self, df_ready):
        """Close Below is a state check — SHOULD fire when price < 500 always.
        Every trade exits via Target on bar+1."""
        strategy = self._make_strategy("Close Below")
        _, stats_df = execute_custom_strategy(df_ready.copy(), strategy)
        num = int(stats_df.loc["Number of trades", "value"])
        if num == 0:
            pytest.skip("No trades")
        target_pct = float(stats_df.loc["Target exit (%)", "value"])
        assert target_pct == 100.0, (
            f"Close Below (state) should fire on every bar where price < 500, "
            f"got {target_pct:.1f}% target exits"
        )

    def test_pandas_numpy_parity(self, df_ready):
        """Both engines agree on Close Below state behavior."""
        strategy = self._make_strategy("Close Below")
        _, stats_pd, stats_np = _run_both(df_ready, strategy)
        n_pd = int(stats_pd.loc["Number of trades", "value"])
        n_np = int(stats_np.loc["Number of trades", "value"])
        assert n_pd == n_np, f"Parity: pandas={n_pd}, numpy={n_np}"


# ---------------------------------------------------------------------------
# TEST 2: Close Above IS a state check (symmetric)
# ---------------------------------------------------------------------------

class TestCloseAbove_IsState:
    """Close Above = state. Symmetric test for Close Above."""

    @pytest.fixture(scope="class")
    def df_ready(self):
        df = _generate_oscillation(n=300, center=150.0, amplitude=30.0, period=40)
        return calculate_indicators(df, **DEFAULT_INDICATOR_SETTINGS)

    def test_cross_above_does_not_fire_on_persistent(self, df_ready):
        """Cross Above is transition — should NOT fire when price > 1 perpetually."""
        target = _make_target("Price", "Cross Above", "Fixed Value", value=1.0)
        exit_group = make_exit_group(allocation_pct=100.0, targets=[target])
        strategy = make_strategy(
            direction="Long",
            entry_group="RSI Group",
            entry_element1="RSI",
            entry_event="Cross Above",
            entry_compare_type="Fixed Value",
            entry_value=30.0,
            exit_groups=[exit_group],
        )
        _, stats_df = execute_custom_strategy(df_ready.copy(), strategy)
        num = int(stats_df.loc["Number of trades", "value"])
        if num == 0:
            pytest.skip("No trades")
        assert float(stats_df.loc["Target exit (%)", "value"]) == 0.0

    def test_close_above_fires_on_persistent(self, df_ready):
        """Close Above is state — SHOULD fire when price > 1 always."""
        target = _make_target("Price", "Close Above", "Fixed Value", value=1.0)
        exit_group = make_exit_group(allocation_pct=100.0, targets=[target])
        strategy = make_strategy(
            direction="Long",
            entry_group="RSI Group",
            entry_element1="RSI",
            entry_event="Cross Above",
            entry_compare_type="Fixed Value",
            entry_value=30.0,
            exit_groups=[exit_group],
        )
        _, stats_df = execute_custom_strategy(df_ready.copy(), strategy)
        num = int(stats_df.loc["Number of trades", "value"])
        if num == 0:
            pytest.skip("No trades")
        target_pct = float(stats_df.loc["Target exit (%)", "value"])
        assert target_pct == 100.0, (
            f"Close Above (state) should fire every bar where price > 1, "
            f"got {target_pct:.1f}%"
        )


# ---------------------------------------------------------------------------
# TEST 3: R Profit/Loss "Close Above" must be state (was transition — bug)
# ---------------------------------------------------------------------------

class TestRProfitCloseIsState:
    """R Profit 'Close Above' should fire as state: whenever R value exceeds
    the target. Previously it required a transition (prev <= target AND
    current > target), which is the Cross semantic."""

    @pytest.fixture(scope="class")
    def df_ready(self):
        df = _generate_oscillation(n=300, center=150.0, amplitude=30.0, period=40)
        return calculate_indicators(df, **DEFAULT_INDICATOR_SETTINGS)

    def test_r_profit_close_above_fires_as_state(self, df_ready):
        """R Profit (Close Above) 0.01 — with a tiny R target, the condition
        should be true as soon as the trade has any profit. As a state check,
        it fires on the first bar after entry where R > 0.01."""
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
        n_pd = int(stats_pd.loc["Number of trades", "value"])
        if n_pd == 0:
            pytest.skip("No trades")

        # With state semantics and a 0.01R target, most trades should exit
        # via target almost immediately when they have any profit at all
        target_pct = float(stats_pd.loc["Target exit (%)", "value"])
        assert target_pct > 0.0, (
            f"R Profit (Close Above) 0.01 should fire as state whenever R > 0.01, "
            f"but got 0% target exits. Likely still using transition logic."
        )

        # Parity
        n_np = int(stats_np.loc["Number of trades", "value"])
        assert n_pd == n_np, f"Parity: pandas={n_pd}, numpy={n_np}"


# ---------------------------------------------------------------------------
# TEST 4: "Close" (bidirectional) remains a transition
# ---------------------------------------------------------------------------

class TestCloseBidirectionalIsTransition:
    """The 'Close' event (bidirectional) uses transition logic."""

    @pytest.fixture(scope="class")
    def df_ready(self):
        df = _generate_oscillation(n=300, center=150.0, amplitude=30.0, period=40)
        return calculate_indicators(df, **DEFAULT_INDICATOR_SETTINGS)

    def test_close_bidir_does_not_fire_on_persistent(self, df_ready):
        """'Close' (bidir) with Price vs 500: price never crosses 500,
        so the transition never happens."""
        target = _make_target("Price", "Close", "Fixed Value", value=500.0)
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
        num = int(stats_df.loc["Number of trades", "value"])
        if num == 0:
            pytest.skip("No trades")
        target_pct = float(stats_df.loc["Target exit (%)", "value"])
        assert target_pct == 0.0, (
            f"'Close' (bidir) should be transition-based, got {target_pct:.1f}%"
        )

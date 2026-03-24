"""
Test for ATR Target exit with null event — reproduces client-reported issue.

When a strategy has an ATR Target exit with event: null (which happens when
the user doesn't explicitly set an event in the UI), the target should still
fire using a sensible default (Cross Above for Long, Cross Below for Short).

Currently, event=null causes the ATR Target to NEVER trigger, meaning trades
only exit via the static stop or end-of-data. This produces erratic results:
100% win rate or 100% loss rate depending on price direction at end-of-data,
making grid search results meaningless.

Both pandas and numpy engines are affected.
"""
import pytest
import pandas as pd
import numpy as np

from indicators.calculate_indicators import calculate_indicators
from strategies.first_strategy import execute_custom_strategy
from strategies.first_strategy_numpy import execute_custom_strategy_numpy
from tests.conftest import (
    DEFAULT_INDICATOR_SETTINGS,
    _generate_oscillation,
    _generate_uptrend,
)


def _make_strategy_with_atr_target(event_value, direction="Long"):
    """Build the client's exact strategy structure with a configurable ATR Target event."""
    return {
        "strategy_name": "ATR Target Test",
        "direction": direction,
        "patterns": [],
        "max_positions": 1,
        "created_at": "2024-01-01 00:00:00",
        "entry": {
            "trigger": {
                "group": "Price & Indicators",
                "element1": "Price",
                "event": "Cross Above",
                "compare_type": "Indicator",
                "element2": "BB Upper Band",
                "value": None,
            },
            "position_size": 1.0,
            "conditions_count": 0,
            "conditions": [],
        },
        "initial_stop": {
            "element1": "Price",
            "stop_type": "ATR",
            "event": "Cross Above",
            "atr_period": 14,
            "atr_multiplier": 2.0,
        },
        "exit_groups": [{
            "group_id": 1,
            "allocation_pct": 100.0,
            "targets": [{
                "type": "Target",
                "trigger": {
                    "group": "ATR Target",
                    "element1": "ATR Target",
                    "event": event_value,       # This is the key — null vs "Cross Above"
                    "compare_type": "ATR",
                    "element2": None,
                    "value": None,
                    "atr_period": 14,
                    "atr_multiplier": 3.0,
                },
                "conditions": [],
            }],
            "stops": [],
        }],
        "indicator_settings": dict(DEFAULT_INDICATOR_SETTINGS),
    }


@pytest.fixture(scope="module")
def df_oscillation():
    df = _generate_oscillation(n=500, freq="15min")
    return calculate_indicators(df, **DEFAULT_INDICATOR_SETTINGS)


@pytest.fixture(scope="module")
def df_uptrend():
    df = _generate_uptrend(n=500, freq="15min")
    return calculate_indicators(df, **DEFAULT_INDICATOR_SETTINGS)


class TestATRTargetNullEvent:
    """Tests that ATR Target with event=null behaves identically to an explicit event."""

    def test_null_event_blocked_by_validator_pandas(self, df_oscillation):
        """Long strategy with ATR Target event=null is blocked by validator — returns 0 trades."""
        strat_null = _make_strategy_with_atr_target(event_value=None, direction="Long")

        import warnings
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            _, stats_null = execute_custom_strategy(df_oscillation.copy(), strat_null)

        pnls_null = stats_null.attrs.get("trade_pnls_r", [])
        assert len(pnls_null) == 0, "Null event strategy should be blocked by validator"

    def test_null_event_blocked_by_validator_short(self, df_oscillation):
        """Short strategy with ATR Target event=null is blocked by validator."""
        strat_null = _make_strategy_with_atr_target(event_value=None, direction="Short")
        strat_null["entry"]["trigger"]["event"] = "Cross Below"
        strat_null["entry"]["trigger"]["element2"] = "BB Lower Band"

        import warnings
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            _, stats_null = execute_custom_strategy(df_oscillation.copy(), strat_null)

        pnls_null = stats_null.attrs.get("trade_pnls_r", [])
        assert len(pnls_null) == 0, "Null event strategy should be blocked by validator"

    def test_null_event_blocked_by_validator_numpy(self, df_oscillation):
        """NumPy engine also blocks null event strategies."""
        strat_null = _make_strategy_with_atr_target(event_value=None, direction="Long")

        import warnings
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            _, stats_null = execute_custom_strategy_numpy(df_oscillation.copy(), strat_null)

        pnls_null = stats_null.attrs.get("trade_pnls_r", [])
        assert len(pnls_null) == 0, "Null event strategy should be blocked by validator"

    def test_null_event_caught_by_validator(self):
        """Validator explicitly catches null event on ATR Target."""
        from strategies.strategy_validator import validate_strategy
        strat = _make_strategy_with_atr_target(event_value=None, direction="Long")
        is_valid, errors = validate_strategy(strat)
        assert not is_valid
        assert any("event" in e for e in errors)

    def test_explicit_event_target_fires(self, df_uptrend):
        """Sanity check: ATR Target with explicit Cross Above event works correctly."""
        strat = _make_strategy_with_atr_target(event_value="Cross Above", direction="Long")

        _, stats = execute_custom_strategy(df_uptrend.copy(), strat)
        pnls = stats.attrs.get("trade_pnls_r", [])

        if len(pnls) > 0:
            target_hits = [p for p in pnls if p > 0]
            assert len(target_hits) > 0, \
                f"Even explicit Cross Above didn't produce target hits. P&Ls: {pnls}"

    def test_null_event_blocked_on_uptrend(self, df_uptrend):
        """ATR Target with null event is blocked even on uptrend data."""
        strat = _make_strategy_with_atr_target(event_value=None, direction="Long")

        import warnings
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            _, stats = execute_custom_strategy(df_uptrend.copy(), strat)

        pnls = stats.attrs.get("trade_pnls_r", [])
        assert len(pnls) == 0, "Null event strategy should produce 0 trades"


class TestATRTargetNullEventParity:
    """Both engines should handle null event identically — both block it."""

    def test_pandas_numpy_parity_null_event(self, df_oscillation):
        strat = _make_strategy_with_atr_target(event_value=None, direction="Long")

        import warnings
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            _, stats_pd = execute_custom_strategy(df_oscillation.copy(), strat)
            _, stats_np = execute_custom_strategy_numpy(df_oscillation.copy(), strat)

        pnls_pd = stats_pd.attrs.get("trade_pnls_r", [])
        pnls_np = stats_np.attrs.get("trade_pnls_r", [])

        # Both should return 0 trades (blocked by validator)
        assert len(pnls_pd) == 0, "Pandas engine should block null event"
        assert len(pnls_np) == 0, "NumPy engine should block null event"
        assert len(pnls_pd) == len(pnls_np)

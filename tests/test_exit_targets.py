"""
Parametrized tests for all exit target combinations.
102 combos: 7 groups × 6 events × 2 compare types + R Profit/Loss + ATR Target.
"""
import pytest
import pandas as pd

from indicators.calculate_indicators import calculate_indicators
from strategies.first_strategy import execute_custom_strategy
from tests.conftest import (
    EXIT_TARGET_COMBOS,
    DEFAULT_INDICATOR_SETTINGS,
    make_strategy,
    make_exit_group,
    make_exit_trigger,
    _generate_oscillation,
    _generate_v_shape,
)


@pytest.fixture(scope="module")
def df_oscillation_ready():
    df = _generate_oscillation(n=300, freq="15min")
    return calculate_indicators(df, **DEFAULT_INDICATOR_SETTINGS)


@pytest.fixture(scope="module")
def df_v_shape_ready():
    df = _generate_v_shape(n=300, freq="15min")
    return calculate_indicators(df, **DEFAULT_INDICATOR_SETTINGS)


def _build_exit_target(group, el1, event, compare_type, el2, fixed_val):
    """Build an exit target dict from parametrized values."""
    trigger = {
        "group": group,
        "element1": el1,
        "event": event,
        "compare_type": compare_type,
        "element2": el2 if compare_type == "Indicator" else None,
        "value": fixed_val if compare_type == "Fixed Value" else None,
    }
    if el1 == "ATR Target":
        trigger["atr_period"] = 14
        trigger["atr_multiplier"] = 2.0
        trigger["compare_type"] = "ATR"
    return {"type": "Target", "trigger": trigger, "conditions": []}


@pytest.mark.parametrize("group,el1,event,compare_type,el2,fixed_val", EXIT_TARGET_COMBOS)
def test_exit_target_long(df_oscillation_ready, group, el1, event, compare_type, el2, fixed_val):
    """Exit target does not crash for Long strategies."""
    target = _build_exit_target(group, el1, event, compare_type, el2, fixed_val)
    exit_group = make_exit_group(
        allocation_pct=100.0,
        targets=[target],
    )

    strategy = make_strategy(
        direction="Long",
        entry_group="RSI Group",
        entry_element1="RSI",
        entry_event="Cross Above",
        entry_compare_type="Fixed Value",
        entry_value=30.0,
        exit_groups=[exit_group],
    )

    _, stats_df = execute_custom_strategy(df_oscillation_ready.copy(), strategy)
    assert isinstance(stats_df, pd.DataFrame)


@pytest.mark.parametrize("group,el1,event,compare_type,el2,fixed_val", EXIT_TARGET_COMBOS)
def test_exit_target_short(df_oscillation_ready, group, el1, event, compare_type, el2, fixed_val):
    """Exit target does not crash for Short strategies."""
    target = _build_exit_target(group, el1, event, compare_type, el2, fixed_val)
    exit_group = make_exit_group(
        allocation_pct=100.0,
        targets=[target],
    )

    strategy = make_strategy(
        direction="Short",
        entry_group="RSI Group",
        entry_element1="RSI",
        entry_event="Cross Below",
        entry_compare_type="Fixed Value",
        entry_value=70.0,
        exit_groups=[exit_group],
    )

    _, stats_df = execute_custom_strategy(df_oscillation_ready.copy(), strategy)
    assert isinstance(stats_df, pd.DataFrame)


@pytest.mark.parametrize("group,el1,event,compare_type,el2,fixed_val", EXIT_TARGET_COMBOS)
def test_exit_target_on_v_shape(df_v_shape_ready, group, el1, event, compare_type, el2, fixed_val):
    """Exit targets work on V-shape data."""
    target = _build_exit_target(group, el1, event, compare_type, el2, fixed_val)
    exit_group = make_exit_group(
        allocation_pct=100.0,
        targets=[target],
    )

    strategy = make_strategy(
        direction="Long",
        entry_group="RSI Group",
        entry_element1="RSI",
        entry_event="Cross Above",
        entry_compare_type="Fixed Value",
        entry_value=30.0,
        exit_groups=[exit_group],
    )

    _, stats_df = execute_custom_strategy(df_v_shape_ready.copy(), strategy)
    assert isinstance(stats_df, pd.DataFrame)

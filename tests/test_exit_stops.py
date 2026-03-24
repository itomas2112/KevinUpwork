"""
Parametrized tests for all exit dynamic stop combinations.
68 combos: 7 groups × 4 events × 2 compare types + R Profit/Loss + ATR Target.
"""
import pytest
import pandas as pd

from indicators.calculate_indicators import calculate_indicators
from strategies.first_strategy import execute_custom_strategy
from tests.conftest import (
    EXIT_STOP_COMBOS,
    DEFAULT_INDICATOR_SETTINGS,
    make_strategy,
    make_exit_group,
    make_exit_trigger,
    make_exit_stop,
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


def _build_exit_stop(group, el1, event, compare_type, el2, fixed_val):
    """Build an exit dynamic stop dict from parametrized values."""
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
    return {"type": "Stop", "trigger": trigger, "conditions": []}


@pytest.mark.parametrize("group,el1,event,compare_type,el2,fixed_val", EXIT_STOP_COMBOS)
def test_exit_stop_long(df_oscillation_ready, group, el1, event, compare_type, el2, fixed_val):
    """Dynamic stop does not crash for Long strategies."""
    stop = _build_exit_stop(group, el1, event, compare_type, el2, fixed_val)
    # Include a far-away R Profit target so the stop is what matters
    target = {
        "type": "Target",
        "trigger": {
            "group": "R Profit / R Loss",
            "element1": "R Profit",
            "event": "Cross Above",
            "compare_type": "Fixed Value",
            "element2": None,
            "value": 100.0,  # Very far target, unlikely to hit
        },
        "conditions": [],
    }
    exit_group = make_exit_group(
        allocation_pct=100.0,
        targets=[target],
        stops=[stop],
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


@pytest.mark.parametrize("group,el1,event,compare_type,el2,fixed_val", EXIT_STOP_COMBOS)
def test_exit_stop_short(df_oscillation_ready, group, el1, event, compare_type, el2, fixed_val):
    """Dynamic stop does not crash for Short strategies."""
    stop = _build_exit_stop(group, el1, event, compare_type, el2, fixed_val)
    target = {
        "type": "Target",
        "trigger": {
            "group": "R Profit / R Loss",
            "element1": "R Profit",
            "event": "Cross Above",
            "compare_type": "Fixed Value",
            "element2": None,
            "value": 100.0,
        },
        "conditions": [],
    }
    exit_group = make_exit_group(
        allocation_pct=100.0,
        targets=[target],
        stops=[stop],
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


@pytest.mark.parametrize("group,el1,event,compare_type,el2,fixed_val", EXIT_STOP_COMBOS)
def test_exit_stop_on_v_shape(df_v_shape_ready, group, el1, event, compare_type, el2, fixed_val):
    """Dynamic stops work on V-shape data."""
    stop = _build_exit_stop(group, el1, event, compare_type, el2, fixed_val)
    target = {
        "type": "Target",
        "trigger": {
            "group": "R Profit / R Loss",
            "element1": "R Profit",
            "event": "Cross Above",
            "compare_type": "Fixed Value",
            "element2": None,
            "value": 100.0,
        },
        "conditions": [],
    }
    exit_group = make_exit_group(
        allocation_pct=100.0,
        targets=[target],
        stops=[stop],
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

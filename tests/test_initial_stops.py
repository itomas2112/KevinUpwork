"""
Parametrized tests for all initial stop combinations.
8 combos: 4 events × 2 stop types (Indicator, ATR).
"""
import pytest
import pandas as pd

from indicators.calculate_indicators import calculate_indicators
from strategies.first_strategy import execute_custom_strategy
from tests.conftest import (
    INITIAL_STOP_COMBOS,
    DEFAULT_INDICATOR_SETTINGS,
    make_strategy,
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


def _validate_result(stats_df):
    assert isinstance(stats_df, pd.DataFrame)


@pytest.mark.parametrize("stop_type,event,element2,atr_period,atr_mult", INITIAL_STOP_COMBOS)
def test_initial_stop_long(df_oscillation_ready, stop_type, event, element2, atr_period, atr_mult):
    """Initial stop does not crash for Long strategies."""
    strategy = make_strategy(
        direction="Long",
        # Use a simple entry that will likely trigger
        entry_group="RSI Group",
        entry_element1="RSI",
        entry_event="Cross Above",
        entry_compare_type="Fixed Value",
        entry_value=30.0,
        # The stop under test
        initial_stop_type=stop_type,
        initial_stop_event=event,
        initial_stop_element2=element2 or "BB Lower Band",
        initial_stop_atr_period=atr_period or 14,
        initial_stop_atr_multiplier=atr_mult or 1.5,
    )

    _, stats_df = execute_custom_strategy(df_oscillation_ready.copy(), strategy)
    _validate_result(stats_df)


@pytest.mark.parametrize("stop_type,event,element2,atr_period,atr_mult", INITIAL_STOP_COMBOS)
def test_initial_stop_short(df_oscillation_ready, stop_type, event, element2, atr_period, atr_mult):
    """Initial stop does not crash for Short strategies."""
    strategy = make_strategy(
        direction="Short",
        entry_group="RSI Group",
        entry_element1="RSI",
        entry_event="Cross Below",
        entry_compare_type="Fixed Value",
        entry_value=70.0,
        initial_stop_type=stop_type,
        initial_stop_event=event,
        initial_stop_element2=element2 or "BB Upper Band",
        initial_stop_atr_period=atr_period or 14,
        initial_stop_atr_multiplier=atr_mult or 1.5,
    )

    _, stats_df = execute_custom_strategy(df_oscillation_ready.copy(), strategy)
    _validate_result(stats_df)


@pytest.mark.parametrize("stop_type,event,element2,atr_period,atr_mult", INITIAL_STOP_COMBOS)
def test_initial_stop_on_v_shape(df_v_shape_ready, stop_type, event, element2, atr_period, atr_mult):
    """Initial stops work on V-shape data."""
    strategy = make_strategy(
        direction="Long",
        entry_group="RSI Group",
        entry_element1="RSI",
        entry_event="Cross Above",
        entry_compare_type="Fixed Value",
        entry_value=30.0,
        initial_stop_type=stop_type,
        initial_stop_event=event,
        initial_stop_element2=element2 or "BB Lower Band",
        initial_stop_atr_period=atr_period or 14,
        initial_stop_atr_multiplier=atr_mult or 1.5,
    )

    _, stats_df = execute_custom_strategy(df_v_shape_ready.copy(), strategy)
    _validate_result(stats_df)

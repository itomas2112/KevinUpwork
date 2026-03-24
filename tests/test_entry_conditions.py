"""
Parametrized tests for all entry condition combinations.
28 combos: 7 groups × 2 operators × 2 compare types.
"""
import pytest
import pandas as pd

from indicators.calculate_indicators import calculate_indicators
from strategies.first_strategy import execute_custom_strategy
from tests.conftest import (
    ENTRY_CONDITION_COMBOS,
    DEFAULT_INDICATOR_SETTINGS,
    make_strategy,
    _generate_oscillation,
)


@pytest.fixture(scope="module")
def df_oscillation_ready():
    df = _generate_oscillation(n=300, freq="15min")
    return calculate_indicators(df, **DEFAULT_INDICATOR_SETTINGS)


@pytest.mark.parametrize("group,el1,operator,compare_type,el2,fixed_val", ENTRY_CONDITION_COMBOS)
def test_entry_condition_long(df_oscillation_ready, group, el1, operator, compare_type, el2, fixed_val):
    """Entry with a condition does not crash and produces valid output."""
    condition = {
        "group": group,
        "element1": el1,
        "operator": operator,
        "compare_type": compare_type,
        "element2": el2,
        "value": fixed_val if fixed_val is not None else 50.0,
    }

    strategy = make_strategy(
        direction="Long",
        # Simple entry trigger
        entry_group="RSI Group",
        entry_element1="RSI",
        entry_event="Cross Above",
        entry_compare_type="Fixed Value",
        entry_value=30.0,
        entry_conditions=[condition],
    )

    _, stats_df = execute_custom_strategy(df_oscillation_ready.copy(), strategy)
    assert isinstance(stats_df, pd.DataFrame)


@pytest.mark.parametrize("group,el1,operator,compare_type,el2,fixed_val", ENTRY_CONDITION_COMBOS)
def test_entry_condition_short(df_oscillation_ready, group, el1, operator, compare_type, el2, fixed_val):
    """Entry conditions work for Short strategies."""
    condition = {
        "group": group,
        "element1": el1,
        "operator": operator,
        "compare_type": compare_type,
        "element2": el2,
        "value": fixed_val if fixed_val is not None else 50.0,
    }

    strategy = make_strategy(
        direction="Short",
        entry_group="RSI Group",
        entry_element1="RSI",
        entry_event="Cross Below",
        entry_compare_type="Fixed Value",
        entry_value=70.0,
        entry_conditions=[condition],
    )

    _, stats_df = execute_custom_strategy(df_oscillation_ready.copy(), strategy)
    assert isinstance(stats_df, pd.DataFrame)


def test_impossible_condition_blocks_entry(df_oscillation_ready):
    """A condition that can never be true should produce zero trades."""
    # RSI must be above 999 — impossible
    condition = {
        "group": "RSI Group",
        "element1": "RSI",
        "operator": "Above",
        "compare_type": "Fixed Value",
        "value": 999.0,
    }

    strategy = make_strategy(
        direction="Long",
        entry_group="RSI Group",
        entry_element1="RSI",
        entry_event="Cross Above",
        entry_compare_type="Fixed Value",
        entry_value=30.0,
        entry_conditions=[condition],
    )

    _, stats_df = execute_custom_strategy(df_oscillation_ready.copy(), strategy)
    # Should have no trades (or empty stats)
    if len(stats_df) > 0:
        trade_pnls = stats_df.attrs.get("trade_pnls_r", [])
        assert len(trade_pnls) == 0


def test_multiple_conditions(df_oscillation_ready):
    """Multiple conditions should all be checked."""
    conditions = [
        {
            "group": "RSI Group",
            "element1": "RSI",
            "operator": "Above",
            "compare_type": "Fixed Value",
            "value": 20.0,
        },
        {
            "group": "ADX Group",
            "element1": "ADX",
            "operator": "Above",
            "compare_type": "Fixed Value",
            "value": 10.0,
        },
    ]

    strategy = make_strategy(
        direction="Long",
        entry_group="RSI Group",
        entry_element1="RSI",
        entry_event="Cross Above",
        entry_compare_type="Fixed Value",
        entry_value=30.0,
        entry_conditions=conditions,
    )

    _, stats_df = execute_custom_strategy(df_oscillation_ready.copy(), strategy)
    assert isinstance(stats_df, pd.DataFrame)

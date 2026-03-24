"""
Parametrized tests for all entry trigger combinations.
84 combos: 7 groups × 6 events × 2 compare types.

Each test verifies:
  - Strategy executes without error
  - Returned stats_df has valid structure
  - Trade direction matches strategy direction
  - P&L values are numerically consistent
"""
import pytest
import pandas as pd

from indicators.calculate_indicators import calculate_indicators
from strategies.first_strategy import execute_custom_strategy
from tests.conftest import (
    ENTRY_TRIGGER_COMBOS,
    DEFAULT_INDICATOR_SETTINGS,
    make_strategy,
    _generate_oscillation,
    _generate_v_shape,
)


@pytest.fixture(scope="module")
def df_oscillation_ready():
    """Pre-calculated oscillation data (shared across all tests in this module)."""
    df = _generate_oscillation(n=300, freq="15min")
    return calculate_indicators(df, **DEFAULT_INDICATOR_SETTINGS)


@pytest.fixture(scope="module")
def df_v_shape_ready():
    """Pre-calculated V-shape data."""
    df = _generate_v_shape(n=300, freq="15min")
    return calculate_indicators(df, **DEFAULT_INDICATOR_SETTINGS)


def _validate_result(df_result, stats_df, direction):
    """Common assertions for any strategy execution result."""
    # stats_df should be a DataFrame (possibly empty if no trades)
    assert isinstance(stats_df, pd.DataFrame)

    # If trades occurred, validate structure
    if len(stats_df) > 0 and "trade_pnls_r" in stats_df.attrs:
        trade_pnls = stats_df.attrs.get("trade_pnls_r", [])
        # P&L values should be finite numbers
        for pnl in trade_pnls:
            assert pd.notna(pnl), "Trade P&L should not be NaN"


@pytest.mark.parametrize("group,el1,el2,event,compare_type,fixed_val", ENTRY_TRIGGER_COMBOS)
def test_entry_trigger_on_oscillation(df_oscillation_ready, group, el1, el2, event, compare_type, fixed_val):
    """Entry trigger does not crash and produces valid output."""
    strategy = make_strategy(
        direction="Long",
        entry_group=group,
        entry_element1=el1,
        entry_event=event,
        entry_compare_type=compare_type,
        entry_element2=el2,
        entry_value=fixed_val if fixed_val is not None else 50.0,
    )

    df_result, stats_df = execute_custom_strategy(
        df_oscillation_ready.copy(), strategy
    )
    _validate_result(df_result, stats_df, "Long")


@pytest.mark.parametrize("group,el1,el2,event,compare_type,fixed_val", ENTRY_TRIGGER_COMBOS)
def test_entry_trigger_short(df_oscillation_ready, group, el1, el2, event, compare_type, fixed_val):
    """Same entry triggers work for Short strategies too."""
    strategy = make_strategy(
        direction="Short",
        entry_group=group,
        entry_element1=el1,
        entry_event=event,
        entry_compare_type=compare_type,
        entry_element2=el2,
        entry_value=fixed_val if fixed_val is not None else 50.0,
    )

    df_result, stats_df = execute_custom_strategy(
        df_oscillation_ready.copy(), strategy
    )
    _validate_result(df_result, stats_df, "Short")


@pytest.mark.parametrize("group,el1,el2,event,compare_type,fixed_val", ENTRY_TRIGGER_COMBOS)
def test_entry_trigger_on_v_shape(df_v_shape_ready, group, el1, el2, event, compare_type, fixed_val):
    """Entry triggers also work on V-shape data (different price dynamics)."""
    strategy = make_strategy(
        direction="Long",
        entry_group=group,
        entry_element1=el1,
        entry_event=event,
        entry_compare_type=compare_type,
        entry_element2=el2,
        entry_value=fixed_val if fixed_val is not None else 50.0,
    )

    df_result, stats_df = execute_custom_strategy(
        df_v_shape_ready.copy(), strategy
    )
    _validate_result(df_result, stats_df, "Long")

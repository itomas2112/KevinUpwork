"""
Tests that the pandas and NumPy strategy execution engines produce identical results.

For every strategy, both execute_custom_strategy (pandas) and
execute_custom_strategy_numpy (NumPy vectorized) should return the same
trade P&L values and statistics.
"""
import pytest
import pandas as pd
import numpy as np

from indicators.calculate_indicators import calculate_indicators
from strategies.first_strategy import execute_custom_strategy
from strategies.first_strategy_numpy import execute_custom_strategy_numpy
from tests.conftest import (
    DEFAULT_INDICATOR_SETTINGS,
    GROUP_REPRESENTATIVES,
    ENTRY_EVENTS,
    GROUP_FIXED_VALUES,
    make_strategy,
    make_exit_group,
    _generate_oscillation,
    _generate_v_shape,
    _generate_uptrend,
)


@pytest.fixture(scope="module")
def df_oscillation():
    df = _generate_oscillation(n=300, freq="15min")
    return calculate_indicators(df, **DEFAULT_INDICATOR_SETTINGS)


@pytest.fixture(scope="module")
def df_v_shape():
    df = _generate_v_shape(n=300, freq="15min")
    return calculate_indicators(df, **DEFAULT_INDICATOR_SETTINGS)


@pytest.fixture(scope="module")
def df_uptrend():
    df = _generate_uptrend(n=300, freq="15min")
    return calculate_indicators(df, **DEFAULT_INDICATOR_SETTINGS)


def _compare_results(df_data, strategy):
    """Run both engines and compare outputs."""
    _, stats_pandas = execute_custom_strategy(df_data.copy(), strategy)
    _, stats_numpy = execute_custom_strategy_numpy(df_data.copy(), strategy)

    # Both should return DataFrames
    assert isinstance(stats_pandas, pd.DataFrame)
    assert isinstance(stats_numpy, pd.DataFrame)

    # Compare trade P&L lists from attrs
    pnls_pandas = stats_pandas.attrs.get("trade_pnls_r", [])
    pnls_numpy = stats_numpy.attrs.get("trade_pnls_r", [])

    assert len(pnls_pandas) == len(pnls_numpy), \
        f"Trade count mismatch: pandas={len(pnls_pandas)}, numpy={len(pnls_numpy)}"

    if len(pnls_pandas) > 0:
        np.testing.assert_allclose(
            pnls_pandas, pnls_numpy, rtol=1e-6,
            err_msg="Trade P&L values differ between pandas and numpy engines"
        )


# ---------------------------------------------------------------------------
# Parity tests per indicator group
# ---------------------------------------------------------------------------

# Build a small set of representative strategies for parity testing
PARITY_COMBOS = []
for group_name, el1, el2 in GROUP_REPRESENTATIVES:
    fixed_val = GROUP_FIXED_VALUES[group_name]
    # Indicator compare
    PARITY_COMBOS.append(pytest.param(
        group_name, el1, el2, "Cross Above", "Indicator", None,
        id=f"Parity-{el1}-CrossAbove-vs-{el2}"
    ))
    # Fixed Value compare
    PARITY_COMBOS.append(pytest.param(
        group_name, el1, None, "Cross Above", "Fixed Value", fixed_val,
        id=f"Parity-{el1}-CrossAbove-vs-Fixed({fixed_val})"
    ))


@pytest.mark.parametrize("group,el1,el2,event,compare_type,fixed_val", PARITY_COMBOS)

def test_parity_entry_triggers(df_oscillation, group, el1, el2, event, compare_type, fixed_val):
    """Pandas and NumPy produce same results for each entry trigger combo."""
    strategy = make_strategy(
        direction="Long",
        entry_group=group,
        entry_element1=el1,
        entry_event=event,
        entry_compare_type=compare_type,
        entry_element2=el2,
        entry_value=fixed_val if fixed_val is not None else 50.0,
    )
    _compare_results(df_oscillation, strategy)



class TestParityOnDifferentData:

    def test_parity_v_shape(self, df_v_shape):
        strategy = make_strategy(direction="Long")
        _compare_results(df_v_shape, strategy)

    def test_parity_uptrend(self, df_uptrend):
        strategy = make_strategy(direction="Long")
        _compare_results(df_uptrend, strategy)

    def test_parity_short(self, df_oscillation):
        strategy = make_strategy(
            direction="Short",
            entry_group="RSI Group",
            entry_element1="RSI",
            entry_event="Cross Below",
            entry_compare_type="Fixed Value",
            entry_value=70.0,
        )
        _compare_results(df_oscillation, strategy)



class TestParityMultiExitGroups:

    def test_parity_two_groups(self, df_oscillation):
        group1 = make_exit_group(
            allocation_pct=50.0, group_id=1,
            targets=[{
                "type": "Target",
                "trigger": {
                    "group": "R Profit / R Loss", "element1": "R Profit",
                    "event": "Cross Above", "compare_type": "Fixed Value",
                    "element2": None, "value": 1.5,
                },
                "conditions": [],
            }],
        )
        group2 = make_exit_group(
            allocation_pct=50.0, group_id=2,
            targets=[{
                "type": "Target",
                "trigger": {
                    "group": "R Profit / R Loss", "element1": "R Profit",
                    "event": "Cross Above", "compare_type": "Fixed Value",
                    "element2": None, "value": 3.0,
                },
                "conditions": [],
            }],
        )

        strategy = make_strategy(
            direction="Long",
            exit_groups=[group1, group2],
        )
        _compare_results(df_oscillation, strategy)

    def test_parity_with_conditions(self, df_oscillation):
        strategy = make_strategy(
            direction="Long",
            entry_conditions=[{
                "group": "ADX Group",
                "element1": "ADX",
                "operator": "Above",
                "compare_type": "Fixed Value",
                "value": 10.0,
            }],
        )
        _compare_results(df_oscillation, strategy)

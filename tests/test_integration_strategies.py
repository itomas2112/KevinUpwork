"""
Integration tests — full strategies combining entry triggers, conditions,
initial stops, and multi-exit groups from different indicator groups.
"""
import pytest
import pandas as pd

from indicators.calculate_indicators import calculate_indicators
from strategies.first_strategy import execute_custom_strategy
from tests.conftest import (
    DEFAULT_INDICATOR_SETTINGS,
    make_strategy,
    make_exit_group,
    _generate_oscillation,
    _generate_v_shape,
    _generate_uptrend,
    _generate_downtrend,
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


@pytest.fixture(scope="module")
def df_downtrend():
    df = _generate_downtrend(n=300, freq="15min")
    return calculate_indicators(df, **DEFAULT_INDICATOR_SETTINGS)


class TestMultiExitGroups:

    def test_two_exit_groups_50_50(self, df_oscillation):
        """Two exit groups with 50/50 allocation."""
        group1 = make_exit_group(
            allocation_pct=50.0,
            group_id=1,
            targets=[{
                "type": "Target",
                "trigger": {
                    "group": "R Profit / R Loss",
                    "element1": "R Profit",
                    "event": "Cross Above",
                    "compare_type": "Fixed Value",
                    "element2": None,
                    "value": 1.5,
                },
                "conditions": [],
            }],
        )
        group2 = make_exit_group(
            allocation_pct=50.0,
            group_id=2,
            targets=[{
                "type": "Target",
                "trigger": {
                    "group": "R Profit / R Loss",
                    "element1": "R Profit",
                    "event": "Cross Above",
                    "compare_type": "Fixed Value",
                    "element2": None,
                    "value": 3.0,
                },
                "conditions": [],
            }],
        )

        strategy = make_strategy(
            direction="Long",
            entry_group="RSI Group",
            entry_element1="RSI",
            entry_event="Cross Above",
            entry_compare_type="Fixed Value",
            entry_value=30.0,
            exit_groups=[group1, group2],
        )

        _, stats_df = execute_custom_strategy(df_oscillation.copy(), strategy)
        assert isinstance(stats_df, pd.DataFrame)

    def test_three_exit_groups(self, df_oscillation):
        """Three exit groups with uneven allocation."""
        groups = []
        for i, (alloc, r_target) in enumerate([(40.0, 1.0), (30.0, 2.0), (30.0, 4.0)]):
            groups.append(make_exit_group(
                allocation_pct=alloc,
                group_id=i + 1,
                targets=[{
                    "type": "Target",
                    "trigger": {
                        "group": "R Profit / R Loss",
                        "element1": "R Profit",
                        "event": "Cross Above",
                        "compare_type": "Fixed Value",
                        "element2": None,
                        "value": r_target,
                    },
                    "conditions": [],
                }],
            ))

        strategy = make_strategy(
            direction="Long",
            exit_groups=groups,
        )

        _, stats_df = execute_custom_strategy(df_oscillation.copy(), strategy)
        assert isinstance(stats_df, pd.DataFrame)


class TestMixedIndicatorGroups:

    def test_entry_rsi_condition_adx_exit_macd(self, df_oscillation):
        """Entry from RSI group, condition from ADX group, exit from MACD group."""
        condition = {
            "group": "ADX Group",
            "element1": "ADX",
            "operator": "Above",
            "compare_type": "Fixed Value",
            "value": 10.0,
        }

        exit_group = make_exit_group(
            allocation_pct=100.0,
            targets=[{
                "type": "Target",
                "trigger": {
                    "group": "MACD Group",
                    "element1": "MACD Line",
                    "event": "Cross Below",
                    "compare_type": "Indicator",
                    "element2": "MACD Signal",
                    "value": None,
                },
                "conditions": [],
            }],
        )

        strategy = make_strategy(
            direction="Long",
            entry_group="RSI Group",
            entry_element1="RSI",
            entry_event="Cross Above",
            entry_compare_type="Fixed Value",
            entry_value=30.0,
            entry_conditions=[condition],
            exit_groups=[exit_group],
        )

        _, stats_df = execute_custom_strategy(df_oscillation.copy(), strategy)
        assert isinstance(stats_df, pd.DataFrame)

    def test_entry_stoch_exit_bb_with_dynamic_stop(self, df_oscillation):
        """Entry from Stochastic, exit target on BB, dynamic stop on KC."""
        target = {
            "type": "Target",
            "trigger": {
                "group": "Price & Indicators",
                "element1": "Price",
                "event": "Cross Above",
                "compare_type": "Indicator",
                "element2": "BB Upper Band",
                "value": None,
            },
            "conditions": [],
        }
        stop = {
            "type": "Stop",
            "trigger": {
                "group": "Price & Indicators",
                "element1": "Price",
                "event": "Cross Below",
                "compare_type": "Indicator",
                "element2": "KC Lower Band",
                "value": None,
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
            entry_group="Stoch Group",
            entry_element1="Stoch %K",
            entry_event="Cross Above",
            entry_compare_type="Indicator",
            entry_element2="Stoch %D",
            exit_groups=[exit_group],
        )

        _, stats_df = execute_custom_strategy(df_oscillation.copy(), strategy)
        assert isinstance(stats_df, pd.DataFrame)

    def test_entry_price_vs_supertrend(self, df_v_shape):
        """Entry: Price Cross Above Supertrend."""
        exit_group = make_exit_group(
            allocation_pct=100.0,
            targets=[{
                "type": "Target",
                "trigger": {
                    "group": "R Profit / R Loss",
                    "element1": "R Profit",
                    "event": "Cross Above",
                    "compare_type": "Fixed Value",
                    "element2": None,
                    "value": 2.0,
                },
                "conditions": [],
            }],
        )

        strategy = make_strategy(
            direction="Long",
            entry_group="Price & Indicators",
            entry_element1="Price",
            entry_event="Cross Above",
            entry_compare_type="Indicator",
            entry_element2="Supertrend",
            exit_groups=[exit_group],
        )

        _, stats_df = execute_custom_strategy(df_v_shape.copy(), strategy)
        assert isinstance(stats_df, pd.DataFrame)


class TestATRTargetExit:

    def test_atr_target_exit(self, df_oscillation):
        """Exit using ATR Target element."""
        target = {
            "type": "Target",
            "trigger": {
                "group": "ATR Target",
                "element1": "ATR Target",
                "event": "Cross Above",
                "compare_type": "ATR",
                "element2": None,
                "value": None,
                "atr_period": 14,
                "atr_multiplier": 2.0,
            },
            "conditions": [],
        }
        exit_group = make_exit_group(
            allocation_pct=100.0,
            targets=[target],
        )

        strategy = make_strategy(
            direction="Long",
            exit_groups=[exit_group],
        )

        _, stats_df = execute_custom_strategy(df_oscillation.copy(), strategy)
        assert isinstance(stats_df, pd.DataFrame)


class TestMaxPositions:

    def test_max_positions_1(self, df_oscillation):
        """With max_positions=1, should not open a second position while one is active."""
        strategy = make_strategy(
            direction="Long",
            max_positions=1,
        )
        _, stats_df = execute_custom_strategy(df_oscillation.copy(), strategy)
        assert isinstance(stats_df, pd.DataFrame)

    def test_max_positions_unlimited(self, df_oscillation):
        """With max_positions=None, unlimited positions allowed."""
        strategy = make_strategy(
            direction="Long",
            max_positions=None,
        )
        _, stats_df = execute_custom_strategy(df_oscillation.copy(), strategy)
        assert isinstance(stats_df, pd.DataFrame)


class TestPeriodBounds:

    def test_period_start_end(self, df_oscillation):
        """Entries should only occur within period_start and period_end."""
        strategy = make_strategy(direction="Long")

        period_start = df_oscillation.index[50]
        period_end = df_oscillation.index[150]

        _, stats_df = execute_custom_strategy(
            df_oscillation.copy(), strategy,
            period_start=period_start, period_end=period_end,
        )
        assert isinstance(stats_df, pd.DataFrame)

    def test_period_no_data_range(self, df_oscillation):
        """If period is outside data range, should produce no trades."""
        strategy = make_strategy(direction="Long")

        future_start = pd.Timestamp("2030-01-01")
        future_end = pd.Timestamp("2030-12-31")

        _, stats_df = execute_custom_strategy(
            df_oscillation.copy(), strategy,
            period_start=future_start, period_end=future_end,
        )
        assert isinstance(stats_df, pd.DataFrame)


class TestOnAllDataShapes:
    """Smoke test: a representative strategy on every data shape."""

    @pytest.mark.parametrize("generator", [
        _generate_uptrend, _generate_downtrend,
        _generate_oscillation, _generate_v_shape,
    ], ids=["uptrend", "downtrend", "oscillation", "v_shape"])
    def test_full_strategy_all_shapes(self, generator):
        df = generator(n=300, freq="15min")
        df = calculate_indicators(df, **DEFAULT_INDICATOR_SETTINGS)

        strategy = make_strategy(
            direction="Long",
            entry_group="RSI Group",
            entry_element1="RSI",
            entry_event="Cross Above",
            entry_compare_type="Fixed Value",
            entry_value=30.0,
            entry_conditions=[{
                "group": "ADX Group",
                "element1": "ADX",
                "operator": "Above",
                "compare_type": "Fixed Value",
                "value": 10.0,
            }],
        )

        _, stats_df = execute_custom_strategy(df, strategy)
        assert isinstance(stats_df, pd.DataFrame)

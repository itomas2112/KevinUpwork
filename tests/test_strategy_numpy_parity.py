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

    # Per-trade stop distance (used by the dollar-based Monte Carlo)
    rd_pandas = stats_pandas.attrs.get("trade_r_distances")
    rd_numpy = stats_numpy.attrs.get("trade_r_distances")
    assert rd_pandas is not None and rd_numpy is not None
    assert len(rd_pandas) == len(pnls_pandas)
    assert len(rd_numpy) == len(pnls_numpy)
    if len(rd_pandas) > 0:
        np.testing.assert_allclose(
            rd_pandas, rd_numpy, rtol=1e-9,
            err_msg="Trade R distances differ between pandas and numpy engines"
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


class TestTradeRDistances:
    """attrs['trade_r_distances']: one |entry - locked stop| per trade."""

    def test_multi_group_trade_has_one_distance(self, df_oscillation):
        # Two exit groups -> two legs per trade, still one distance per trade
        groups = [
            make_exit_group(allocation_pct=50.0, group_id=g, targets=[{
                "type": "Target",
                "trigger": {"group": "R Profit / R Loss", "element1": "R Profit",
                            "event": "Cross Above", "compare_type": "Fixed Value",
                            "element2": None, "value": v},
                "conditions": [],
            }])
            for g, v in ((1, 1.5), (2, 3.0))
        ]
        strategy = make_strategy(direction="Long", exit_groups=groups)
        for engine in (execute_custom_strategy, execute_custom_strategy_numpy):
            _, stats = engine(df_oscillation.copy(), strategy)
            pnls = stats.attrs["trade_pnls_r"]
            dists = stats.attrs["trade_r_distances"]
            assert len(pnls) > 0
            assert len(dists) == len(pnls)
            assert all(d > 0 for d in dists)

    def test_distance_equals_entry_minus_locked_stop(self, df_oscillation):
        """ATR initial stop: locked stop = entry - ATR(14)*1.5 at the entry bar,
        so the first trade's distance must equal ATR*1.5 at that bar."""
        from indicators.atr_indicator import atr_indicator
        strategy = make_strategy(direction="Long", initial_stop_type="ATR",
                                 initial_stop_atr_period=14,
                                 initial_stop_atr_multiplier=1.5)
        df_res, stats_pd = execute_custom_strategy(df_oscillation.copy(), strategy)
        _, stats_np = execute_custom_strategy_numpy(df_oscillation.copy(), strategy)
        dists = stats_pd.attrs["trade_r_distances"]
        assert len(dists) > 0

        first_entry = int(np.flatnonzero(df_res["entry_signal"].to_numpy())[0])
        atr = atr_indicator(df_oscillation["high"], df_oscillation["low"],
                            df_oscillation["latest"], period=14)
        expected = float(atr.iloc[first_entry]) * 1.5
        assert dists[0] == pytest.approx(expected, rel=1e-9)
        assert stats_np.attrs["trade_r_distances"][0] == pytest.approx(expected, rel=1e-9)

    def test_empty_when_no_trades(self, df_oscillation):
        # RSI can never cross above 101 -> no trades
        strategy = make_strategy(direction="Long", entry_value=101.0)
        for engine in (execute_custom_strategy, execute_custom_strategy_numpy):
            _, stats = engine(df_oscillation.copy(), strategy)
            assert stats.attrs["trade_pnls_r"] == []
            assert stats.attrs["trade_r_distances"] == []


# ---------------------------------------------------------------------------
# within_last (0-based): 0 = this bar only, N = also the previous N bars
# ---------------------------------------------------------------------------

class TestWithinLastParity:
    """RSI crosses above 50 exactly at bar T; CCI > 0 only on one chosen bar."""

    T = 80

    @pytest.fixture(scope="class")
    def base_df(self):
        from tests.conftest import _generate_flat
        return calculate_indicators(_generate_flat(n=120), **DEFAULT_INDICATOR_SETTINGS)

    def _df(self, base_df, cci_true_bar):
        df = base_df.copy()
        rsi = np.full(len(df), 40.0)
        rsi[self.T:] = 60.0
        cci = np.full(len(df), -100.0)
        cci[cci_true_bar] = 100.0
        df["rsi"] = rsi
        df["cci"] = cci
        return df

    def _strategy(self, trigger_wl, cond_wl):
        s = make_strategy(
            entry_value=50.0,
            entry_conditions=[{"group": "CCI Group", "element1": "CCI", "operator": "Above",
                               "compare_type": "Fixed Value", "value": 0.0}],
        )
        s["entry"]["trigger"]["within_last"] = trigger_wl
        s["entry"]["conditions"][0]["within_last"] = cond_wl
        return s

    def _entry_bars(self, df, strategy):
        _compare_results(df, strategy)
        df_res, stats_pd = execute_custom_strategy(df.copy(), strategy)
        _, stats_np = execute_custom_strategy_numpy(df.copy(), strategy)
        assert len(stats_pd.attrs["trade_pnls_r"]) == len(stats_np.attrs["trade_pnls_r"])
        return list(np.flatnonzero(df_res["entry_signal"].to_numpy())), len(stats_np.attrs["trade_pnls_r"])

    def test_condition_within_last_2_accepts_t_minus_2(self, base_df):
        df = self._df(base_df, self.T - 2)
        bars, n_np = self._entry_bars(df, self._strategy(0, 2))
        assert bars == [self.T]
        assert n_np == 1

    @pytest.mark.parametrize("cond_wl", [0, 1])
    def test_condition_window_too_short_rejects(self, base_df, cond_wl):
        df = self._df(base_df, self.T - 2)
        bars, n_np = self._entry_bars(df, self._strategy(0, cond_wl))
        assert bars == []
        assert n_np == 0

    def test_trigger_within_last_1_fires_on_bar_after_cross(self, base_df):
        df = self._df(base_df, self.T + 1)  # condition only holds the bar after the cross
        bars, n_np = self._entry_bars(df, self._strategy(1, 0))
        assert bars == [self.T + 1]
        assert n_np == 1

    def test_trigger_within_last_0_needs_cross_on_this_bar(self, base_df):
        df = self._df(base_df, self.T + 1)
        bars, n_np = self._entry_bars(df, self._strategy(0, 0))
        assert bars == []
        assert n_np == 0

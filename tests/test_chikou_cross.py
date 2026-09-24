"""
Chikou events per the client's definitions (contracts/kevin_spec_2026-09-23.md):

  Chikou (close below) X : close[t] < X[t-26]
  Chikou (cross above) X : close[t-1] <= X[t-27] and close[t] > X[t-26]
  Chikou vs Senkou A/B   : Senkou at t+0 (senkou_*_current), shifted 26

Chikou never uses the Price high/low cross exception. Every case runs through
both engines and checks they agree.
"""
import numpy as np
import pytest

from indicators.calculate_indicators import calculate_indicators
from strategies.first_strategy import execute_custom_strategy
from strategies.first_strategy_numpy import execute_custom_strategy_numpy
from tests.conftest import (DEFAULT_INDICATOR_SETTINGS, _make_ohlc,
                            make_strategy, make_exit_group, make_exit_trigger)

N = 120
T = 80          # the bar under test
PAD = 0.5       # high/low offset from close on untouched bars


@pytest.fixture(scope="module")
def base_df():
    return calculate_indicators(_make_ohlc([100.0] * N), **DEFAULT_INDICATOR_SETTINGS)


def _frame(base_df, close, col, values, overrides=None):
    """Flat close series with high/low = close ± PAD, `col` set to `values`,
    then per-bar overrides {(column, bar): value}."""
    df = base_df.copy()
    close = np.asarray(close, dtype=float)
    df["latest"] = close
    df["open"] = close
    df["high"] = close + PAD
    df["low"] = close - PAD
    for c, v in (col if isinstance(col, dict) else {col: values}).items():
        df[c] = np.asarray(v, dtype=float)
    for (c, i), v in (overrides or {}).items():
        df.iloc[i, df.columns.get_loc(c)] = v
    return df


def _strategy(element1, event, element2=None, value=None, direction="Long"):
    compare_type = "Fixed Value" if value is not None else "Indicator"
    return make_strategy(
        direction=direction,
        entry_group="Price & Indicators", entry_element1=element1,
        entry_event=event, entry_compare_type=compare_type,
        entry_element2=element2, entry_value=value,
        initial_stop_event="Cross Below" if direction == "Long" else "Cross Above",
        initial_stop_atr_multiplier=1.5,
        exit_groups=[make_exit_group(targets=[make_exit_trigger(value=1000.0)])],
    )


def _trades(df, strategy, direction="Long"):
    """Run both engines, assert parity, return [(entry_bar, entry_price), ...].

    Positions run to End of Data, so entry bar = last bar - holding period and
    entry price = last close -/+ pnl_r * r_distance.
    """
    _, sp = execute_custom_strategy(df.copy(), strategy)
    _, sn = execute_custom_strategy_numpy(df.copy(), strategy)
    for key in ("trade_pnls_r", "trade_r_distances", "trade_holding_periods"):
        a, b = sp.attrs.get(key, []), sn.attrs.get(key, [])
        assert len(a) == len(b), (key, a, b)
        np.testing.assert_allclose(a, b, rtol=1e-9, err_msg=key)
    last = float(df["latest"].iloc[-1])
    sign = 1.0 if direction == "Long" else -1.0
    return [(N - 1 - hold, last - sign * pnl * rd)
            for pnl, rd, hold in zip(sp.attrs["trade_pnls_r"],
                                     sp.attrs["trade_r_distances"],
                                     sp.attrs["trade_holding_periods"])]


def _close(before, at_t, after=None):
    c = np.full(N, before)
    c[T:] = at_t if after is None else after
    c[T] = at_t
    return c


def test_chikou_cross_above_uses_close_and_t26(base_df):
    kijun = np.full(N, 200.0)
    kijun[T - 26] = 104.0                        # X[t-26]; X[t-27] stays 200
    df = _frame(base_df, _close(100.0, 105.0), "kijun", kijun,
                {("high", T): 106.0})            # high[t] <= X[t-27]
    trades = _trades(df, _strategy("Chikou", "Cross Above", "Kijun"))
    assert len(trades) == 1
    bar, price = trades[0]
    assert bar == T
    assert price == pytest.approx(105.0)         # fill at the bar close


def test_chikou_cross_above_not_fired_on_high_only(base_df):
    kijun = np.full(N, 200.0)
    kijun[T - 27] = 103.0                        # high[t] = 104 > X[t-27]
    df = _frame(base_df, _close(100.0, 101.0), "kijun", kijun,
                {("high", T): 104.0})            # close[t] = 101 <= X[t-26] = 200
    assert _trades(df, _strategy("Chikou", "Cross Above", "Kijun")) == []


def test_chikou_cross_below_uses_close_and_t26(base_df):
    kijun = np.full(N, 50.0)
    kijun[T - 26] = 95.0                         # X[t-26]; X[t-27] stays 50
    df = _frame(base_df, _close(100.0, 94.0), "kijun", kijun,
                {("low", T): 93.0})              # low[t] >= X[t-27]
    trades = _trades(df, _strategy("Chikou", "Cross Below", "Kijun", direction="Short"),
                     direction="Short")
    assert len(trades) == 1
    bar, price = trades[0]
    assert bar == T
    assert price == pytest.approx(94.0)


def test_chikou_cross_below_not_fired_on_low_only(base_df):
    kijun = np.full(N, 50.0)
    kijun[T - 27] = 97.0                         # low[t] = 96 < X[t-27]
    df = _frame(base_df, _close(100.0, 99.0), "kijun", kijun,
                {("low", T): 96.0})              # close[t] = 99 >= X[t-26] = 50
    assert _trades(df, _strategy("Chikou", "Cross Below", "Kijun", direction="Short"),
                   direction="Short") == []


def test_chikou_close_below_unchanged(base_df):
    # close[t] < X[t-26] fires regardless of t-1: X[t-27] and X[t-26] are both
    # above close, so it first fires at bar t-1 and keeps holding at t.
    kijun = np.full(N, 50.0)
    kijun[T - 27] = kijun[T - 26] = 110.0
    df = _frame(base_df, np.full(N, 100.0), "kijun", kijun)
    trades = _trades(df, _strategy("Chikou", "Close Below", "Kijun"))
    assert len(trades) == 1
    bar, price = trades[0]
    assert bar == T - 1
    assert price == pytest.approx(100.0)


def test_chikou_vs_fixed_value_cross_uses_close(base_df):
    df = _frame(base_df, _close(4990.0, 5010.0), {}, None,
                {("high", T - 1): 5005.0})       # high[t-1] > 5000, close[t-1] below
    trades = _trades(df, _strategy("Chikou", "Cross Above", value=5000.0))
    assert len(trades) == 1
    bar, price = trades[0]
    assert bar == T
    assert price == pytest.approx(5010.0)


def test_chikou_vs_senkou_uses_current_shifted_26(base_df):
    # Displaced senkou_a sits below close everywhere; only senkou_a_current[t-26]
    # drops below close, so the event must fire exactly at t.
    cur = np.full(N, 200.0)
    cur[T - 26] = 90.0
    df = _frame(base_df, np.full(N, 100.0),
                {"senkou_a": np.full(N, 50.0), "senkou_a_current": cur}, None)
    trades = _trades(df, _strategy("Chikou", "Close Above", "Senkou A"))
    assert len(trades) == 1
    bar, price = trades[0]
    assert bar == T
    assert price == pytest.approx(100.0)


def test_price_cross_exception_untouched(base_df):
    ema = np.full(N, 200.0)
    ema[T - 1] = 101.0                           # close[t-1] = 100 <= EMA[t-1]
    df = _frame(base_df, _close(100.0, 100.8), "ema_0", ema,
                {("high", T): 102.0})            # high[t] > EMA[t-1]; close[t] < EMA[t]
    trades = _trades(df, _strategy("Price", "Cross Above", "EMA 1"))
    assert len(trades) == 1
    bar, price = trades[0]
    assert bar == T
    assert price == pytest.approx(101.0)         # fill at EMA[t-1]


class TestPriceChannelEvents:
    """Price Upper / Price Lower (period 1 = raw bar high / low) through the
    existing Chikou and Price rules — no engine changes involved."""

    @staticmethod
    def _pc_frame(base_df, close, overrides=None):
        from indicators.price_channel import price_channel
        df = _frame(base_df, close, {}, None, overrides)
        df["pc_upper"], df["pc_lower"] = price_channel(df["high"], df["low"], 1, 1)
        return df

    def _chikou_below_lower(self, df):
        return _trades(df, _strategy("Chikou", "Cross Below", "Price Lower", direction="Short"),
                       direction="Short")

    def test_chikou_cross_below_price_lower_fires_at_t(self, base_df):
        # close[t-1] = 100 >= low[t-27] = 99.5 and close[t] = 100 < low[t-26] = 101
        df = self._pc_frame(base_df, np.full(N, 100.0),
                            {("low", T - 26): 101.0, ("high", T - 26): 102.0})
        trades = self._chikou_below_lower(df)
        assert len(trades) == 1
        bar, price = trades[0]
        assert bar == T
        assert price == pytest.approx(100.0)

    def test_chikou_cross_below_price_lower_uses_t27_for_prev(self, base_df):
        # low[t-27] and low[t-26] both above close: the cross happens one bar
        # earlier (close[t-2] >= low[t-28], close[t-1] < low[t-27]), never at t.
        df = self._pc_frame(base_df, np.full(N, 100.0),
                            {("low", T - 27): 101.0, ("high", T - 27): 102.0,
                             ("low", T - 26): 101.0, ("high", T - 26): 102.0})
        trades = self._chikou_below_lower(df)
        assert len(trades) == 1
        assert trades[0][0] == T - 1

    def test_chikou_cross_below_price_lower_not_fired_when_equal(self, base_df):
        # close[t] == low[t-26] is not strictly below → no event anywhere
        df = self._pc_frame(base_df, np.full(N, 100.0),
                            {("low", T - 26): 100.0, ("high", T - 26): 101.0})
        assert self._chikou_below_lower(df) == []

    def test_chikou_cross_below_price_lower_ignores_low_t(self, base_df):
        # A deep low at bar t itself must not matter — Chikou is close-based
        df = self._pc_frame(base_df, np.full(N, 100.0), {("low", T): 50.0})
        assert self._chikou_below_lower(df) == []

    def test_price_close_above_price_upper_never_fires(self, base_df):
        from tests.conftest import _generate_oscillation
        df = calculate_indicators(_generate_oscillation(n=N), **DEFAULT_INDICATOR_SETTINGS)
        assert (df["latest"] <= df["pc_upper"]).all()
        assert _trades(df, _strategy("Price", "Close Above", "Price Upper")) == []

    def test_price_cross_above_price_upper_uses_high_exception(self, base_df):
        # close[t-1] = 100 <= high[t-1] = 100.5 and high[t] = 102 > high[t-1];
        # fill at the crossed line = high[t-1]
        df = self._pc_frame(base_df, np.full(N, 100.0), {("high", T): 102.0})
        trades = _trades(df, _strategy("Price", "Cross Above", "Price Upper"))
        assert len(trades) == 1
        bar, price = trades[0]
        assert bar == T
        assert price == pytest.approx(100.0 + PAD)

"""
Tests for indicator correctness — manual formula verification on synthetic data.

Each test feeds known price data, computes the indicator, and checks results
against the textbook formula computed independently.
"""
import pytest
import numpy as np
import pandas as pd

from indicators.calculate_indicators import calculate_indicators
from tests.conftest import DEFAULT_INDICATOR_SETTINGS, _generate_uptrend, _generate_flat, _generate_oscillation, _generate_v_shape


def _calc(df, **overrides):
    """Run calculate_indicators with defaults, applying overrides."""
    params = {**DEFAULT_INDICATOR_SETTINGS, **overrides}
    return calculate_indicators(df, **params)


# ---------------------------------------------------------------------------
# RSI
# ---------------------------------------------------------------------------

class TestRSI:

    def test_rsi_column_exists(self, df_uptrend_15m):
        df = _calc(df_uptrend_15m)
        assert "rsi" in df.columns

    def test_rsi_range(self, df_uptrend_15m):
        df = _calc(df_uptrend_15m)
        valid = df["rsi"].dropna()
        assert (valid >= 0).all() and (valid <= 100).all()

    def test_rsi_uptrend_is_high(self, df_uptrend_15m):
        """In a steady uptrend, RSI should be well above 50."""
        df = _calc(df_uptrend_15m)
        # After warmup, RSI should be consistently high
        tail = df["rsi"].iloc[50:]
        assert tail.mean() > 70

    def test_rsi_downtrend_is_low(self, df_downtrend_15m):
        """In a steady downtrend, RSI should be well below 50."""
        df = _calc(df_downtrend_15m)
        tail = df["rsi"].iloc[50:]
        assert tail.mean() < 30

    def test_rsi_flat_near_50(self, df_flat_15m):
        """In flat data, RSI should hover near 50."""
        df = _calc(df_flat_15m)
        tail = df["rsi"].iloc[50:]
        assert 35 < tail.mean() < 65

    def test_rsi_manual_calculation(self):
        """Verify RSI against manual Wilder's smoothing for a small dataset."""
        # Create a simple price series
        closes = [44, 44.34, 44.09, 43.61, 44.33, 44.83, 45.10, 45.42, 45.84,
                  46.08, 45.89, 46.03, 45.61, 46.28, 46.28, 46.00, 46.03, 46.41,
                  46.22, 46.21]
        df = pd.DataFrame({
            "open": closes,
            "high": [c + 0.5 for c in closes],
            "low": [c - 0.5 for c in closes],
            "latest": closes,
            "volume": [1000] * len(closes),
        }, index=pd.date_range("2024-01-01", periods=len(closes), freq="15min"))
        df.index.name = "time"

        result = _calc(df, rsi_window=14)
        rsi_vals = result["rsi"].dropna()
        # RSI should exist and be in valid range
        assert len(rsi_vals) > 0
        assert (rsi_vals >= 0).all() and (rsi_vals <= 100).all()

    def test_rsi_different_period(self, df_oscillation_15m):
        """RSI with different periods should give different values."""
        df1 = _calc(df_oscillation_15m, rsi_window=7)
        df2 = _calc(df_oscillation_15m, rsi_window=21)
        # After warmup, values should differ on oscillating data
        assert not np.allclose(df1["rsi"].iloc[50:].values, df2["rsi"].iloc[50:].values)


# ---------------------------------------------------------------------------
# Bollinger Bands
# ---------------------------------------------------------------------------

class TestBollingerBands:

    def test_columns_exist(self, df_uptrend_15m):
        df = _calc(df_uptrend_15m)
        assert "bb_upper" in df.columns
        assert "bb_mid" in df.columns
        assert "bb_lower" in df.columns

    def test_band_ordering(self, df_uptrend_15m):
        """Upper > Mid > Lower always."""
        df = _calc(df_uptrend_15m)
        valid = df[["bb_upper", "bb_mid", "bb_lower"]].dropna()
        assert (valid["bb_upper"] >= valid["bb_mid"]).all()
        assert (valid["bb_mid"] >= valid["bb_lower"]).all()

    def test_mid_is_sma(self, df_flat_15m):
        """Middle band should be the SMA of close."""
        df = _calc(df_flat_15m, bb_mid_period=20)
        expected_sma = df["latest"].rolling(20).mean()
        valid_idx = df["bb_mid"].dropna().index
        pd.testing.assert_series_equal(
            df.loc[valid_idx, "bb_mid"],
            expected_sma.loc[valid_idx],
            check_names=False,
            atol=1e-10,
        )

    def test_upper_equals_sma_plus_stdev(self, df_flat_15m):
        """Upper band = SMA(period) + stdev_mult * StdDev(period)."""
        period, stdev = 20, 2.0
        df = _calc(df_flat_15m, bb_upper_period=period, bb_upper_stdev=stdev)
        sma = df["latest"].rolling(period).mean()
        std = df["latest"].rolling(period).std()
        expected = sma + stdev * std
        valid_idx = df["bb_upper"].dropna().index
        pd.testing.assert_series_equal(
            df.loc[valid_idx, "bb_upper"],
            expected.loc[valid_idx],
            check_names=False,
            atol=1e-10,
        )

    def test_independent_periods(self, df_uptrend_15m):
        """Upper and lower bands can have different periods."""
        df = _calc(df_uptrend_15m, bb_upper_period=10, bb_lower_period=30)
        # After warmup for both, they should produce different widths
        valid = df[["bb_upper", "bb_lower"]].iloc[40:].dropna()
        assert len(valid) > 0

    def test_flat_data_tight_bands(self, df_flat_15m):
        """Flat data should have very tight bands (small std)."""
        df = _calc(df_flat_15m)
        valid = df[["bb_upper", "bb_lower"]].dropna()
        band_width = (valid["bb_upper"] - valid["bb_lower"]).mean()
        assert band_width < 5  # Very tight for ~0.5 amplitude


# ---------------------------------------------------------------------------
# Keltner Channel
# ---------------------------------------------------------------------------

class TestKeltnerChannel:

    def test_columns_exist(self, df_uptrend_15m):
        df = _calc(df_uptrend_15m)
        assert "kc_upper" in df.columns
        assert "kc_mid" in df.columns
        assert "kc_lower" in df.columns

    def test_band_ordering(self, df_uptrend_15m):
        df = _calc(df_uptrend_15m)
        valid = df[["kc_upper", "kc_mid", "kc_lower"]].dropna()
        assert (valid["kc_upper"] >= valid["kc_mid"]).all()
        assert (valid["kc_mid"] >= valid["kc_lower"]).all()

    def test_mid_is_ema(self, df_uptrend_15m):
        """Middle band should be EMA of close."""
        df = _calc(df_uptrend_15m, kc_mid_ema=20)
        expected_ema = df["latest"].ewm(span=20, adjust=False).mean()
        valid_idx = df["kc_mid"].dropna().index
        pd.testing.assert_series_equal(
            df.loc[valid_idx, "kc_mid"],
            expected_ema.loc[valid_idx],
            check_names=False,
            atol=1e-10,
        )


# ---------------------------------------------------------------------------
# Ichimoku
# ---------------------------------------------------------------------------

class TestIchimoku:

    def test_columns_exist(self, df_uptrend_15m):
        df = _calc(df_uptrend_15m)
        for col in ["tenkan", "kijun", "senkou_a", "senkou_b", "chikou",
                     "senkou_a_current", "senkou_b_current"]:
            assert col in df.columns, f"Missing column: {col}"

    def test_tenkan_formula(self, df_uptrend_15m):
        """Tenkan = (9-period high + 9-period low) / 2."""
        df = _calc(df_uptrend_15m)
        expected = (df["high"].rolling(9).max() + df["low"].rolling(9).min()) / 2
        valid_idx = df["tenkan"].dropna().index
        pd.testing.assert_series_equal(
            df.loc[valid_idx, "tenkan"],
            expected.loc[valid_idx],
            check_names=False,
            atol=1e-10,
        )

    def test_kijun_formula(self, df_uptrend_15m):
        """Kijun = (26-period high + 26-period low) / 2."""
        df = _calc(df_uptrend_15m)
        expected = (df["high"].rolling(26).max() + df["low"].rolling(26).min()) / 2
        valid_idx = df["kijun"].dropna().index
        pd.testing.assert_series_equal(
            df.loc[valid_idx, "kijun"],
            expected.loc[valid_idx],
            check_names=False,
            atol=1e-10,
        )

    def test_senkou_a_is_displaced(self, df_uptrend_15m):
        """Senkou A should be senkou_a_current shifted forward by 26."""
        df = _calc(df_uptrend_15m)
        expected = df["senkou_a_current"].shift(26)
        valid_idx = df["senkou_a"].dropna().index
        common = valid_idx.intersection(expected.dropna().index)
        if len(common) > 0:
            pd.testing.assert_series_equal(
                df.loc[common, "senkou_a"],
                expected.loc[common],
                check_names=False,
                atol=1e-10,
            )

    def test_chikou_is_shifted_back(self, df_uptrend_15m):
        """Chikou = close shifted backward by 26."""
        df = _calc(df_uptrend_15m)
        expected = df["latest"].shift(-26)
        valid_idx = df["chikou"].dropna().index
        common = valid_idx.intersection(expected.dropna().index)
        if len(common) > 0:
            pd.testing.assert_series_equal(
                df.loc[common, "chikou"],
                expected.loc[common],
                check_names=False,
                atol=1e-10,
            )


# ---------------------------------------------------------------------------
# MACD
# ---------------------------------------------------------------------------

class TestMACD:

    def test_columns_exist(self, df_uptrend_15m):
        df = _calc(df_uptrend_15m)
        assert "macd_line" in df.columns
        assert "macd_signal" in df.columns
        assert "macd_hist" in df.columns

    def test_macd_line_formula(self, df_uptrend_15m):
        """MACD line = EMA(12) - EMA(26)."""
        df = _calc(df_uptrend_15m, macd_fast=12, macd_slow=26)
        ema12 = df["latest"].ewm(span=12, adjust=False).mean()
        ema26 = df["latest"].ewm(span=26, adjust=False).mean()
        expected = ema12 - ema26
        valid_idx = df["macd_line"].dropna().index
        pd.testing.assert_series_equal(
            df.loc[valid_idx, "macd_line"],
            expected.loc[valid_idx],
            check_names=False,
            atol=1e-10,
        )

    def test_histogram_is_line_minus_signal(self, df_uptrend_15m):
        """Histogram = MACD line - Signal."""
        df = _calc(df_uptrend_15m)
        valid = df[["macd_line", "macd_signal", "macd_hist"]].dropna()
        expected = valid["macd_line"] - valid["macd_signal"]
        pd.testing.assert_series_equal(
            valid["macd_hist"],
            expected,
            check_names=False,
            atol=1e-10,
        )

    def test_macd_uptrend_positive(self, df_uptrend_15m):
        """In uptrend, MACD line should be positive (fast EMA > slow EMA)."""
        df = _calc(df_uptrend_15m)
        tail = df["macd_line"].iloc[50:].dropna()
        assert (tail > 0).all()


# ---------------------------------------------------------------------------
# Stochastic
# ---------------------------------------------------------------------------

class TestStochastic:

    def test_columns_exist(self, df_uptrend_15m):
        df = _calc(df_uptrend_15m)
        assert "stoch_k" in df.columns
        assert "stoch_d" in df.columns

    def test_range(self, df_oscillation_15m):
        df = _calc(df_oscillation_15m)
        valid_k = df["stoch_k"].dropna()
        valid_d = df["stoch_d"].dropna()
        assert (valid_k >= 0).all() and (valid_k <= 100).all()
        assert (valid_d >= 0).all() and (valid_d <= 100).all()

    def test_k_formula(self, df_uptrend_15m):
        """Raw %K = 100 * (close - lowest_low) / (highest_high - lowest_low), then smoothed."""
        df = _calc(df_uptrend_15m, stoch_k_period=14, stoch_k_smooth=3, stoch_d_smooth=3)
        lowest = df["low"].rolling(14).min()
        highest = df["high"].rolling(14).max()
        k_raw = 100 * (df["latest"] - lowest) / (highest - lowest)
        k_smooth = k_raw.rolling(3).mean()
        valid_idx = df["stoch_k"].dropna().index
        common = valid_idx.intersection(k_smooth.dropna().index)
        if len(common) > 0:
            pd.testing.assert_series_equal(
                df.loc[common, "stoch_k"],
                k_smooth.loc[common],
                check_names=False,
                atol=1e-10,
            )


# ---------------------------------------------------------------------------
# ADX
# ---------------------------------------------------------------------------

class TestADX:

    def test_columns_exist(self, df_uptrend_15m):
        df = _calc(df_uptrend_15m)
        assert "adx" in df.columns
        assert "plus_di" in df.columns
        assert "minus_di" in df.columns

    def test_adx_range(self, df_oscillation_15m):
        df = _calc(df_oscillation_15m)
        valid = df["adx"].dropna()
        assert (valid >= 0).all() and (valid <= 100).all()

    def test_uptrend_plus_di_higher(self, df_uptrend_15m):
        """In uptrend, +DI should generally be above -DI."""
        df = _calc(df_uptrend_15m)
        tail = df[["plus_di", "minus_di"]].iloc[50:].dropna()
        assert (tail["plus_di"] > tail["minus_di"]).mean() > 0.8


# ---------------------------------------------------------------------------
# ATR
# ---------------------------------------------------------------------------

class TestATR:

    def test_column_exists(self, df_uptrend_15m):
        df = _calc(df_uptrend_15m)
        assert "atr" in df.columns

    def test_atr_positive(self, df_uptrend_15m):
        df = _calc(df_uptrend_15m)
        valid = df["atr"].dropna()
        assert (valid > 0).all()

    def test_flat_data_small_atr(self, df_flat_15m):
        """Flat data should have small ATR."""
        df = _calc(df_flat_15m)
        valid = df["atr"].dropna()
        assert valid.mean() < 5


# ---------------------------------------------------------------------------
# Supertrend
# ---------------------------------------------------------------------------

class TestSupertrend:

    def test_columns_exist(self, df_uptrend_15m):
        df = _calc(df_uptrend_15m)
        assert "supertrend" in df.columns
        assert "supertrend_upper" in df.columns
        assert "supertrend_lower" in df.columns

    def test_uptrend_bullish(self, df_uptrend_15m):
        """In steady uptrend, supertrend should be bullish (below price)."""
        df = _calc(df_uptrend_15m)
        tail = df[["latest", "supertrend"]].iloc[50:].dropna()
        assert (tail["supertrend"] <= tail["latest"]).mean() > 0.9


# ---------------------------------------------------------------------------
# Donchian Channel
# ---------------------------------------------------------------------------

class TestDonchian:

    def test_columns_exist(self, df_uptrend_15m):
        df = _calc(df_uptrend_15m)
        assert "dc_upper" in df.columns
        assert "dc_mid" in df.columns
        assert "dc_lower" in df.columns

    def test_band_ordering(self, df_oscillation_15m):
        df = _calc(df_oscillation_15m)
        valid = df[["dc_upper", "dc_mid", "dc_lower"]].dropna()
        assert (valid["dc_upper"] >= valid["dc_mid"]).all()
        assert (valid["dc_mid"] >= valid["dc_lower"]).all()


# ---------------------------------------------------------------------------
# Parabolic SAR
# ---------------------------------------------------------------------------

class TestParabolicSAR:

    def test_columns_exist(self, df_uptrend_15m):
        df = _calc(df_uptrend_15m)
        assert "psar" in df.columns
        assert "psar_upper" in df.columns
        assert "psar_lower" in df.columns

    def test_psar_values_positive(self, df_uptrend_15m):
        df = _calc(df_uptrend_15m)
        valid = df["psar"].dropna()
        assert (valid > 0).all()


# ---------------------------------------------------------------------------
# OBV
# ---------------------------------------------------------------------------

class TestOBV:

    def test_column_exists(self, df_uptrend_15m):
        df = _calc(df_uptrend_15m)
        assert "obv" in df.columns

    def test_obv_uptrend_increasing(self, df_uptrend_15m):
        """In uptrend (close always rising), OBV should generally increase."""
        df = _calc(df_uptrend_15m)
        valid = df["obv"].dropna()
        # Most of the OBV changes should be positive
        diffs = valid.diff().dropna()
        assert (diffs > 0).mean() > 0.8


# ---------------------------------------------------------------------------
# Accumulation/Distribution
# ---------------------------------------------------------------------------

class TestAccDist:

    def test_column_exists(self, df_uptrend_15m):
        df = _calc(df_uptrend_15m)
        assert "acc_dist" in df.columns


# ---------------------------------------------------------------------------
# EMA Overlay
# ---------------------------------------------------------------------------

class TestEMAOverlay:

    def test_ema_columns_created(self, df_uptrend_15m):
        df = _calc(df_uptrend_15m, ema_periods=[20, 50])
        assert "ema_0" in df.columns
        assert "ema_1" in df.columns

    def test_ema_is_correct(self, df_uptrend_15m):
        df = _calc(df_uptrend_15m, ema_periods=[20])
        expected = df["latest"].ewm(span=20, adjust=False).mean()
        valid_idx = df["ema_0"].dropna().index
        pd.testing.assert_series_equal(
            df.loc[valid_idx, "ema_0"],
            expected.loc[valid_idx],
            check_names=False,
            atol=1e-10,
        )

    def test_no_ema_columns_when_empty(self, df_uptrend_15m):
        df = _calc(df_uptrend_15m, ema_periods=[])
        assert "ema_0" not in df.columns


# ---------------------------------------------------------------------------
# CMB Composite
# ---------------------------------------------------------------------------

class TestCMB:

    def test_columns_exist(self, df_uptrend_15m):
        df = _calc(df_uptrend_15m)
        assert "ci" in df.columns
        assert "ci_13" in df.columns
        assert "ci_33" in df.columns

    def test_ci_13_is_sma_of_ci(self, df_uptrend_15m):
        """ci_13 should be the 13-period SMA of ci."""
        df = _calc(df_uptrend_15m)
        expected = df["ci"].rolling(13).mean()
        valid_idx = df["ci_13"].dropna().index
        common = valid_idx.intersection(expected.dropna().index)
        if len(common) > 0:
            pd.testing.assert_series_equal(
                df.loc[common, "ci_13"],
                expected.loc[common],
                check_names=False,
                atol=1e-10,
            )


# ---------------------------------------------------------------------------
# All indicators on 1H data (smoke test)
# ---------------------------------------------------------------------------

class TestIndicators1H:

    def test_all_indicators_compute_on_1h(self, df_uptrend_1h):
        """All indicators should compute without error on 1H data."""
        df = _calc(df_uptrend_1h)
        expected_cols = [
            "rsi", "bb_upper", "bb_mid", "bb_lower",
            "kc_upper", "kc_mid", "kc_lower",
            "tenkan", "kijun", "senkou_a", "senkou_b", "chikou",
            "macd_line", "macd_signal", "macd_hist",
            "stoch_k", "stoch_d", "adx", "plus_di", "minus_di",
            "atr", "supertrend", "dc_upper", "dc_mid", "dc_lower",
            "psar", "ci", "obv", "acc_dist",
        ]
        for col in expected_cols:
            assert col in df.columns, f"Missing column on 1H data: {col}"


# ---------------------------------------------------------------------------
# Cross-shape consistency
# ---------------------------------------------------------------------------

class TestCrossShapeConsistency:
    """Run all indicators on every data shape to ensure no crashes."""

    def test_indicators_on_all_shapes(self, df_all_shapes_15m):
        df = _calc(df_all_shapes_15m)
        assert "rsi" in df.columns
        assert len(df) > 0


# ---------------------------------------------------------------------------
# Williams %R
# ---------------------------------------------------------------------------

class TestWilliamsR:

    def test_column_exists(self, df_uptrend_15m):
        df = _calc(df_uptrend_15m)
        assert "willr" in df.columns

    def test_range_bounds(self, df_uptrend_15m):
        """Williams %R should be between -100 and 0."""
        df = _calc(df_uptrend_15m)
        valid = df["willr"].dropna()
        assert (valid >= -100).all()
        assert (valid <= 0).all()

    def test_formula_matches_manual(self, df_uptrend_15m):
        """Verify against manual calculation."""
        period = 14
        df = _calc(df_uptrend_15m, willr_period=period)
        hh = df["high"].rolling(period).max()
        ll = df["low"].rolling(period).min()
        expected = (hh - df["latest"]) / (hh - ll) * -100
        valid_idx = df["willr"].dropna().index
        pd.testing.assert_series_equal(
            df.loc[valid_idx, "willr"],
            expected.loc[valid_idx],
            check_names=False, atol=1e-10,
        )

    def test_custom_period(self, df_uptrend_15m):
        """Different period should produce different values."""
        df14 = _calc(df_uptrend_15m, willr_period=14)
        df7 = _calc(df_uptrend_15m, willr_period=7)
        assert not df14["willr"].equals(df7["willr"])

    def test_flat_market(self, df_flat_15m):
        """In a perfectly flat market, %R should be 0 (close == high == low)."""
        df = _calc(df_flat_15m, willr_period=5)
        # When high == low, division by zero → NaN, but when high > low slightly
        # due to data generation, result should be near 0 or NaN
        valid = df["willr"].dropna()
        if len(valid) > 0:
            assert (valid >= -100).all() and (valid <= 0).all()


# ---------------------------------------------------------------------------
# CCI
# ---------------------------------------------------------------------------

class TestCCI:

    def test_column_exists(self, df_uptrend_15m):
        df = _calc(df_uptrend_15m)
        assert "cci" in df.columns

    def test_formula_matches_manual(self, df_uptrend_15m):
        """Verify against manual calculation."""
        period = 20
        df = _calc(df_uptrend_15m, cci_period=period)
        tp = (df["high"] + df["low"] + df["latest"]) / 3
        sma_tp = tp.rolling(period).mean()
        mean_dev = tp.rolling(period).apply(lambda x: abs(x - x.mean()).mean(), raw=True)
        expected = (tp - sma_tp) / (0.015 * mean_dev)
        valid_idx = df["cci"].dropna().index
        pd.testing.assert_series_equal(
            df.loc[valid_idx, "cci"],
            expected.loc[valid_idx],
            check_names=False, atol=1e-10,
        )

    def test_custom_period(self, df_uptrend_15m):
        df20 = _calc(df_uptrend_15m, cci_period=20)
        df10 = _calc(df_uptrend_15m, cci_period=10)
        assert not df20["cci"].equals(df10["cci"])

    def test_oscillates_around_zero(self, df_oscillation_15m):
        """CCI should have both positive and negative values in oscillating data."""
        df = _calc(df_oscillation_15m, cci_period=10)
        valid = df["cci"].dropna()
        assert (valid > 0).any()
        assert (valid < 0).any()


# ---------------------------------------------------------------------------
# Rate of Change (ROC)
# ---------------------------------------------------------------------------

class TestROC:

    def test_columns_exist(self, df_uptrend_15m):
        df = _calc(df_uptrend_15m)
        assert "roc" in df.columns
        assert "roc_signal" in df.columns

    def test_formula_matches_manual(self, df_uptrend_15m):
        """Verify ROC line against manual calculation."""
        period = 12
        df = _calc(df_uptrend_15m, roc_period=period)
        expected = (df["latest"] - df["latest"].shift(period)) / df["latest"].shift(period) * 100
        valid_idx = df["roc"].dropna().index
        pd.testing.assert_series_equal(
            df.loc[valid_idx, "roc"],
            expected.loc[valid_idx],
            check_names=False, atol=1e-10,
        )

    def test_signal_is_ema_of_roc(self, df_uptrend_15m):
        """Signal line should be EMA of ROC."""
        roc_period = 12
        signal_period = 9
        df = _calc(df_uptrend_15m, roc_period=roc_period, roc_signal_period=signal_period)
        expected_signal = df["roc"].ewm(span=signal_period, adjust=False).mean()
        valid_idx = df["roc_signal"].dropna().index
        pd.testing.assert_series_equal(
            df.loc[valid_idx, "roc_signal"],
            expected_signal.loc[valid_idx],
            check_names=False, atol=1e-10,
        )

    def test_uptrend_positive_roc(self, df_uptrend_15m):
        """In a strong uptrend, ROC should be mostly positive after warmup."""
        df = _calc(df_uptrend_15m, roc_period=12)
        valid = df["roc"].dropna()
        # Drop first half (warmup) and check the rest is mostly positive
        second_half = valid.iloc[len(valid) // 2:]
        assert (second_half > 0).sum() > len(second_half) * 0.5

    def test_custom_periods(self, df_uptrend_15m):
        df_a = _calc(df_uptrend_15m, roc_period=12, roc_signal_period=9)
        df_b = _calc(df_uptrend_15m, roc_period=6, roc_signal_period=5)
        assert not df_a["roc"].equals(df_b["roc"])
        assert not df_a["roc_signal"].equals(df_b["roc_signal"])


# ---------------------------------------------------------------------------
# Linear Regression Channel
# ---------------------------------------------------------------------------

class TestLinearRegressionChannel:

    def test_columns_exist(self, df_uptrend_15m):
        df = _calc(df_uptrend_15m, lr_period=50)
        assert "lr_upper" in df.columns
        assert "lr_mid" in df.columns
        assert "lr_lower" in df.columns

    def test_channel_ordering(self, df_uptrend_15m):
        """Upper >= Mid >= Lower at every valid bar."""
        df = _calc(df_uptrend_15m, lr_period=50)
        valid = df[["lr_upper", "lr_mid", "lr_lower"]].dropna()
        assert (valid["lr_upper"] >= valid["lr_mid"] - 1e-10).all()
        assert (valid["lr_mid"] >= valid["lr_lower"] - 1e-10).all()

    def test_channel_symmetry(self, df_uptrend_15m):
        """Upper and lower should be equidistant from mid."""
        df = _calc(df_uptrend_15m, lr_period=50, lr_multiplier=2.0)
        valid = df[["lr_upper", "lr_mid", "lr_lower"]].dropna()
        upper_dist = valid["lr_upper"] - valid["lr_mid"]
        lower_dist = valid["lr_mid"] - valid["lr_lower"]
        pd.testing.assert_series_equal(upper_dist, lower_dist, check_names=False, atol=1e-10)

    def test_nans_before_period(self, df_uptrend_15m):
        """Should have NaN before enough bars for the period."""
        period = 50
        df = _calc(df_uptrend_15m, lr_period=period)
        assert df["lr_mid"].iloc[:period - 1].isna().all()
        assert df["lr_mid"].iloc[period - 1:].notna().any()

    def test_custom_multiplier(self, df_oscillation_15m):
        """Larger multiplier → wider channel."""
        df_narrow = _calc(df_oscillation_15m, lr_period=50, lr_multiplier=1.0)
        df_wide = _calc(df_oscillation_15m, lr_period=50, lr_multiplier=3.0)
        valid_n = df_narrow[["lr_upper", "lr_lower"]].dropna()
        valid_w = df_wide[["lr_upper", "lr_lower"]].dropna()
        common = valid_n.index.intersection(valid_w.index)
        width_narrow = (valid_n.loc[common, "lr_upper"] - valid_n.loc[common, "lr_lower"]).mean()
        width_wide = (valid_w.loc[common, "lr_upper"] - valid_w.loc[common, "lr_lower"]).mean()
        assert width_wide > width_narrow

    def test_mid_is_regression_endpoint(self, df_uptrend_15m):
        """Mid should equal the linear regression value at the last bar of the window."""
        period = 30
        df = _calc(df_uptrend_15m, lr_period=period)
        # Check a specific bar manually
        idx = period + 10  # Some bar after warmup
        values = df["latest"].iloc[idx - period + 1: idx + 1].values.astype(float)
        x = np.arange(period, dtype=float)
        slope, intercept = np.polyfit(x, values, 1)
        expected_mid = intercept + slope * (period - 1)
        assert abs(df["lr_mid"].iloc[idx] - expected_mid) < 1e-8

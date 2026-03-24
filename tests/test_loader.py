"""
Tests for data loading, DRM parsing, and OHLC resampling.
"""
import pytest
import pandas as pd
import numpy as np
import io
import tempfile
import os

from data.loader import load_ohlc, resample_ohlc, parse_drm_periods


# ---------------------------------------------------------------------------
# CSV Loading
# ---------------------------------------------------------------------------

class TestLoadOhlc:

    def _csv_bytes(self, text):
        """Create a file-like object with a .name attribute from CSV text."""
        f = io.BytesIO(text.encode())
        f.name = "test.csv"
        return f

    def test_basic_load(self):
        csv = "time,open,high,low,latest,volume\n2024-01-01 00:00,100,105,95,102,1000\n2024-01-01 00:15,102,106,98,104,1200\n"
        df = load_ohlc(self._csv_bytes(csv))

        assert df.index.name == "time"
        assert list(df.columns) == ["open", "high", "low", "latest", "volume"]
        assert len(df) == 2
        assert str(df.index.dtype).startswith("datetime64")

    def test_columns_lowercased(self):
        csv = "Time,Open,High,Low,Latest,Volume\n2024-01-01 00:00,100,105,95,102,1000\n"
        df = load_ohlc(self._csv_bytes(csv))
        assert all(c.islower() for c in df.columns)

    def test_sorted_by_time(self):
        csv = "time,open,high,low,latest\n2024-01-02 00:00,100,105,95,102\n2024-01-01 00:00,100,105,95,102\n"
        df = load_ohlc(self._csv_bytes(csv))
        assert df.index[0] < df.index[1]

    def test_drops_rows_with_nan_high(self):
        csv = "time,open,high,low,latest\n2024-01-01 00:00,100,,95,102\n2024-01-01 00:15,100,105,95,102\n"
        df = load_ohlc(self._csv_bytes(csv))
        assert len(df) == 1

    def test_rejects_non_csv(self):
        f = io.BytesIO(b"data")
        f.name = "test.xlsx"
        with pytest.raises(ValueError, match="CSV"):
            load_ohlc(f)


# ---------------------------------------------------------------------------
# Resampling
# ---------------------------------------------------------------------------

class TestResampleOhlc:

    def test_same_timeframe_returns_copy(self, df_uptrend_15m):
        result = resample_ohlc(df_uptrend_15m, "15m", base_timeframe="15m")
        assert len(result) == len(df_uptrend_15m)
        # Should be a copy, not the same object
        assert result is not df_uptrend_15m

    def test_15m_to_1h(self, df_uptrend_15m):
        result = resample_ohlc(df_uptrend_15m, "1H", base_timeframe="15m")
        # 200 bars of 15m = 50 hours, so ~50 bars at 1H
        assert len(result) < len(df_uptrend_15m)
        assert "latest" in result.columns
        assert "high" in result.columns

    def test_ohlc_aggregation_correct(self, df_uptrend_15m):
        result = resample_ohlc(df_uptrend_15m, "1H", base_timeframe="15m")

        # For the first hour, check aggregation rules
        first_hour = df_uptrend_15m.iloc[:4]  # 4 bars of 15m = 1 hour
        first_resampled = result.iloc[0]

        assert first_resampled["open"] == first_hour["open"].iloc[0]     # first
        assert first_resampled["high"] == first_hour["high"].max()       # max
        assert first_resampled["low"] == first_hour["low"].min()         # min
        assert first_resampled["latest"] == first_hour["latest"].iloc[-1]  # last

    def test_volume_summed(self, df_uptrend_15m):
        result = resample_ohlc(df_uptrend_15m, "1H", base_timeframe="15m")
        first_hour_vol = df_uptrend_15m.iloc[:4]["volume"].sum()
        assert result.iloc[0]["volume"] == first_hour_vol

    def test_15m_to_4h(self, df_uptrend_15m):
        result = resample_ohlc(df_uptrend_15m, "4H", base_timeframe="15m")
        assert len(result) < len(df_uptrend_15m)

    def test_15m_to_1d(self, df_uptrend_15m):
        result = resample_ohlc(df_uptrend_15m, "1D", base_timeframe="15m")
        assert len(result) < len(df_uptrend_15m)

    def test_1h_to_4h(self, df_uptrend_1h):
        result = resample_ohlc(df_uptrend_1h, "4H", base_timeframe="1H")
        assert len(result) < len(df_uptrend_1h)

    def test_1h_to_1d(self, df_uptrend_1h):
        result = resample_ohlc(df_uptrend_1h, "1D", base_timeframe="1H")
        assert len(result) < len(df_uptrend_1h)

    def test_invalid_timeframe_raises(self, df_uptrend_15m):
        with pytest.raises(ValueError):
            resample_ohlc(df_uptrend_15m, "2H", base_timeframe="15m")


# ---------------------------------------------------------------------------
# DRM Parsing
# ---------------------------------------------------------------------------

class TestParseDrmPeriods:

    def _make_drm_df(self, sheet_name, primary, secondary, periods_str):
        """Build a minimal DRM DataFrame."""
        rows = []
        for p in periods_str:
            rows.append({sheet_name: primary, "secondary": secondary, "period": p})
        return pd.DataFrame(rows)

    def test_basic_parsing(self):
        drm = self._make_drm_df("Bullish", "Wave3", "Sub1",
                                ["28.09.2025_17:00, 30.09.2025_19:00"])
        periods = parse_drm_periods(drm, "Bullish", "Wave3", "Sub1")

        assert len(periods) == 1
        start, end = periods[0]
        assert start == pd.Timestamp("2025-09-28 17:00")
        assert end == pd.Timestamp("2025-09-30 19:00")

    def test_multiple_periods(self):
        drm = self._make_drm_df("Bullish", "Wave3", "Sub1", [
            "01.01.2024_00:00, 02.01.2024_00:00",
            "10.01.2024_00:00, 11.01.2024_00:00",
        ])
        periods = parse_drm_periods(drm, "Bullish", "Wave3", "Sub1")
        assert len(periods) == 2

    def test_skips_non_string_values(self):
        drm = self._make_drm_df("Bullish", "Wave3", "Sub1", [np.nan, 42])
        periods = parse_drm_periods(drm, "Bullish", "Wave3", "Sub1")
        assert len(periods) == 0

    def test_skips_malformed_strings(self):
        drm = self._make_drm_df("Bullish", "Wave3", "Sub1", ["not a date range"])
        periods = parse_drm_periods(drm, "Bullish", "Wave3", "Sub1")
        assert len(periods) == 0

    def test_filters_by_primary_and_secondary(self):
        rows = [
            {"Bullish": "Wave3", "secondary": "Sub1", "period": "01.01.2024_00:00, 02.01.2024_00:00"},
            {"Bullish": "Wave5", "secondary": "Sub1", "period": "05.01.2024_00:00, 06.01.2024_00:00"},
        ]
        drm = pd.DataFrame(rows)
        periods = parse_drm_periods(drm, "Bullish", "Wave3", "Sub1")
        assert len(periods) == 1

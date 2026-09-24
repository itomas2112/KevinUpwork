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

    def _csv_bytes(self, text, name="test.csv"):
        """Create a file-like object with a .name attribute from CSV text."""
        f = io.BytesIO(text.encode("utf-8"))
        f.name = name
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
        with pytest.raises(ValueError, match="CSV or TXT"):
            load_ohlc(f)

    def test_accepts_txt_extension(self):
        csv = "time,open,high,low,latest,volume\n2024-01-01 00:00,100,105,95,102,1000\n"
        df = load_ohlc(self._csv_bytes(csv, name="test.txt"))
        assert len(df) == 1

    def test_legacy_close_column_renamed_to_latest(self):
        csv = "time,open,high,low,close,volume\n2024-01-01 00:00,100,105,95,102,1000\n"
        df = load_ohlc(self._csv_bytes(csv))
        assert "latest" in df.columns and "close" not in df.columns
        assert df["latest"].iloc[0] == 102

    # -- Kibot headerless format ------------------------------------------

    KIBOT_ROWS = (
        "09/27/2009,18:00,1042.25,1044.75,1042.25,1043.75,4905\n"
        "09/27/2009,18:15,1043.75,1044.00,1042.50,1043.00,1200\n"
        "09/27/2009,18:30,1043.00,1045.50,1042.75,1045.25,2310\n"
    )

    def test_kibot_headerless_txt_loads(self):
        df = load_ohlc(self._csv_bytes(self.KIBOT_ROWS, name="ES.txt"))

        assert df.index.name == "time"
        assert list(df.columns) == ["open", "high", "low", "latest", "volume"]
        assert len(df) == 3
        assert str(df.index.dtype).startswith("datetime64")
        assert df.index[0] == pd.Timestamp("2009-09-27 18:00")
        assert df["latest"].iloc[0] == 1043.75
        assert df["volume"].iloc[0] == 4905

    def test_kibot_dates_are_month_first(self):
        row = "03/04/2024,09:30,100,105,95,102,1000\n"
        df = load_ohlc(self._csv_bytes(row, name="ES.txt"))
        # March 4, not April 3: a day-first parse would give 2024-04-03.
        assert df.index[0] == pd.Timestamp("2024-03-04 09:30")
        assert df.index[0] != pd.Timestamp("2024-04-03 09:30")
        assert df.index[0].month == 3 and df.index[0].day == 4

    def test_kibot_detected_by_content_not_extension(self):
        df = load_ohlc(self._csv_bytes(self.KIBOT_ROWS, name="ES.csv"))
        assert "latest" in df.columns
        assert "date" not in df.columns
        assert len(df) == 3

    def test_kibot_unsorted_rows_are_sorted(self):
        rows = (
            "09/27/2009,18:15,1043.75,1044.00,1042.50,1043.00,1200\n"
            "09/27/2009,18:00,1042.25,1044.75,1042.25,1043.75,4905\n"
        )
        df = load_ohlc(self._csv_bytes(rows, name="ES.txt"))
        assert df.index.is_monotonic_increasing
        assert df.index[0] == pd.Timestamp("2009-09-27 18:00")

    def test_kibot_bom_is_tolerated(self):
        row = "\ufeff09/27/2009,18:00,1042.25,1044.75,1042.25,1043.75,4905\n"
        df = load_ohlc(self._csv_bytes(row, name="ES.txt"))
        assert len(df) == 1
        assert df.index[0] == pd.Timestamp("2009-09-27 18:00")

    def test_kibot_timestamps_unshifted(self):
        # Kibot stamps are naive US Eastern; the loader must pass them through
        # untouched, so the Sunday session open stays at 18:00.
        df = load_ohlc(self._csv_bytes(self.KIBOT_ROWS, name="ES.txt"))
        assert df.index[0].hour == 18 and df.index[0].minute == 0
        assert df.index.tz is None

    def test_kibot_nan_high_rows_dropped(self):
        rows = (
            "09/27/2009,18:00,1042.25,,1042.25,1043.75,4905\n"
            "09/27/2009,18:15,1043.75,1044.00,1042.50,1043.00,1200\n"
        )
        df = load_ohlc(self._csv_bytes(rows, name="ES.txt"))
        assert len(df) == 1


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

    def test_kibot_frame_resamples_to_1h(self):
        # Two full hours of Kibot rows, run through the loader so the renamed
        # ``latest`` column is proven to flow through resample_ohlc unchanged.
        rows = "".join(
            f"09/28/2009,{h:02d}:{m:02d},{100 + i},{110 + i},{90 + i},{105 + i},{10 * (i + 1)}\n"
            for i, (h, m) in enumerate((h, m) for h in (9, 10) for m in (0, 15, 30, 45))
        )
        f = io.BytesIO(rows.encode("utf-8"))
        f.name = "ES.txt"
        df = load_ohlc(f)
        assert len(df) == 8

        result = resample_ohlc(df, "1H", base_timeframe="15m")

        assert len(result) == 2
        assert list(result.columns) == ["open", "high", "low", "latest", "volume"]
        assert result.index[0] == pd.Timestamp("2009-09-28 09:00")
        # First hour: bars i=0..3
        assert result.iloc[0]["open"] == 100
        assert result.iloc[0]["high"] == 113
        assert result.iloc[0]["low"] == 90
        assert result.iloc[0]["latest"] == df["latest"].iloc[3]
        assert result.iloc[0]["volume"] == df["volume"].iloc[:4].sum()
        # Second hour: bars i=4..7
        assert result.iloc[1]["latest"] == df["latest"].iloc[7]
        assert result.iloc[1]["volume"] == df["volume"].iloc[4:].sum()


# ---------------------------------------------------------------------------
# Weekly and monthly resampling
#
# pandas defaults ``label`` and ``closed`` per frequency -- left for hours and
# days, right for weeks and months -- so ``resample_ohlc`` passes both
# explicitly and every timeframe stamps its bars with the *start* of the
# period. These tests pin that, and pin that saying so out loud changed nothing
# about the three timeframes that already existed.
# ---------------------------------------------------------------------------

def _daily_frame():
    """Daily bars from Sunday 2023-12-31 to Saturday 2024-03-30 inclusive.

    91 rows, which is 13 whole Sunday-to-Saturday weeks by construction, and
    which spans four calendar months: one day of December, the whole of January
    and February, and the first 30 days of March.

    OHLC is a straight ramp so every aggregate can be read off the position:
    the open of a bin is its first row, the high its last, the low its first.
    """
    index = pd.date_range("2023-12-31", "2024-03-30", freq="1D")
    position = np.arange(len(index), dtype=float)
    return pd.DataFrame({
        "open": position,
        "high": position + 100.0,
        "low": -position,
        "latest": position + 1.0,
        "volume": np.ones(len(index)),
    }, index=index)


def _period_start(ts, timeframe):
    """The start of the period ``ts`` falls in, computed independently."""
    day = ts.floor("D")
    if timeframe == "1H":
        return ts.floor("h")
    if timeframe == "4H":
        return ts.floor("4h")
    if timeframe == "1D":
        return day
    if timeframe == "1W":
        # Back to the Sunday that opens the week (Monday is dayofweek 0).
        return day - pd.Timedelta(days=(day.dayofweek + 1) % 7)
    if timeframe == "1M":
        return day.replace(day=1)
    raise AssertionError(timeframe)


class TestWeeklyAndMonthlyResample:

    def test_weekly_bar_count_and_labels(self):
        df = _daily_frame()
        result = resample_ohlc(df, "1W", base_timeframe="1D")

        # 91 daily rows starting on a Sunday: 13 weeks of exactly 7.
        assert len(result) == 13
        assert list(result.index) == list(
            pd.date_range("2023-12-31", periods=13, freq="7D"))

    def test_weekly_ohlc_aggregation(self):
        df = _daily_frame()
        result = resample_ohlc(df, "1W", base_timeframe="1D")

        for week in range(13):
            rows = df.iloc[week * 7:(week + 1) * 7]
            bar = result.iloc[week]
            assert bar["open"] == rows["open"].iloc[0]
            assert bar["high"] == rows["high"].max()
            assert bar["low"] == rows["low"].min()
            assert bar["latest"] == rows["latest"].iloc[-1]
            assert bar["volume"] == rows["volume"].sum()

    def test_monthly_bar_count_and_labels(self):
        df = _daily_frame()
        result = resample_ohlc(df, "1M", base_timeframe="1D")

        # December (one day of it), January, February, March.
        assert list(result.index) == [pd.Timestamp("2023-12-01"),
                                      pd.Timestamp("2024-01-01"),
                                      pd.Timestamp("2024-02-01"),
                                      pd.Timestamp("2024-03-01")]

    def test_monthly_ohlc_aggregation(self):
        df = _daily_frame()
        result = resample_ohlc(df, "1M", base_timeframe="1D")

        # Row counts by construction: 31 Dec, then Jan, Feb (leap), Mar 1-30.
        spans = [(0, 1), (1, 32), (32, 61), (61, 91)]
        for position, (start, stop) in enumerate(spans):
            rows = df.iloc[start:stop]
            bar = result.iloc[position]
            assert bar["open"] == rows["open"].iloc[0]
            assert bar["high"] == rows["high"].max()
            assert bar["low"] == rows["low"].min()
            assert bar["latest"] == rows["latest"].iloc[-1]
            assert bar["volume"] == rows["volume"].sum()

    def test_partial_first_and_last_periods_are_kept_whole(self):
        # The fixture's December is one day long and its March is 30. Neither is
        # dropped and neither is padded -- a bar is whatever rows fell in it.
        df = _daily_frame()
        result = resample_ohlc(df, "1M", base_timeframe="1D")

        assert result.iloc[0]["volume"] == 1.0
        assert result.iloc[-1]["volume"] == 30.0

    def test_bars_are_labelled_with_the_period_start_at_every_timeframe(self):
        # Three months of 15m bars, so every timeframe has several bars to
        # place. The claim is exact in both directions: every bar the resample
        # produces is the start of a period, and every base bar's period start
        # is a bar.
        index = pd.date_range("2024-01-01", "2024-03-31 23:45", freq="15min")
        position = np.arange(len(index), dtype=float)
        df = pd.DataFrame({"open": position, "high": position + 1.0,
                           "low": position - 1.0, "latest": position},
                          index=index)

        for timeframe in ["1H", "4H", "1D", "1W", "1M"]:
            result = resample_ohlc(df, timeframe, base_timeframe="15m")
            expected = {_period_start(ts, timeframe) for ts in df.index}
            assert set(result.index) == expected, timeframe

    def test_hourly_daily_resampling_is_unchanged_by_the_explicit_labelling(self):
        # The change that made weeks and months behave had to leave the three
        # timeframes the whole application already runs on byte-identical. The
        # expectation is built the way the old implementation built it -- pandas'
        # own defaults, nothing passed -- so this compares against the previous
        # behaviour rather than against a restatement of the new one.
        index = pd.date_range("2024-01-01 00:00", periods=96 * 40, freq="15min")
        position = np.arange(len(index), dtype=float)
        df = pd.DataFrame({"open": position, "high": position + 3.0,
                           "low": position - 3.0, "latest": position + 1.0,
                           "volume": position + 10.0}, index=index)
        agg = {"open": "first", "high": "max", "low": "min",
               "latest": "last", "volume": "sum"}

        for timeframe, rule in [("1H", "1h"), ("4H", "4h"), ("1D", "1D")]:
            before = df.resample(rule).agg(agg).dropna(subset=["high"])
            after = resample_ohlc(df, timeframe, base_timeframe="15m")

            assert after.index.equals(before.index), timeframe
            assert list(after.columns) == list(before.columns), timeframe
            assert after.equals(before), timeframe

    def test_the_monthly_alias_is_the_one_this_pandas_accepts(self):
        # Pinned deliberately: the repo runs pandas 3, where the bare "M" alias
        # is an error rather than a deprecation, and "ME" bins on month *ends* --
        # with left-hand labelling that would name a bar running from the 31st to
        # the 31st after the wrong month. "MS" is the calendar month.
        df = _daily_frame()

        with pytest.raises(ValueError):
            df.resample("M")

        assert resample_ohlc(df, "1M", base_timeframe="1D").index[0].day == 1


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

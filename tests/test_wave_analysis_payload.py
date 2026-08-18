"""
Tests for the Wave Analysis payload builder.
"""

import json
import math
import numpy as np
import pandas as pd
import pytest

from config.wave_projection import period_map
from data.loader import resample_ohlc
from ui.wave_analysis_tab import (build_wave_payload, build_fingerprint,
                                  choose_payload, dedupe_bars, held_fingerprint)

SERIES_KEYS = ["time", "open", "high", "low", "close",
               "rsi", "rsi_13", "rsi_33", "ci", "ci_13", "ci_33"]
ALL_KEYS = SERIES_KEYS + ["timeframe", "fingerprint"]


def test_payload_keys_and_lengths(df_oscillation_15m):
    payload = build_wave_payload(df_oscillation_15m, "15m", "gold.csv")

    assert set(payload.keys()) == set(ALL_KEYS)
    assert len(ALL_KEYS) == 13

    for key in SERIES_KEYS:
        assert len(payload[key]) == len(df_oscillation_15m), key

    assert payload["timeframe"] == "15m"


def test_time_values_match_index(df_oscillation_15m):
    payload = build_wave_payload(df_oscillation_15m, "15m", "gold.csv")
    times = payload["time"]

    assert all(isinstance(t, int) for t in times)
    assert times == [int(ts.timestamp()) for ts in df_oscillation_15m.index]
    assert all(times[i] < times[i + 1] for i in range(len(times) - 1))


def test_nans_become_none_and_payload_is_json_safe(df_oscillation_15m):
    payload = build_wave_payload(df_oscillation_15m, "15m", "gold.csv")

    # Leading rolling-window gaps show up as None, not NaN.
    # RSI itself is undefined on the first bar (no prior close to diff against),
    # so its 13/33 SMAs start one bar later than the raw window length.
    def first_value(key):
        return next(i for i, v in enumerate(payload[key]) if v is not None)

    assert first_value("rsi") == 1
    assert first_value("rsi_13") == 13
    assert first_value("rsi_33") == 33
    assert first_value("ci") == 10
    assert first_value("ci_13") == 22
    assert first_value("ci_33") == 42

    for key in SERIES_KEYS:
        for v in payload[key]:
            assert v is None or not (isinstance(v, float) and math.isnan(v)), key

    # allow_nan=False turns any surviving NaN into a ValueError.
    json.dumps(payload, allow_nan=False)


def test_fingerprint_identity_and_changes(df_oscillation_15m):
    df = df_oscillation_15m

    assert build_fingerprint(df, "15m", "gold.csv") == build_fingerprint(df, "15m", "gold.csv")
    assert build_fingerprint(df, "15m", "gold.csv") != build_fingerprint(df, "1H", "gold.csv")

    extra = df.iloc[[-1]].copy()
    extra.index = [df.index[-1] + (df.index[-1] - df.index[-2])]
    df_longer = pd.concat([df, extra])

    assert build_fingerprint(df_longer, "15m", "gold.csv") != \
        build_fingerprint(df, "15m", "gold.csv")
    assert build_wave_payload(df_longer, "15m", "gold.csv")["fingerprint"] != \
        build_wave_payload(df, "15m", "gold.csv")["fingerprint"]


def test_fingerprint_separates_datasets_of_identical_shape(df_oscillation_15m):
    # Two instruments exported over the same period: same row count, same first
    # and last timestamp. Only the dataset key tells them apart, and without it
    # the cached payload keeps the previous upload's bars on screen.
    df = df_oscillation_15m

    assert build_fingerprint(df, "15m", "gold.csv") != build_fingerprint(df, "15m", "silver.csv")
    assert build_wave_payload(df, "15m", "gold.csv")["fingerprint"] != \
        build_wave_payload(df, "15m", "silver.csv")["fingerprint"]


# ---------------------------------------------------------------------------
# Duplicate timestamps
#
# The client's real export carries 1,985 of them: on a contract-roll day two
# futures contracts both print a bar at the same minute. Lightweight Charts
# rejects the whole dataset on the first non-strictly-increasing time and shows
# a blank pane, so the payload builder has to resolve them here.
# ---------------------------------------------------------------------------

def _insert_duplicate(df, position, **overrides):
    """Repeat the row at ``position`` immediately after itself, with overrides."""
    row = df.iloc[[position]].copy()
    for column, value in overrides.items():
        row[column] = value
    return pd.concat([df.iloc[:position + 1], row, df.iloc[position + 1:]])


def _strictly_increasing(times):
    return all(times[i] < times[i + 1] for i in range(len(times) - 1))


def test_duplicate_timestamp_keeps_the_highest_volume_row(df_oscillation_15m):
    df = df_oscillation_15m
    winner_high = float(df["high"].iloc[10]) + 500.0
    duped = _insert_duplicate(df, 10, high=winner_high,
                              volume=float(df["volume"].max()) + 1.0)

    payload = build_wave_payload(duped, "15m", "gold.csv")

    assert len(payload["time"]) == len(df)
    assert payload["high"][10] == winner_high
    assert _strictly_increasing(payload["time"])


def test_highest_volume_row_wins_from_either_side(df_oscillation_15m):
    df = df_oscillation_15m
    # This time the loser is the *second* row of the pair.
    duped = _insert_duplicate(df, 10, high=float(df["high"].iloc[10]) + 500.0,
                              volume=0.0)

    payload = build_wave_payload(duped, "15m", "gold.csv")

    assert len(payload["time"]) == len(df)
    assert payload["high"][10] == float(df["high"].iloc[10])


def test_tied_or_unusable_volume_keeps_the_last_row(df_oscillation_15m):
    df = df_oscillation_15m
    tie_high = float(df["high"].iloc[5]) + 3.0

    tied = _insert_duplicate(df, 5, high=tie_high, volume=float(df["volume"].iloc[5]))
    assert build_wave_payload(tied, "15m", "gold.csv")["high"][5] == tie_high

    # No volume column at all -- the export's own ordering is all there is.
    no_volume = _insert_duplicate(df.drop(columns=["volume"]), 5, high=tie_high)
    assert build_wave_payload(no_volume, "15m", "gold.csv")["high"][5] == tie_high

    # A volume column that is non-numeric on exactly the duplicated rows.
    unusable = _insert_duplicate(df, 5, high=tie_high)
    unusable["volume"] = unusable["volume"].astype(object)
    column = unusable.columns.get_loc("volume")
    unusable.iloc[5, column] = "n/a"
    unusable.iloc[6, column] = "n/a"
    assert build_wave_payload(unusable, "15m", "gold.csv")["high"][5] == tie_high


def test_frame_without_duplicates_is_passed_through_untouched(df_oscillation_15m):
    df = df_oscillation_15m

    # Fast path: no copy of a quarter-million rows on every payload build.
    assert dedupe_bars(df) is df

    payload = build_wave_payload(df, "15m", "gold.csv")
    for key in SERIES_KEYS:
        assert len(payload[key]) == len(df), key
    assert payload["high"] == [float(v) for v in df["high"]]
    assert payload["close"] == [float(v) for v in df["latest"]]


def test_indicators_ignore_the_row_that_lost_the_volume_contest(df_oscillation_15m):
    df = df_oscillation_15m
    # An absurd bar that loses on volume must not reach a rolling window.
    duped = _insert_duplicate(df, 20, open=1e9, high=1e9, low=-1e9, latest=1e9,
                              volume=0.0)

    from_duplicated = build_wave_payload(duped, "15m", "gold.csv")
    from_clean = build_wave_payload(df, "15m", "gold.csv")

    for key in SERIES_KEYS:
        assert from_duplicated[key] == from_clean[key], key


def test_adversarial_input_still_yields_strictly_increasing_json_safe_times(
        df_oscillation_15m):
    df = df_oscillation_15m
    # Every third bar repeated, then the whole frame shuffled: duplicates are
    # interleaved rather than adjacent and the index arrives out of order too.
    scrambled = pd.concat([df, df.iloc[::3]])
    scrambled = scrambled.iloc[np.random.RandomState(0).permutation(len(scrambled))]

    payload = build_wave_payload(scrambled, "15m", "gold.csv")

    times = payload["time"]
    assert len(times) == len(df)
    assert times == sorted(set(times))
    for key in SERIES_KEYS:
        assert len(payload[key]) == len(df), key
    json.dumps(payload, allow_nan=False)


def test_unsorted_duplicate_free_frame_is_sorted(df_oscillation_15m):
    reversed_df = df_oscillation_15m.iloc[::-1]

    result = dedupe_bars(reversed_df)

    assert len(result) == len(reversed_df)
    assert result.index.is_monotonic_increasing
    assert result.index.is_unique


# ---------------------------------------------------------------------------
# One frame, one grid
#
# The bars the chart holds and the period map a projection is built on must come
# from the same base frame. If they disagree by so much as one bar, a projected
# pivot lands on a bar the chart does not have and the marking simply vanishes.
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("timeframe", ["1H", "4H", "1D"])
def test_payload_times_are_exactly_the_period_maps_display_bars(
        df_oscillation_15m, timeframe):
    base_df = dedupe_bars(df_oscillation_15m)
    display_df = resample_ohlc(base_df, timeframe, "15m")

    payload = build_wave_payload(display_df, timeframe, "gold.csv")
    pmap = period_map(base_df, timeframe, "15m")

    assert payload["time"] == pmap.display_times


def test_the_raw_and_deduped_frames_do_not_resample_alike(df_oscillation_15m):
    # Why the tab derives its display frame from the deduped base frame instead
    # of reusing the sidebar's ``df_ohlc``: the sidebar resamples the *raw*
    # frame, duplicates and all. The client's current export happens to repeat
    # bars verbatim, so on his file the two agree by luck -- a duplicate that is
    # not a verbatim repeat is all it takes for them to part company.
    df = df_oscillation_15m
    row = df.iloc[[10]].copy()
    row["high"] = float(df["high"].iloc[10]) + 500.0
    row["volume"] = 0.0                     # loses the dedup contest
    raw = pd.concat([df.iloc[:11], row, df.iloc[11:]])

    from_raw = resample_ohlc(raw, "1D", "15m")
    from_base = resample_ohlc(dedupe_bars(raw), "1D", "15m")

    assert list(from_raw["high"]) != list(from_base["high"])
    # ...and it is the deduped frame that agrees with the bars the chart shows.
    payload = build_wave_payload(raw, "15m", "gold.csv")
    assert payload["high"][10] == float(df["high"].iloc[10])
    assert float(from_base["high"].iloc[0]) == max(payload["high"])


# ---------------------------------------------------------------------------
# Stub handshake
# ---------------------------------------------------------------------------

def test_choose_payload_stubs_only_on_a_matching_fingerprint(df_oscillation_15m):
    payload = build_wave_payload(df_oscillation_15m, "15m", "gold.csv")
    fingerprint = payload["fingerprint"]

    stub = choose_payload(payload, fingerprint)
    assert stub == {"fingerprint": fingerprint, "stub": True}
    # The whole point: what crosses the wire is bytes, not megabytes.
    assert len(json.dumps(stub)) < 500

    # No value yet, a stale fingerprint, or a frontend that lost its state.
    assert choose_payload(payload, None) is payload
    assert choose_payload(payload, "gold.csv|15m|1|x|y") is payload
    assert choose_payload(payload, "") is payload


def test_held_fingerprint_reads_the_component_value():
    assert held_fingerprint({"seq": 3, "events": [], "held": "fp"}) == "fp"
    assert held_fingerprint({"seq": 3, "events": []}) is None
    assert held_fingerprint(None) is None
    assert held_fingerprint("not a dict") is None


def test_close_column_fallback(df_oscillation_15m):
    df_latest = df_oscillation_15m
    df_close = df_latest.rename(columns={"latest": "close"})

    payload_latest = build_wave_payload(df_latest, "15m", "gold.csv")
    payload_close = build_wave_payload(df_close, "15m", "gold.csv")

    assert payload_close["close"] == payload_latest["close"]
    assert payload_close["rsi"] == payload_latest["rsi"]
    assert payload_close["ci_33"] == payload_latest["ci_33"]


# ---------------------------------------------------------------------------
# Weekly and monthly aggregations
#
# The two timeframes pandas would otherwise label on the right. The payload is
# what the chart reads bar-for-bar, so if the grids had parted company it would
# show up here as a time the period map has never heard of.
# ---------------------------------------------------------------------------

def _two_years_1h():
    """Two years of hourly bars -- enough to make 1W and 1M real series."""
    index = pd.date_range("2023-01-01", "2024-12-31 23:00", freq="1h")
    position = np.arange(len(index), dtype=float)
    wave = 150.0 + 30.0 * np.sin(position / 90.0)
    return pd.DataFrame({
        "open": wave,
        "high": wave + 2.0,
        "low": wave - 2.0,
        "latest": wave + 0.5,
        "volume": np.ones(len(index)),
    }, index=index)


@pytest.mark.parametrize("timeframe", ["1W", "1M"])
def test_a_weekly_or_monthly_payload_is_ordered_and_json_safe(timeframe):
    display_df = resample_ohlc(_two_years_1h(), timeframe, "1H")

    payload = build_wave_payload(display_df, timeframe, "gold.csv")

    times = payload["time"]
    assert len(times) > 20                      # a real series, not one bar
    assert all(isinstance(t, int) for t in times)
    assert all(times[i] < times[i + 1] for i in range(len(times) - 1))
    for key in SERIES_KEYS:
        assert len(payload[key]) == len(display_df), key
    assert payload["timeframe"] == timeframe

    json.dumps(payload, allow_nan=False)


@pytest.mark.parametrize("timeframe", ["1W", "1M"])
def test_weekly_and_monthly_payload_times_are_the_period_maps_display_bars(
        timeframe):
    base_df = dedupe_bars(_two_years_1h())
    display_df = resample_ohlc(base_df, timeframe, "1H")

    payload = build_wave_payload(display_df, timeframe, "gold.csv")
    pmap = period_map(base_df, timeframe, "1H")

    assert payload["time"] == pmap.display_times

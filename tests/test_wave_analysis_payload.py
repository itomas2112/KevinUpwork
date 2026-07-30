"""
Tests for the Wave Analysis payload builder.
"""

import json
import math
import pandas as pd
import pytest

from ui.wave_analysis_tab import build_wave_payload, build_fingerprint

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


def test_close_column_fallback(df_oscillation_15m):
    df_latest = df_oscillation_15m
    df_close = df_latest.rename(columns={"latest": "close"})

    payload_latest = build_wave_payload(df_latest, "15m", "gold.csv")
    payload_close = build_wave_payload(df_close, "15m", "gold.csv")

    assert payload_close["close"] == payload_latest["close"]
    assert payload_close["rsi"] == payload_latest["rsi"]
    assert payload_close["ci_33"] == payload_latest["ci_33"]

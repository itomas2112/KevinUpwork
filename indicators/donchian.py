import pandas as pd


def donchian_channel(high: pd.Series, low: pd.Series,
                     upper_period: int = 20, mid_period: int = 20,
                     lower_period: int = 20, offset: int = 0):
    """
    Donchian Channel with per-band periods and optional offset/shift.

    Returns (dc_upper, dc_mid, dc_lower) as pandas Series.
    """
    dc_upper = high.rolling(window=upper_period).max()
    dc_lower = low.rolling(window=lower_period).min()

    # Midline uses average of its own upper/lower computed from mid_period
    mid_high = high.rolling(window=mid_period).max()
    mid_low = low.rolling(window=mid_period).min()
    dc_mid = (mid_high + mid_low) / 2

    if offset != 0:
        dc_upper = dc_upper.shift(offset)
        dc_mid = dc_mid.shift(offset)
        dc_lower = dc_lower.shift(offset)

    return dc_upper, dc_mid, dc_lower

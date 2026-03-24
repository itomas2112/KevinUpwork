import pandas as pd
import numpy as np


def supertrend(
    high: pd.Series,
    low: pd.Series,
    close: pd.Series,
    period: int = 7,
    multiplier: float = 3.0,
):
    """
    Supertrend indicator.

    Returns:
        supertrend_line, direction (1=up/bullish, -1=down/bearish)
    """
    prev_close = close.shift(1)

    tr = pd.concat([
        high - low,
        (high - prev_close).abs(),
        (low - prev_close).abs(),
    ], axis=1).max(axis=1)

    atr = tr.ewm(alpha=1 / period, adjust=False).mean()

    hl2 = (high + low) / 2
    upper_band = hl2 + multiplier * atr
    lower_band = hl2 - multiplier * atr

    n = len(close)
    st_line = np.full(n, np.nan)
    direction = np.full(n, 1)  # 1 = up (bullish), -1 = down (bearish)

    close_arr = close.values.copy()
    upper_arr = upper_band.values.copy()
    lower_arr = lower_band.values.copy()

    for i in range(1, n):
        if np.isnan(upper_arr[i]) or np.isnan(lower_arr[i]):
            continue

        # Tighten bands based on previous values
        if not np.isnan(upper_arr[i - 1]) and upper_arr[i] > upper_arr[i - 1]:
            if close_arr[i - 1] <= upper_arr[i - 1]:
                upper_arr[i] = upper_arr[i - 1]

        if not np.isnan(lower_arr[i - 1]) and lower_arr[i] < lower_arr[i - 1]:
            if close_arr[i - 1] >= lower_arr[i - 1]:
                lower_arr[i] = lower_arr[i - 1]

        # Determine direction
        if direction[i - 1] == 1:
            if close_arr[i] < lower_arr[i]:
                direction[i] = -1
                st_line[i] = upper_arr[i]
            else:
                direction[i] = 1
                st_line[i] = lower_arr[i]
        else:
            if close_arr[i] > upper_arr[i]:
                direction[i] = 1
                st_line[i] = lower_arr[i]
            else:
                direction[i] = -1
                st_line[i] = upper_arr[i]

    return (
        pd.Series(st_line, index=close.index),
        pd.Series(direction, index=close.index),
        pd.Series(upper_arr, index=close.index),
        pd.Series(lower_arr, index=close.index),
    )

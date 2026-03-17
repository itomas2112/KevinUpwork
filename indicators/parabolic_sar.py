import pandas as pd
import numpy as np


def parabolic_sar(high: pd.Series, low: pd.Series, close: pd.Series,
                  af_start: float = 0.02, af_increment: float = 0.02,
                  af_max: float = 0.20):
    """
    Parabolic SAR (Stop and Reverse).

    Returns (psar, psar_dir) as pandas Series.
      psar     – the SAR value
      psar_dir – 1 for uptrend (SAR below price), -1 for downtrend (SAR above price)
    """
    n = len(close)
    psar = np.full(n, np.nan)
    psar_dir = np.full(n, np.nan)
    psar_upper = np.full(n, np.nan)
    psar_lower = np.full(n, np.nan)

    if n < 2:
        return (pd.Series(psar, index=close.index),
                pd.Series(psar_dir, index=close.index),
                pd.Series(psar_upper, index=close.index),
                pd.Series(psar_lower, index=close.index))

    h = high.values
    l = low.values

    # Initialize: assume uptrend if second bar closes higher
    is_long = h[1] >= h[0]
    af = af_start
    if is_long:
        sar = l[0]
        ep = h[0]
    else:
        sar = h[0]
        ep = l[0]

    # Initialize independent upper (bearish) and lower (bullish) SAR tracks
    upper_sar = h[0]
    upper_ep = l[0]
    upper_af = af_start

    lower_sar = l[0]
    lower_ep = h[0]
    lower_af = af_start

    psar[0] = sar
    psar_dir[0] = 1 if is_long else -1
    psar_upper[0] = upper_sar
    psar_lower[0] = lower_sar

    for i in range(1, n):
        prev_sar = sar

        # --- Standard flipping PSAR ---
        if is_long:
            sar = prev_sar + af * (ep - prev_sar)
            sar = min(sar, l[i - 1])
            if i >= 2:
                sar = min(sar, l[i - 2])

            if l[i] < sar:
                is_long = False
                sar = ep
                ep = l[i]
                af = af_start
            else:
                if h[i] > ep:
                    ep = h[i]
                    af = min(af + af_increment, af_max)
        else:
            sar = prev_sar + af * (ep - prev_sar)
            sar = max(sar, h[i - 1])
            if i >= 2:
                sar = max(sar, h[i - 2])

            if h[i] > sar:
                is_long = True
                sar = ep
                ep = h[i]
                af = af_start
            else:
                if l[i] < ep:
                    ep = l[i]
                    af = min(af + af_increment, af_max)

        psar[i] = sar
        psar_dir[i] = 1 if is_long else -1

        # --- Upper SAR (always bearish / above price) ---
        prev_upper = upper_sar
        upper_sar = prev_upper + upper_af * (upper_ep - prev_upper)
        # SAR must not be below prior two highs
        upper_sar = max(upper_sar, h[i - 1])
        if i >= 2:
            upper_sar = max(upper_sar, h[i - 2])

        if h[i] > upper_sar:
            # Price broke above — reset upper SAR from current extreme
            upper_sar = h[i]
            upper_ep = l[i]
            upper_af = af_start
        else:
            if l[i] < upper_ep:
                upper_ep = l[i]
                upper_af = min(upper_af + af_increment, af_max)

        psar_upper[i] = upper_sar

        # --- Lower SAR (always bullish / below price) ---
        prev_lower = lower_sar
        lower_sar = prev_lower + lower_af * (lower_ep - prev_lower)
        # SAR must not be above prior two lows
        lower_sar = min(lower_sar, l[i - 1])
        if i >= 2:
            lower_sar = min(lower_sar, l[i - 2])

        if l[i] < lower_sar:
            # Price broke below — reset lower SAR from current extreme
            lower_sar = l[i]
            lower_ep = h[i]
            lower_af = af_start
        else:
            if h[i] > lower_ep:
                lower_ep = h[i]
                lower_af = min(lower_af + af_increment, af_max)

        psar_lower[i] = lower_sar

    return (pd.Series(psar, index=close.index),
            pd.Series(psar_dir, index=close.index),
            pd.Series(psar_upper, index=close.index),
            pd.Series(psar_lower, index=close.index))

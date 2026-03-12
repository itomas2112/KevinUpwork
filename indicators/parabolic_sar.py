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

    if n < 2:
        return pd.Series(psar, index=close.index), pd.Series(psar_dir, index=close.index)

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

    psar[0] = sar
    psar_dir[0] = 1 if is_long else -1

    for i in range(1, n):
        prev_sar = sar

        if is_long:
            sar = prev_sar + af * (ep - prev_sar)
            # SAR must not be above prior two lows
            sar = min(sar, l[i - 1])
            if i >= 2:
                sar = min(sar, l[i - 2])

            if l[i] < sar:
                # Reversal to short
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
            # SAR must not be below prior two highs
            sar = max(sar, h[i - 1])
            if i >= 2:
                sar = max(sar, h[i - 2])

            if h[i] > sar:
                # Reversal to long
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

    return pd.Series(psar, index=close.index), pd.Series(psar_dir, index=close.index)

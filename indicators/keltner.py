import pandas as pd


def keltner_channel(
    high: pd.Series,
    low: pd.Series,
    close: pd.Series,
    ema_period: int = 20,
    atr_period: int = 10,
    atr_mult: float = 2.0,
):
    """
    Keltner Channel

    Returns:
    - kc_mid   : EMA(close)
    - kc_upper : EMA + atr_mult * ATR
    - kc_lower : EMA - atr_mult * ATR
    """

    # -------------------------------------------------
    # EMA (middle line)
    # -------------------------------------------------
    kc_mid = close.ewm(span=ema_period, adjust=False).mean()

    # -------------------------------------------------
    # True Range
    # -------------------------------------------------
    prev_close = close.shift(1)

    tr = pd.concat(
        [
            high - low,
            (high - prev_close).abs(),
            (low - prev_close).abs(),
        ],
        axis=1,
    ).max(axis=1)

    # -------------------------------------------------
    # ATR (Wilder-style EMA)
    # -------------------------------------------------
    atr = tr.ewm(alpha=1 / atr_period, adjust=False).mean()

    # -------------------------------------------------
    # Channels
    # -------------------------------------------------
    kc_upper = kc_mid + atr_mult * atr
    kc_lower = kc_mid - atr_mult * atr

    return kc_mid, kc_upper, kc_lower

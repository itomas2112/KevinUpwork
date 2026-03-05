import pandas as pd


def keltner_channel(
    high: pd.Series,
    low: pd.Series,
    close: pd.Series,
    kc_upper_ema: int = 20,
    kc_mid_ema: int = 20,
    kc_lower_ema: int = 20,
    kc_atr_period: int = 10,
    kc_upper_mult: float = 2.0,
    kc_lower_mult: float = 2.0,
):
    """
    Keltner Channel with independent EMA and multiplier per band.

    Returns:
    - kc_mid   : EMA(close, kc_mid_ema)
    - kc_upper : EMA(close, kc_upper_ema) + kc_upper_mult * ATR
    - kc_lower : EMA(close, kc_lower_ema) - kc_lower_mult * ATR
    """

    # -------------------------------------------------
    # EMA per band
    # -------------------------------------------------
    kc_mid = close.ewm(span=kc_mid_ema, adjust=False).mean()
    kc_upper_ema_line = close.ewm(span=kc_upper_ema, adjust=False).mean()
    kc_lower_ema_line = close.ewm(span=kc_lower_ema, adjust=False).mean()

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
    # ATR (Wilder-style EMA) — shared across bands
    # -------------------------------------------------
    atr = tr.ewm(alpha=1 / kc_atr_period, adjust=False).mean()

    # -------------------------------------------------
    # Channels
    # -------------------------------------------------
    kc_upper = kc_upper_ema_line + kc_upper_mult * atr
    kc_lower = kc_lower_ema_line - kc_lower_mult * atr

    return kc_mid, kc_upper, kc_lower

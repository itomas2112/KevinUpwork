import pandas as pd
from indicators.rsi import rsi

def cmb_composite(
    close: pd.Series,
    rsi_long: int = 14,
    mom_len: int = 9,
    rsi_short: int = 3,
    rsi3_sma: int = 3,
    ci_sma_fast: int = 13,
    ci_sma_slow: int = 33,
):
    """
    Constance Brown Composite (CMB Composite)

    Returns three series:
    - ci      : Composite Index
    - ci_fast : SMA(ci, 13)
    - ci_slow : SMA(ci, 33)
    """

    # -------------------------------------------------
    # RSI calculations (Wilder)
    # -------------------------------------------------
    rsi14 = rsi(close, rsi_long)
    rsi3 = rsi(close, rsi_short)

    # -------------------------------------------------
    # Momentum(9) of RSI(14)
    # -------------------------------------------------
    rsi14_mom9 = rsi14 - rsi14.shift(mom_len)

    # -------------------------------------------------
    # SMA(3) of RSI(3)
    # -------------------------------------------------
    rsi3_sma3 = rsi3.rolling(rsi3_sma).mean()

    # -------------------------------------------------
    # Composite Index
    # -------------------------------------------------
    ci = rsi14_mom9 + rsi3_sma3

    # -------------------------------------------------
    # Signal lines
    # -------------------------------------------------
    ci_fast = ci.rolling(ci_sma_fast).mean()
    ci_slow = ci.rolling(ci_sma_slow).mean()

    return ci, ci_fast, ci_slow

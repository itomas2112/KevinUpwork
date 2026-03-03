import pandas as pd


def stochastic(
    high: pd.Series,
    low: pd.Series,
    close: pd.Series,
    k_period: int = 14,
    k_smooth: int = 3,
    d_smooth: int = 3,
):
    """
    Full Stochastic Oscillator.

    %K_raw = 100 * (Close - Lowest Low(kPeriod)) / (Highest High(kPeriod) - Lowest Low(kPeriod))
    %K     = SMA(%K_raw, k_smooth)
    %D     = SMA(%K, d_smooth)

    Returns:
        stoch_k, stoch_d
    """
    lowest_low = low.rolling(k_period).min()
    highest_high = high.rolling(k_period).max()

    k_raw = 100 * (close - lowest_low) / (highest_high - lowest_low)

    stoch_k = k_raw.rolling(k_smooth).mean()
    stoch_d = stoch_k.rolling(d_smooth).mean()

    return stoch_k, stoch_d

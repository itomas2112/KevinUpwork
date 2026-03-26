import pandas as pd


def williams_r(
    high: pd.Series,
    low: pd.Series,
    close: pd.Series,
    period: int = 14,
):
    """
    Williams %R oscillator.

    %R = (Highest High(n) - Close) / (Highest High(n) - Lowest Low(n)) * -100

    Range: -100 to 0
    Overbought: above -20, Oversold: below -80

    Returns:
        willr: pd.Series
    """
    highest_high = high.rolling(period).max()
    lowest_low = low.rolling(period).min()

    willr = (highest_high - close) / (highest_high - lowest_low) * -100

    return willr

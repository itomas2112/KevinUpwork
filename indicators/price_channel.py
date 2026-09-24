import pandas as pd


def price_channel(high: pd.Series, low: pd.Series,
                  upper_period: int = 1, lower_period: int = 1):
    """Price Channel: Price Upper = SMA(high, upper_period), Price Lower = SMA(low, lower_period).
    Period 1 returns the raw high / low series (client's definition). Returns (pc_upper, pc_lower)."""
    upper_period = int(upper_period)
    lower_period = int(lower_period)
    pc_upper = high.rolling(window=upper_period, min_periods=upper_period).mean()
    pc_lower = low.rolling(window=lower_period, min_periods=lower_period).mean()
    return pc_upper, pc_lower

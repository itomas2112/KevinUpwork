import pandas as pd


def bollinger_bands(
    price: pd.Series,
    bb_upper_period: int = 20,
    bb_upper_stdev: float = 2.0,
    bb_mid_period: int = 20,
    bb_lower_period: int = 20,
    bb_lower_stdev: float = 2.0,
):
    """
    Bollinger Bands with independent settings per band.

    Returns:
    - bb_mid   : SMA(bb_mid_period)
    - bb_upper : SMA(bb_upper_period) + bb_upper_stdev * std(bb_upper_period)
    - bb_lower : SMA(bb_lower_period) - bb_lower_stdev * std(bb_lower_period)
    """

    bb_mid = price.rolling(bb_mid_period).mean()

    bb_upper_sma = price.rolling(bb_upper_period).mean()
    bb_upper_std = price.rolling(bb_upper_period).std()
    bb_upper = bb_upper_sma + bb_upper_stdev * bb_upper_std

    bb_lower_sma = price.rolling(bb_lower_period).mean()
    bb_lower_std = price.rolling(bb_lower_period).std()
    bb_lower = bb_lower_sma - bb_lower_stdev * bb_lower_std

    return bb_mid, bb_upper, bb_lower

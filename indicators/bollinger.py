import pandas as pd


def bollinger_bands(
    price: pd.Series,
    period: int = 20,
    stdev: float = 2.0,
):
    """
    Bollinger Bands

    Returns:
    - bb_mid   : SMA(period)
    - bb_upper : mid + stdev * std
    - bb_lower : mid - stdev * std
    """

    bb_mid = price.rolling(period).mean()
    bb_std = price.rolling(period).std()

    bb_upper = bb_mid + stdev * bb_std
    bb_lower = bb_mid - stdev * bb_std

    return bb_mid, bb_upper, bb_lower

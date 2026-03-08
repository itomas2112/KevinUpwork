import pandas as pd


def accumulation_distribution(
    high: pd.Series,
    low: pd.Series,
    close: pd.Series,
    volume: pd.Series,
):
    """
    Accumulation/Distribution Line.

    Returns:
        ad Series
    """
    hl_range = high - low
    # Avoid division by zero
    mfm = ((close - low) - (high - close)) / hl_range.replace(0, float('nan'))
    mfm = mfm.fillna(0)
    mfv = mfm * volume
    ad_line = mfv.cumsum()
    return ad_line

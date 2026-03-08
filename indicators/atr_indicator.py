import pandas as pd


def atr_indicator(
    high: pd.Series,
    low: pd.Series,
    close: pd.Series,
    period: int = 14,
):
    """
    Average True Range.

    Returns:
        atr Series
    """
    prev_close = close.shift(1)

    tr = pd.concat([
        high - low,
        (high - prev_close).abs(),
        (low - prev_close).abs(),
    ], axis=1).max(axis=1)

    atr_line = tr.ewm(alpha=1 / period, adjust=False).mean()

    return atr_line

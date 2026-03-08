import pandas as pd


def ema_overlay(close: pd.Series, periods: list):
    """
    Calculate multiple EMA lines for overlay on price chart.

    Parameters:
        close: close price series
        periods: list of EMA periods (e.g. [9, 21, 50])

    Returns:
        list of (period, ema_series) tuples
    """
    results = []
    for p in periods:
        ema = close.ewm(span=p, adjust=False).mean()
        results.append((p, ema))
    return results

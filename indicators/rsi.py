import pandas as pd

def rsi(close: pd.Series, window: int = 14) -> pd.Series:
    """
    Compute Wilder's RSI.

    Parameters
    ----------
    close : pd.Series
        Close price series
    window : int
        RSI window size (default 14)

    Returns
    -------
    pd.Series
        RSI values
    """

    delta = close.diff()

    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)

    # Wilder's smoothing
    avg_gain = gain.ewm(alpha=1 / window, adjust=False).mean()
    avg_loss = loss.ewm(alpha=1 / window, adjust=False).mean()

    rs = avg_gain / avg_loss
    rsi = 100 - (100 / (1 + rs))

    return rsi

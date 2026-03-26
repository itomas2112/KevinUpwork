import pandas as pd


def roc(
    close: pd.Series,
    period: int = 12,
    signal_period: int = 9,
):
    """
    Rate of Change with EMA signal line.

    ROC = ((Close - Close[n]) / Close[n]) * 100
    Signal = EMA(ROC, signal_period)

    Returns:
        roc_line, roc_signal: (pd.Series, pd.Series)
    """
    roc_line = (close - close.shift(period)) / close.shift(period) * 100

    roc_signal = roc_line.ewm(span=signal_period, adjust=False).mean()

    return roc_line, roc_signal

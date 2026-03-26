import pandas as pd


def cci(
    high: pd.Series,
    low: pd.Series,
    close: pd.Series,
    period: int = 20,
):
    """
    Commodity Channel Index.

    TP = (High + Low + Close) / 3
    CCI = (TP - SMA(TP, n)) / (0.015 * MeanDeviation(TP, n))

    Returns:
        cci_series: pd.Series
    """
    tp = (high + low + close) / 3
    sma_tp = tp.rolling(period).mean()
    mean_dev = tp.rolling(period).apply(lambda x: abs(x - x.mean()).mean(), raw=True)

    cci_series = (tp - sma_tp) / (0.015 * mean_dev)

    return cci_series

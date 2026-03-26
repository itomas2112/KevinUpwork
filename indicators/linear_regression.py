import numpy as np
import pandas as pd


def linear_regression_channel(
    close: pd.Series,
    period: int = 100,
    multiplier: float = 2.0,
):
    """
    Linear Regression Channel.

    For each bar, compute a linear regression over the last `period` bars.
    - Middle = regression value at the current bar
    - Upper  = middle + multiplier * standard error
    - Lower  = middle - multiplier * standard error

    Returns:
        lr_upper, lr_mid, lr_lower: (pd.Series, pd.Series, pd.Series)
    """
    n = len(close)
    lr_mid = pd.Series(np.nan, index=close.index, dtype=np.float64)
    lr_upper = pd.Series(np.nan, index=close.index, dtype=np.float64)
    lr_lower = pd.Series(np.nan, index=close.index, dtype=np.float64)

    values = close.values.astype(np.float64)
    x = np.arange(period, dtype=np.float64)
    x_mean = x.mean()
    ss_xx = ((x - x_mean) ** 2).sum()

    for i in range(period - 1, n):
        y = values[i - period + 1: i + 1]
        if np.isnan(y).any():
            continue
        y_mean = y.mean()
        slope = ((x - x_mean) * (y - y_mean)).sum() / ss_xx
        intercept = y_mean - slope * x_mean
        fitted = intercept + slope * x
        residuals = y - fitted
        std_err = np.sqrt((residuals ** 2).sum() / (period - 2)) if period > 2 else 0.0

        lr_mid.iloc[i] = fitted[-1]
        lr_upper.iloc[i] = fitted[-1] + multiplier * std_err
        lr_lower.iloc[i] = fitted[-1] - multiplier * std_err

    return lr_upper, lr_mid, lr_lower

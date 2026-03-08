import pandas as pd
import numpy as np


def obv(close: pd.Series, volume: pd.Series):
    """
    On Balance Volume.

    Returns:
        obv Series
    """
    direction = np.sign(close.diff())
    obv_line = (volume * direction).fillna(0).cumsum()
    return obv_line

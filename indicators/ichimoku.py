import pandas as pd


def ichimoku(
    high: pd.Series,
    low: pd.Series,
    tenkan_len: int = 9,
    kijun_len: int = 26,
    senkou_b_len: int = 52,
    displacement: int = 26,
):
    """
    Ichimoku Cloud (without Chikou span)

    Returns:
    - tenkan
    - kijun
    - senkou_a
    - senkou_b
    """

    # -------------------------------------------------
    # Tenkan-sen (Conversion Line)
    # -------------------------------------------------
    tenkan = (
        high.rolling(tenkan_len).max()
        + low.rolling(tenkan_len).min()
    ) / 2

    # -------------------------------------------------
    # Kijun-sen (Base Line)
    # -------------------------------------------------
    kijun = (
        high.rolling(kijun_len).max()
        + low.rolling(kijun_len).min()
    ) / 2

    # -------------------------------------------------
    # Senkou Span A (Leading Span A)
    # -------------------------------------------------
    senkou_a = ((tenkan + kijun) / 2).shift(displacement)

    # -------------------------------------------------
    # Senkou Span B (Leading Span B)
    # -------------------------------------------------
    senkou_b = (
        (
            high.rolling(senkou_b_len).max()
            + low.rolling(senkou_b_len).min()
        ) / 2
    ).shift(displacement)

    return tenkan, kijun, senkou_a, senkou_b

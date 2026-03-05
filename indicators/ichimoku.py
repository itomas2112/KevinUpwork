import pandas as pd


def ichimoku(
    high: pd.Series,
    low: pd.Series,
    close: pd.Series = None,
    tenkan_len: int = 9,
    kijun_len: int = 26,
    senkou_b_len: int = 52,
    displacement: int = 26,
):
    """
    Ichimoku Cloud

    Returns:
    - tenkan
    - kijun
    - senkou_a
    - senkou_b
    - chikou (close shifted back by displacement periods)
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
    senkou_a_raw = (tenkan + kijun) / 2
    senkou_a = senkou_a_raw.shift(displacement)

    # -------------------------------------------------
    # Senkou Span B (Leading Span B)
    # -------------------------------------------------
    senkou_b_raw = (
        high.rolling(senkou_b_len).max()
        + low.rolling(senkou_b_len).min()
    ) / 2
    senkou_b = senkou_b_raw.shift(displacement)

    # -------------------------------------------------
    # Chikou Span (Lagging Span) — close shifted back
    # -------------------------------------------------
    chikou = close.shift(-displacement) if close is not None else None

    return tenkan, kijun, senkou_a, senkou_b, chikou, senkou_a_raw, senkou_b_raw

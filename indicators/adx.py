import pandas as pd


def adx(
    high: pd.Series,
    low: pd.Series,
    close: pd.Series,
    period: int = 14,
):
    """
    Average Directional Index.

    Returns:
        adx_line, plus_di, minus_di
    """
    prev_high = high.shift(1)
    prev_low = low.shift(1)
    prev_close = close.shift(1)

    # Directional movement
    plus_dm = high - prev_high
    minus_dm = prev_low - low

    plus_dm = plus_dm.where((plus_dm > minus_dm) & (plus_dm > 0), 0.0)
    minus_dm = minus_dm.where((minus_dm > plus_dm) & (minus_dm > 0), 0.0)

    # True Range
    tr = pd.concat([
        high - low,
        (high - prev_close).abs(),
        (low - prev_close).abs(),
    ], axis=1).max(axis=1)

    # Wilder smoothing (EMA with alpha=1/period)
    alpha = 1 / period
    atr = tr.ewm(alpha=alpha, adjust=False).mean()
    smooth_plus_dm = plus_dm.ewm(alpha=alpha, adjust=False).mean()
    smooth_minus_dm = minus_dm.ewm(alpha=alpha, adjust=False).mean()

    plus_di = 100 * smooth_plus_dm / atr
    minus_di = 100 * smooth_minus_dm / atr

    dx = 100 * (plus_di - minus_di).abs() / (plus_di + minus_di)
    adx_line = dx.ewm(alpha=alpha, adjust=False).mean()

    return adx_line, plus_di, minus_di

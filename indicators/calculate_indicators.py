import pandas as pd

from indicators.rsi import rsi
from indicators.cmb import cmb_composite
from indicators.ichimoku import ichimoku
from indicators.bollinger import bollinger_bands
from indicators.keltner import keltner_channel


def calculate_indicators(
    df: pd.DataFrame,
    rsi_window: int,
    bb_period: int,
    bb_stdev: float,
    kc_ema_period: int,
    kc_atr_period: int,
    kc_atr_mult: float,
) -> pd.DataFrame:
    """
    Calculate all indicators on full data, then slice and clean.

    Returns:
        all features created
    """

    df = df.copy()

    # -------------------------------------------------
    # RSI
    # -------------------------------------------------
    df["rsi"] = rsi(df["latest"], rsi_window)

    # -------------------------------------------------
    # CMB Composite
    # -------------------------------------------------
    ci, ci_13, ci_33 = cmb_composite(df["latest"])
    df["ci"] = ci
    df["ci_13"] = ci_13
    df["ci_33"] = ci_33

    # -------------------------------------------------
    # Ichimoku
    # -------------------------------------------------
    (
        df["tenkan"],
        df["kijun"],
        df["senkou_a"],
        df["senkou_b"],
    ) = ichimoku(df["high"], df["low"])

    # -------------------------------------------------
    # Bollinger Bands
    # -------------------------------------------------
    (
        df["bb_mid"],
        df["bb_upper"],
        df["bb_lower"],
    ) = bollinger_bands(
        df["latest"],
        period=bb_period,
        stdev=bb_stdev,
    )

    # -------------------------------------------------
    # Keltner Channel
    # -------------------------------------------------
    (
        df["kc_mid"],
        df["kc_upper"],
        df["kc_lower"],
    ) = keltner_channel(
        df["high"],
        df["low"],
        df["latest"],
        ema_period=kc_ema_period,
        atr_period=kc_atr_period,
        atr_mult=kc_atr_mult,
    )

    return df

def slice_for_graph(
        df: pd.DataFrame,
        start_date,
        end_date,
        show_ichimoku: bool,
        show_bb: bool,
        show_kc: bool,
        context_bars: int = 50,
) -> pd.DataFrame:
    # -------------------------------------------------
    # Selected window (true period)
    # -------------------------------------------------
    df_sel = df.loc[start_date:end_date]

    if df_sel.empty:
        return df_sel, None, None

    period_start = df_sel.index[0]
    period_end = df_sel.index[-1]

    # -------------------------------------------------
    # Extend by ±context bars
    # -------------------------------------------------
    full_index = df.index
    start_pos = full_index.get_loc(period_start)
    end_pos = full_index.get_loc(period_end)

    ext_start = max(0, start_pos - context_bars)
    ext_end = min(len(df) - 1, end_pos + context_bars)

    df_plot = df.iloc[ext_start : ext_end + 1].copy()

    # -------------------------------------------------
    # Drop NaNs only on required cols
    # -------------------------------------------------
    required_cols = ["latest", "rsi", "ci", "ci_13", "ci_33"]

    if show_ichimoku:
        required_cols += ["tenkan", "kijun", "senkou_a", "senkou_b"]
    if show_bb:
        required_cols += ["bb_mid", "bb_upper", "bb_lower"]
    if show_kc:
        required_cols += ["kc_mid", "kc_upper", "kc_lower"]

    df_plot = df_plot.dropna(subset=required_cols)

    # -------------------------------------------------
    # Categorical x-axis helpers
    # -------------------------------------------------
    df_plot["x"] = df_plot.index.strftime("%Y-%m-%d %H:%M")
    df_plot["date_only"] = df_plot.index.strftime("%Y-%m-%d")

    period_start = _snap_to_plot_index(period_start, df_plot.index)
    period_end = _snap_to_plot_index(period_end, df_plot.index)

    return(df_plot, period_start, period_end)

def _snap_to_plot_index(ts, plot_index):
    # ts is a pandas Timestamp
    if ts in plot_index:
        return ts
    # find the first bar >= ts; if none, use last
    pos = plot_index.searchsorted(ts, side="left")
    if pos >= len(plot_index):
        return plot_index[-1]
    return plot_index[pos]
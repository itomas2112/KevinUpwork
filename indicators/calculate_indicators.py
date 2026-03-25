import pandas as pd

from indicators.rsi import rsi
from indicators.cmb import cmb_composite
from indicators.ichimoku import ichimoku
from indicators.bollinger import bollinger_bands
from indicators.keltner import keltner_channel
from indicators.stochastic import stochastic
from indicators.adx import adx
from indicators.atr_indicator import atr_indicator
from indicators.macd import macd
from indicators.obv import obv
from indicators.accumulation_distribution import accumulation_distribution
from indicators.supertrend import supertrend
from indicators.ema_overlay import ema_overlay
from indicators.donchian import donchian_channel
from indicators.parabolic_sar import parabolic_sar


# ---------------------------------------------------------------------------
# Param → indicator group mapping for incremental recalculation
# ---------------------------------------------------------------------------
PARAM_TO_GROUP = {
    'rsi_window': 'rsi',
    'bb_upper_period': 'bb', 'bb_upper_stdev': 'bb', 'bb_mid_period': 'bb',
    'bb_lower_period': 'bb', 'bb_lower_stdev': 'bb',
    'kc_upper_ema': 'kc', 'kc_mid_ema': 'kc', 'kc_lower_ema': 'kc',
    'kc_atr_period': 'kc', 'kc_upper_mult': 'kc', 'kc_lower_mult': 'kc',
    'stoch_k_period': 'stoch', 'stoch_k_smooth': 'stoch', 'stoch_d_smooth': 'stoch',
    'adx_period': 'adx',
    'atr_period': 'atr',
    'macd_fast': 'macd', 'macd_slow': 'macd', 'macd_signal': 'macd',
    'supertrend_period': 'supertrend', 'supertrend_multiplier': 'supertrend',
    'ema_periods': 'ema',
    'dc_upper_period': 'donchian', 'dc_mid_period': 'donchian',
    'dc_lower_period': 'donchian', 'dc_offset': 'donchian',
    'psar_af_start': 'psar', 'psar_af_increment': 'psar', 'psar_af_max': 'psar',
}


def recalculate_groups(df, groups, **params):
    """Recalculate only the specified indicator groups on *df* (in-place)."""

    if 'rsi' in groups:
        df["rsi"] = rsi(df["latest"], params.get('rsi_window', 14))
        df["rsi_13"] = df["rsi"].rolling(13).mean()
        df["rsi_33"] = df["rsi"].rolling(33).mean()

    # CMB and Ichimoku have no user-tunable params, so they never appear in
    # a "changed groups" set.  Included here only for completeness.
    if 'cmb' in groups:
        ci, ci_13, ci_33 = cmb_composite(df["latest"])
        df["ci"] = ci; df["ci_13"] = ci_13; df["ci_33"] = ci_33

    if 'ichimoku' in groups:
        (df["tenkan"], df["kijun"], df["senkou_a"], df["senkou_b"],
         df["chikou"], df["senkou_a_current"], df["senkou_b_current"]
        ) = ichimoku(df["high"], df["low"], df["latest"])

    if 'bb' in groups:
        df["bb_mid"], df["bb_upper"], df["bb_lower"] = bollinger_bands(
            df["latest"],
            bb_upper_period=params.get('bb_upper_period', 20),
            bb_upper_stdev=params.get('bb_upper_stdev', 2.0),
            bb_mid_period=params.get('bb_mid_period', 20),
            bb_lower_period=params.get('bb_lower_period', 20),
            bb_lower_stdev=params.get('bb_lower_stdev', 2.0),
        )

    if 'kc' in groups:
        df["kc_mid"], df["kc_upper"], df["kc_lower"] = keltner_channel(
            df["high"], df["low"], df["latest"],
            kc_upper_ema=params.get('kc_upper_ema', 20),
            kc_mid_ema=params.get('kc_mid_ema', 20),
            kc_lower_ema=params.get('kc_lower_ema', 20),
            kc_atr_period=params.get('kc_atr_period', 10),
            kc_upper_mult=params.get('kc_upper_mult', 2.0),
            kc_lower_mult=params.get('kc_lower_mult', 2.0),
        )

    if 'stoch' in groups:
        df["stoch_k"], df["stoch_d"] = stochastic(
            df["high"], df["low"], df["latest"],
            k_period=params.get('stoch_k_period', 14),
            k_smooth=params.get('stoch_k_smooth', 3),
            d_smooth=params.get('stoch_d_smooth', 3),
        )

    if 'adx' in groups:
        df["adx"], df["plus_di"], df["minus_di"] = adx(
            df["high"], df["low"], df["latest"],
            period=params.get('adx_period', 14),
        )

    if 'atr' in groups:
        df["atr"] = atr_indicator(
            df["high"], df["low"], df["latest"],
            period=params.get('atr_period', 14),
        )

    if 'macd' in groups:
        df["macd_line"], df["macd_signal"], df["macd_hist"] = macd(
            df["latest"],
            fast_period=params.get('macd_fast', 12),
            slow_period=params.get('macd_slow', 26),
            signal_period=params.get('macd_signal', 9),
        )

    if 'supertrend' in groups:
        df["supertrend"], df["supertrend_dir"], st_upper, st_lower = supertrend(
            df["high"], df["low"], df["latest"],
            period=params.get('supertrend_period', 10),
            multiplier=params.get('supertrend_multiplier', 3.0),
        )
        df["supertrend_upper"] = st_upper.shift(1)
        df["supertrend_lower"] = st_lower.shift(1)

    if 'ema' in groups:
        # Remove old ema columns
        old_ema_cols = [c for c in df.columns if c.startswith("ema_")]
        if old_ema_cols:
            df.drop(columns=old_ema_cols, inplace=True)
        ema_periods = params.get('ema_periods') or []
        for i, (period, ema_series) in enumerate(ema_overlay(df["latest"], ema_periods)):
            df[f"ema_{i}"] = ema_series

    if 'donchian' in groups:
        df["dc_upper"], df["dc_mid"], df["dc_lower"] = donchian_channel(
            df["high"], df["low"],
            upper_period=params.get('dc_upper_period', 20),
            mid_period=params.get('dc_mid_period', 20),
            lower_period=params.get('dc_lower_period', 20),
            offset=params.get('dc_offset', 0),
        )

    if 'psar' in groups:
        df["psar"], df["psar_dir"], psar_up, psar_lo = parabolic_sar(
            df["high"], df["low"], df["latest"],
            af_start=params.get('psar_af_start', 0.02),
            af_increment=params.get('psar_af_increment', 0.02),
            af_max=params.get('psar_af_max', 0.20),
        )
        df["psar_upper"] = psar_up.shift(1)
        df["psar_lower"] = psar_lo.shift(1)

    return df


def changed_groups(old_params, new_params):
    """Return the set of indicator group names whose params differ."""
    groups = set()
    all_keys = set(old_params.keys()) | set(new_params.keys())
    for k in all_keys:
        if old_params.get(k) != new_params.get(k):
            group = PARAM_TO_GROUP.get(k)
            if group:
                groups.add(group)
    return groups


def migrate_indicator_settings(settings):
    """
    Convert old-format indicator settings (bb_period, bb_stdev, kc_ema_period,
    kc_atr_mult) to the new per-band format. Returns a new dict; does not
    mutate the original.
    """
    if settings is None:
        return settings
    s = dict(settings)

    # BB migration: bb_period -> all 3 band periods, bb_stdev -> both stdevs
    if 'bb_period' in s:
        val = s.pop('bb_period')
        s.setdefault('bb_upper_period', val)
        s.setdefault('bb_mid_period', val)
        s.setdefault('bb_lower_period', val)
    if 'bb_stdev' in s:
        val = s.pop('bb_stdev')
        s.setdefault('bb_upper_stdev', val)
        s.setdefault('bb_lower_stdev', val)

    # KC migration: kc_ema_period -> all 3 EMAs, kc_atr_mult -> both mults
    if 'kc_ema_period' in s:
        val = s.pop('kc_ema_period')
        s.setdefault('kc_upper_ema', val)
        s.setdefault('kc_mid_ema', val)
        s.setdefault('kc_lower_ema', val)
    if 'kc_atr_mult' in s:
        val = s.pop('kc_atr_mult')
        s.setdefault('kc_upper_mult', val)
        s.setdefault('kc_lower_mult', val)

    return s


def calculate_indicators(
    df: pd.DataFrame,
    rsi_window: int,
    bb_upper_period: int = 20,
    bb_upper_stdev: float = 2.0,
    bb_mid_period: int = 20,
    bb_lower_period: int = 20,
    bb_lower_stdev: float = 2.0,
    kc_upper_ema: int = 20,
    kc_mid_ema: int = 20,
    kc_lower_ema: int = 20,
    kc_atr_period: int = 10,
    kc_upper_mult: float = 2.0,
    kc_lower_mult: float = 2.0,
    stoch_k_period: int = 14,
    stoch_k_smooth: int = 3,
    stoch_d_smooth: int = 3,
    adx_period: int = 14,
    atr_period: int = 14,
    macd_fast: int = 12,
    macd_slow: int = 26,
    macd_signal: int = 9,
    supertrend_period: int = 10,
    supertrend_multiplier: float = 3.0,
    ema_periods: list = None,
    dc_upper_period: int = 20,
    dc_mid_period: int = 20,
    dc_lower_period: int = 20,
    dc_offset: int = 0,
    psar_af_start: float = 0.02,
    psar_af_increment: float = 0.02,
    psar_af_max: float = 0.20,
) -> pd.DataFrame:
    """
    Calculate all indicators on full data.

    Returns:
        DataFrame with all features created
    """

    df = df.copy()

    # -------------------------------------------------
    # RSI
    # -------------------------------------------------
    df["rsi"] = rsi(df["latest"], rsi_window)
    df["rsi_13"] = df["rsi"].rolling(13).mean()
    df["rsi_33"] = df["rsi"].rolling(33).mean()

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
        df["chikou"],
        df["senkou_a_current"],
        df["senkou_b_current"],
    ) = ichimoku(df["high"], df["low"], df["latest"])

    # -------------------------------------------------
    # Bollinger Bands
    # -------------------------------------------------
    (
        df["bb_mid"],
        df["bb_upper"],
        df["bb_lower"],
    ) = bollinger_bands(
        df["latest"],
        bb_upper_period=bb_upper_period,
        bb_upper_stdev=bb_upper_stdev,
        bb_mid_period=bb_mid_period,
        bb_lower_period=bb_lower_period,
        bb_lower_stdev=bb_lower_stdev,
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
        kc_upper_ema=kc_upper_ema,
        kc_mid_ema=kc_mid_ema,
        kc_lower_ema=kc_lower_ema,
        kc_atr_period=kc_atr_period,
        kc_upper_mult=kc_upper_mult,
        kc_lower_mult=kc_lower_mult,
    )

    # -------------------------------------------------
    # Stochastic
    # -------------------------------------------------
    (
        df["stoch_k"],
        df["stoch_d"],
    ) = stochastic(
        df["high"],
        df["low"],
        df["latest"],
        k_period=stoch_k_period,
        k_smooth=stoch_k_smooth,
        d_smooth=stoch_d_smooth,
    )

    # -------------------------------------------------
    # ADX
    # -------------------------------------------------
    (
        df["adx"],
        df["plus_di"],
        df["minus_di"],
    ) = adx(df["high"], df["low"], df["latest"], period=adx_period)

    # -------------------------------------------------
    # ATR
    # -------------------------------------------------
    df["atr"] = atr_indicator(df["high"], df["low"], df["latest"], period=atr_period)

    # -------------------------------------------------
    # MACD
    # -------------------------------------------------
    (
        df["macd_line"],
        df["macd_signal"],
        df["macd_hist"],
    ) = macd(df["latest"], fast_period=macd_fast, slow_period=macd_slow, signal_period=macd_signal)

    # -------------------------------------------------
    # OBV & Accumulation/Distribution (require volume)
    # -------------------------------------------------
    has_volume = "volume" in df.columns and df["volume"].notna().any()
    if has_volume:
        df["obv"] = obv(df["latest"], df["volume"])
        df["acc_dist"] = accumulation_distribution(df["high"], df["low"], df["latest"], df["volume"])
    else:
        df["obv"] = float('nan')
        df["acc_dist"] = float('nan')

    # -------------------------------------------------
    # Supertrend
    # -------------------------------------------------
    (
        df["supertrend"],
        df["supertrend_dir"],
        st_upper,
        st_lower,
    ) = supertrend(df["high"], df["low"], df["latest"],
                   period=supertrend_period, multiplier=supertrend_multiplier)
    df["supertrend_upper"] = st_upper.shift(1)
    df["supertrend_lower"] = st_lower.shift(1)

    # -------------------------------------------------
    # EMA Overlay
    # -------------------------------------------------
    if ema_periods:
        for i, (period, ema_series) in enumerate(ema_overlay(df["latest"], ema_periods)):
            df[f"ema_{i}"] = ema_series
    else:
        ema_periods = []

    # -------------------------------------------------
    # Donchian Channel
    # -------------------------------------------------
    (
        df["dc_upper"],
        df["dc_mid"],
        df["dc_lower"],
    ) = donchian_channel(
        df["high"], df["low"],
        upper_period=dc_upper_period,
        mid_period=dc_mid_period,
        lower_period=dc_lower_period,
        offset=dc_offset,
    )

    # -------------------------------------------------
    # Parabolic SAR
    # -------------------------------------------------
    (
        df["psar"],
        df["psar_dir"],
        psar_up,
        psar_lo,
    ) = parabolic_sar(
        df["high"], df["low"], df["latest"],
        af_start=psar_af_start,
        af_increment=psar_af_increment,
        af_max=psar_af_max,
    )
    df["psar_upper"] = psar_up.shift(1)
    df["psar_lower"] = psar_lo.shift(1)

    return df

_ICHIMOKU_ELEMENTS = {"Tenkan", "Kijun", "Senkou A", "Senkou B", "Chikou"}
_BB_ELEMENTS = {"BB Upper Band", "BB Middle Band", "BB Lower Band"}
_KC_ELEMENTS = {"KC Upper Band", "KC Middle Band", "KC Lower Band"}
_DC_ELEMENTS = {"DC Upper Band", "DC Middle Band", "DC Lower Band"}
_PSAR_ELEMENTS = {"PSAR", "PSAR Upper", "PSAR Lower"}


def strategy_indicator_flags(strategy):
    """Inspect a strategy dict and return which overlay indicators it references.

    Returns a dict with keys: show_ichimoku, show_bb, show_kc, show_donchian, show_psar.
    """
    flags = {
        "show_ichimoku": False,
        "show_bb": False,
        "show_kc": False,
        "show_donchian": False,
        "show_psar": False,
    }

    def _check(name):
        if name in _ICHIMOKU_ELEMENTS:
            flags["show_ichimoku"] = True
        elif name in _BB_ELEMENTS:
            flags["show_bb"] = True
        elif name in _KC_ELEMENTS:
            flags["show_kc"] = True
        elif name in _DC_ELEMENTS:
            flags["show_donchian"] = True
        elif name in _PSAR_ELEMENTS:
            flags["show_psar"] = True

    def _scan_trigger(trigger):
        if not trigger:
            return
        _check(trigger.get("element1", ""))
        _check(trigger.get("element2", ""))

    def _scan_conditions(conditions):
        for cond in (conditions or []):
            _check(cond.get("element1", ""))
            _check(cond.get("element2", ""))

    entry = strategy.get("entry", {})
    _scan_trigger(entry.get("trigger"))
    _scan_conditions(entry.get("conditions"))

    initial_stop = strategy.get("initial_stop", {})
    _check(initial_stop.get("element2", ""))

    for group in strategy.get("exit_groups", []):
        for target in group.get("targets", []):
            _scan_trigger(target.get("trigger"))
            _scan_conditions(target.get("conditions"))
        for stop in group.get("stops", []):
            _scan_trigger(stop.get("trigger"))
            _scan_conditions(stop.get("conditions"))

    return flags


def slice_for_graph(
        df: pd.DataFrame,
        start_date,
        end_date,
        show_ichimoku: bool,
        show_bb: bool,
        show_kc: bool,
        show_donchian: bool = False,
        show_psar: bool = False,
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

    # get_loc may return a slice when there are duplicate index values
    if isinstance(start_pos, slice):
        start_pos = start_pos.start
    if isinstance(end_pos, slice):
        end_pos = end_pos.stop - 1

    ext_start = max(0, start_pos - context_bars)
    ext_end = min(len(df) - 1, end_pos + context_bars)

    df_plot = df.iloc[ext_start : ext_end + 1].copy()

    # -------------------------------------------------
    # Drop NaNs only on required cols
    # -------------------------------------------------
    required_cols = ["latest", "rsi", "rsi_13", "rsi_33", "ci", "ci_13", "ci_33",
                     "stoch_k", "stoch_d", "adx", "plus_di", "minus_di",
                     "atr", "macd_line", "macd_signal", "macd_hist"]

    if show_ichimoku:
        required_cols += ["tenkan", "kijun", "senkou_a", "senkou_b",
                          "senkou_a_current", "senkou_b_current"]
    if show_bb:
        required_cols += ["bb_mid", "bb_upper", "bb_lower"]
    if show_kc:
        required_cols += ["kc_mid", "kc_upper", "kc_lower"]
    if show_donchian:
        required_cols += ["dc_upper", "dc_mid", "dc_lower"]
    if show_psar:
        required_cols += ["psar"]

    df_plot = df_plot.dropna(subset=required_cols)

    # -------------------------------------------------
    # Categorical x-axis helpers
    # -------------------------------------------------
    df_plot["x"] = df_plot.index.strftime("%d.%m.%Y_%H:%M")
    df_plot["date_only"] = df_plot.index.strftime("%d.%m.%Y")

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

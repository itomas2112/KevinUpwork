"""
Configuration constants for the trading platform
"""

STRATEGIES_FILE = "saved_strategies.json"

# Indicator groups for strategy builder
PRICE_AND_INDICATORS = [
    "Price",
    "BB Upper Band",
    "BB Middle Band",
    "BB Lower Band",
    "KC Upper Band",
    "KC Middle Band",
    "KC Lower Band",
    "Tenkan",
    "Kijun",
    "Senkou A",
    "Senkou B",
    "Chikou",
    "Supertrend",
]

ATR_VOLUME_GROUP = [
    "ATR",
    "OBV",
    "Acc/Dist",
]

RSI_GROUP = [
    "RSI",
    "RSI 13 SMA",
    "RSI 33 SMA"
]

CMB_GROUP = [
    "CMB",
    "CMB 13 SMA",
    "CMB 33 SMA"
]

STOCH_GROUP = [
    "Stoch %K",
    "Stoch %D",
]

ADX_GROUP = [
    "ADX",
    "+DI",
    "-DI",
]

MACD_GROUP = [
    "MACD Line",
    "MACD Signal",
    "MACD Histogram",
]

EVENT_TYPES = [
    "Cross",
    "Cross Above",
    "Cross Below",
    "Close",
    "Close Above",
    "Close Below",
]

STOP_EVENT_TYPES = [
    "Cross Above",
    "Cross Below",
    "Close Above",
    "Close Below",
]

CONDITION_OPERATORS = [
    "Above",
    "Below"
]

# Condition comparison types
CONDITION_COMPARE_TYPES = [
    "Indicator",
    "Fixed Value"
]

# Exit types - NEW
EXIT_TYPES = [
    "Target",
    "Stop"
]

# Group names for strategy builder dropdowns
GROUP_NAMES = [
    "Price & Indicators",
    "RSI Group",
    "CMB Group",
    "Stoch Group",
    "ADX Group",
    "MACD Group",
    "ATR / Volume Group",
]

# Group name -> element list mapping
GROUP_MAP = {
    "Price & Indicators": PRICE_AND_INDICATORS,
    "RSI Group": RSI_GROUP,
    "CMB Group": CMB_GROUP,
    "Stoch Group": STOCH_GROUP,
    "ADX Group": ADX_GROUP,
    "MACD Group": MACD_GROUP,
    "ATR / Volume Group": ATR_VOLUME_GROUP,
}


def get_group_elements(group_name, ema_count=0):
    """
    Return the element list for a given group name.
    Dynamically appends EMA entries to Price & Indicators.
    """
    elements = list(GROUP_MAP.get(group_name, []))
    if group_name == "Price & Indicators" and ema_count > 0:
        for i in range(ema_count):
            elements.append(f"EMA {i + 1}")
    return elements


# Indicator mapping for strategy execution
INDICATOR_MAP = {
    "Price": "latest",
    "BB Upper Band": "bb_upper",
    "BB Middle Band": "bb_mid",  # NOT bb_middle
    "BB Lower Band": "bb_lower",
    "KC Upper Band": "kc_upper",
    "KC Middle Band": "kc_mid",  # NOT kc_middle
    "KC Lower Band": "kc_lower",
    "Tenkan": "tenkan",
    "Kijun": "kijun",
    "Senkou A": "senkou_a_current",
    "Senkou B": "senkou_b_current",
    "RSI": "rsi",
    "RSI 13 SMA": "rsi_13",
    "RSI 33 SMA": "rsi_33",
    "CMB": "ci",
    "CMB 13 SMA": "ci_13",
    "CMB 33 SMA": "ci_33",
    "Stoch %K": "stoch_k",
    "Stoch %D": "stoch_d",
    "Chikou": "chikou",
    "ADX": "adx",
    "+DI": "plus_di",
    "-DI": "minus_di",
    "MACD Line": "macd_line",
    "MACD Signal": "macd_signal",
    "MACD Histogram": "macd_hist",
    "ATR": "atr",
    "OBV": "obv",
    "Acc/Dist": "acc_dist",
    "Supertrend": "supertrend",
}


def get_indicator_map(ema_count=0):
    """
    Return the full indicator map, including dynamic EMA entries.
    """
    m = dict(INDICATOR_MAP)
    for i in range(ema_count):
        m[f"EMA {i + 1}"] = f"ema_{i}"
    return m

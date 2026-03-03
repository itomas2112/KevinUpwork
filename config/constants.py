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
    "Senkou B"
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
    "Senkou A": "senkou_a",
    "Senkou B": "senkou_b",
    "RSI": "rsi",
    "RSI 13 SMA": "rsi_13",
    "RSI 33 SMA": "rsi_33",
    "CMB": "ci",
    "CMB 13 SMA": "ci_13",
    "CMB 33 SMA": "ci_33",
    "Stoch %K": "stoch_k",
    "Stoch %D": "stoch_d",
}
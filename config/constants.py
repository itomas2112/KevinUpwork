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

EVENT_TYPES = [
    "Cross",
    "Cross Above",
    "Cross Below",
    "At Level"
]

CONDITION_OPERATORS = [
    "Above",
    "Below"
]

# Indicator mapping for strategy execution
INDICATOR_MAP = {
    "Price": "latest",
    "BB Upper Band": "bb_upper",
    "BB Middle Band": "bb_middle",
    "BB Lower Band": "bb_lower",
    "KC Upper Band": "kc_upper",
    "KC Middle Band": "kc_middle",
    "KC Lower Band": "kc_lower",
    "Tenkan": "tenkan",
    "Kijun": "kijun",
    "Senkou A": "senkou_a",
    "Senkou B": "senkou_b",
    "RSI": "rsi",
    "RSI 13 SMA": "rsi_13_sma",
    "RSI 33 SMA": "rsi_33_sma",
    "CMB": "cmb",
    "CMB 13 SMA": "cmb_13_sma",
    "CMB 33 SMA": "cmb_33_sma",
}
"""
Configuration constants for the trading platform
"""

STRATEGIES_FILE = "saved_strategies.json"

# Default EMA overlay periods (used by strategy builder, grid search, performance, etc.)
DEFAULT_EMA_PERIODS = [10, 20, 50, 200]

# Indicator groups for strategy builder
PRICE_AND_INDICATORS = [
    "Price",
    "Price Upper",
    "Price Lower",
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
    "Supertrend Upper",
    "Supertrend Lower",
    "DC Upper Band",
    "DC Middle Band",
    "DC Lower Band",
    "PSAR",
    "PSAR Upper",
    "PSAR Lower",
    "LR Upper",
    "LR Middle",
    "LR Lower",
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

WILLR_GROUP = [
    "Williams %R",
]

ROC_GROUP = [
    "ROC",
    "ROC Signal",
]

CCI_GROUP = [
    "CCI",
]

# Special exit-only elements (computed at runtime from entry price and R distance)
R_PROFIT_LOSS_ELEMENTS = [
    "R Profit",
    "R Loss",
]

# Special exit-only element: ATR-based fixed target/stop (locked at entry time)
ATR_TARGET_ELEMENTS = [
    "ATR Target",
]

# Special stop-only element: standard Chandelier Exit (level updates each bar
# from the highest-high-since-entry for longs, lowest-low for shorts).
ATR_TRAILING_ELEMENTS = [
    "ATR Trailing",
]

EVENT_TYPES = [
    "Cross",
    "Cross Above",
    "Cross Below",
    "Close",
    "Close Above",
    "Close Below",
]

# "Within last (periods)" for events, 0-based. 0 = current bar only. Stored as
# `within_last` on every trigger/condition/stop dict; the executor treats the
# event as fired if it was true on this bar or any of the previous N bars.
DEFAULT_WITHIN_LAST = 0
MIN_WITHIN_LAST = 0
MAX_WITHIN_LAST = 99

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
    "Williams %R Group",
    "ROC Group",
    "CCI Group",
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
    "Williams %R Group": WILLR_GROUP,
    "ROC Group": ROC_GROUP,
    "CCI Group": CCI_GROUP,
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
    "Supertrend Upper": "supertrend_upper",
    "Supertrend Lower": "supertrend_lower",
    "DC Upper Band": "dc_upper",
    "DC Middle Band": "dc_mid",
    "DC Lower Band": "dc_lower",
    "Price Upper": "pc_upper",
    "Price Lower": "pc_lower",
    "PSAR": "psar",
    "PSAR Upper": "psar_upper",
    "PSAR Lower": "psar_lower",
    "LR Upper": "lr_upper",
    "LR Middle": "lr_mid",
    "LR Lower": "lr_lower",
    "Williams %R": "willr",
    "ROC": "roc",
    "ROC Signal": "roc_signal",
    "CCI": "cci",
}


def get_indicator_map(ema_count=0):
    """
    Return the full indicator map, including dynamic EMA entries.
    """
    m = dict(INDICATOR_MAP)
    for i in range(ema_count):
        m[f"EMA {i + 1}"] = f"ema_{i}"
    return m


# ---------------------------------------------------------------------------
# Walk Forward Optimization — parameter ranges
# ---------------------------------------------------------------------------

# Element display name → WFO param group
# (CMB, Ichimoku, OBV, Acc/Dist are on ignore list — no tunable params for WFO)
ELEMENT_TO_WFO_GROUP = {
    "BB Upper Band": "bb", "BB Middle Band": "bb", "BB Lower Band": "bb",
    "KC Upper Band": "kc", "KC Middle Band": "kc", "KC Lower Band": "kc",
    "RSI": "rsi", "RSI 13 SMA": "rsi", "RSI 33 SMA": "rsi",
    "Stoch %K": "stoch", "Stoch %D": "stoch",
    "ADX": "adx", "+DI": "adx", "-DI": "adx",
    "ATR": "atr",
    "MACD Line": "macd", "MACD Signal": "macd", "MACD Histogram": "macd",
    "Supertrend": "supertrend", "Supertrend Upper": "supertrend", "Supertrend Lower": "supertrend",
    "DC Upper Band": "donchian", "DC Middle Band": "donchian", "DC Lower Band": "donchian",
    "PSAR": "psar", "PSAR Upper": "psar", "PSAR Lower": "psar",
    "Williams %R": "willr",
    "ROC": "roc", "ROC Signal": "roc",
    "CCI": "cci",
    "LR Upper": "lr", "LR Middle": "lr", "LR Lower": "lr",
    # EMA handled dynamically: "EMA 1" → "ema", etc.
}

# Default parameter ranges for WFO.  (min, max, step) or list of discrete values.
# Only optimised groups appear here; CMB / Ichimoku / OBV / Acc-Dist are ignored.
WFO_DEFAULT_RANGES = {
    "ema": {
        # Per-EMA ranges (index 0 = EMA 1, etc.)
        "_ema_ranges": [
            (1, 15, 2),
            (16, 35, 5),
            (36, 125, 10),
            (126, 250, 25),
        ],
    },
    "rsi": {
        "rsi_window": (5, 50, 5),
    },
    "stoch": {
        "stoch_k_period": (5, 50, 10),
        "stoch_k_smooth": (1, 10, 3),
        "stoch_d_smooth": (1, 10, 3),
    },
    "adx": {
        "adx_period": (5, 50, 5),
    },
    "atr": {
        "atr_period": (5, 50, 5),
    },
    "macd": {
        "macd_fast": (5, 50, 10),
        "macd_slow": (10, 50, 10),
        "macd_signal": (5, 50, 10),
    },
    "supertrend": {
        "supertrend_period": (5, 50, 5),
        "supertrend_multiplier": (1.0, 10.0, 2.0),
    },
    "bb": {
        "bb_upper_period": (10, 50, 10),
        "bb_upper_stdev": (1.0, 3.0, 1.0),
        "bb_mid_period": (10, 50, 10),
        "bb_lower_period": (10, 50, 10),
        "bb_lower_stdev": (1.0, 3.0, 1.0),
    },
    "kc": {
        "kc_upper_ema": (10, 50, 10),
        "kc_upper_mult": (1.0, 3.0, 1.0),
        "kc_mid_ema": (10, 50, 10),
        "kc_lower_ema": (10, 50, 10),
        "kc_lower_mult": (1.0, 3.0, 1.0),
    },
    "donchian": {
        "dc_upper_period": (10, 50, 10),
        "dc_mid_period": (10, 50, 10),
        "dc_lower_period": (10, 50, 10),
    },
    "psar": {
        "psar_af_start": (0.005, 0.05, 0.01),
        "psar_af_increment": (0.005, 0.05, 0.01),
        "psar_af_max": (0.1, 0.5, 0.1),
    },
    "willr": {
        "willr_period": (5, 50, 5),
    },
    "roc": {
        "roc_period": (5, 50, 5),
        "roc_signal_period": (5, 50, 5),
    },
    "cci": {
        "cci_period": (5, 50, 5),
    },
    "lr": {
        "lr_period": (10, 150, 10),
        "lr_multiplier": [0.674, 1.0, 1.282, 1.5, 1.645, 1.96, 2.0, 2.58, 2.81],
    },
}


# ── Instruments (sidebar "Instrument" section; used by dollar-based Monte Carlo) ──
from collections import OrderedDict

# symbol -> (min_tick in price units of the data, tick_value $, margin $ per contract)
INSTRUMENTS = OrderedDict([
    ("ES",  (0.25,       12.50,   28000)),
    ("NQ",  (0.25,        5.00,   46000)),
    ("YM",  (1.0,         5.00,   17000)),
    ("RTY", (0.10,        5.00,   12000)),
    ("GC",  (0.10,       10.00,   22000)),
    ("SI",  (0.005,      25.00,   36000)),
    ("HG",  (0.0005,     12.50,   13000)),
    ("PL",  (0.10,        5.00,    9000)),
    ("MBT", (5.0,         0.50,    1800)),
    ("ETH", (0.50,       25.00,   41000)),
    ("6J",  (0.0000005,   6.25,    3100)),
    ("6E",  (0.00005,     6.25,    2300)),
    ("CL",  (0.01,       10.00,    9600)),
    ("NG",  (0.001,      10.00,    3100)),
    ("ZN",  (1/64,       15.625,   2100)),
    ("ZF",  (1/128,       7.8125,  1400)),
    ("ZC",  (0.25,       12.50,    1200)),   # quoted in cents/bushel; 1/4 cent
    ("ZS",  (0.25,       12.50,    2300)),
    ("ZW",  (0.25,       12.50,    2100)),
    ("SB",  (0.0001,     11.20,    1200)),
    ("CT",  (0.0001,      5.00,    1900)),
    ("CC",  (1.0,        10.00,    8300)),
])
DEFAULT_INSTRUMENT = "ES"
MC_DEFAULT_BALANCE = 100_000.0
MC_DEFAULT_TRADES_PER_SIM = 100
# Phased risk sizing (strategies/monte_carlo_core.py::simulate_trades_phased):
# trades 1..MC_INITIAL_TRADES target MC_INITIAL_RISK_PCT (whole contracts must
# land inside MC_INITIAL_RISK_BAND, else the trade is skipped); thereafter every
# MC_REASSESS_EVERY trades the risk % is re-derived from the trades so far by an
# MC_REASSESS_SIMS-sim MC of the next block (capped at the trades remaining)
# targeting MC_TARGET_DD avg max DD.
MC_INITIAL_RISK_PCT = 1.0
MC_INITIAL_RISK_BAND = (0.75, 1.25)
MC_INITIAL_TRADES = 30
MC_REASSESS_EVERY = 200
MC_REASSESS_SIMS = 30
MC_TARGET_DD = 5.0


def instrument_min_tick(symbol):
    """Minimum price increment in the data's price units."""
    return INSTRUMENTS[symbol][0]


def instrument_point_value(symbol):
    """Dollars per 1.0 price unit = tick_value / min_tick (ES -> 50, GC -> 100, ZN -> 1000)."""
    min_tick, tick_value, _margin = INSTRUMENTS[symbol]
    return tick_value / min_tick


def instrument_margin(symbol):
    """Margin in dollars required per contract."""
    return float(INSTRUMENTS[symbol][2])

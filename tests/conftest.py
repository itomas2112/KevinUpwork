"""
Shared fixtures for the Trading Analysis Platform test suite.

Provides:
  - Deterministic synthetic OHLC DataFrames (15m and 1H) in several shapes
  - A strategy factory for building valid strategy dicts programmatically
  - Combo lists for parametrized tests
"""

import pytest
import numpy as np
import pandas as pd
from copy import deepcopy

# ---------------------------------------------------------------------------
# Constants used across tests
# ---------------------------------------------------------------------------

# Representative element per indicator group (element1, element2 for Indicator compare)
GROUP_REPRESENTATIVES = [
    ("Price & Indicators", "Price",     "BB Upper Band"),
    ("RSI Group",          "RSI",       "RSI 13 SMA"),
    ("CMB Group",          "CMB",       "CMB 13 SMA"),
    ("Stoch Group",        "Stoch %K",  "Stoch %D"),
    ("ADX Group",          "ADX",       "+DI"),
    ("MACD Group",         "MACD Line", "MACD Signal"),
    ("ATR / Volume Group", "ATR",       "OBV"),
]

ENTRY_EVENTS = ["Cross", "Cross Above", "Cross Below", "Close", "Close Above", "Close Below"]
STOP_EVENTS  = ["Cross Above", "Cross Below", "Close Above", "Close Below"]
CONDITION_OPERATORS = ["Above", "Below"]

# Fixed values that make sense per group (so the strategy can actually trigger)
GROUP_FIXED_VALUES = {
    "Price & Indicators": 150.0,
    "RSI Group":          50.0,
    "CMB Group":          50.0,
    "Stoch Group":        50.0,
    "ADX Group":          25.0,
    "MACD Group":         0.0,
    "ATR / Volume Group": 1.0,
}

# Default indicator params matching calculate_indicators defaults
DEFAULT_INDICATOR_SETTINGS = {
    "rsi_window": 14,
    "bb_upper_period": 20, "bb_upper_stdev": 2.0, "bb_mid_period": 20,
    "bb_lower_period": 20, "bb_lower_stdev": 2.0,
    "kc_upper_ema": 20, "kc_mid_ema": 20, "kc_lower_ema": 20,
    "kc_atr_period": 10, "kc_upper_mult": 2.0, "kc_lower_mult": 2.0,
    "stoch_k_period": 14, "stoch_k_smooth": 3, "stoch_d_smooth": 3,
    "adx_period": 14, "atr_period": 14,
    "macd_fast": 12, "macd_slow": 26, "macd_signal": 9,
    "supertrend_period": 10, "supertrend_multiplier": 3.0,
    "ema_periods": [],
    "dc_upper_period": 20, "dc_mid_period": 20, "dc_lower_period": 20, "dc_offset": 0,
    "psar_af_start": 0.02, "psar_af_increment": 0.02, "psar_af_max": 0.20,
}


# ---------------------------------------------------------------------------
# Synthetic OHLC data generators
# ---------------------------------------------------------------------------

def _make_ohlc(closes, freq="15min", start="2024-01-01 00:00:00", volume=True):
    """
    Build a DataFrame from a sequence of close prices.

    Generates realistic open/high/low from close using small deterministic offsets.
    """
    n = len(closes)
    times = pd.date_range(start=start, periods=n, freq=freq)

    closes = np.array(closes, dtype=float)
    opens = np.roll(closes, 1)
    opens[0] = closes[0]

    # High is max(open, close) + small offset; Low is min(open, close) - small offset
    np.random.seed(42)
    noise = np.random.uniform(0.2, 1.0, size=n)

    highs = np.maximum(opens, closes) + noise
    lows  = np.minimum(opens, closes) - noise

    data = {
        "open":   opens,
        "high":   highs,
        "low":    lows,
        "latest": closes,
    }
    if volume:
        data["volume"] = np.random.randint(100, 10000, size=n).astype(float)

    df = pd.DataFrame(data, index=times)
    df.index.name = "time"
    return df


def _generate_uptrend(n=200, start_price=100.0, step=0.5, **kwargs):
    """Steady uptrend: price goes from start_price upward."""
    closes = [start_price + i * step for i in range(n)]
    return _make_ohlc(closes, **kwargs)


def _generate_downtrend(n=200, start_price=200.0, step=0.5, **kwargs):
    """Steady downtrend: price goes from start_price downward."""
    closes = [start_price - i * step for i in range(n)]
    return _make_ohlc(closes, **kwargs)


def _generate_flat(n=200, price=150.0, amplitude=0.5, **kwargs):
    """Flat/sideways: small sine wave around a fixed price."""
    np.random.seed(99)
    closes = [price + amplitude * np.sin(i * 0.1) for i in range(n)]
    return _make_ohlc(closes, **kwargs)


def _generate_v_shape(n=200, start_price=200.0, bottom_price=120.0, **kwargs):
    """
    V-shaped reversal: goes down to bottom then back up.
    The reversal point is at n//2.
    """
    mid = n // 2
    down_step = (start_price - bottom_price) / mid
    up_step   = (start_price - bottom_price) / (n - mid)

    closes = []
    for i in range(mid):
        closes.append(start_price - i * down_step)
    for i in range(n - mid):
        closes.append(bottom_price + i * up_step)

    return _make_ohlc(closes, **kwargs)


def _generate_oscillation(n=200, center=150.0, amplitude=30.0, period=40, **kwargs):
    """
    Repeating oscillation (sine wave) for predictable crossovers.
    Full cycle every `period` bars.
    """
    closes = [center + amplitude * np.sin(2 * np.pi * i / period) for i in range(n)]
    return _make_ohlc(closes, **kwargs)


# ---------------------------------------------------------------------------
# Fixtures — 15-minute data
# ---------------------------------------------------------------------------

@pytest.fixture
def df_uptrend_15m():
    return _generate_uptrend(freq="15min")

@pytest.fixture
def df_downtrend_15m():
    return _generate_downtrend(freq="15min")

@pytest.fixture
def df_flat_15m():
    return _generate_flat(freq="15min")

@pytest.fixture
def df_v_shape_15m():
    return _generate_v_shape(freq="15min")

@pytest.fixture
def df_oscillation_15m():
    return _generate_oscillation(freq="15min")


# ---------------------------------------------------------------------------
# Fixtures — 1-hour data
# ---------------------------------------------------------------------------

@pytest.fixture
def df_uptrend_1h():
    return _generate_uptrend(freq="1h")

@pytest.fixture
def df_downtrend_1h():
    return _generate_downtrend(freq="1h")

@pytest.fixture
def df_flat_1h():
    return _generate_flat(freq="1h")

@pytest.fixture
def df_v_shape_1h():
    return _generate_v_shape(freq="1h")

@pytest.fixture
def df_oscillation_1h():
    return _generate_oscillation(freq="1h")


# ---------------------------------------------------------------------------
# Convenience fixture: all shapes at once (for parametrize)
# ---------------------------------------------------------------------------

@pytest.fixture(params=["uptrend", "downtrend", "flat", "v_shape", "oscillation"],
                ids=lambda s: s)
def df_all_shapes_15m(request):
    generators = {
        "uptrend":     _generate_uptrend,
        "downtrend":   _generate_downtrend,
        "flat":        _generate_flat,
        "v_shape":     _generate_v_shape,
        "oscillation": _generate_oscillation,
    }
    return generators[request.param](freq="15min")


@pytest.fixture(params=["uptrend", "downtrend", "flat", "v_shape", "oscillation"],
                ids=lambda s: s)
def df_all_shapes_1h(request):
    generators = {
        "uptrend":     _generate_uptrend,
        "downtrend":   _generate_downtrend,
        "flat":        _generate_flat,
        "v_shape":     _generate_v_shape,
        "oscillation": _generate_oscillation,
    }
    return generators[request.param](freq="1h")


# ---------------------------------------------------------------------------
# Pre-calculated fixtures (indicators already computed)
# ---------------------------------------------------------------------------

@pytest.fixture
def df_oscillation_15m_with_indicators():
    from indicators.calculate_indicators import calculate_indicators
    df = _generate_oscillation(freq="15min")
    return calculate_indicators(df, **DEFAULT_INDICATOR_SETTINGS)

@pytest.fixture
def df_v_shape_15m_with_indicators():
    from indicators.calculate_indicators import calculate_indicators
    df = _generate_v_shape(freq="15min")
    return calculate_indicators(df, **DEFAULT_INDICATOR_SETTINGS)

@pytest.fixture
def df_uptrend_15m_with_indicators():
    from indicators.calculate_indicators import calculate_indicators
    df = _generate_uptrend(freq="15min")
    return calculate_indicators(df, **DEFAULT_INDICATOR_SETTINGS)

@pytest.fixture
def df_downtrend_15m_with_indicators():
    from indicators.calculate_indicators import calculate_indicators
    df = _generate_downtrend(freq="15min")
    return calculate_indicators(df, **DEFAULT_INDICATOR_SETTINGS)

@pytest.fixture
def df_flat_15m_with_indicators():
    from indicators.calculate_indicators import calculate_indicators
    df = _generate_flat(freq="15min")
    return calculate_indicators(df, **DEFAULT_INDICATOR_SETTINGS)


# ---------------------------------------------------------------------------
# Strategy factory
# ---------------------------------------------------------------------------

def make_strategy(
    # Basic
    direction="Long",
    strategy_name="Test Strategy",
    max_positions=1,
    # Entry trigger
    entry_group="RSI Group",
    entry_element1="RSI",
    entry_event="Cross Above",
    entry_compare_type="Fixed Value",
    entry_element2=None,
    entry_value=30.0,
    position_size=1.0,
    # Entry conditions (list of dicts)
    entry_conditions=None,
    # Initial stop
    initial_stop_type="ATR",          # "Indicator" or "ATR"
    initial_stop_event="Cross Below",
    initial_stop_element2="BB Lower Band",  # used when stop_type == "Indicator"
    initial_stop_atr_period=14,       # used when stop_type == "ATR"
    initial_stop_atr_multiplier=1.5,  # used when stop_type == "ATR"
    # Exit groups (list of dicts, or None for a sensible default)
    exit_groups=None,
    # Indicator settings override
    indicator_settings=None,
):
    """
    Build a valid strategy dict matching the format used by save_strategy_to_session.

    If exit_groups is None, creates one group at 100% with an R Profit target at 2.0.
    """
    # Entry trigger
    trigger = {
        "group": entry_group,
        "element1": entry_element1,
        "event": entry_event,
        "compare_type": entry_compare_type,
        "element2": entry_element2 if entry_compare_type == "Indicator" else None,
        "value": entry_value if entry_compare_type == "Fixed Value" else None,
    }

    # Initial stop
    if initial_stop_type == "ATR":
        initial_stop = {
            "element1": "Price",
            "stop_type": "ATR",
            "event": initial_stop_event,
            "atr_period": initial_stop_atr_period,
            "atr_multiplier": initial_stop_atr_multiplier,
        }
    else:
        initial_stop = {
            "element1": "Price",
            "stop_type": "Indicator",
            "event": initial_stop_event,
            "compare_type": "Indicator",
            "element2": initial_stop_element2,
        }

    # Entry conditions
    conditions = []
    if entry_conditions:
        for cond in entry_conditions:
            conditions.append({
                "group": cond.get("group", "RSI Group"),
                "element1": cond.get("element1", "RSI"),
                "operator": cond.get("operator", "Above"),
                "compare_type": cond.get("compare_type", "Fixed Value"),
                "element2": cond.get("element2") if cond.get("compare_type") == "Indicator" else None,
                "value": cond.get("value", 50.0) if cond.get("compare_type", "Fixed Value") == "Fixed Value" else None,
            })

    # Default exit group: R Profit target at 2.0R
    if exit_groups is None:
        exit_groups = [
            {
                "group_id": 1,
                "allocation_pct": 100.0,
                "targets": [
                    {
                        "type": "Target",
                        "trigger": {
                            "group": "R Profit / R Loss",
                            "element1": "R Profit",
                            "event": "Cross Above",
                            "compare_type": "Fixed Value",
                            "element2": None,
                            "value": 2.0,
                        },
                        "conditions": [],
                    }
                ],
                "stops": [],
            }
        ]

    strategy = {
        "strategy_name": strategy_name,
        "direction": direction,
        "patterns": [],
        "max_positions": max_positions,
        "created_at": "2024-01-01 00:00:00",
        "entry": {
            "trigger": trigger,
            "position_size": position_size,
            "conditions_count": len(conditions),
            "conditions": conditions,
        },
        "initial_stop": initial_stop,
        "exit_groups": deepcopy(exit_groups),
        "indicator_settings": deepcopy(indicator_settings or DEFAULT_INDICATOR_SETTINGS),
    }

    return strategy


# ---------------------------------------------------------------------------
# Exit group helpers
# ---------------------------------------------------------------------------

def make_exit_group(
    allocation_pct=100.0,
    group_id=1,
    targets=None,
    stops=None,
):
    """Build a single exit group dict."""
    return {
        "group_id": group_id,
        "allocation_pct": allocation_pct,
        "targets": targets or [],
        "stops": stops or [],
    }


def make_exit_trigger(
    group="R Profit / R Loss",
    element1="R Profit",
    event="Cross Above",
    compare_type="Fixed Value",
    element2=None,
    value=2.0,
    atr_period=None,
    atr_multiplier=None,
):
    """Build a single exit target or stop dict."""
    trigger = {
        "group": group,
        "element1": element1,
        "event": event,
        "compare_type": compare_type,
        "element2": element2 if compare_type == "Indicator" else None,
        "value": value if compare_type == "Fixed Value" else None,
    }
    if element1 == "ATR Target":
        trigger["atr_period"] = atr_period or 14
        trigger["atr_multiplier"] = atr_multiplier or 2.0
    return {"type": "Target", "trigger": trigger, "conditions": []}


def make_exit_stop(
    group="Price & Indicators",
    element1="Price",
    event="Cross Below",
    compare_type="Indicator",
    element2="BB Lower Band",
    value=None,
):
    """Build a single dynamic stop dict."""
    trigger = {
        "group": group,
        "element1": element1,
        "event": event,
        "compare_type": compare_type,
        "element2": element2 if compare_type == "Indicator" else None,
        "value": value if compare_type == "Fixed Value" else None,
    }
    return {"type": "Stop", "trigger": trigger, "conditions": []}


# ---------------------------------------------------------------------------
# Parametrize combo builders
# ---------------------------------------------------------------------------

def build_entry_trigger_combos():
    """84 combos: 7 groups x 6 events x 2 compare types."""
    combos = []
    for group_name, el1, el2 in GROUP_REPRESENTATIVES:
        fixed_val = GROUP_FIXED_VALUES[group_name]
        for event in ENTRY_EVENTS:
            # Indicator compare
            combos.append(pytest.param(
                group_name, el1, el2, event, "Indicator", None,
                id=f"{el1}-{event}-vs-{el2}"
            ))
            # Fixed Value compare
            combos.append(pytest.param(
                group_name, el1, None, event, "Fixed Value", fixed_val,
                id=f"{el1}-{event}-vs-Fixed({fixed_val})"
            ))
    return combos


def build_initial_stop_combos():
    """8 combos: 4 events x 2 stop types (Indicator, ATR)."""
    combos = []
    for event in STOP_EVENTS:
        # Indicator stop
        combos.append(pytest.param(
            "Indicator", event, "BB Lower Band", None, None,
            id=f"Stop-Indicator-{event}"
        ))
        # ATR stop
        combos.append(pytest.param(
            "ATR", event, None, 14, 1.5,
            id=f"Stop-ATR-{event}"
        ))
    return combos


def build_entry_condition_combos():
    """28 combos: 7 groups x 2 operators x 2 compare types."""
    combos = []
    for group_name, el1, el2 in GROUP_REPRESENTATIVES:
        fixed_val = GROUP_FIXED_VALUES[group_name]
        for operator in CONDITION_OPERATORS:
            # Indicator compare
            combos.append(pytest.param(
                group_name, el1, operator, "Indicator", el2, None,
                id=f"Cond-{el1}-{operator}-{el2}"
            ))
            # Fixed Value compare
            combos.append(pytest.param(
                group_name, el1, operator, "Fixed Value", None, fixed_val,
                id=f"Cond-{el1}-{operator}-Fixed({fixed_val})"
            ))
    return combos


def build_exit_target_combos():
    """
    102 combos:
      7 groups x 6 events x 2 compare types = 84
      + R Profit x 6 events = 6
      + R Loss x 6 events = 6
      + ATR Target x 6 events = 6
    """
    combos = []

    # Regular groups
    for group_name, el1, el2 in GROUP_REPRESENTATIVES:
        fixed_val = GROUP_FIXED_VALUES[group_name]
        for event in ENTRY_EVENTS:
            combos.append(pytest.param(
                group_name, el1, event, "Indicator", el2, None,
                id=f"Target-{el1}-{event}-vs-{el2}"
            ))
            combos.append(pytest.param(
                group_name, el1, event, "Fixed Value", None, fixed_val,
                id=f"Target-{el1}-{event}-vs-Fixed({fixed_val})"
            ))

    # R Profit
    for event in ENTRY_EVENTS:
        combos.append(pytest.param(
            "R Profit / R Loss", "R Profit", event, "Fixed Value", None, 2.0,
            id=f"Target-RProfit-{event}"
        ))

    # R Loss
    for event in ENTRY_EVENTS:
        combos.append(pytest.param(
            "R Profit / R Loss", "R Loss", event, "Fixed Value", None, 1.0,
            id=f"Target-RLoss-{event}"
        ))

    # ATR Target
    for event in ENTRY_EVENTS:
        combos.append(pytest.param(
            "ATR Target", "ATR Target", event, "ATR", None, None,
            id=f"Target-ATRTarget-{event}"
        ))

    return combos


def build_exit_stop_combos():
    """
    68 combos:
      7 groups x 4 events x 2 compare types = 56
      + R Profit x 4 events = 4
      + R Loss x 4 events = 4
      + ATR Target x 4 events = 4
    """
    combos = []

    # Regular groups
    for group_name, el1, el2 in GROUP_REPRESENTATIVES:
        fixed_val = GROUP_FIXED_VALUES[group_name]
        for event in STOP_EVENTS:
            combos.append(pytest.param(
                group_name, el1, event, "Indicator", el2, None,
                id=f"DynStop-{el1}-{event}-vs-{el2}"
            ))
            combos.append(pytest.param(
                group_name, el1, event, "Fixed Value", None, fixed_val,
                id=f"DynStop-{el1}-{event}-vs-Fixed({fixed_val})"
            ))

    # R Profit
    for event in STOP_EVENTS:
        combos.append(pytest.param(
            "R Profit / R Loss", "R Profit", event, "Fixed Value", None, 2.0,
            id=f"DynStop-RProfit-{event}"
        ))

    # R Loss
    for event in STOP_EVENTS:
        combos.append(pytest.param(
            "R Profit / R Loss", "R Loss", event, "Fixed Value", None, 1.0,
            id=f"DynStop-RLoss-{event}"
        ))

    # ATR Target
    for event in STOP_EVENTS:
        combos.append(pytest.param(
            "ATR Target", "ATR Target", event, "ATR", None, None,
            id=f"DynStop-ATRTarget-{event}"
        ))

    return combos


# Pre-build combo lists so they can be imported by test files
ENTRY_TRIGGER_COMBOS    = build_entry_trigger_combos()
INITIAL_STOP_COMBOS     = build_initial_stop_combos()
ENTRY_CONDITION_COMBOS  = build_entry_condition_combos()
EXIT_TARGET_COMBOS      = build_exit_target_combos()
EXIT_STOP_COMBOS        = build_exit_stop_combos()

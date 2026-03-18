"""
Grid Search helpers — strategy construction, candidate management, labels.
"""
import copy


# ---------------------------------------------------------------------------
# Default skeleton
# ---------------------------------------------------------------------------

def build_skeleton_strategy(direction, indicator_settings):
    """Return the default Training Strategy skeleton.

    Long:  2×ATR static stop below entry, 2×ATR target above entry.
    Short: 2×ATR static stop above entry, 2×ATR target below entry.
    """
    is_long = direction == "Long"

    return {
        "strategy_name": "Grid Search",
        "direction": direction,
        "patterns": [],
        "max_positions": 1,
        "entry": {
            "trigger": None,
            "position_size": 1.0,
            "conditions_count": 0,
            "conditions": [],
        },
        "initial_stop": {
            "stop_type": "ATR",
            "atr_period": 14,
            "atr_multiplier": 2.0,
        },
        "exit_groups": [{
            "group_id": 1,
            "allocation_pct": 100.0,
            "targets": [{
                "type": "Target",
                "trigger": {
                    "group": "Price & Indicators",
                    "element1": "ATR Target",
                    "event": "Cross Above" if is_long else "Cross Below",
                    "compare_type": "Indicator",
                    "element2": None,
                    "value": None,
                    "atr_period": 14,
                    "atr_multiplier": 2.0,
                },
                "conditions": [],
            }],
            "stops": [],
        }],
        "indicator_settings": dict(indicator_settings),
    }


# ---------------------------------------------------------------------------
# Candidate → full strategy dict
# ---------------------------------------------------------------------------

def build_candidate_strategy(base_strategy, step, candidate_config):
    """Return a complete strategy dict with *candidate_config* inserted at *step*.

    Steps:
        1 – entry trigger
        2 – entry condition (single)
        3 – dynamic stop (exit_groups[0].stops)
        4 – static / initial stop
        5 – target (exit_groups[0].targets)
    """
    strategy = copy.deepcopy(base_strategy)

    if step == 1:
        strategy["entry"]["trigger"] = candidate_config
    elif step == 2:
        if candidate_config is not None:
            strategy["entry"]["conditions"] = [candidate_config]
            strategy["entry"]["conditions_count"] = 1
        else:
            strategy["entry"]["conditions"] = []
            strategy["entry"]["conditions_count"] = 0
    elif step == 3:
        if candidate_config is not None:
            strategy["exit_groups"][0]["stops"] = [{
                "type": "Stop",
                "trigger": candidate_config,
                "conditions": [],
            }]
        else:
            strategy["exit_groups"][0]["stops"] = []
    elif step == 4:
        strategy["initial_stop"] = candidate_config
    elif step == 5:
        if candidate_config is not None:
            strategy["exit_groups"][0]["targets"] = [{
                "type": "Target",
                "trigger": candidate_config,
                "conditions": [],
            }]
        else:
            strategy["exit_groups"][0]["targets"] = []

    return strategy


# ---------------------------------------------------------------------------
# Readable label for a candidate
# ---------------------------------------------------------------------------

def candidate_label(config, step):
    """Generate a short human-readable label for a candidate dict."""
    if config is None:
        return "None"

    if step == 4:  # static / initial stop
        st_type = config.get("stop_type", "Indicator")
        if st_type == "ATR":
            return f"ATR({config.get('atr_period', 14)}) × {config.get('atr_multiplier', 2.0)}"
        e2 = config.get("element2", "?")
        return f"Price vs {e2}"

    # Steps 1, 2, 3, 5 all share a trigger-like shape
    e1 = config.get("element1", "?")

    if step == 2:
        op = config.get("operator", "?")
    else:
        op = config.get("event", "?")

    cmp = config.get("compare_type", "Indicator")
    if cmp == "Indicator":
        e2 = config.get("element2", "?")
        return f"{e1} {op} {e2}"
    else:
        val = config.get("value", "?")
        return f"{e1} {op} {val}"


# ---------------------------------------------------------------------------
# Collect indicator settings from gs_ prefixed session state
# ---------------------------------------------------------------------------

def collect_gs_indicator_settings(ss):
    """Read gs_ prefixed indicator params from Streamlit session state."""
    pfx = "gs_"
    keys = [
        "rsi_window",
        "bb_upper_period", "bb_upper_stdev", "bb_mid_period",
        "bb_lower_period", "bb_lower_stdev",
        "kc_upper_ema", "kc_mid_ema", "kc_lower_ema",
        "kc_atr_period", "kc_upper_mult", "kc_lower_mult",
        "stoch_k_period", "stoch_k_smooth", "stoch_d_smooth",
        "adx_period", "atr_period",
        "macd_fast", "macd_slow", "macd_signal",
        "supertrend_period", "supertrend_multiplier",
        "ema_periods",
        "dc_upper_period", "dc_mid_period", "dc_lower_period", "dc_offset",
        "psar_af_start", "psar_af_increment", "psar_af_max",
    ]
    from ui.performance_tab import _DEFAULT_INDICATOR_PARAMS
    result = dict(_DEFAULT_INDICATOR_PARAMS)
    for k in keys:
        val = ss.get(f"{pfx}{k}")
        if val is not None:
            result[k] = val
    return result

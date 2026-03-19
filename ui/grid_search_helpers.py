"""
Grid Search helpers — strategy replacement logic, candidate labeling,
cross-combination generation.
"""
import copy


# ---------------------------------------------------------------------------
# Format candidate label: "Element1 (action) Element2"
# ---------------------------------------------------------------------------

def format_candidate_label(candidate, group_type):
    """Generate label in the format: Element1 (action) Element2.

    - First element capitalized, action lower case in ( ), second element capitalized.
    - ATR stops: "ATR(period) x multiplier"
    """
    if candidate is None:
        return "None"

    # ATR static stop
    if group_type == "static_stop" and candidate.get("stop_type") == "ATR":
        return f"ATR({candidate.get('atr_period', 14)}) x {candidate.get('atr_multiplier', 2.0)}"

    e1 = candidate.get("element1", "?")

    # ATR Target element
    if e1 == "ATR Target":
        action = candidate.get("event", "?").lower()
        atr_p = candidate.get("atr_period", 14)
        atr_m = candidate.get("atr_multiplier", 2.0)
        return f"ATR Target({atr_p} x {atr_m}) ({action})"

    # Conditions use operator, others use event
    if group_type == "condition":
        action = candidate.get("operator", "?").lower()
    else:
        action = candidate.get("event", "?").lower()

    compare = candidate.get("compare_type", "Indicator")
    if compare == "Indicator":
        e2 = candidate.get("element2", "?")
        return f"{e1} ({action}) {e2}"
    else:
        val = candidate.get("value", "?")
        return f"{e1} ({action}) {val}"


# ---------------------------------------------------------------------------
# Build a replacement strategy from a base + candidate
# ---------------------------------------------------------------------------

def build_replacement_strategy(base_strategy, search_group, candidate, extra_condition=None):
    """Deep-copy base strategy and insert candidate at the right component.

    search_group: 'trigger' | 'condition' | 'static_stop' | 'dynamic_stop' | 'target'
    extra_condition: optional condition dict to APPEND to entry conditions (for cross-combo).
    """
    strategy = copy.deepcopy(base_strategy)

    if search_group == "trigger":
        strategy["entry"]["trigger"] = _candidate_to_entry_trigger(candidate)

    elif search_group == "condition":
        # APPEND to existing conditions (never replace)
        cond = _candidate_to_condition(candidate)
        strategy["entry"]["conditions"].append(cond)
        strategy["entry"]["conditions_count"] = len(strategy["entry"]["conditions"])

    elif search_group == "static_stop":
        strategy["initial_stop"] = _candidate_to_static_stop(candidate, strategy["direction"])

    elif search_group == "dynamic_stop":
        trigger_dict = _candidate_to_exit_trigger(candidate)
        strategy["exit_groups"][0]["stops"] = [{
            "type": "Stop",
            "trigger": trigger_dict,
            "conditions": [],
        }]

    elif search_group == "target":
        trigger_dict = _candidate_to_exit_trigger(candidate)
        strategy["exit_groups"][0]["targets"] = [{
            "type": "Target",
            "trigger": trigger_dict,
            "conditions": [],
        }]

    # Append extra condition if provided (cross-combination)
    if extra_condition is not None:
        cond = _candidate_to_condition(extra_condition)
        strategy["entry"]["conditions"].append(cond)
        strategy["entry"]["conditions_count"] = len(strategy["entry"]["conditions"])

    return strategy


# ---------------------------------------------------------------------------
# Generate all run configs (handles cross-combination)
# ---------------------------------------------------------------------------

def generate_run_configs(base_strategy, search_group, search_candidates,
                         condition_candidates=None):
    """Return list of (label, strategy_dict) pairs.

    - trigger, condition, static_stop: one run per candidate.
    - target, dynamic_stop: cross-combine with condition candidates if provided.
      Each search candidate runs standalone + with each condition candidate added.
    """
    runs = []

    if search_group in ("trigger", "condition", "static_stop"):
        for candidate in search_candidates:
            label = format_candidate_label(candidate, search_group)
            strategy = build_replacement_strategy(base_strategy, search_group, candidate)
            runs.append((label, strategy))

    elif search_group in ("target", "dynamic_stop"):
        for candidate in search_candidates:
            cand_label = format_candidate_label(candidate, search_group)

            # Standalone run (no extra condition)
            strategy = build_replacement_strategy(base_strategy, search_group, candidate)
            runs.append((cand_label, strategy))

            # Cross-combination with each condition candidate
            if condition_candidates:
                for cond_cand in condition_candidates:
                    strategy = build_replacement_strategy(
                        base_strategy, search_group, candidate,
                        extra_condition=cond_cand)
                    cond_label = format_candidate_label(cond_cand, "condition")
                    combined_label = f"{cand_label} + {cond_label}"
                    runs.append((combined_label, strategy))

    # Deduplicate labels
    seen = {}
    deduped = []
    for label, strat in runs:
        if label in seen:
            seen[label] += 1
            deduped.append((f"{label} ({seen[label]})", strat))
        else:
            seen[label] = 1
            deduped.append((label, strat))

    return deduped


# ---------------------------------------------------------------------------
# Candidate conversion helpers
# ---------------------------------------------------------------------------

def _candidate_to_entry_trigger(candidate):
    """Convert a group set candidate dict to an entry trigger dict."""
    return {
        "group": candidate.get("group", "Price & Indicators"),
        "element1": candidate.get("element1"),
        "event": candidate.get("event"),
        "compare_type": candidate.get("compare_type", "Indicator"),
        "element2": candidate.get("element2"),
        "value": candidate.get("value"),
    }


def _candidate_to_condition(candidate):
    """Convert a group set candidate dict to an entry condition dict."""
    return {
        "group": candidate.get("group", "Price & Indicators"),
        "element1": candidate.get("element1"),
        "operator": candidate.get("operator", "Above"),
        "compare_type": candidate.get("compare_type", "Indicator"),
        "element2": candidate.get("element2"),
        "value": candidate.get("value"),
    }


def _candidate_to_static_stop(candidate, direction):
    """Convert a group set candidate dict to an initial_stop dict.

    Ensures element1 and event are always present (the ATR fix).
    """
    stop_type = candidate.get("stop_type", "Indicator")
    default_event = "Cross Below" if direction == "Long" else "Cross Above"

    if stop_type == "ATR":
        return {
            "element1": "Price",
            "stop_type": "ATR",
            "event": candidate.get("event", default_event),
            "atr_period": candidate.get("atr_period", 14),
            "atr_multiplier": candidate.get("atr_multiplier", 2.0),
        }
    else:
        return {
            "stop_type": "Indicator",
            "group": candidate.get("group", "Price & Indicators"),
            "element1": candidate.get("element1", "Price"),
            "element2": candidate.get("element2"),
            "event": candidate.get("event", default_event),
            "compare_type": candidate.get("compare_type", "Indicator"),
        }


def _candidate_to_exit_trigger(candidate):
    """Convert a group set candidate dict to an exit trigger dict (target or dynamic stop)."""
    result = {
        "group": candidate.get("group", "Price & Indicators"),
        "element1": candidate.get("element1"),
        "event": candidate.get("event"),
        "compare_type": candidate.get("compare_type", "Indicator"),
        "element2": candidate.get("element2"),
        "value": candidate.get("value"),
    }
    # ATR Target needs period and multiplier
    if candidate.get("element1") == "ATR Target":
        result["atr_period"] = candidate.get("atr_period", 14)
        result["atr_multiplier"] = candidate.get("atr_multiplier", 2.0)
    return result


# ---------------------------------------------------------------------------
# Collect gs_ prefixed indicator settings from session state
# ---------------------------------------------------------------------------

def collect_gs_indicator_settings(ss):
    """Read gs_ prefixed indicator params from Streamlit session state."""
    from ui.performance_tab import _DEFAULT_INDICATOR_PARAMS
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
    result = dict(_DEFAULT_INDICATOR_PARAMS)
    for k in keys:
        val = ss.get(f"{pfx}{k}")
        if val is not None:
            result[k] = val
    return result

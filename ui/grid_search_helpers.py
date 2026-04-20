"""
Grid Search helpers — strategy replacement logic, candidate labeling,
run generation for universal group sets.
"""
import copy

from config.constants import R_PROFIT_LOSS_ELEMENTS, ATR_TARGET_ELEMENTS
from strategies.strategy_validator import validate_strategy


# ---------------------------------------------------------------------------
# Event → Operator mapping (for converting events to condition operators)
# ---------------------------------------------------------------------------

_EVENT_TO_OPERATOR = {
    "Cross Above": "Above",
    "Cross Below": "Below",
    "Close Above": "Above",
    "Close Below": "Below",
    "Cross": "Above",      # default fallback
    "Close": "Above",      # default fallback
    "Above": "Above",      # pass-through for condition operators
    "Below": "Below",
}


# ---------------------------------------------------------------------------
# Format candidate label (no event — universal candidates)
# ---------------------------------------------------------------------------

def format_candidate_label(candidate):
    """Generate label for a universal candidate: 'Element1 vs Element2' or 'Element1 vs value'.

    No event/operator in the label since universal candidates don't store them.
    """
    if candidate is None:
        return "None"

    # ATR static stop
    if candidate.get("stop_type") == "ATR":
        return f"ATR({candidate.get('atr_period', 14)}) x {candidate.get('atr_multiplier', 2.0)}"

    e1 = candidate.get("element1", "?")

    # ATR Target element
    if e1 == "ATR Target":
        atr_p = candidate.get("atr_period", 14)
        atr_m = candidate.get("atr_multiplier", 2.0)
        return f"ATR Target({atr_p} x {atr_m})"

    compare = candidate.get("compare_type", "Indicator")
    if compare == "Indicator":
        e2 = candidate.get("element2", "?")
        return f"{e1} vs {e2}"
    else:
        val = candidate.get("value", "?")
        return f"{e1} vs {val}"


def format_run_label(candidate, event):
    """Generate label for a grid search run: 'Element1 (event) Element2'.

    Includes the event since runs have a specific event applied.
    """
    if candidate is None:
        return "None"

    action = event.lower() if event else "?"

    # ATR static stop
    if candidate.get("stop_type") == "ATR":
        return f"ATR({candidate.get('atr_period', 14)}) x {candidate.get('atr_multiplier', 2.0)} ({action})"

    e1 = candidate.get("element1", "?")

    # ATR Target element
    if e1 == "ATR Target":
        atr_p = candidate.get("atr_period", 14)
        atr_m = candidate.get("atr_multiplier", 2.0)
        return f"ATR Target({atr_p} x {atr_m}) ({action})"

    compare = candidate.get("compare_type", "Indicator")
    if compare == "Indicator":
        e2 = candidate.get("element2", "?")
        return f"{e1} ({action}) {e2}"
    else:
        val = candidate.get("value", "?")
        return f"{e1} ({action}) {val}"


# ---------------------------------------------------------------------------
# Build a replacement strategy from a base + candidate + event
# ---------------------------------------------------------------------------

def build_replacement_strategy(base_strategy, search_group, candidate, event,
                               extra_condition=None, extra_condition_event=None):
    """Deep-copy base strategy and insert candidate at the right component.

    search_group: 'trigger' | 'condition' | 'static_stop' | 'dynamic_stop' | 'target'
    event: the event/operator to apply to the candidate
    extra_condition: optional condition candidate to APPEND to entry conditions (for cross-combo).
    extra_condition_event: event/operator for the extra condition.
    """
    strategy = copy.deepcopy(base_strategy)

    if search_group == "trigger":
        strategy["entry"]["trigger"] = _candidate_to_entry_trigger(candidate, event)

    elif search_group == "condition":
        cond = _candidate_to_condition(candidate, event)
        strategy["entry"]["conditions"].append(cond)
        strategy["entry"]["conditions_count"] = len(strategy["entry"]["conditions"])

    elif search_group == "static_stop":
        strategy["initial_stop"] = _candidate_to_static_stop(candidate, strategy["direction"], event)

    elif search_group == "dynamic_stop":
        trigger_dict = _candidate_to_exit_trigger(candidate, event)
        strategy["exit_groups"][0]["stops"] = [{
            "type": "Stop",
            "trigger": trigger_dict,
            "conditions": [],
        }]

    elif search_group == "target":
        trigger_dict = _candidate_to_exit_trigger(candidate, event)
        strategy["exit_groups"][0]["targets"] = [{
            "type": "Target",
            "trigger": trigger_dict,
            "conditions": [],
        }]

    # Append extra condition if provided (cross-combination)
    if extra_condition is not None:
        cond_event = extra_condition_event or "Above"
        cond = _candidate_to_condition(extra_condition, cond_event)
        strategy["entry"]["conditions"].append(cond)
        strategy["entry"]["conditions_count"] = len(strategy["entry"]["conditions"])

    return strategy


# ---------------------------------------------------------------------------
# Generate all run configs (handles candidate × event cross-product)
# ---------------------------------------------------------------------------

def generate_run_configs(base_strategy, search_group, search_candidates,
                         events, condition_candidates=None, condition_event=None):
    """Return list of (label, strategy_dict) pairs.

    Generates candidate × event cross-product.
    For target/dynamic_stop: further cross-combines with condition candidates if provided.
    """
    runs = []

    for event in events:
        if search_group in ("trigger", "condition", "static_stop"):
            for candidate in search_candidates:
                label = format_run_label(candidate, event)
                strategy = build_replacement_strategy(base_strategy, search_group, candidate, event)
                runs.append((label, strategy))

        elif search_group in ("target", "dynamic_stop"):
            for candidate in search_candidates:
                cand_label = format_run_label(candidate, event)

                # Standalone run (no extra condition)
                strategy = build_replacement_strategy(base_strategy, search_group, candidate, event)
                runs.append((cand_label, strategy))

                # Cross-combination with each condition candidate
                if condition_candidates:
                    for cond_cand in condition_candidates:
                        strategy = build_replacement_strategy(
                            base_strategy, search_group, candidate, event,
                            extra_condition=cond_cand,
                            extra_condition_event=condition_event)
                        cond_label = format_candidate_label(cond_cand)
                        cond_op = _EVENT_TO_OPERATOR.get(condition_event, "above") if condition_event else "above"
                        combined_label = f"{cand_label} + {cond_label} ({cond_op.lower()})"
                        runs.append((combined_label, strategy))

    # Validate and filter out invalid strategies
    validated = []
    skipped = 0
    for label, strat in runs:
        ema_count = len(strat.get("indicator_settings", {}).get("ema_periods", []))
        is_valid, errors = validate_strategy(strat, ema_count=ema_count)
        if is_valid:
            validated.append((label, strat))
        else:
            skipped += 1

    if skipped > 0:
        import streamlit as st
        st.warning(f"⚠️ {skipped} candidate(s) skipped due to invalid strategy configuration.")

    # Deduplicate labels
    seen = {}
    deduped = []
    for label, strat in validated:
        if label in seen:
            seen[label] += 1
            deduped.append((f"{label} ({seen[label]})", strat))
        else:
            seen[label] = 1
            deduped.append((label, strat))

    return deduped


# ---------------------------------------------------------------------------
# Candidate conversion helpers (event applied externally)
# ---------------------------------------------------------------------------

def _candidate_to_entry_trigger(candidate, event):
    """Convert a universal candidate + event to an entry trigger dict."""
    return {
        "group": candidate.get("group", "Price & Indicators"),
        "element1": candidate.get("element1"),
        "event": event,
        "compare_type": candidate.get("compare_type", "Indicator"),
        "element2": candidate.get("element2"),
        "value": candidate.get("value"),
    }


def _candidate_to_condition(candidate, event):
    """Convert a universal candidate + event to an entry condition dict.

    Derives operator from event using _EVENT_TO_OPERATOR mapping.
    """
    operator = _EVENT_TO_OPERATOR.get(event, "Above")
    return {
        "group": candidate.get("group", "Price & Indicators"),
        "element1": candidate.get("element1"),
        "operator": operator,
        "compare_type": candidate.get("compare_type", "Indicator"),
        "element2": candidate.get("element2"),
        "value": candidate.get("value"),
    }


def _candidate_to_static_stop(candidate, direction, event):
    """Convert a universal candidate + event to an initial_stop dict."""
    stop_type = candidate.get("stop_type", "Indicator")

    if stop_type == "ATR":
        return {
            "element1": "Price",
            "stop_type": "ATR",
            "event": event,
            "atr_period": candidate.get("atr_period", 14),
            "atr_multiplier": candidate.get("atr_multiplier", 2.0),
        }
    else:
        return {
            "stop_type": "Indicator",
            "group": candidate.get("group", "Price & Indicators"),
            "element1": candidate.get("element1", "Price"),
            "element2": candidate.get("element2"),
            "event": event,
            "compare_type": candidate.get("compare_type", "Indicator"),
        }


def _candidate_to_exit_trigger(candidate, event):
    """Convert a universal candidate + event to an exit trigger dict (target or dynamic stop)."""
    result = {
        "group": candidate.get("group", "Price & Indicators"),
        "element1": candidate.get("element1"),
        "event": event,
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
        "willr_period",
        "cci_period",
        "roc_period", "roc_signal_period",
        "lr_period", "lr_multiplier",
    ]
    result = dict(_DEFAULT_INDICATOR_PARAMS)
    for k in keys:
        val = ss.get(f"{pfx}{k}")
        if val is not None:
            result[k] = val
    return result

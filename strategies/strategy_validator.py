"""
Strategy validation — enforces strict rules on strategy structure before save.

validate_strategy() returns (is_valid, errors) where errors is a list of
human-readable strings describing each problem found.
"""

from config.constants import (
    EVENT_TYPES,
    STOP_EVENT_TYPES,
    CONDITION_OPERATORS,
    INDICATOR_MAP,
    R_PROFIT_LOSS_ELEMENTS,
    ATR_TARGET_ELEMENTS,
    get_indicator_map,
)

# All valid indicator element names (without EMA, those are dynamic)
_VALID_ELEMENTS = set(INDICATOR_MAP.keys())
_VALID_R_ELEMENTS = set(R_PROFIT_LOSS_ELEMENTS)
_VALID_ATR_TARGET_ELEMENTS = set(ATR_TARGET_ELEMENTS)


def _is_valid_element(name, ema_count=0):
    """Check if an element name is a recognized indicator."""
    if name in _VALID_ELEMENTS:
        return True
    if name in _VALID_R_ELEMENTS:
        return True
    if name in _VALID_ATR_TARGET_ELEMENTS:
        return True
    # Dynamic EMA names: "EMA 1", "EMA 2", etc.
    if name and name.startswith("EMA "):
        try:
            idx = int(name.split(" ")[1])
            return 1 <= idx <= ema_count
        except (ValueError, IndexError):
            return False
    return False


def validate_strategy(strategy, ema_count=0):
    """
    Validate a strategy dict for completeness and correctness.

    Parameters:
        strategy: dict — the strategy to validate
        ema_count: int — number of EMA overlays configured (for validating EMA element names)

    Returns:
        (is_valid: bool, errors: list[str])
    """
    errors = []

    if not isinstance(strategy, dict):
        return False, ["Strategy must be a dictionary"]

    # ------------------------------------------------------------------
    # Top level
    # ------------------------------------------------------------------
    direction = strategy.get("direction")
    if direction not in ("Long", "Short"):
        errors.append(f"direction must be 'Long' or 'Short', got: {direction!r}")

    max_pos = strategy.get("max_positions")
    if max_pos is not None:
        if not isinstance(max_pos, (int, float)) or max_pos < 1:
            errors.append(f"max_positions must be None (unlimited) or >= 1, got: {max_pos!r}")

    # ------------------------------------------------------------------
    # Entry
    # ------------------------------------------------------------------
    entry = strategy.get("entry")
    if not isinstance(entry, dict):
        errors.append("entry must be a dictionary")
        return False, errors

    pos_size = entry.get("position_size")
    if pos_size is None or (isinstance(pos_size, (int, float)) and pos_size <= 0):
        errors.append(f"entry.position_size must be a number > 0, got: {pos_size!r}")

    # Entry trigger
    trigger = entry.get("trigger")
    if not isinstance(trigger, dict):
        errors.append("entry.trigger must be a dictionary")
    else:
        _validate_trigger(trigger, "entry.trigger", errors, ema_count,
                          allowed_events=EVENT_TYPES,
                          allow_r_elements=False,
                          allow_atr_target=False)

    # Entry conditions
    conditions = entry.get("conditions", [])
    for i, cond in enumerate(conditions):
        _validate_condition(cond, f"entry.conditions[{i}]", errors, ema_count)

    # ------------------------------------------------------------------
    # Initial stop
    # ------------------------------------------------------------------
    initial_stop = strategy.get("initial_stop")
    if not initial_stop or not isinstance(initial_stop, dict):
        errors.append("initial_stop must be a non-empty dictionary")
    else:
        stop_type = initial_stop.get("stop_type")
        if stop_type not in ("Indicator", "ATR"):
            errors.append(f"initial_stop.stop_type must be 'Indicator' or 'ATR', got: {stop_type!r}")

        event = initial_stop.get("event")
        if event not in STOP_EVENT_TYPES:
            errors.append(f"initial_stop.event must be one of {STOP_EVENT_TYPES}, got: {event!r}")

        if stop_type == "Indicator":
            el2 = initial_stop.get("element2")
            if not el2 or not _is_valid_element(el2, ema_count):
                errors.append(f"initial_stop.element2 must be a valid indicator name, got: {el2!r}")

        elif stop_type == "ATR":
            atr_period = initial_stop.get("atr_period")
            if not isinstance(atr_period, (int, float)) or atr_period < 1:
                errors.append(f"initial_stop.atr_period must be >= 1, got: {atr_period!r}")

            atr_mult = initial_stop.get("atr_multiplier")
            if not isinstance(atr_mult, (int, float)) or atr_mult <= 0:
                errors.append(f"initial_stop.atr_multiplier must be > 0, got: {atr_mult!r}")

    # ------------------------------------------------------------------
    # Exit groups
    # ------------------------------------------------------------------
    exit_groups = strategy.get("exit_groups", [])
    if not exit_groups:
        errors.append("exit_groups must have at least one group")
    else:
        total_alloc = 0.0
        for g_idx, group in enumerate(exit_groups):
            prefix = f"exit_groups[{g_idx}]"

            alloc = group.get("allocation_pct")
            if not isinstance(alloc, (int, float)) or alloc <= 0 or alloc > 100:
                errors.append(f"{prefix}.allocation_pct must be > 0 and <= 100, got: {alloc!r}")
            else:
                total_alloc += alloc

            targets = group.get("targets", [])
            stops = group.get("stops", [])

            if len(targets) == 0 and len(stops) == 0:
                errors.append(f"{prefix} must have at least one target or stop")

            for t_idx, target in enumerate(targets):
                t_prefix = f"{prefix}.targets[{t_idx}]"
                t_trigger = target.get("trigger")
                if not isinstance(t_trigger, dict):
                    errors.append(f"{t_prefix}.trigger must be a dictionary")
                else:
                    _validate_trigger(t_trigger, f"{t_prefix}.trigger", errors, ema_count,
                                      allowed_events=EVENT_TYPES,
                                      allow_r_elements=True,
                                      allow_atr_target=True)

                for c_idx, cond in enumerate(target.get("conditions", [])):
                    _validate_condition(cond, f"{t_prefix}.conditions[{c_idx}]", errors, ema_count)

            for s_idx, stop in enumerate(stops):
                s_prefix = f"{prefix}.stops[{s_idx}]"
                s_trigger = stop.get("trigger")
                if not isinstance(s_trigger, dict):
                    errors.append(f"{s_prefix}.trigger must be a dictionary")
                else:
                    _validate_trigger(s_trigger, f"{s_prefix}.trigger", errors, ema_count,
                                      allowed_events=STOP_EVENT_TYPES,
                                      allow_r_elements=True,
                                      allow_atr_target=True)

                for c_idx, cond in enumerate(stop.get("conditions", [])):
                    _validate_condition(cond, f"{s_prefix}.conditions[{c_idx}]", errors, ema_count)

        if exit_groups and abs(total_alloc - 100.0) >= 0.01:
            errors.append(f"exit_groups total allocation must equal 100%, got: {total_alloc:.1f}%")

    # ------------------------------------------------------------------
    # Indicator settings
    # ------------------------------------------------------------------
    ind = strategy.get("indicator_settings")
    if not isinstance(ind, dict):
        errors.append("indicator_settings must be a dictionary")
    else:
        _validate_indicator_settings(ind, errors)

    return len(errors) == 0, errors


# ------------------------------------------------------------------
# Internal helpers
# ------------------------------------------------------------------

def _validate_trigger(trigger, path, errors, ema_count,
                      allowed_events, allow_r_elements, allow_atr_target):
    """Validate a single trigger dict (entry, exit target, or exit stop)."""

    element1 = trigger.get("element1")
    event = trigger.get("event")
    compare_type = trigger.get("compare_type", "Indicator")

    # Check element1
    is_r = element1 in _VALID_R_ELEMENTS
    is_atr_target = element1 in _VALID_ATR_TARGET_ELEMENTS

    if is_r and not allow_r_elements:
        errors.append(f"{path}.element1: R Profit/R Loss not allowed in entry triggers")
    elif is_atr_target and not allow_atr_target:
        errors.append(f"{path}.element1: ATR Target not allowed in entry triggers")
    elif not is_r and not is_atr_target:
        if not element1 or not _is_valid_element(element1, ema_count):
            errors.append(f"{path}.element1 must be a valid indicator name, got: {element1!r}")

    # Check event
    if event is None:
        errors.append(f"{path}.event must not be null")
    elif event not in allowed_events:
        errors.append(f"{path}.event must be one of {allowed_events}, got: {event!r}")

    # Check compare side
    if is_r:
        # R Profit/Loss must use Fixed Value
        if compare_type != "Fixed Value":
            errors.append(f"{path}: R Profit/R Loss must use compare_type='Fixed Value'")
        value = trigger.get("value")
        if value is None or not isinstance(value, (int, float)):
            errors.append(f"{path}.value must be a number for R Profit/R Loss, got: {value!r}")

    elif is_atr_target:
        # ATR Target needs period and multiplier
        atr_period = trigger.get("atr_period")
        if not isinstance(atr_period, (int, float)) or atr_period < 1:
            errors.append(f"{path}.atr_period must be >= 1, got: {atr_period!r}")
        atr_mult = trigger.get("atr_multiplier")
        if not isinstance(atr_mult, (int, float)) or atr_mult <= 0:
            errors.append(f"{path}.atr_multiplier must be > 0, got: {atr_mult!r}")

    elif compare_type == "Indicator":
        element2 = trigger.get("element2")
        if not element2 or not _is_valid_element(element2, ema_count):
            errors.append(f"{path}.element2 must be a valid indicator name when compare_type='Indicator', got: {element2!r}")

    elif compare_type == "Fixed Value":
        value = trigger.get("value")
        if value is None or not isinstance(value, (int, float)):
            errors.append(f"{path}.value must be a number when compare_type='Fixed Value', got: {value!r}")

    else:
        errors.append(f"{path}.compare_type must be 'Indicator' or 'Fixed Value', got: {compare_type!r}")


def _validate_condition(condition, path, errors, ema_count):
    """Validate a single condition dict."""

    element1 = condition.get("element1")
    if not element1 or not _is_valid_element(element1, ema_count):
        errors.append(f"{path}.element1 must be a valid indicator name, got: {element1!r}")

    operator = condition.get("operator")
    if operator not in CONDITION_OPERATORS:
        errors.append(f"{path}.operator must be one of {CONDITION_OPERATORS}, got: {operator!r}")

    compare_type = condition.get("compare_type", "Indicator")
    if compare_type == "Indicator":
        element2 = condition.get("element2")
        if not element2 or not _is_valid_element(element2, ema_count):
            errors.append(f"{path}.element2 must be a valid indicator name when compare_type='Indicator', got: {element2!r}")
    elif compare_type == "Fixed Value":
        value = condition.get("value")
        if value is None or not isinstance(value, (int, float)):
            errors.append(f"{path}.value must be a number when compare_type='Fixed Value', got: {value!r}")
    else:
        errors.append(f"{path}.compare_type must be 'Indicator' or 'Fixed Value', got: {compare_type!r}")


def _validate_indicator_settings(settings, errors):
    """Validate indicator settings have required keys with valid types."""

    _required_int = [
        "rsi_window", "bb_upper_period", "bb_mid_period", "bb_lower_period",
        "kc_upper_ema", "kc_mid_ema", "kc_lower_ema", "kc_atr_period",
        "stoch_k_period", "stoch_k_smooth", "stoch_d_smooth",
        "adx_period", "atr_period",
        "macd_fast", "macd_slow", "macd_signal",
        "supertrend_period",
        "dc_upper_period", "dc_mid_period", "dc_lower_period",
    ]
    _required_float = [
        "bb_upper_stdev", "bb_lower_stdev",
        "kc_upper_mult", "kc_lower_mult",
        "supertrend_multiplier",
        "psar_af_start", "psar_af_increment", "psar_af_max",
    ]

    for key in _required_int:
        val = settings.get(key)
        if val is None or not isinstance(val, (int, float)) or val < 1:
            errors.append(f"indicator_settings.{key} must be >= 1, got: {val!r}")

    for key in _required_float:
        val = settings.get(key)
        if val is None or not isinstance(val, (int, float)) or val <= 0:
            errors.append(f"indicator_settings.{key} must be > 0, got: {val!r}")

    # dc_offset can be negative, zero, or positive
    dc_offset = settings.get("dc_offset")
    if dc_offset is None or not isinstance(dc_offset, (int, float)):
        errors.append(f"indicator_settings.dc_offset must be a number, got: {dc_offset!r}")

    # ema_periods must be a list of positive integers
    ema_periods = settings.get("ema_periods")
    if ema_periods is None:
        errors.append("indicator_settings.ema_periods must be a list (can be empty)")
    elif not isinstance(ema_periods, list):
        errors.append(f"indicator_settings.ema_periods must be a list, got: {type(ema_periods).__name__}")
    else:
        for i, p in enumerate(ema_periods):
            if not isinstance(p, (int, float)) or p < 2:
                errors.append(f"indicator_settings.ema_periods[{i}] must be >= 2, got: {p!r}")

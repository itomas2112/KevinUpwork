"""
Tests for strategy validation — ensures only correctly defined strategies pass.

Tests every field that can be null, verifying the validator catches it.
Tests that all valid strategies (from our existing combos) pass validation.
Tests the client's exact broken strategy (ATR Target event: null) is caught.
"""
import pytest
import json
import copy

from strategies.strategy_validator import validate_strategy
from tests.conftest import (
    make_strategy, make_exit_group, make_exit_trigger, make_exit_stop,
    DEFAULT_INDICATOR_SETTINGS,
    ENTRY_TRIGGER_COMBOS, INITIAL_STOP_COMBOS, ENTRY_CONDITION_COMBOS,
    EXIT_TARGET_COMBOS, EXIT_STOP_COMBOS,
    GROUP_FIXED_VALUES,
)
from config.constants import EVENT_TYPES, STOP_EVENT_TYPES


# =====================================================================
# Valid strategies should all pass
# =====================================================================

class TestValidStrategiesPass:
    """Every correctly-built strategy from our combo lists must pass validation."""

    def test_default_strategy_valid(self):
        strategy = make_strategy()
        is_valid, errors = validate_strategy(strategy)
        assert is_valid, f"Default strategy invalid: {errors}"

    @pytest.mark.parametrize("group,el1,el2,event,compare_type,fixed_val", ENTRY_TRIGGER_COMBOS)
    def test_all_entry_triggers_valid(self, group, el1, el2, event, compare_type, fixed_val):
        strategy = make_strategy(
            entry_group=group, entry_element1=el1, entry_event=event,
            entry_compare_type=compare_type, entry_element2=el2,
            entry_value=fixed_val if fixed_val is not None else 50.0,
        )
        is_valid, errors = validate_strategy(strategy)
        assert is_valid, f"Entry trigger [{el1} {event}] invalid: {errors}"

    @pytest.mark.parametrize("stop_type,event,element2,atr_period,atr_mult", INITIAL_STOP_COMBOS)
    def test_all_initial_stops_valid(self, stop_type, event, element2, atr_period, atr_mult):
        strategy = make_strategy(
            initial_stop_type=stop_type, initial_stop_event=event,
            initial_stop_element2=element2 or "BB Lower Band",
            initial_stop_atr_period=atr_period or 14,
            initial_stop_atr_multiplier=atr_mult or 1.5,
        )
        is_valid, errors = validate_strategy(strategy)
        assert is_valid, f"Stop [{stop_type} {event}] invalid: {errors}"

    @pytest.mark.parametrize("group,el1,operator,compare_type,el2,fixed_val", ENTRY_CONDITION_COMBOS)
    def test_all_entry_conditions_valid(self, group, el1, operator, compare_type, el2, fixed_val):
        condition = {
            "group": group, "element1": el1, "operator": operator,
            "compare_type": compare_type, "element2": el2,
            "value": fixed_val if fixed_val is not None else 50.0,
        }
        strategy = make_strategy(entry_conditions=[condition])
        is_valid, errors = validate_strategy(strategy)
        assert is_valid, f"Condition [{el1} {operator}] invalid: {errors}"

    def test_long_and_short_valid(self):
        for direction in ("Long", "Short"):
            strategy = make_strategy(direction=direction)
            is_valid, errors = validate_strategy(strategy)
            assert is_valid, f"{direction} strategy invalid: {errors}"

    def test_multi_exit_groups_valid(self):
        group1 = make_exit_group(allocation_pct=50.0, group_id=1,
                                  targets=[make_exit_trigger(value=1.5)])
        group2 = make_exit_group(allocation_pct=50.0, group_id=2,
                                  targets=[make_exit_trigger(value=3.0)])
        strategy = make_strategy(exit_groups=[group1, group2])
        is_valid, errors = validate_strategy(strategy)
        assert is_valid, f"Multi-exit invalid: {errors}"

    def test_max_positions_unlimited_valid(self):
        strategy = make_strategy(max_positions=None)
        is_valid, errors = validate_strategy(strategy)
        assert is_valid, f"Unlimited positions invalid: {errors}"

    def test_strategy_survives_json_roundtrip_and_validates(self):
        strategy = make_strategy()
        loaded = json.loads(json.dumps(strategy))
        is_valid, errors = validate_strategy(loaded)
        assert is_valid, f"After JSON roundtrip: {errors}"


# =====================================================================
# Null/missing top-level fields
# =====================================================================

class TestTopLevelNulls:

    def test_null_direction(self):
        strategy = make_strategy()
        strategy["direction"] = None
        is_valid, errors = validate_strategy(strategy)
        assert not is_valid
        assert any("direction" in e for e in errors)

    def test_invalid_direction(self):
        strategy = make_strategy()
        strategy["direction"] = "Up"
        is_valid, errors = validate_strategy(strategy)
        assert not is_valid

    def test_max_positions_zero(self):
        strategy = make_strategy()
        strategy["max_positions"] = 0
        is_valid, errors = validate_strategy(strategy)
        assert not is_valid
        assert any("max_positions" in e for e in errors)

    def test_max_positions_negative(self):
        strategy = make_strategy()
        strategy["max_positions"] = -1
        is_valid, errors = validate_strategy(strategy)
        assert not is_valid

    def test_missing_entry(self):
        strategy = make_strategy()
        del strategy["entry"]
        is_valid, errors = validate_strategy(strategy)
        assert not is_valid

    def test_null_position_size(self):
        strategy = make_strategy()
        strategy["entry"]["position_size"] = None
        is_valid, errors = validate_strategy(strategy)
        assert not is_valid
        assert any("position_size" in e for e in errors)

    def test_zero_position_size(self):
        strategy = make_strategy()
        strategy["entry"]["position_size"] = 0
        is_valid, errors = validate_strategy(strategy)
        assert not is_valid

    def test_not_a_dict(self):
        is_valid, errors = validate_strategy("not a dict")
        assert not is_valid

    def test_missing_indicator_settings(self):
        strategy = make_strategy()
        del strategy["indicator_settings"]
        is_valid, errors = validate_strategy(strategy)
        assert not is_valid


# =====================================================================
# Null/missing entry trigger fields
# =====================================================================

class TestEntryTriggerNulls:

    def test_null_element1(self):
        strategy = make_strategy()
        strategy["entry"]["trigger"]["element1"] = None
        is_valid, errors = validate_strategy(strategy)
        assert not is_valid
        assert any("element1" in e for e in errors)

    def test_null_event(self):
        strategy = make_strategy()
        strategy["entry"]["trigger"]["event"] = None
        is_valid, errors = validate_strategy(strategy)
        assert not is_valid
        assert any("event" in e for e in errors)

    def test_invalid_event(self):
        strategy = make_strategy()
        strategy["entry"]["trigger"]["event"] = "Explode"
        is_valid, errors = validate_strategy(strategy)
        assert not is_valid

    def test_null_element2_when_indicator_compare(self):
        strategy = make_strategy(
            entry_compare_type="Indicator", entry_element2="BB Upper Band"
        )
        strategy["entry"]["trigger"]["element2"] = None
        is_valid, errors = validate_strategy(strategy)
        assert not is_valid
        assert any("element2" in e for e in errors)

    def test_null_value_when_fixed_value_compare(self):
        strategy = make_strategy(
            entry_compare_type="Fixed Value", entry_value=50.0
        )
        strategy["entry"]["trigger"]["value"] = None
        is_valid, errors = validate_strategy(strategy)
        assert not is_valid
        assert any("value" in e for e in errors)

    def test_invalid_element_name(self):
        strategy = make_strategy()
        strategy["entry"]["trigger"]["element1"] = "Nonexistent Indicator"
        is_valid, errors = validate_strategy(strategy)
        assert not is_valid

    def test_r_profit_not_allowed_in_entry(self):
        strategy = make_strategy()
        strategy["entry"]["trigger"]["element1"] = "R Profit"
        strategy["entry"]["trigger"]["compare_type"] = "Fixed Value"
        strategy["entry"]["trigger"]["value"] = 2.0
        is_valid, errors = validate_strategy(strategy)
        assert not is_valid
        assert any("R Profit" in e for e in errors)

    def test_atr_target_not_allowed_in_entry(self):
        strategy = make_strategy()
        strategy["entry"]["trigger"]["element1"] = "ATR Target"
        is_valid, errors = validate_strategy(strategy)
        assert not is_valid
        assert any("ATR Target" in e for e in errors)


# =====================================================================
# Null/missing initial stop fields
# =====================================================================

class TestInitialStopNulls:

    def test_null_initial_stop(self):
        strategy = make_strategy()
        strategy["initial_stop"] = None
        is_valid, errors = validate_strategy(strategy)
        assert not is_valid
        assert any("initial_stop" in e for e in errors)

    def test_empty_initial_stop(self):
        strategy = make_strategy()
        strategy["initial_stop"] = {}
        is_valid, errors = validate_strategy(strategy)
        assert not is_valid

    def test_null_stop_type(self):
        strategy = make_strategy(initial_stop_type="Indicator")
        strategy["initial_stop"]["stop_type"] = None
        is_valid, errors = validate_strategy(strategy)
        assert not is_valid

    def test_null_stop_event(self):
        strategy = make_strategy()
        strategy["initial_stop"]["event"] = None
        is_valid, errors = validate_strategy(strategy)
        assert not is_valid
        assert any("event" in e for e in errors)

    def test_indicator_stop_null_element2(self):
        strategy = make_strategy(initial_stop_type="Indicator", initial_stop_element2="BB Lower Band")
        strategy["initial_stop"]["element2"] = None
        is_valid, errors = validate_strategy(strategy)
        assert not is_valid
        assert any("element2" in e for e in errors)

    def test_atr_stop_null_period(self):
        strategy = make_strategy(initial_stop_type="ATR")
        strategy["initial_stop"]["atr_period"] = None
        is_valid, errors = validate_strategy(strategy)
        assert not is_valid

    def test_atr_stop_null_multiplier(self):
        strategy = make_strategy(initial_stop_type="ATR")
        strategy["initial_stop"]["atr_multiplier"] = None
        is_valid, errors = validate_strategy(strategy)
        assert not is_valid

    def test_atr_stop_zero_multiplier(self):
        strategy = make_strategy(initial_stop_type="ATR")
        strategy["initial_stop"]["atr_multiplier"] = 0
        is_valid, errors = validate_strategy(strategy)
        assert not is_valid


# =====================================================================
# Null/missing entry condition fields
# =====================================================================

class TestEntryConditionNulls:

    def _strategy_with_condition(self, **overrides):
        cond = {
            "group": "RSI Group", "element1": "RSI", "operator": "Above",
            "compare_type": "Fixed Value", "value": 50.0,
        }
        cond.update(overrides)
        return make_strategy(entry_conditions=[cond])

    def test_null_element1(self):
        strategy = self._strategy_with_condition(element1=None)
        is_valid, errors = validate_strategy(strategy)
        assert not is_valid
        assert any("element1" in e and "conditions" in e for e in errors)

    def test_null_operator(self):
        strategy = self._strategy_with_condition(operator=None)
        is_valid, errors = validate_strategy(strategy)
        assert not is_valid
        assert any("operator" in e for e in errors)

    def test_invalid_operator(self):
        strategy = self._strategy_with_condition(operator="Greater")
        is_valid, errors = validate_strategy(strategy)
        assert not is_valid

    def test_null_element2_indicator_compare(self):
        strategy = self._strategy_with_condition(
            compare_type="Indicator", element2=None, value=None
        )
        is_valid, errors = validate_strategy(strategy)
        assert not is_valid

    def test_null_value_fixed_compare(self):
        strategy = self._strategy_with_condition(
            compare_type="Fixed Value", value=None
        )
        is_valid, errors = validate_strategy(strategy)
        assert not is_valid


# =====================================================================
# Null/missing exit target/stop trigger fields
# =====================================================================

class TestExitTriggerNulls:

    def _strategy_with_exit_target(self, trigger_overrides=None):
        trigger = {
            "group": "R Profit / R Loss", "element1": "R Profit",
            "event": "Cross Above", "compare_type": "Fixed Value",
            "element2": None, "value": 2.0,
        }
        if trigger_overrides:
            trigger.update(trigger_overrides)
        target = {"type": "Target", "trigger": trigger, "conditions": []}
        group = make_exit_group(allocation_pct=100.0, targets=[target])
        return make_strategy(exit_groups=[group])

    def test_null_event_on_exit_target(self):
        """This is the client's exact bug — ATR Target with event: null."""
        strategy = self._strategy_with_exit_target({
            "group": "ATR Target", "element1": "ATR Target",
            "event": None, "compare_type": "ATR",
            "element2": None, "value": None,
            "atr_period": 14, "atr_multiplier": 3.0,
        })
        is_valid, errors = validate_strategy(strategy)
        assert not is_valid
        assert any("event" in e and "null" in e.lower() for e in errors)

    def test_null_event_on_r_profit(self):
        strategy = self._strategy_with_exit_target({"event": None})
        is_valid, errors = validate_strategy(strategy)
        assert not is_valid

    def test_null_element1_on_exit(self):
        strategy = self._strategy_with_exit_target({"element1": None})
        is_valid, errors = validate_strategy(strategy)
        assert not is_valid

    def test_null_value_on_r_profit(self):
        strategy = self._strategy_with_exit_target({"value": None})
        is_valid, errors = validate_strategy(strategy)
        assert not is_valid

    def test_r_profit_wrong_compare_type(self):
        strategy = self._strategy_with_exit_target({"compare_type": "Indicator"})
        is_valid, errors = validate_strategy(strategy)
        assert not is_valid

    def test_atr_target_null_period(self):
        strategy = self._strategy_with_exit_target({
            "group": "ATR Target", "element1": "ATR Target",
            "event": "Cross Above", "compare_type": "ATR",
            "atr_period": None, "atr_multiplier": 2.0,
        })
        is_valid, errors = validate_strategy(strategy)
        assert not is_valid

    def test_atr_target_null_multiplier(self):
        strategy = self._strategy_with_exit_target({
            "group": "ATR Target", "element1": "ATR Target",
            "event": "Cross Above", "compare_type": "ATR",
            "atr_period": 14, "atr_multiplier": None,
        })
        is_valid, errors = validate_strategy(strategy)
        assert not is_valid

    def test_indicator_exit_null_element2(self):
        strategy = self._strategy_with_exit_target({
            "group": "Price & Indicators", "element1": "Price",
            "event": "Cross Above", "compare_type": "Indicator",
            "element2": None, "value": None,
        })
        is_valid, errors = validate_strategy(strategy)
        assert not is_valid

    def test_fixed_value_exit_null_value(self):
        strategy = self._strategy_with_exit_target({
            "group": "RSI Group", "element1": "RSI",
            "event": "Cross Above", "compare_type": "Fixed Value",
            "element2": None, "value": None,
        })
        is_valid, errors = validate_strategy(strategy)
        assert not is_valid


class TestExitStopNulls:

    def _strategy_with_dynamic_stop(self, trigger_overrides=None):
        trigger = {
            "group": "Price & Indicators", "element1": "Price",
            "event": "Cross Below", "compare_type": "Indicator",
            "element2": "BB Lower Band", "value": None,
        }
        if trigger_overrides:
            trigger.update(trigger_overrides)
        stop = {"type": "Stop", "trigger": trigger, "conditions": []}
        # Need a target too for the group to be valid
        target = make_exit_trigger(value=2.0)
        group = make_exit_group(allocation_pct=100.0, targets=[target], stops=[stop])
        return make_strategy(exit_groups=[group])

    def test_null_event_on_stop(self):
        strategy = self._strategy_with_dynamic_stop({"event": None})
        is_valid, errors = validate_strategy(strategy)
        assert not is_valid

    def test_invalid_event_on_stop(self):
        """Stops only allow STOP_EVENT_TYPES, not full EVENT_TYPES."""
        strategy = self._strategy_with_dynamic_stop({"event": "Cross"})
        is_valid, errors = validate_strategy(strategy)
        assert not is_valid

    def test_null_element2_on_indicator_stop(self):
        strategy = self._strategy_with_dynamic_stop({"element2": None})
        is_valid, errors = validate_strategy(strategy)
        assert not is_valid


# =====================================================================
# Exit group structure validation
# =====================================================================

class TestExitGroupStructure:

    def test_no_exit_groups(self):
        strategy = make_strategy()
        strategy["exit_groups"] = []
        is_valid, errors = validate_strategy(strategy)
        assert not is_valid
        assert any("exit_groups" in e for e in errors)

    def test_empty_exit_group(self):
        """Group with no targets and no stops."""
        group = make_exit_group(allocation_pct=100.0, targets=[], stops=[])
        strategy = make_strategy(exit_groups=[group])
        is_valid, errors = validate_strategy(strategy)
        assert not is_valid
        assert any("at least one target or stop" in e for e in errors)

    def test_allocation_not_100(self):
        group1 = make_exit_group(allocation_pct=60.0, targets=[make_exit_trigger()])
        group2 = make_exit_group(allocation_pct=30.0, targets=[make_exit_trigger()])
        strategy = make_strategy(exit_groups=[group1, group2])
        is_valid, errors = validate_strategy(strategy)
        assert not is_valid
        assert any("100%" in e for e in errors)

    def test_allocation_zero(self):
        group = make_exit_group(allocation_pct=0, targets=[make_exit_trigger()])
        strategy = make_strategy(exit_groups=[group])
        is_valid, errors = validate_strategy(strategy)
        assert not is_valid

    def test_allocation_over_100(self):
        group = make_exit_group(allocation_pct=150.0, targets=[make_exit_trigger()])
        strategy = make_strategy(exit_groups=[group])
        is_valid, errors = validate_strategy(strategy)
        assert not is_valid


# =====================================================================
# Indicator settings validation
# =====================================================================

class TestIndicatorSettingsValidation:

    def test_valid_defaults(self):
        strategy = make_strategy()
        is_valid, errors = validate_strategy(strategy)
        assert is_valid

    def test_null_rsi_window(self):
        strategy = make_strategy()
        strategy["indicator_settings"]["rsi_window"] = None
        is_valid, errors = validate_strategy(strategy)
        assert not is_valid

    def test_zero_rsi_window(self):
        strategy = make_strategy()
        strategy["indicator_settings"]["rsi_window"] = 0
        is_valid, errors = validate_strategy(strategy)
        assert not is_valid

    def test_null_bb_stdev(self):
        strategy = make_strategy()
        strategy["indicator_settings"]["bb_upper_stdev"] = None
        is_valid, errors = validate_strategy(strategy)
        assert not is_valid

    def test_zero_bb_stdev(self):
        strategy = make_strategy()
        strategy["indicator_settings"]["bb_upper_stdev"] = 0
        is_valid, errors = validate_strategy(strategy)
        assert not is_valid

    def test_null_ema_periods(self):
        strategy = make_strategy()
        strategy["indicator_settings"]["ema_periods"] = None
        is_valid, errors = validate_strategy(strategy)
        assert not is_valid

    def test_ema_period_too_small(self):
        strategy = make_strategy()
        strategy["indicator_settings"]["ema_periods"] = [1]
        is_valid, errors = validate_strategy(strategy)
        assert not is_valid

    def test_dc_offset_negative_valid(self):
        """dc_offset can be negative."""
        strategy = make_strategy()
        strategy["indicator_settings"]["dc_offset"] = -5
        is_valid, errors = validate_strategy(strategy)
        assert is_valid

    def test_dc_offset_null_invalid(self):
        strategy = make_strategy()
        strategy["indicator_settings"]["dc_offset"] = None
        is_valid, errors = validate_strategy(strategy)
        assert not is_valid


# =====================================================================
# Client's exact broken strategy
# =====================================================================

class TestClientStrategy:

    def test_client_strategy_with_null_event(self):
        """The exact strategy from the client's exported JSON — should fail validation."""
        strategy = {
            "strategy_name": "Long",
            "direction": "Long",
            "patterns": [],
            "max_positions": 1,
            "created_at": "2026-03-23 09:47:25",
            "entry": {
                "trigger": {
                    "group": "Price & Indicators",
                    "element1": "Price",
                    "event": "Cross Above",
                    "compare_type": "Indicator",
                    "element2": "BB Upper Band",
                    "value": None,
                },
                "position_size": 1.0,
                "conditions_count": 0,
                "conditions": [],
            },
            "initial_stop": {
                "element1": "Price",
                "stop_type": "ATR",
                "event": "Cross Above",
                "atr_period": 14,
                "atr_multiplier": 2.0,
            },
            "exit_groups": [{
                "group_id": 1,
                "allocation_pct": 100.0,
                "targets": [{
                    "type": "Target",
                    "trigger": {
                        "group": "ATR Target",
                        "element1": "ATR Target",
                        "event": None,           # <-- THE BUG
                        "compare_type": "ATR",
                        "element2": None,
                        "value": None,
                        "atr_period": 14,
                        "atr_multiplier": 3.0,
                    },
                    "conditions": [],
                }],
                "stops": [],
            }],
            "indicator_settings": dict(DEFAULT_INDICATOR_SETTINGS),
        }

        is_valid, errors = validate_strategy(strategy)
        assert not is_valid
        assert len(errors) == 1, f"Expected exactly 1 error, got: {errors}"
        assert "event" in errors[0]

    def test_client_strategy_fixed(self):
        """Same strategy but with event set — should pass."""
        strategy = {
            "strategy_name": "Long",
            "direction": "Long",
            "patterns": [],
            "max_positions": 1,
            "created_at": "2026-03-23 09:47:25",
            "entry": {
                "trigger": {
                    "group": "Price & Indicators",
                    "element1": "Price",
                    "event": "Cross Above",
                    "compare_type": "Indicator",
                    "element2": "BB Upper Band",
                    "value": None,
                },
                "position_size": 1.0,
                "conditions_count": 0,
                "conditions": [],
            },
            "initial_stop": {
                "element1": "Price",
                "stop_type": "ATR",
                "event": "Cross Above",
                "atr_period": 14,
                "atr_multiplier": 2.0,
            },
            "exit_groups": [{
                "group_id": 1,
                "allocation_pct": 100.0,
                "targets": [{
                    "type": "Target",
                    "trigger": {
                        "group": "ATR Target",
                        "element1": "ATR Target",
                        "event": "Cross Above",  # <-- FIXED
                        "compare_type": "ATR",
                        "element2": None,
                        "value": None,
                        "atr_period": 14,
                        "atr_multiplier": 3.0,
                    },
                    "conditions": [],
                }],
                "stops": [],
            }],
            "indicator_settings": dict(DEFAULT_INDICATOR_SETTINGS),
        }

        is_valid, errors = validate_strategy(strategy)
        assert is_valid, f"Fixed strategy should be valid: {errors}"


# =====================================================================
# Multiple errors reported
# =====================================================================

class TestMultipleErrors:

    def test_reports_all_errors(self):
        """A strategy with many problems should report all of them."""
        strategy = {
            "direction": None,
            "entry": {
                "trigger": {"element1": None, "event": None},
                "position_size": None,
                "conditions": [],
            },
            "initial_stop": {},
            "exit_groups": [],
            "indicator_settings": {"rsi_window": None},
        }
        is_valid, errors = validate_strategy(strategy)
        assert not is_valid
        assert len(errors) >= 5, f"Expected multiple errors, got {len(errors)}: {errors}"

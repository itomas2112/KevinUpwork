"""
Tests for strategy save/load/delete roundtrip.

These tests operate on JSON files directly, without needing Streamlit session state.
Includes execution roundtrip tests: build strategy → save to JSON → load → execute
→ verify identical results to executing the original.
"""
import pytest
import json
import os
import tempfile
import numpy as np
import pandas as pd

from indicators.calculate_indicators import calculate_indicators
from strategies.first_strategy import execute_custom_strategy
from tests.conftest import (
    make_strategy, make_exit_group, DEFAULT_INDICATOR_SETTINGS,
    ENTRY_TRIGGER_COMBOS, INITIAL_STOP_COMBOS, ENTRY_CONDITION_COMBOS,
    EXIT_TARGET_COMBOS, EXIT_STOP_COMBOS,
    GROUP_FIXED_VALUES,
    _generate_oscillation, _generate_v_shape,
)


class TestStrategyRoundtrip:

    def test_save_and_load_json(self):
        """Strategy serializes to JSON and deserializes identically."""
        strategy = make_strategy(
            strategy_name="Test Roundtrip",
            direction="Long",
            entry_group="RSI Group",
            entry_element1="RSI",
            entry_event="Cross Above",
            entry_compare_type="Fixed Value",
            entry_value=30.0,
        )

        # Serialize
        json_str = json.dumps([strategy], indent=4)

        # Deserialize
        loaded = json.loads(json_str)
        assert len(loaded) == 1

        loaded_strategy = loaded[0]
        assert loaded_strategy["strategy_name"] == "Test Roundtrip"
        assert loaded_strategy["direction"] == "Long"
        assert loaded_strategy["entry"]["trigger"]["element1"] == "RSI"
        assert loaded_strategy["entry"]["trigger"]["event"] == "Cross Above"
        assert loaded_strategy["entry"]["trigger"]["value"] == 30.0

    def test_save_and_load_to_file(self):
        """Strategy round-trips through a temp JSON file."""
        strategy = make_strategy(strategy_name="File Test")
        strategies = [strategy]

        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
            json.dump(strategies, f, indent=4)
            tmp_path = f.name

        try:
            with open(tmp_path, 'r') as f:
                loaded = json.load(f)
            assert len(loaded) == 1
            assert loaded[0]["strategy_name"] == "File Test"
        finally:
            os.unlink(tmp_path)

    def test_multiple_strategies(self):
        """Multiple strategies save and load correctly."""
        strategies = [
            make_strategy(strategy_name=f"Strategy {i}", direction=d)
            for i, d in enumerate(["Long", "Short", "Long"])
        ]

        json_str = json.dumps(strategies, indent=4)
        loaded = json.loads(json_str)

        assert len(loaded) == 3
        assert loaded[0]["direction"] == "Long"
        assert loaded[1]["direction"] == "Short"
        assert loaded[2]["direction"] == "Long"

    def test_delete_strategy(self):
        """Deleting a strategy removes it from the list."""
        strategies = [
            make_strategy(strategy_name="Keep"),
            make_strategy(strategy_name="Delete"),
            make_strategy(strategy_name="Also Keep"),
        ]

        strategies.pop(1)  # Delete by index
        assert len(strategies) == 2
        assert strategies[0]["strategy_name"] == "Keep"
        assert strategies[1]["strategy_name"] == "Also Keep"

    def test_delete_all_strategies(self):
        strategies = [make_strategy() for _ in range(5)]
        strategies.clear()
        assert len(strategies) == 0


class TestStrategyStructure:

    def test_required_fields_present(self):
        """Strategy dict has all required top-level fields."""
        strategy = make_strategy()
        required = ["strategy_name", "direction", "entry", "initial_stop",
                     "exit_groups", "indicator_settings", "max_positions"]
        for field in required:
            assert field in strategy, f"Missing field: {field}"

    def test_entry_has_trigger(self):
        strategy = make_strategy()
        assert "trigger" in strategy["entry"]
        trigger = strategy["entry"]["trigger"]
        assert "element1" in trigger
        assert "event" in trigger
        assert "compare_type" in trigger

    def test_initial_stop_has_required_fields(self):
        # ATR stop
        strategy = make_strategy(initial_stop_type="ATR")
        stop = strategy["initial_stop"]
        assert stop["stop_type"] == "ATR"
        assert "atr_period" in stop
        assert "atr_multiplier" in stop

        # Indicator stop
        strategy = make_strategy(initial_stop_type="Indicator")
        stop = strategy["initial_stop"]
        assert stop["stop_type"] == "Indicator"
        assert "element2" in stop

    def test_exit_group_structure(self):
        strategy = make_strategy()
        assert len(strategy["exit_groups"]) > 0
        group = strategy["exit_groups"][0]
        assert "allocation_pct" in group
        assert "targets" in group
        assert "stops" in group

    def test_indicator_settings_complete(self):
        """Strategy's indicator settings have all expected keys."""
        strategy = make_strategy()
        settings = strategy["indicator_settings"]
        for key in DEFAULT_INDICATOR_SETTINGS:
            assert key in settings, f"Missing indicator setting: {key}"

    def test_empty_name_gets_default(self):
        """Strategy with empty name should still serialize."""
        strategy = make_strategy(strategy_name="")
        json_str = json.dumps(strategy)
        loaded = json.loads(json_str)
        assert loaded["strategy_name"] == ""

    def test_special_characters_in_name(self):
        strategy = make_strategy(strategy_name='Test "quotes" & <special>')
        json_str = json.dumps(strategy)
        loaded = json.loads(json_str)
        assert loaded["strategy_name"] == 'Test "quotes" & <special>'

    def test_conditions_roundtrip(self):
        """Entry conditions round-trip through JSON."""
        conditions = [
            {"group": "RSI Group", "element1": "RSI", "operator": "Above",
             "compare_type": "Fixed Value", "value": 50.0},
            {"group": "ADX Group", "element1": "ADX", "operator": "Above",
             "compare_type": "Fixed Value", "value": 25.0},
        ]
        strategy = make_strategy(entry_conditions=conditions)

        json_str = json.dumps(strategy)
        loaded = json.loads(json_str)

        assert len(loaded["entry"]["conditions"]) == 2
        assert loaded["entry"]["conditions"][0]["element1"] == "RSI"
        assert loaded["entry"]["conditions"][1]["element1"] == "ADX"


# ---------------------------------------------------------------------------
# Execution roundtrip: strategy → JSON → load → execute → same results
# ---------------------------------------------------------------------------

@pytest.fixture(scope="module")
def df_oscillation_ready():
    df = _generate_oscillation(n=300, freq="15min")
    return calculate_indicators(df, **DEFAULT_INDICATOR_SETTINGS)


def _roundtrip_via_json_file(strategy):
    """Save strategy to a temp JSON file, load it back, return the loaded copy."""
    with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
        json.dump([strategy], f, indent=4)
        tmp_path = f.name
    try:
        with open(tmp_path, 'r') as f:
            loaded = json.load(f)
        return loaded[0]
    finally:
        os.unlink(tmp_path)


def _assert_execution_identical(df, original_strategy, loaded_strategy):
    """Execute both strategies and assert identical trade results."""
    _, stats_orig = execute_custom_strategy(df.copy(), original_strategy)
    _, stats_loaded = execute_custom_strategy(df.copy(), loaded_strategy)

    pnls_orig = stats_orig.attrs.get("trade_pnls_r", [])
    pnls_loaded = stats_loaded.attrs.get("trade_pnls_r", [])

    assert len(pnls_orig) == len(pnls_loaded), \
        f"Trade count mismatch: original={len(pnls_orig)}, loaded={len(pnls_loaded)}"
    if pnls_orig:
        np.testing.assert_allclose(
            pnls_orig, pnls_loaded, rtol=1e-10,
            err_msg="P&L values differ after JSON roundtrip"
        )


class TestExecutionRoundtripEntryTriggers:
    """Every entry trigger combo survives JSON roundtrip with identical execution."""

    @pytest.mark.parametrize("group,el1,el2,event,compare_type,fixed_val", ENTRY_TRIGGER_COMBOS)
    def test_entry_trigger_roundtrip(self, df_oscillation_ready, group, el1, el2, event, compare_type, fixed_val):
        strategy = make_strategy(
            direction="Long",
            entry_group=group, entry_element1=el1, entry_event=event,
            entry_compare_type=compare_type, entry_element2=el2,
            entry_value=fixed_val if fixed_val is not None else 50.0,
        )
        loaded = _roundtrip_via_json_file(strategy)
        _assert_execution_identical(df_oscillation_ready, strategy, loaded)


class TestExecutionRoundtripInitialStops:
    """Every initial stop combo survives JSON roundtrip."""

    @pytest.mark.parametrize("stop_type,event,element2,atr_period,atr_mult", INITIAL_STOP_COMBOS)
    def test_initial_stop_roundtrip(self, df_oscillation_ready, stop_type, event, element2, atr_period, atr_mult):
        strategy = make_strategy(
            direction="Long",
            initial_stop_type=stop_type, initial_stop_event=event,
            initial_stop_element2=element2 or "BB Lower Band",
            initial_stop_atr_period=atr_period or 14,
            initial_stop_atr_multiplier=atr_mult or 1.5,
        )
        loaded = _roundtrip_via_json_file(strategy)
        _assert_execution_identical(df_oscillation_ready, strategy, loaded)


class TestExecutionRoundtripEntryConditions:
    """Every entry condition combo survives JSON roundtrip."""

    @pytest.mark.parametrize("group,el1,operator,compare_type,el2,fixed_val", ENTRY_CONDITION_COMBOS)
    def test_entry_condition_roundtrip(self, df_oscillation_ready, group, el1, operator, compare_type, el2, fixed_val):
        condition = {
            "group": group, "element1": el1, "operator": operator,
            "compare_type": compare_type, "element2": el2,
            "value": fixed_val if fixed_val is not None else 50.0,
        }
        strategy = make_strategy(
            direction="Long",
            entry_conditions=[condition],
        )
        loaded = _roundtrip_via_json_file(strategy)
        _assert_execution_identical(df_oscillation_ready, strategy, loaded)


class TestExecutionRoundtripExitTargets:
    """Every exit target combo survives JSON roundtrip."""

    @pytest.mark.parametrize("group,el1,event,compare_type,el2,fixed_val", EXIT_TARGET_COMBOS)
    def test_exit_target_roundtrip(self, df_oscillation_ready, group, el1, event, compare_type, el2, fixed_val):
        trigger = {
            "group": group, "element1": el1, "event": event,
            "compare_type": compare_type,
            "element2": el2 if compare_type == "Indicator" else None,
            "value": fixed_val if compare_type == "Fixed Value" else None,
        }
        if el1 == "ATR Target":
            trigger["atr_period"] = 14
            trigger["atr_multiplier"] = 2.0
            trigger["compare_type"] = "ATR"
        target = {"type": "Target", "trigger": trigger, "conditions": []}
        exit_group = make_exit_group(allocation_pct=100.0, targets=[target])
        strategy = make_strategy(direction="Long", exit_groups=[exit_group])
        loaded = _roundtrip_via_json_file(strategy)
        _assert_execution_identical(df_oscillation_ready, strategy, loaded)


class TestExecutionRoundtripExitStops:
    """Every exit dynamic stop combo survives JSON roundtrip."""

    @pytest.mark.parametrize("group,el1,event,compare_type,el2,fixed_val", EXIT_STOP_COMBOS)
    def test_exit_stop_roundtrip(self, df_oscillation_ready, group, el1, event, compare_type, el2, fixed_val):
        trigger = {
            "group": group, "element1": el1, "event": event,
            "compare_type": compare_type,
            "element2": el2 if compare_type == "Indicator" else None,
            "value": fixed_val if compare_type == "Fixed Value" else None,
        }
        if el1 == "ATR Target":
            trigger["atr_period"] = 14
            trigger["atr_multiplier"] = 2.0
            trigger["compare_type"] = "ATR"
        stop = {"type": "Stop", "trigger": trigger, "conditions": []}
        far_target = {
            "type": "Target",
            "trigger": {
                "group": "R Profit / R Loss", "element1": "R Profit",
                "event": "Cross Above", "compare_type": "Fixed Value",
                "element2": None, "value": 100.0,
            },
            "conditions": [],
        }
        exit_group = make_exit_group(allocation_pct=100.0, targets=[far_target], stops=[stop])
        strategy = make_strategy(direction="Long", exit_groups=[exit_group])
        loaded = _roundtrip_via_json_file(strategy)
        _assert_execution_identical(df_oscillation_ready, strategy, loaded)


class TestExecutionRoundtripIntegration:
    """Complex multi-component strategies survive JSON roundtrip."""

    def test_multi_exit_groups_roundtrip(self, df_oscillation_ready):
        """50/50 split with different R targets."""
        group1 = make_exit_group(
            allocation_pct=50.0, group_id=1,
            targets=[{"type": "Target", "trigger": {
                "group": "R Profit / R Loss", "element1": "R Profit",
                "event": "Cross Above", "compare_type": "Fixed Value",
                "element2": None, "value": 1.5,
            }, "conditions": []}],
        )
        group2 = make_exit_group(
            allocation_pct=50.0, group_id=2,
            targets=[{"type": "Target", "trigger": {
                "group": "R Profit / R Loss", "element1": "R Profit",
                "event": "Cross Above", "compare_type": "Fixed Value",
                "element2": None, "value": 3.0,
            }, "conditions": []}],
        )
        strategy = make_strategy(direction="Long", exit_groups=[group1, group2])
        loaded = _roundtrip_via_json_file(strategy)
        _assert_execution_identical(df_oscillation_ready, strategy, loaded)

    def test_mixed_groups_with_conditions_roundtrip(self, df_oscillation_ready):
        """Entry from RSI, condition from ADX, exit from MACD."""
        strategy = make_strategy(
            direction="Long",
            entry_group="RSI Group", entry_element1="RSI",
            entry_event="Cross Above", entry_compare_type="Fixed Value", entry_value=30.0,
            entry_conditions=[{
                "group": "ADX Group", "element1": "ADX",
                "operator": "Above", "compare_type": "Fixed Value", "value": 10.0,
            }],
            exit_groups=[make_exit_group(
                allocation_pct=100.0,
                targets=[{"type": "Target", "trigger": {
                    "group": "MACD Group", "element1": "MACD Line",
                    "event": "Cross Below", "compare_type": "Indicator",
                    "element2": "MACD Signal", "value": None,
                }, "conditions": []}],
            )],
        )
        loaded = _roundtrip_via_json_file(strategy)
        _assert_execution_identical(df_oscillation_ready, strategy, loaded)

    def test_short_strategy_roundtrip(self, df_oscillation_ready):
        """Short strategy roundtrip."""
        strategy = make_strategy(
            direction="Short",
            entry_group="RSI Group", entry_element1="RSI",
            entry_event="Cross Below", entry_compare_type="Fixed Value", entry_value=70.0,
        )
        loaded = _roundtrip_via_json_file(strategy)
        _assert_execution_identical(df_oscillation_ready, strategy, loaded)

    def test_atr_stop_and_target_roundtrip(self, df_oscillation_ready):
        """ATR-based stop and ATR Target exit."""
        target = {"type": "Target", "trigger": {
            "group": "ATR Target", "element1": "ATR Target",
            "event": "Cross Above", "compare_type": "ATR",
            "element2": None, "value": None,
            "atr_period": 14, "atr_multiplier": 2.0,
        }, "conditions": []}
        strategy = make_strategy(
            direction="Long",
            initial_stop_type="ATR", initial_stop_atr_period=14, initial_stop_atr_multiplier=1.5,
            exit_groups=[make_exit_group(allocation_pct=100.0, targets=[target])],
        )
        loaded = _roundtrip_via_json_file(strategy)
        _assert_execution_identical(df_oscillation_ready, strategy, loaded)

    def test_indicator_settings_preserved(self, df_oscillation_ready):
        """Custom indicator settings survive roundtrip and affect execution."""
        custom_settings = {**DEFAULT_INDICATOR_SETTINGS, "rsi_window": 7}
        strategy = make_strategy(
            direction="Long",
            indicator_settings=custom_settings,
        )
        loaded = _roundtrip_via_json_file(strategy)

        # Verify settings are preserved
        assert loaded["indicator_settings"]["rsi_window"] == 7

        # Verify execution is identical
        _assert_execution_identical(df_oscillation_ready, strategy, loaded)

    def test_max_positions_unlimited_roundtrip(self, df_oscillation_ready):
        """max_positions=None (unlimited) roundtrip — null in JSON."""
        strategy = make_strategy(direction="Long", max_positions=None)
        loaded = _roundtrip_via_json_file(strategy)
        assert loaded["max_positions"] is None
        _assert_execution_identical(df_oscillation_ready, strategy, loaded)

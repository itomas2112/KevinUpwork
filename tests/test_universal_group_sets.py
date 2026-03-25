"""
Tests for the universal group set system.
Covers: candidate labeling, strategy building, run generation, import/export.
"""
import json
import sys
import pytest
from copy import deepcopy
from unittest.mock import MagicMock

# Mock streamlit before importing modules that depend on it
if "streamlit" not in sys.modules:
    sys.modules["streamlit"] = MagicMock()

from ui.grid_search_helpers import (
    format_candidate_label,
    format_run_label,
    build_replacement_strategy,
    generate_run_configs,
    _candidate_to_entry_trigger,
    _candidate_to_condition,
    _candidate_to_static_stop,
    _candidate_to_exit_trigger,
)
from strategies.group_set_manager import (
    import_group_set,
    _deduplicate_candidates,
    _strip_legacy_fields,
)
from tests.conftest import make_strategy


# =====================================================================
# Candidate labeling
# =====================================================================

class TestFormatCandidateLabel:
    def test_indicator_pair(self):
        c = {"element1": "Price", "compare_type": "Indicator", "element2": "Tenkan"}
        assert format_candidate_label(c) == "Price vs Tenkan"

    def test_fixed_value(self):
        c = {"element1": "RSI", "compare_type": "Fixed Value", "value": 70}
        assert format_candidate_label(c) == "RSI vs 70"

    def test_atr_stop(self):
        c = {"stop_type": "ATR", "atr_period": 14, "atr_multiplier": 2.0}
        assert format_candidate_label(c) == "ATR(14) x 2.0"

    def test_atr_target(self):
        c = {"element1": "ATR Target", "atr_period": 10, "atr_multiplier": 3.0}
        assert format_candidate_label(c) == "ATR Target(10 x 3.0)"

    def test_none(self):
        assert format_candidate_label(None) == "None"


class TestFormatRunLabel:
    def test_indicator_with_event(self):
        c = {"element1": "Price", "compare_type": "Indicator", "element2": "Tenkan"}
        assert format_run_label(c, "Cross Above") == "Price (cross above) Tenkan"

    def test_fixed_value_with_event(self):
        c = {"element1": "RSI", "compare_type": "Fixed Value", "value": 70}
        assert format_run_label(c, "Close Above") == "RSI (close above) 70"

    def test_atr_stop_with_event(self):
        c = {"stop_type": "ATR", "atr_period": 14, "atr_multiplier": 2.0}
        assert format_run_label(c, "Cross Below") == "ATR(14) x 2.0 (cross below)"


# =====================================================================
# Candidate-to-strategy conversion
# =====================================================================

class TestCandidateConversion:
    def test_to_entry_trigger(self):
        c = {"group": "Price & Indicators", "element1": "Price",
             "compare_type": "Indicator", "element2": "Tenkan"}
        result = _candidate_to_entry_trigger(c, "Cross Above")
        assert result["event"] == "Cross Above"
        assert result["element1"] == "Price"
        assert result["element2"] == "Tenkan"

    def test_to_condition_cross_above(self):
        c = {"group": "RSI Group", "element1": "RSI",
             "compare_type": "Fixed Value", "value": 50}
        result = _candidate_to_condition(c, "Cross Above")
        assert result["operator"] == "Above"
        assert "event" not in result

    def test_to_condition_cross_below(self):
        c = {"element1": "RSI", "compare_type": "Fixed Value", "value": 30}
        result = _candidate_to_condition(c, "Cross Below")
        assert result["operator"] == "Below"

    def test_to_condition_operator_passthrough(self):
        """When event is already an operator name, it passes through."""
        c = {"element1": "Price", "compare_type": "Indicator", "element2": "Kijun"}
        result = _candidate_to_condition(c, "Above")
        assert result["operator"] == "Above"

    def test_to_static_stop_atr(self):
        c = {"stop_type": "ATR", "atr_period": 14, "atr_multiplier": 2.0}
        result = _candidate_to_static_stop(c, "Long", "Cross Below")
        assert result["stop_type"] == "ATR"
        assert result["event"] == "Cross Below"
        assert result["atr_period"] == 14

    def test_to_static_stop_indicator(self):
        c = {"group": "Price & Indicators", "element1": "Price",
             "compare_type": "Indicator", "element2": "BB Lower Band"}
        result = _candidate_to_static_stop(c, "Long", "Close Below")
        assert result["stop_type"] == "Indicator"
        assert result["event"] == "Close Below"
        assert result["element2"] == "BB Lower Band"

    def test_to_exit_trigger(self):
        c = {"group": "RSI Group", "element1": "RSI",
             "compare_type": "Fixed Value", "value": 70}
        result = _candidate_to_exit_trigger(c, "Cross Above")
        assert result["event"] == "Cross Above"
        assert result["value"] == 70

    def test_to_exit_trigger_atr_target(self):
        c = {"element1": "ATR Target", "atr_period": 14, "atr_multiplier": 3.0}
        result = _candidate_to_exit_trigger(c, "Cross Above")
        assert result["atr_period"] == 14
        assert result["atr_multiplier"] == 3.0
        assert result["event"] == "Cross Above"


# =====================================================================
# Build replacement strategy
# =====================================================================

class TestBuildReplacementStrategy:
    def _base(self):
        return make_strategy(direction="Long")

    def test_trigger_replacement(self):
        c = {"element1": "Price", "compare_type": "Indicator", "element2": "Kijun"}
        s = build_replacement_strategy(self._base(), "trigger", c, "Cross Above")
        assert s["entry"]["trigger"]["element1"] == "Price"
        assert s["entry"]["trigger"]["element2"] == "Kijun"
        assert s["entry"]["trigger"]["event"] == "Cross Above"

    def test_condition_appended(self):
        base = self._base()
        orig_count = len(base["entry"]["conditions"])
        c = {"element1": "ADX", "compare_type": "Fixed Value", "value": 25}
        s = build_replacement_strategy(base, "condition", c, "Above")
        assert len(s["entry"]["conditions"]) == orig_count + 1
        assert s["entry"]["conditions"][-1]["operator"] == "Above"

    def test_static_stop_replacement(self):
        c = {"stop_type": "ATR", "atr_period": 20, "atr_multiplier": 1.5}
        s = build_replacement_strategy(self._base(), "static_stop", c, "Cross Below")
        assert s["initial_stop"]["stop_type"] == "ATR"
        assert s["initial_stop"]["atr_period"] == 20

    def test_dynamic_stop_replacement(self):
        c = {"element1": "Price", "compare_type": "Indicator", "element2": "BB Lower Band"}
        s = build_replacement_strategy(self._base(), "dynamic_stop", c, "Cross Below")
        assert s["exit_groups"][0]["stops"][0]["trigger"]["element2"] == "BB Lower Band"
        assert s["exit_groups"][0]["stops"][0]["trigger"]["event"] == "Cross Below"

    def test_target_replacement(self):
        c = {"element1": "RSI", "compare_type": "Fixed Value", "value": 80}
        s = build_replacement_strategy(self._base(), "target", c, "Cross Above")
        assert s["exit_groups"][0]["targets"][0]["trigger"]["value"] == 80
        assert s["exit_groups"][0]["targets"][0]["trigger"]["event"] == "Cross Above"

    def test_does_not_mutate_base(self):
        base = self._base()
        original_trigger = deepcopy(base["entry"]["trigger"])
        c = {"element1": "Price", "compare_type": "Indicator", "element2": "Kijun"}
        build_replacement_strategy(base, "trigger", c, "Cross Above")
        assert base["entry"]["trigger"] == original_trigger

    def test_extra_condition(self):
        c = {"element1": "RSI", "compare_type": "Fixed Value", "value": 70}
        cond = {"element1": "ADX", "compare_type": "Fixed Value", "value": 25}
        s = build_replacement_strategy(self._base(), "target", c, "Cross Above",
                                       extra_condition=cond, extra_condition_event="Above")
        # Last condition should be the extra one
        assert s["entry"]["conditions"][-1]["element1"] == "ADX"
        assert s["entry"]["conditions"][-1]["operator"] == "Above"


# =====================================================================
# Generate run configs
# =====================================================================

class TestGenerateRunConfigs:
    def _base(self):
        return make_strategy(direction="Long")

    def test_single_candidate_single_event(self):
        candidates = [{"element1": "Price", "compare_type": "Indicator", "element2": "Tenkan"}]
        runs = generate_run_configs(self._base(), "trigger", candidates, ["Cross Above"])
        assert len(runs) == 1
        assert "cross above" in runs[0][0].lower()

    def test_single_candidate_multiple_events(self):
        candidates = [{"element1": "Price", "compare_type": "Indicator", "element2": "Tenkan"}]
        runs = generate_run_configs(self._base(), "trigger", candidates,
                                    ["Cross Above", "Close Above"])
        assert len(runs) == 2

    def test_multiple_candidates_multiple_events(self):
        candidates = [
            {"element1": "Price", "compare_type": "Indicator", "element2": "Tenkan"},
            {"element1": "Price", "compare_type": "Indicator", "element2": "Kijun"},
        ]
        runs = generate_run_configs(self._base(), "trigger", candidates,
                                    ["Cross Above", "Cross Below"])
        assert len(runs) == 4  # 2 candidates × 2 events

    def test_condition_uses_operators(self):
        candidates = [{"element1": "RSI", "compare_type": "Fixed Value", "value": 50}]
        runs = generate_run_configs(self._base(), "condition", candidates, ["Above", "Below"])
        assert len(runs) == 2

    def test_target_with_cross_combination(self):
        candidates = [{"element1": "RSI", "compare_type": "Fixed Value", "value": 70}]
        cond_candidates = [
            {"element1": "ADX", "compare_type": "Fixed Value", "value": 25},
        ]
        runs = generate_run_configs(self._base(), "target", candidates, ["Cross Above"],
                                    condition_candidates=cond_candidates, condition_event="Above")
        # 1 candidate × 1 event × (1 standalone + 1 condition) = 2
        assert len(runs) == 2

    def test_empty_candidates(self):
        runs = generate_run_configs(self._base(), "trigger", [], ["Cross Above"])
        assert len(runs) == 0

    def test_empty_events(self):
        candidates = [{"element1": "Price", "compare_type": "Indicator", "element2": "Tenkan"}]
        runs = generate_run_configs(self._base(), "trigger", candidates, [])
        assert len(runs) == 0

    def test_invalid_candidates_skipped(self):
        """Candidates that produce invalid strategies are skipped."""
        candidates = [{"element1": "NonexistentIndicator", "compare_type": "Indicator",
                       "element2": "AlsoNonexistent"}]
        runs = generate_run_configs(self._base(), "trigger", candidates, ["Cross Above"])
        assert len(runs) == 0  # Should be skipped by validation


# =====================================================================
# Import / export
# =====================================================================

class TestImportGroupSet:
    def test_new_format(self):
        data = json.dumps({"name": "Test", "candidates": [
            {"element1": "Price", "compare_type": "Indicator", "element2": "Tenkan"}
        ]})
        result = import_group_set(data)
        assert result["name"] == "Test"
        assert len(result["candidates"]) == 1

    def test_old_format_stripped(self):
        """Old format with type/event/operator should be cleaned."""
        data = json.dumps({
            "name": "Old",
            "type": "trigger",
            "candidates": [
                {"element1": "Price", "event": "Cross Above", "compare_type": "Indicator",
                 "element2": "Tenkan"},
                {"element1": "Price", "operator": "Above", "compare_type": "Indicator",
                 "element2": "Tenkan"},
            ]
        })
        result = import_group_set(data)
        assert "type" not in result
        for c in result["candidates"]:
            assert "event" not in c
            assert "operator" not in c
        # After stripping, both candidates are identical → deduplicated to 1
        assert len(result["candidates"]) == 1

    def test_missing_name_raises(self):
        with pytest.raises(ValueError, match="name"):
            import_group_set(json.dumps({"candidates": []}))

    def test_missing_candidates_raises(self):
        with pytest.raises(ValueError, match="candidates"):
            import_group_set(json.dumps({"name": "Test"}))

    def test_non_dict_raises(self):
        with pytest.raises(ValueError, match="JSON object"):
            import_group_set(json.dumps([1, 2, 3]))


class TestStripLegacyFields:
    def test_strips_type_event_operator(self):
        data = {
            "name": "Test",
            "type": "condition",
            "candidates": [
                {"element1": "Price", "event": "Cross Above", "operator": "Above",
                 "compare_type": "Indicator", "element2": "Tenkan"}
            ]
        }
        _strip_legacy_fields(data)
        assert "type" not in data
        assert "event" not in data["candidates"][0]
        assert "operator" not in data["candidates"][0]
        # Non-legacy fields preserved
        assert data["candidates"][0]["element1"] == "Price"
        assert data["candidates"][0]["compare_type"] == "Indicator"


class TestDeduplicateCandidates:
    def test_removes_duplicates(self):
        gs = {"candidates": [
            {"element1": "Price", "element2": "Tenkan"},
            {"element1": "Price", "element2": "Tenkan"},
            {"element1": "Price", "element2": "Kijun"},
        ]}
        removed = _deduplicate_candidates(gs)
        assert removed == 1
        assert len(gs["candidates"]) == 2

    def test_no_duplicates(self):
        gs = {"candidates": [
            {"element1": "Price", "element2": "Tenkan"},
            {"element1": "Price", "element2": "Kijun"},
        ]}
        removed = _deduplicate_candidates(gs)
        assert removed == 0
        assert len(gs["candidates"]) == 2

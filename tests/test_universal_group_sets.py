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
    get_mode,
    MODE_RUNTIME,
    MODE_PER_CANDIDATE,
    candidate_groups,
    extract_groups_from_set,
    extract_value_eligible_groups,
    value_in_filter,
    apply_view,
    candidate_wfo_groups,
    enumerate_variant_values,
    compute_offsets,
    offset_label,
    group_variant_combos,
    INDICATOR_PRIMARY_PARAM,
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

    def test_ema_candidates_pass_validation(self):
        """Candidates using EMA elements must validate against base strategy's ema_periods."""
        # Base strategy has 4 EMAs (ema_periods=[10, 20, 50, 200]) per DEFAULT_INDICATOR_SETTINGS
        candidates = [
            {"element1": "Price", "compare_type": "Indicator", "element2": "EMA 1"},
            {"element1": "Price", "compare_type": "Indicator", "element2": "EMA 2"},
            {"element1": "EMA 3", "compare_type": "Indicator", "element2": "EMA 4"},
        ]
        runs = generate_run_configs(self._base(), "trigger", candidates, ["Cross Above"])
        assert len(runs) == 3  # All three EMA-based candidates must pass

    def test_ema_candidates_beyond_config_rejected(self):
        """Candidates referencing an EMA beyond the configured count are still rejected."""
        # Base has 4 EMAs, so "EMA 5" must fail
        candidates = [
            {"element1": "Price", "compare_type": "Indicator", "element2": "EMA 5"},
        ]
        runs = generate_run_configs(self._base(), "trigger", candidates, ["Cross Above"])
        assert len(runs) == 0


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


# =====================================================================
# Two-mode group sets
# =====================================================================

class TestGroupSetMode:
    def test_legacy_set_defaults_to_runtime_mode(self):
        gs = {"name": "Old", "candidates": []}
        assert get_mode(gs) == MODE_RUNTIME

    def test_explicit_per_candidate_mode_preserved(self):
        gs = {"name": "New", "mode": MODE_PER_CANDIDATE, "candidates": []}
        assert get_mode(gs) == MODE_PER_CANDIDATE

    def test_strip_legacy_keeps_event_for_per_candidate_mode(self):
        """In per-candidate mode the candidate's `event` is real data, not legacy."""
        data = {
            "name": "Per",
            "mode": MODE_PER_CANDIDATE,
            "candidates": [
                {"element1": "Price", "compare_type": "Indicator",
                 "element2": "Tenkan", "event": "Cross Above"},
            ],
        }
        _strip_legacy_fields(data)
        assert data["candidates"][0]["event"] == "Cross Above"

    def test_strip_legacy_drops_event_for_runtime_mode(self):
        """Runtime mode treats `event` as legacy and removes it."""
        data = {
            "name": "Runtime",
            "mode": MODE_RUNTIME,
            "candidates": [
                {"element1": "Price", "compare_type": "Indicator",
                 "element2": "Tenkan", "event": "Cross Above"},
            ],
        }
        _strip_legacy_fields(data)
        assert "event" not in data["candidates"][0]


class TestImportPerCandidateMode:
    def test_round_trip_preserves_per_candidate_event(self):
        data = json.dumps({
            "name": "Per",
            "mode": MODE_PER_CANDIDATE,
            "candidates": [
                {"element1": "RSI", "compare_type": "Fixed Value",
                 "value": 30, "event": "Cross Above"},
                {"element1": "Price", "compare_type": "Indicator",
                 "element2": "Kijun", "event": "Close Below"},
            ],
        })
        result = import_group_set(data)
        assert result["mode"] == MODE_PER_CANDIDATE
        assert len(result["candidates"]) == 2
        assert result["candidates"][0]["event"] == "Cross Above"
        assert result["candidates"][1]["event"] == "Close Below"

    def test_per_candidate_missing_event_raises(self):
        data = json.dumps({
            "name": "Per",
            "mode": MODE_PER_CANDIDATE,
            "candidates": [
                {"element1": "RSI", "compare_type": "Fixed Value", "value": 30},
            ],
        })
        with pytest.raises(ValueError, match="event"):
            import_group_set(data)

    def test_invalid_mode_raises(self):
        data = json.dumps({
            "name": "Bad", "mode": "nonsense", "candidates": [],
        })
        with pytest.raises(ValueError, match="Invalid mode"):
            import_group_set(data)

    def test_legacy_import_without_mode_defaults_to_runtime(self):
        data = json.dumps({
            "name": "Legacy",
            "candidates": [
                {"element1": "Price", "compare_type": "Indicator", "element2": "Tenkan"}
            ],
        })
        result = import_group_set(data)
        assert result["mode"] == MODE_RUNTIME


class TestGenerateRunConfigsPerCandidateEvents:
    def _base(self):
        return make_strategy(direction="Long")

    def test_per_candidate_events_used_no_cross_product(self):
        """Each candidate's embedded event drives one run; `events` arg is ignored."""
        candidates = [
            {"element1": "Price", "compare_type": "Indicator",
             "element2": "Tenkan", "event": "Cross Above"},
            {"element1": "Price", "compare_type": "Indicator",
             "element2": "Kijun", "event": "Close Below"},
        ]
        runs = generate_run_configs(
            self._base(), "trigger", candidates,
            events=["IGNORED_AT_RUNTIME"],
            events_per_candidate=True)
        assert len(runs) == 2  # NOT 2 x 1 cross-product, just 2 runs
        labels = [lbl for lbl, _, _ in runs]
        # First run uses Cross Above, second uses Close Below
        assert any("cross above" in l.lower() and "tenkan" in l.lower() for l in labels)
        assert any("close below" in l.lower() and "kijun" in l.lower() for l in labels)

    def test_per_candidate_strategy_uses_embedded_event(self):
        candidates = [
            {"element1": "Price", "compare_type": "Indicator",
             "element2": "Kijun", "event": "Cross Above"},
        ]
        runs = generate_run_configs(
            self._base(), "trigger", candidates, events=[],
            events_per_candidate=True)
        assert len(runs) == 1
        _, strat, _ = runs[0]
        assert strat["entry"]["trigger"]["event"] == "Cross Above"

    def test_runtime_mode_still_cross_products(self):
        """Sanity: turning the new flag off keeps legacy behaviour intact."""
        candidates = [
            {"element1": "Price", "compare_type": "Indicator", "element2": "Tenkan"},
            {"element1": "Price", "compare_type": "Indicator", "element2": "Kijun"},
        ]
        runs = generate_run_configs(
            self._base(), "trigger", candidates,
            events=["Cross Above", "Cross Below"],
            events_per_candidate=False)
        assert len(runs) == 4  # 2 candidates × 2 events


# =====================================================================
# Set "view" filter — toggle indicator groups on/off + per-group value ranges
# =====================================================================

class TestCandidateGroups:
    def test_standard_indicator_pair(self):
        c = {"group": "Price & Indicators", "element1": "Price",
             "compare_type": "Indicator", "element2": "RSI"}
        assert candidate_groups(c) == {"Price & Indicators", "RSI Group"}

    def test_fixed_value_only_uses_e1_group(self):
        c = {"group": "RSI Group", "element1": "RSI",
             "compare_type": "Fixed Value", "value": 30}
        assert candidate_groups(c) == {"RSI Group"}

    def test_atr_stop(self):
        c = {"stop_type": "ATR", "atr_period": 14, "atr_multiplier": 2.0}
        assert candidate_groups(c) == {"ATR / Volume Group"}

    def test_atr_target(self):
        c = {"element1": "ATR Target", "atr_period": 10, "atr_multiplier": 3.0}
        assert candidate_groups(c) == {"ATR / Volume Group"}

    def test_r_profit_loss_contributes_nothing(self):
        c = {"group": "Price & Indicators", "element1": "R Profit",
             "compare_type": "Fixed Value", "value": 1}
        # R Profit isn't tied to an indicator
        assert candidate_groups(c) == set()

    def test_ema_element_resolves_to_price_group(self):
        c = {"group": "Price & Indicators", "element1": "Price",
             "compare_type": "Indicator", "element2": "EMA 2"}
        assert candidate_groups(c) == {"Price & Indicators"}

    def test_cross_group_pair(self):
        c = {"group": "MACD Group", "element1": "MACD Line",
             "compare_type": "Indicator", "element2": "RSI"}
        assert candidate_groups(c) == {"MACD Group", "RSI Group"}


class TestExtractGroupsFromSet:
    def test_union_across_candidates(self):
        cands = [
            {"group": "RSI Group", "element1": "RSI",
             "compare_type": "Fixed Value", "value": 30},
            {"group": "Price & Indicators", "element1": "Price",
             "compare_type": "Indicator", "element2": "Tenkan"},
            {"stop_type": "ATR", "atr_period": 14, "atr_multiplier": 2.0},
        ]
        assert extract_groups_from_set(cands) == [
            "ATR / Volume Group", "Price & Indicators", "RSI Group",
        ]

    def test_empty_set(self):
        assert extract_groups_from_set([]) == []


class TestExtractValueEligibleGroups:
    def test_only_fixed_value_groups(self):
        cands = [
            {"group": "RSI Group", "element1": "RSI",
             "compare_type": "Fixed Value", "value": 30},
            {"group": "RSI Group", "element1": "RSI",
             "compare_type": "Fixed Value", "value": 70},
            {"group": "Price & Indicators", "element1": "Price",
             "compare_type": "Indicator", "element2": "Tenkan"},
            {"group": "ADX Group", "element1": "ADX",
             "compare_type": "Fixed Value", "value": 25},
        ]
        # Only RSI and ADX have Fixed Value candidates
        assert extract_value_eligible_groups(cands) == ["ADX Group", "RSI Group"]

    def test_ignores_atr_stop(self):
        cands = [{"stop_type": "ATR", "atr_period": 14, "atr_multiplier": 2.0}]
        assert extract_value_eligible_groups(cands) == []


class TestValueInFilter:
    def test_no_filter_passes_anything(self):
        assert value_in_filter(123, None) is True

    def test_in_range_no_step(self):
        vf = {"low": 20, "high": 80}
        assert value_in_filter(30, vf) is True
        assert value_in_filter(20, vf) is True
        assert value_in_filter(80, vf) is True

    def test_out_of_range(self):
        vf = {"low": 20, "high": 80}
        assert value_in_filter(10, vf) is False
        assert value_in_filter(90, vf) is False

    def test_with_step_grid(self):
        vf = {"low": 20, "high": 80, "step": 10}
        # On grid
        for v in (20, 30, 40, 50, 60, 70, 80):
            assert value_in_filter(v, vf) is True, f"{v} should pass"
        # Off grid
        for v in (25, 33, 71):
            assert value_in_filter(v, vf) is False, f"{v} should fail"

    def test_step_zero_means_no_step_constraint(self):
        vf = {"low": 20, "high": 80, "step": 0}
        assert value_in_filter(35, vf) is True

    def test_float_tolerance(self):
        vf = {"low": 0.0, "high": 1.0, "step": 0.1}
        assert value_in_filter(0.3, vf) is True
        # 0.30000001 is on-grid within tolerance
        assert value_in_filter(0.3 + 1e-9, vf) is True


class TestApplyView:
    def _set(self):
        return [
            {"group": "RSI Group", "element1": "RSI",
             "compare_type": "Fixed Value", "value": 30},
            {"group": "RSI Group", "element1": "RSI",
             "compare_type": "Fixed Value", "value": 50},
            {"group": "RSI Group", "element1": "RSI",
             "compare_type": "Fixed Value", "value": 70},
            {"group": "Price & Indicators", "element1": "Price",
             "compare_type": "Indicator", "element2": "Tenkan"},
            {"stop_type": "ATR", "atr_period": 14, "atr_multiplier": 2.0},
        ]

    def test_no_filter_returns_all(self):
        cands = self._set()
        assert apply_view(cands, active_groups=None, value_filters=None) == cands

    def test_toggle_off_atr(self):
        cands = self._set()
        active = {"RSI Group", "Price & Indicators"}  # ATR not included
        result = apply_view(cands, active_groups=active)
        # 4 of 5 survive (ATR Stop is dropped)
        assert len(result) == 4
        for c in result:
            assert c.get("stop_type") != "ATR"

    def test_toggle_off_drops_cross_group_pair(self):
        cands = [
            {"group": "MACD Group", "element1": "MACD Line",
             "compare_type": "Indicator", "element2": "RSI"},
        ]
        # Drop RSI → cross-group candidate goes too
        result = apply_view(cands, active_groups={"MACD Group"})
        assert result == []

    def test_value_range_filter(self):
        cands = self._set()
        vf = {"RSI Group": {"low": 50, "high": 80, "step": 10}}
        result = apply_view(cands, value_filters=vf)
        # RSI 30 dropped, RSI 50 + 70 kept, Price/Tenkan + ATR Stop kept
        rsi_values = [c["value"] for c in result if c.get("element1") == "RSI"]
        assert sorted(rsi_values) == [50, 70]
        assert any(c.get("stop_type") == "ATR" for c in result)

    def test_value_range_with_offgrid_dropped(self):
        cands = [
            {"group": "RSI Group", "element1": "RSI",
             "compare_type": "Fixed Value", "value": 35},  # off-grid
            {"group": "RSI Group", "element1": "RSI",
             "compare_type": "Fixed Value", "value": 30},  # on-grid
        ]
        vf = {"RSI Group": {"low": 20, "high": 80, "step": 10}}
        result = apply_view(cands, value_filters=vf)
        assert len(result) == 1
        assert result[0]["value"] == 30

    def test_combined_toggle_and_range(self):
        cands = self._set()
        active = {"RSI Group"}  # drop Price + ATR
        vf = {"RSI Group": {"low": 50, "high": 70, "step": 10}}
        result = apply_view(cands, active_groups=active, value_filters=vf)
        # Only RSI 50 and RSI 70 survive
        assert len(result) == 2
        assert sorted(c["value"] for c in result) == [50, 70]

    def test_value_filter_does_not_affect_indicator_pair_candidates(self):
        """Indicator-pair candidates have no `value` so the value filter ignores them."""
        cands = [
            {"group": "Price & Indicators", "element1": "Price",
             "compare_type": "Indicator", "element2": "BB Upper Band"},
        ]
        vf = {"Price & Indicators": {"low": 0, "high": 1, "step": 1}}
        # Should NOT be dropped — it has no `value`
        result = apply_view(cands, value_filters=vf)
        assert len(result) == 1

    def test_r_profit_loss_candidate_unaffected_by_indicator_toggles(self):
        """R-unit candidates touch no indicator group, so toggles don't drop them."""
        cands = [
            {"group": "Price & Indicators", "element1": "R Profit",
             "compare_type": "Fixed Value", "value": 1},
        ]
        result = apply_view(cands, active_groups=set())  # everything off
        assert len(result) == 1


# =====================================================================
# Indicator ranges — variant enumeration + offset labels
# =====================================================================

class TestEnumerateVariantValues:
    def test_inclusive_of_max_when_on_grid(self):
        assert enumerate_variant_values(10, 30, 5) == [10.0, 15.0, 20.0, 25.0, 30.0]

    def test_excludes_max_when_off_grid(self):
        # 10, 14, 18, 22, 26 (28 would exceed 30 only if on grid; 26+5=31 > 30)
        assert enumerate_variant_values(10, 30, 4) == [10.0, 14.0, 18.0, 22.0, 26.0, 30.0]

    def test_single_value_when_min_equals_max(self):
        assert enumerate_variant_values(20, 20, 5) == [20.0]

    def test_invalid_step_collapses_to_single(self):
        assert enumerate_variant_values(10, 30, 0) == [10.0]
        assert enumerate_variant_values(10, 30, -5) == [10.0]

    def test_float_range(self):
        result = enumerate_variant_values(1.0, 3.0, 0.5)
        assert result == [1.0, 1.5, 2.0, 2.5, 3.0]


class TestComputeOffsets:
    def test_user_example_rsi_10_30_step_5(self):
        # values 10,15,20,25,30 → midpoint 20 → (-2,-1,0,+1,+2)
        result = compute_offsets([10, 15, 20, 25, 30])
        assert result == [(-2, 10), (-1, 15), (0, 20), (1, 25), (2, 30)]

    def test_even_count_lower_bias(self):
        # values 10,15,20,25 → midpoint 17.5; 15 vs 20 are equidistant
        # lower-bias picks 15 = (0)
        result = compute_offsets([10, 15, 20, 25])
        assert result == [(-1, 10), (0, 15), (1, 20), (2, 25)]

    def test_single_value_is_offset_zero(self):
        assert compute_offsets([42]) == [(0, 42)]

    def test_empty_input(self):
        assert compute_offsets([]) == []

    def test_already_sorted_irrelevant(self):
        # input in any order, output sorted ascending
        result = compute_offsets([30, 10, 20])
        assert result == [(-1, 10), (0, 20), (1, 30)]


class TestOffsetLabel:
    def test_zero(self):
        assert offset_label(0) == "(0)"

    def test_positive(self):
        assert offset_label(2) == "(+2)"
        assert offset_label(1) == "(+1)"

    def test_negative(self):
        assert offset_label(-1) == "(-1)"
        assert offset_label(-3) == "(-3)"


class TestCandidateWfoGroups:
    def test_rsi_vs_fixed(self):
        c = {"group": "RSI Group", "element1": "RSI",
             "compare_type": "Fixed Value", "value": 30}
        assert candidate_wfo_groups(c) == {"rsi"}

    def test_rsi_vs_stoch(self):
        c = {"group": "RSI Group", "element1": "RSI",
             "compare_type": "Indicator", "element2": "Stoch %K"}
        assert candidate_wfo_groups(c) == {"rsi", "stoch"}

    def test_atr_stop_returns_atr(self):
        c = {"stop_type": "ATR", "atr_period": 14, "atr_multiplier": 2.0}
        assert candidate_wfo_groups(c) == {"atr"}

    def test_atr_target_returns_atr(self):
        c = {"element1": "ATR Target", "atr_period": 14, "atr_multiplier": 2.0}
        assert candidate_wfo_groups(c) == {"atr"}

    def test_price_vs_tenkan_no_rangeable_groups(self):
        # Tenkan/Price aren't param-tunable in our model
        c = {"group": "Price & Indicators", "element1": "Price",
             "compare_type": "Indicator", "element2": "Tenkan"}
        assert candidate_wfo_groups(c) == set()

    def test_price_vs_bb_returns_bb(self):
        c = {"group": "Price & Indicators", "element1": "Price",
             "compare_type": "Indicator", "element2": "BB Upper Band"}
        assert candidate_wfo_groups(c) == {"bb"}

    def test_ema_excluded_from_v1(self):
        c = {"group": "Price & Indicators", "element1": "Price",
             "compare_type": "Indicator", "element2": "EMA 2"}
        assert candidate_wfo_groups(c) == set()

    def test_r_profit_excluded(self):
        c = {"element1": "R Profit", "compare_type": "Fixed Value", "value": 1}
        assert candidate_wfo_groups(c) == set()


class TestGroupVariantCombos:
    def test_rsi_user_example(self):
        gs = {
            "name": "T",
            "candidates": [],
            "indicator_ranges": {"rsi": {"rsi_window": [10, 30, 5]}},
        }
        result = group_variant_combos(gs)
        assert "rsi" in result
        rsi = result["rsi"]
        # 5 variants, offsets -2..+2, primary param rsi_window
        assert [offset for offset, _ in rsi] == [-2, -1, 0, 1, 2]
        assert [params["rsi_window"] for _, params in rsi] == [10, 15, 20, 25, 30]
        # ints, not floats (rsi_window is an int param)
        for _, params in rsi:
            assert isinstance(params["rsi_window"], int)

    def test_no_ranges_returns_empty(self):
        gs = {"name": "T", "candidates": []}
        assert group_variant_combos(gs) == {}

    def test_unknown_group_skipped(self):
        gs = {"name": "T", "candidates": [],
              "indicator_ranges": {"nonsense": {"x": [1, 5, 1]}}}
        assert group_variant_combos(gs) == {}

    def test_param_not_primary_skipped(self):
        # Only rsi_window is primary for rsi; stoch_k_smooth isn't primary for stoch
        gs = {"name": "T", "candidates": [],
              "indicator_ranges": {"stoch": {"stoch_k_smooth": [1, 5, 1]}}}
        assert group_variant_combos(gs) == {}

    def test_degenerate_range_one_variant(self):
        gs = {"name": "T", "candidates": [],
              "indicator_ranges": {"rsi": {"rsi_window": [14, 14, 1]}}}
        result = group_variant_combos(gs)
        assert result["rsi"] == [(0, {"rsi_window": 14})]

    def test_int_coercion_dedupes(self):
        # range 10-12 step 0.4 → values 10.0, 10.4, 10.8, 11.2, 11.6, 12.0
        # int-rounded → 10, 10, 11, 11, 12, 12 → deduped to [10, 11, 12]
        gs = {"name": "T", "candidates": [],
              "indicator_ranges": {"rsi": {"rsi_window": [10, 12, 0.4]}}}
        result = group_variant_combos(gs)
        values = [params["rsi_window"] for _, params in result["rsi"]]
        assert values == [10, 11, 12]


class TestPrimaryParamMap:
    def test_v1_groups_have_primary(self):
        # Sanity: every group we expect to range has a primary param
        for grp in ("rsi", "stoch", "adx", "atr", "macd", "bb"):
            assert grp in INDICATOR_PRIMARY_PARAM

    def test_ema_intentionally_omitted(self):
        # EMA per-instance ranging is out of scope for v1
        assert "ema" not in INDICATOR_PRIMARY_PARAM

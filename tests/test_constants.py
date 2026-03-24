"""
Tests for config/constants.py — INDICATOR_MAP consistency, group definitions.
"""
import pytest
from config.constants import (
    INDICATOR_MAP,
    GROUP_MAP,
    GROUP_NAMES,
    PRICE_AND_INDICATORS,
    RSI_GROUP,
    CMB_GROUP,
    STOCH_GROUP,
    ADX_GROUP,
    MACD_GROUP,
    ATR_VOLUME_GROUP,
    EVENT_TYPES,
    STOP_EVENT_TYPES,
    CONDITION_OPERATORS,
    CONDITION_COMPARE_TYPES,
    R_PROFIT_LOSS_ELEMENTS,
    ATR_TARGET_ELEMENTS,
    get_group_elements,
    get_indicator_map,
)


class TestIndicatorMap:

    def test_all_group_elements_have_map_entries(self):
        """Every element in every group must resolve to a column name."""
        all_elements = set()
        for group_name, elements in GROUP_MAP.items():
            all_elements.update(elements)

        missing = all_elements - set(INDICATOR_MAP.keys())
        assert missing == set(), f"Elements missing from INDICATOR_MAP: {missing}"

    def test_no_duplicate_column_names(self):
        """No two indicator names should map to the same DataFrame column."""
        seen = {}
        for display_name, col_name in INDICATOR_MAP.items():
            if col_name in seen:
                pytest.fail(f"Duplicate column '{col_name}': "
                            f"used by both '{seen[col_name]}' and '{display_name}'")
            seen[col_name] = display_name

    def test_all_column_names_are_snake_case(self):
        """Column names should be lowercase snake_case."""
        for display_name, col_name in INDICATOR_MAP.items():
            assert col_name == col_name.lower(), \
                f"'{display_name}' maps to non-lowercase '{col_name}'"
            assert " " not in col_name, \
                f"'{display_name}' maps to column with spaces: '{col_name}'"


class TestGroupDefinitions:

    def test_all_group_names_in_group_map(self):
        """Every GROUP_NAMES entry must exist in GROUP_MAP."""
        for name in GROUP_NAMES:
            assert name in GROUP_MAP, f"'{name}' in GROUP_NAMES but not GROUP_MAP"

    def test_all_groups_non_empty(self):
        """Every group must have at least one element."""
        for name, elements in GROUP_MAP.items():
            assert len(elements) > 0, f"Group '{name}' is empty"

    def test_get_group_elements_returns_correct_list(self):
        for name in GROUP_NAMES:
            elements = get_group_elements(name)
            assert elements == list(GROUP_MAP[name])

    def test_get_group_elements_adds_emas(self):
        elements = get_group_elements("Price & Indicators", ema_count=3)
        assert "EMA 1" in elements
        assert "EMA 2" in elements
        assert "EMA 3" in elements

    def test_get_indicator_map_adds_emas(self):
        m = get_indicator_map(ema_count=2)
        assert m["EMA 1"] == "ema_0"
        assert m["EMA 2"] == "ema_1"

    def test_ema_entries_not_in_base_map(self):
        """EMA entries should only appear when ema_count > 0."""
        assert "EMA 1" not in INDICATOR_MAP


class TestEventTypes:

    def test_stop_events_are_subset_of_entry_events(self):
        """All stop events should also be valid entry events."""
        for event in STOP_EVENT_TYPES:
            assert event in EVENT_TYPES, \
                f"Stop event '{event}' not in EVENT_TYPES"

    def test_condition_operators(self):
        assert "Above" in CONDITION_OPERATORS
        assert "Below" in CONDITION_OPERATORS

    def test_compare_types(self):
        assert "Indicator" in CONDITION_COMPARE_TYPES
        assert "Fixed Value" in CONDITION_COMPARE_TYPES

    def test_r_profit_loss_elements(self):
        assert "R Profit" in R_PROFIT_LOSS_ELEMENTS
        assert "R Loss" in R_PROFIT_LOSS_ELEMENTS

    def test_atr_target_elements(self):
        assert "ATR Target" in ATR_TARGET_ELEMENTS

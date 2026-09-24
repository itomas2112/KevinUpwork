"""
Tests for the group set generator: candidates built from a selection of
elements, fixed-value ranges, exclusions and R / ATR special ranges.
"""
import json
import os
import sys
from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest

# Mock streamlit before importing modules that depend on it
if "streamlit" not in sys.modules:
    sys.modules["streamlit"] = MagicMock()

import strategies.group_set_manager as gsm
from strategies.group_set_manager import (
    generate_candidates, validate_generator, count_candidates,
    import_group_set, export_group_set, save_group_set, update_group_set,
    _deduplicate_candidates, MODE_RUNTIME,
)

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
ATR_JSON = os.path.join(ROOT, "images_client", "ATR.json")
GENERAL_JSON = os.path.join(ROOT, "images_client", "General Group Set.json")

FIVE_RSI_LIKE = ["BB Upper Band", "KC Upper Band", "Tenkan", "Kijun", "Senkou A"]


def _pairs(cands):
    return [(c["element1"], c["element2"]) for c in cands
            if c.get("compare_type") == "Indicator"]


def _keys(cands):
    return {json.dumps(c, sort_keys=True) for c in cands}


# =====================================================================
# Pairing
# =====================================================================

class TestPairs:
    def test_five_elements_one_group(self):
        cands = generate_candidates({"elements": FIVE_RSI_LIKE})
        assert len(cands) == 10
        assert all(c["group"] == "Price & Indicators" for c in cands)
        assert all(c["value"] is None and c["compare_type"] == "Indicator"
                   for c in cands)

    def test_price_both_ways(self):
        cands = generate_candidates({"elements": ["Price"] + FIVE_RSI_LIKE})
        pairs = _pairs(cands)
        price = [p for p in pairs if "Price" in p]
        assert len(price) == 10
        for other in FIVE_RSI_LIKE:
            assert ("Price", other) in pairs
            assert (other, "Price") in pairs
        assert len(cands) == 20
        # Price vs X directly followed by X vs Price, in element-list order
        assert pairs[:2] == [("Price", "BB Upper Band"), ("BB Upper Band", "Price")]

    def test_reverse_candidate_group_is_element1_group(self):
        cands = generate_candidates({"elements": ["Price", "EMA 1"]}, ema_count=1)
        assert _pairs(cands) == [("Price", "EMA 1"), ("EMA 1", "Price")]
        assert all(c["group"] == "Price & Indicators" for c in cands)

    def test_cross_group_off(self):
        gen = {"elements": ["RSI", "RSI 13 SMA", "Tenkan", "Kijun"]}
        pairs = _pairs(generate_candidates(gen))
        assert pairs == [("RSI", "RSI 13 SMA"), ("Tenkan", "Kijun")]

    def test_cross_group_on(self):
        gen = {"elements": ["RSI", "RSI 13 SMA", "Tenkan", "Kijun"],
               "allow_cross_group": True}
        assert len(_pairs(generate_candidates(gen))) == 6

    def test_rsi_group_label(self):
        cands = generate_candidates({"elements": ["RSI", "RSI 13 SMA"]})
        assert cands[0]["group"] == "RSI Group"


# =====================================================================
# Exclusions
# =====================================================================

class TestExclusions:
    @pytest.mark.parametrize("elements", [
        ["KC Upper Band", "KC Lower Band", "Tenkan"],
        ["KC Lower Band", "Tenkan", "KC Upper Band"],
    ])
    def test_exclusion_either_order(self, elements):
        gen = {"elements": elements, "exclude_same_indicator": False,
               "exclusions": [["KC Upper Band", "KC Lower Band"]]}
        pairs = {frozenset(p) for p in _pairs(generate_candidates(gen))}
        assert frozenset(("KC Upper Band", "KC Lower Band")) not in pairs
        assert len(pairs) == 2

    def test_exclusion_listed_reversed(self):
        gen = {"elements": ["KC Upper Band", "KC Lower Band"],
               "exclude_same_indicator": False,
               "exclusions": [["KC Lower Band", "KC Upper Band"]]}
        assert validate_generator(gen) == ["The selection generates no candidates."]
        assert generate_candidates(gen) == []

    def test_exclude_same_indicator(self):
        gen = {"elements": ["BB Upper Band", "BB Lower Band", "KC Upper Band",
                            "KC Lower Band", "Price Upper", "Price Lower",
                            "Tenkan", "Kijun"],
               "exclude_same_indicator": True}
        pairs = {frozenset(p) for p in _pairs(generate_candidates(gen))}
        assert frozenset(("BB Upper Band", "BB Lower Band")) not in pairs
        assert frozenset(("KC Upper Band", "KC Lower Band")) not in pairs
        assert frozenset(("Price Upper", "Price Lower")) not in pairs
        assert frozenset(("Tenkan", "Kijun")) in pairs
        assert frozenset(("BB Upper Band", "KC Upper Band")) in pairs

    def test_same_indicator_off_keeps_family_pairs(self):
        gen = {"elements": ["BB Upper Band", "BB Lower Band"],
               "exclude_same_indicator": False}
        assert len(generate_candidates(gen)) == 1

    def test_same_indicator_defaults_on(self):
        assert generate_candidates({"elements": ["BB Upper Band", "BB Lower Band"]}) == []


# =====================================================================
# Fixed values and special candidates
# =====================================================================

class TestFixedAndSpecial:
    def test_fixed_values_rsi(self):
        gen = {"elements": ["RSI"],
               "fixed_values": {"RSI": {"min": 20, "max": 80, "step": 10}}}
        cands = generate_candidates(gen)
        assert len(cands) == 7
        assert all(c["compare_type"] == "Fixed Value" for c in cands)
        assert all(isinstance(c["value"], float) for c in cands)
        assert all(c["group"] == "RSI Group" and c["element2"] is None for c in cands)
        assert [c["value"] for c in cands] == [20.0, 30.0, 40.0, 50.0, 60.0, 70.0, 80.0]

    def test_step_larger_than_range(self):
        gen = {"elements": ["RSI"],
               "fixed_values": {"RSI": {"min": 20, "max": 25, "step": 10}}}
        cands = generate_candidates(gen)
        assert len(cands) == 1 and cands[0]["value"] == 20.0

    def test_fixed_after_pairs(self):
        gen = {"elements": ["RSI", "RSI 13 SMA"],
               "fixed_values": {"RSI": {"min": 30, "max": 70, "step": 40}}}
        cands = generate_candidates(gen)
        assert [c["compare_type"] for c in cands] == ["Indicator", "Fixed Value", "Fixed Value"]

    def test_r_shapes(self):
        gen = {"r_profit": {"min": 1.0, "max": 2.0, "step": 1.0},
               "r_loss": {"min": 1.0, "max": 1.0, "step": 0.5}}
        cands = generate_candidates(gen)
        assert cands == [
            {"group": "Price & Indicators", "element1": "R Profit",
             "compare_type": "Fixed Value", "element2": None, "value": 1.0},
            {"group": "Price & Indicators", "element1": "R Profit",
             "compare_type": "Fixed Value", "element2": None, "value": 2.0},
            {"group": "Price & Indicators", "element1": "R Loss",
             "compare_type": "Fixed Value", "element2": None, "value": 1.0},
        ]

    def test_atr_target_shape(self):
        gen = {"atr_target": {"period": 10, "min": 1.5, "max": 2.0, "step": 0.5}}
        assert generate_candidates(gen) == [
            {"element1": "ATR Target", "atr_period": 10, "atr_multiplier": 1.5},
            {"element1": "ATR Target", "atr_period": 10, "atr_multiplier": 2.0},
        ]

    def test_atr_stop_matches_client_file(self):
        gen = {"atr_stop": {"period": 14, "min": 1.0, "max": 5.0, "step": 0.2}}
        cands = generate_candidates(gen)
        assert len(cands) == 21
        with open(ATR_JSON) as f:
            client = json.load(f)["candidates"]
        assert _keys(cands) == _keys(client)

    def test_section_order(self):
        gen = {"elements": ["RSI", "RSI 13 SMA"],
               "fixed_values": {"RSI": {"min": 50, "max": 50, "step": 1}},
               "r_profit": {"min": 2, "max": 2, "step": 1},
               "r_loss": {"min": 1, "max": 1, "step": 1},
               "atr_target": {"period": 14, "min": 2, "max": 2, "step": 1},
               "atr_stop": {"period": 14, "min": 3, "max": 3, "step": 1}}
        cands = generate_candidates(gen)
        kinds = [c.get("stop_type") or c.get("element1") if c.get("compare_type") != "Indicator"
                 else "pair" for c in cands]
        assert kinds == ["pair", "RSI", "R Profit", "R Loss", "ATR Target", "ATR"]
        counts = count_candidates(gen)
        assert counts == {"pairs": 1, "fixed": 1, "r": 2, "atr": 2, "total": 6}


# =====================================================================
# Determinism and de-duplication
# =====================================================================

class TestDeterminism:
    def test_same_spec_same_list(self):
        gen = {"elements": ["Price"] + FIVE_RSI_LIKE,
               "fixed_values": {"Tenkan": {"min": 1, "max": 3, "step": 1}}}
        assert generate_candidates(gen) == generate_candidates(dict(gen))

    def test_duplicate_element(self):
        gen = {"elements": ["Tenkan", "Kijun", "Tenkan"]}
        cands = generate_candidates(gen)
        assert _pairs(cands) == [("Tenkan", "Kijun")]

    def test_dedup_is_noop_on_generated(self):
        gen = {"elements": ["Price"] + FIVE_RSI_LIKE,
               "atr_stop": {"period": 14, "min": 1.0, "max": 5.0, "step": 0.2}}
        cands = generate_candidates(gen)
        gs = {"candidates": list(cands)}
        assert _deduplicate_candidates(gs) == 0
        assert gs["candidates"] == cands


# =====================================================================
# Validation
# =====================================================================

class TestValidate:
    def test_valid(self):
        gen = {"elements": ["Price", "Tenkan", "EMA 4"],
               "fixed_values": {"Tenkan": {"min": 1, "max": 2, "step": 1}},
               "exclusions": [["Price", "Tenkan"]],
               "atr_stop": {"period": 14, "min": 1, "max": 2, "step": 0.5}}
        assert validate_generator(gen, ema_count=4) == []

    def test_unknown_element(self):
        errs = validate_generator({"elements": ["Price", "Foo"]})
        assert any("Foo" in e for e in errs)

    def test_ema_beyond_count(self):
        errs = validate_generator({"elements": ["Price", "EMA 5"]}, ema_count=4)
        assert any("EMA 5" in e and "4 EMA" in e for e in errs)

    def test_min_greater_than_max(self):
        errs = validate_generator({"elements": ["RSI"], "fixed_values": {
            "RSI": {"min": 80, "max": 20, "step": 10}}})
        assert any("greater than max" in e for e in errs)

    @pytest.mark.parametrize("step", [0, -1])
    def test_bad_step(self, step):
        errs = validate_generator({"r_profit": {"min": 1, "max": 2, "step": step}})
        assert any("step" in e for e in errs)

    def test_self_exclusion(self):
        errs = validate_generator({"elements": ["Tenkan", "Kijun"],
                                   "exclusions": [["Tenkan", "Tenkan"]]})
        assert any("itself" in e for e in errs)

    def test_fixed_value_for_unselected_element(self):
        errs = validate_generator({"elements": ["Tenkan", "Kijun"], "fixed_values": {
            "RSI": {"min": 20, "max": 80, "step": 10}}})
        assert any("RSI" in e for e in errs)


# =====================================================================
# Manager integration
# =====================================================================

@pytest.fixture
def fake_store(monkeypatch, tmp_path):
    state = {}
    monkeypatch.setattr(gsm, "st", SimpleNamespace(session_state=state))
    monkeypatch.setattr(gsm, "GROUP_SETS_FILE", str(tmp_path / "sets.json"))
    return state


GEN = {"elements": ["Price", "Tenkan", "Kijun", "RSI", "RSI 13 SMA"],
       "fixed_values": {"RSI": {"min": 30, "max": 70, "step": 20}},
       "exclusions": [["Tenkan", "Kijun"]],
       "exclude_same_indicator": True, "allow_cross_group": False,
       "r_profit": None, "r_loss": None, "atr_target": None,
       "atr_stop": {"period": 14, "min": 1.0, "max": 2.0, "step": 0.5}}


class TestManager:
    def test_save_regenerates(self, fake_store):
        gs = {"name": "Gen", "mode": MODE_RUNTIME, "generator": dict(GEN),
              "candidates": [{"bogus": True}], "indicator_ranges": {}}
        save_group_set(gs, ema_count=4)
        saved = fake_store["saved_group_sets"][0]
        assert saved["candidates"] == generate_candidates(GEN)
        with open(gsm.GROUP_SETS_FILE) as f:
            assert json.load(f)[0]["candidates"] == generate_candidates(GEN)

    def test_update_regenerates(self, fake_store):
        save_group_set({"name": "Gen", "generator": dict(GEN)}, ema_count=4)
        gen2 = dict(GEN, exclusions=[])
        update_group_set(0, {"name": "Gen", "generator": gen2}, ema_count=4)
        assert fake_store["saved_group_sets"][0]["candidates"] == generate_candidates(gen2)

    def test_save_invalid_generator_raises(self, fake_store):
        with pytest.raises(ValueError, match="EMA 5"):
            save_group_set({"name": "Gen", "generator": {"elements": ["Price", "EMA 5"]}},
                           ema_count=4)
        assert not fake_store.get("saved_group_sets")

    def test_export_import_roundtrip(self, fake_store):
        gs = {"name": "Gen", "generator": dict(GEN), "indicator_ranges": {}}
        save_group_set(gs, ema_count=4)
        data = import_group_set(export_group_set(fake_store["saved_group_sets"][0]),
                                ema_count=4)
        assert data.pop("_regenerated") == len(gs["candidates"])
        assert data["candidates"] == gs["candidates"]
        assert data["generator"] == GEN

    def test_import_ignores_file_candidates(self):
        payload = json.dumps({"name": "G", "generator": GEN,
                              "candidates": [{"element1": "Junk"}]})
        data = import_group_set(payload)
        assert data["candidates"] == generate_candidates(GEN)
        assert data["mode"] == MODE_RUNTIME

    def test_import_generator_without_candidates(self):
        data = import_group_set(json.dumps({"name": "G", "generator": GEN}))
        assert data["candidates"] == generate_candidates(GEN)

    def test_import_invalid_generator(self):
        with pytest.raises(ValueError):
            import_group_set(json.dumps({"name": "G", "generator": {"elements": ["Foo"]}}))

    @pytest.mark.parametrize("path", [GENERAL_JSON, ATR_JSON])
    def test_legacy_imports_unchanged(self, path):
        with open(path) as f:
            raw = f.read()
        data = import_group_set(raw)
        assert "generator" not in data and "_regenerated" not in data
        assert data["candidates"] == json.loads(raw)["candidates"]

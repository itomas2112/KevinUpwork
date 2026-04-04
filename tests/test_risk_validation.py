"""
Tests for risk distance validation.

Verifies that the risk policy correctly rejects degenerate stop distances
while allowing normal trades through.
"""
import numpy as np
import pytest
from strategies.risk_validation import validate_risk_distance


class TestNormalTrades:
    """Normal trades with healthy stop distances should pass."""

    def test_atr_based_stop_passes(self):
        """Stop 1.5 ATR away — well above 0.25 ATR floor."""
        valid, reason = validate_risk_distance(
            entry_price=100.0, stop_price=98.5, atr=1.0)
        assert valid is True
        assert reason is None

    def test_wide_indicator_stop_passes(self):
        """Stop 2% away from entry — well above price floor."""
        valid, reason = validate_risk_distance(
            entry_price=0.0095, stop_price=0.0093, atr=0.0001)
        assert valid is True
        assert reason is None

    def test_no_atr_but_price_ok(self):
        """ATR unavailable, but stop is far enough from entry by price %."""
        valid, reason = validate_risk_distance(
            entry_price=100.0, stop_price=99.0, atr=None)
        assert valid is True

    def test_exactly_at_atr_floor(self):
        """Stop distance exactly 0.25 ATR — should pass (not < threshold)."""
        valid, reason = validate_risk_distance(
            entry_price=100.0, stop_price=99.75, atr=1.0)
        assert valid is True


class TestDegenerateStops:
    """Degenerate stop distances should be rejected."""

    def test_stop_too_close_vs_atr(self):
        """Stop 0.1 ATR away — below the 0.25 ATR floor."""
        valid, reason = validate_risk_distance(
            entry_price=100.0, stop_price=99.9, atr=1.0)
        assert valid is False
        assert reason == 'too_close_vs_atr'

    def test_stop_equals_entry(self):
        """Stop at exactly entry price — zero distance."""
        valid, reason = validate_risk_distance(
            entry_price=100.0, stop_price=100.0, atr=1.0)
        assert valid is False
        assert reason == 'zero_distance'

    def test_no_stop_price(self):
        """No stop set — should reject."""
        valid, reason = validate_risk_distance(
            entry_price=100.0, stop_price=None)
        assert valid is False
        assert reason == 'no_stop'

    def test_nan_stop(self):
        valid, reason = validate_risk_distance(
            entry_price=100.0, stop_price=float('nan'))
        assert valid is False
        assert reason == 'no_stop'

    def test_tiny_price_tiny_stop(self):
        """6J-style: price=0.0095, stop 0.000003 away (0.03% of price).
        With ATR=0.00001, r_distance=0.000003 < 0.25*0.00001=0.0000025?
        Actually 0.000003 > 0.0000025, so it passes ATR floor.
        But 0.000003 < 0.0095 * 0.0001 = 0.00000095? No, 0.000003 > 0.00000095.
        So this passes. Let me use a tighter example."""
        # r_distance = 0.000001, ATR = 0.00001
        # 0.000001 < 0.25 * 0.00001 = 0.0000025 → rejected
        valid, reason = validate_risk_distance(
            entry_price=0.0095, stop_price=0.009499, atr=0.00001)
        assert valid is False
        assert reason == 'too_close_vs_atr'

    def test_price_floor_without_atr(self):
        """No ATR, stop too close by price percentage."""
        # r_distance = 0.001, entry = 100, threshold = 100 * 0.0001 = 0.01
        # 0.001 < 0.01 → rejected
        valid, reason = validate_risk_distance(
            entry_price=100.0, stop_price=99.999, atr=None)
        assert valid is False
        assert reason == 'too_close_vs_price'


class TestTickSize:
    """Tick-size floor (future proofing)."""

    def test_tick_floor_rejects(self):
        """Stop 1 tick away, but policy requires 3 ticks."""
        valid, reason = validate_risk_distance(
            entry_price=100.0, stop_price=99.99,
            atr=0.001, tick_size=0.01, n_ticks=3)
        assert valid is False
        assert reason == 'too_close_vs_tick'

    def test_tick_floor_disabled_by_default(self):
        """Default n_ticks=0 means tick floor is not applied."""
        valid, reason = validate_risk_distance(
            entry_price=100.0, stop_price=99.99, atr=0.001)
        # Only ATR and price floors apply
        # r_distance = 0.01, ATR = 0.001, 0.25*0.001=0.00025, 0.01 > 0.00025 → passes ATR
        # price: 0.01 > 100*0.0001=0.01? Equal, not less than → passes
        assert valid is True


class TestEngineParity:
    """Both engines should produce the same validation result."""

    def test_parity_on_normal_trade(self):
        """Same inputs → same output regardless of which engine calls it."""
        args = dict(entry_price=150.0, stop_price=148.0, atr=0.5)
        v1, r1 = validate_risk_distance(**args)
        v2, r2 = validate_risk_distance(**args)
        assert v1 == v2
        assert r1 == r2

    def test_parity_on_degenerate(self):
        args = dict(entry_price=0.0095, stop_price=0.009499, atr=0.00001)
        v1, r1 = validate_risk_distance(**args)
        v2, r2 = validate_risk_distance(**args)
        assert v1 == v2 == False
        assert r1 == r2 == 'too_close_vs_atr'


class TestEdgeCases:

    def test_short_direction_stop_above(self):
        """Short trade: stop above entry. r_distance still positive."""
        valid, reason = validate_risk_distance(
            entry_price=100.0, stop_price=102.0, atr=1.0)
        assert valid is True

    def test_inf_entry_price(self):
        valid, reason = validate_risk_distance(
            entry_price=float('inf'), stop_price=100.0)
        assert valid is False

    def test_negative_entry_price(self):
        valid, reason = validate_risk_distance(
            entry_price=-1.0, stop_price=-2.0)
        assert valid is False
        assert reason == 'no_stop'

    def test_zero_atr(self):
        """ATR = 0 → ATR floor disabled, falls through to price floor."""
        valid, reason = validate_risk_distance(
            entry_price=100.0, stop_price=99.5, atr=0.0)
        # ATR check skipped (atr not > 0), price check: 0.5 > 100*0.0001=0.01 → passes
        assert valid is True

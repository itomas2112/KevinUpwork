"""
Tests for the directional Pearson correlation logic used in the Correlation tab.

Verifies that the correlation correctly distinguishes between identical,
opposite, uncorrelated, and partially overlapping strategy outcomes.
"""

import numpy as np
import pytest


# ---------------------------------------------------------------------------
# Replicate the correlation logic from correlation_tab.py
# (the functions are nested inside _run_correlation, so we test the math)
# ---------------------------------------------------------------------------

def _direction_array(pnls_dict, n_periods):
    """Build directional signal: +1 win, -1 loss, 0 no trade."""
    arr = np.zeros(n_periods)
    for i in range(n_periods):
        pnl = pnls_dict.get(i, 0.0)
        if pnl > 0:
            arr[i] = 1.0
        elif pnl < 0:
            arr[i] = -1.0
    return arr


def _compute_correlation(pnls_a, pnls_b, n_periods):
    """Pearson correlation on directional signals, scaled to percentage."""
    a = _direction_array(pnls_a, n_periods)
    b = _direction_array(pnls_b, n_periods)
    if np.std(a) > 0 and np.std(b) > 0:
        return np.corrcoef(a, b)[0, 1] * 100
    return 0.0


# ---------------------------------------------------------------------------
# Identical strategies -> +100%
# ---------------------------------------------------------------------------

class TestIdentical:

    def test_same_direction_every_period(self):
        """Win/lose in the same periods, different magnitudes -> 100%."""
        n = 10
        a = {0: 2.0, 1: -1.0, 2: 3.0, 3: -1.0, 4: 1.5, 5: -0.5, 6: 2.0, 7: -1.0, 8: 1.0, 9: -1.0}
        b = {0: 5.0, 1: -2.0, 2: 1.0, 3: -0.5, 4: 0.1, 5: -3.0, 6: 4.0, 7: -0.1, 8: 0.5, 9: -2.0}
        assert _compute_correlation(a, b, n) == pytest.approx(100.0)

    def test_same_pnl_values(self):
        a = {0: 2.0, 1: -1.0, 2: 3.0, 3: -1.0, 4: 1.0}
        assert _compute_correlation(a, a, 5) == pytest.approx(100.0)


# ---------------------------------------------------------------------------
# Opposite strategies -> -100%
# ---------------------------------------------------------------------------

class TestOpposite:

    def test_opposite_every_period(self):
        n = 6
        a = {0: 2.0, 1: -1.0, 2: 3.0, 3: -1.0, 4: 1.0, 5: -0.5}
        b = {0: -1.0, 1: 2.0, 2: -0.5, 3: 3.0, 4: -1.0, 5: 1.0}
        assert _compute_correlation(a, b, n) == pytest.approx(-100.0)


# ---------------------------------------------------------------------------
# Uncorrelated -> 0%
# ---------------------------------------------------------------------------

class TestUncorrelated:

    def test_one_trades_other_doesnt(self):
        a = {0: 1.0, 1: -1.0, 2: 1.0, 3: -1.0, 4: 1.0}
        assert _compute_correlation(a, {}, 5) == 0.0

    def test_both_no_trades(self):
        assert _compute_correlation({}, {}, 5) == 0.0

    def test_no_variance_in_one(self):
        """All wins (no directional variance) vs mixed -> 0%."""
        a = {0: 1.0, 1: 2.0, 2: 0.5, 3: 3.0}
        b = {0: 1.0, 1: -1.0, 2: 1.0, 3: -1.0}
        assert _compute_correlation(a, b, 4) == 0.0


# ---------------------------------------------------------------------------
# Partial overlap
# ---------------------------------------------------------------------------

class TestPartial:

    def test_mostly_agree_positive(self):
        """Agree on 6/8 periods -> positive correlation."""
        n = 8
        a = {i: (1.0 if i % 2 == 0 else -1.0) for i in range(n)}
        b = dict(a)
        b[6], b[7] = -1.0, 1.0  # flip last two
        corr = _compute_correlation(a, b, n)
        assert 0 < corr < 100

    def test_mostly_disagree_negative(self):
        """Agree on 1/6 periods -> negative correlation."""
        n = 6
        a = {0: 1.0, 1: -1.0, 2: 1.0, 3: -1.0, 4: 1.0, 5: -1.0}
        b = {0: 1.0, 1: 1.0, 2: -1.0, 3: 1.0, 4: -1.0, 5: 1.0}
        assert _compute_correlation(a, b, n) < 0


# ---------------------------------------------------------------------------
# Sparse trading (non-trading periods = 0)
# ---------------------------------------------------------------------------

class TestSparse:

    def test_same_direction_sparse(self):
        """Both trade only 2 of 10 periods, same direction -> 100%."""
        a = {3: 2.0, 7: -1.0}
        b = {3: 1.0, 7: -3.0}
        assert _compute_correlation(a, b, 10) == pytest.approx(100.0)

    def test_opposite_direction_sparse(self):
        a = {3: 2.0, 7: -1.0}
        b = {3: -1.0, 7: 2.0}
        assert _compute_correlation(a, b, 10) == pytest.approx(-100.0)

    def test_no_overlap_in_traded_periods(self):
        """A trades 0-2, B trades 3-5 -> near zero."""
        a = {0: 1.0, 1: -1.0, 2: 1.0}
        b = {3: 1.0, 4: -1.0, 5: 1.0}
        corr = _compute_correlation(a, b, 6)
        assert -50 < corr < 50


# ---------------------------------------------------------------------------
# The user's original bug scenario
# ---------------------------------------------------------------------------

class TestOriginalBug:

    def test_both_trade_all_periods_different_outcomes(self):
        """Old binary overlap gave 100% here. Directional correlation should not."""
        n = 19
        a = {i: (1.0 if i % 2 == 0 else -1.0) for i in range(n)}
        b = {i: (1.0 if i < 10 else -1.0) for i in range(n)}
        corr = _compute_correlation(a, b, n)
        assert corr < 100.0, f"Should not be 100%, got {corr:.1f}%"

    def test_both_always_win_no_variance(self):
        """Both win every period -> no variance -> 0% (not 100%)."""
        n = 10
        a = {i: 2.0 for i in range(n)}
        b = {i: 1.0 for i in range(n)}
        assert _compute_correlation(a, b, n) == 0.0

"""Dollar-based bootstrap Monte Carlo with per-trade contract sizing and a
rolling margin cap (strategies/monte_carlo_core.py)."""
import math

import numpy as np
import pytest

from strategies.monte_carlo_core import (simulate_trades_dollars,
                                         mc_profit_at_target_dd_trades)

ES_PV = 50.0
ES_MARGIN = 28000.0
START = 100_000.0


def _rng(seed=42):
    return np.random.default_rng(seed)


class TestSimulateTradesDollars:

    def test_contracts_capped_by_margin_roll_forward(self):
        """Single-trade pool (+1R, 0.5 pt stop) at risk 100%: the margin cap
        always binds, so the path is deterministic — 3 contracts x $25 until
        the balance crosses $112k, then 4, and so on."""
        balance = START
        for _ in range(100):
            by_risk = math.floor(balance * 1.0 / (0.5 * ES_PV))
            by_margin = math.floor(balance / ES_MARGIN)
            balance += min(by_risk, by_margin) * 1.0 * 0.5 * ES_PV

        res = simulate_trades_dollars([1.0], [0.5], START, 100.0, ES_PV, ES_MARGIN,
                                      trades_per_sim=100, n_sims=5, rng=_rng())
        np.testing.assert_allclose(res["final_balances"], balance)
        np.testing.assert_allclose(res["max_drawdowns"], 0.0)
        # First trade is 3 contracts ($75), never more than floor(balance/margin)
        first = simulate_trades_dollars([1.0], [0.5], START, 100.0, ES_PV, ES_MARGIN,
                                        trades_per_sim=1, n_sims=1, rng=_rng())
        assert first["final_balances"][0] == START + 3 * 25.0

    def test_zero_contracts_when_risk_too_small(self):
        # 10 pt stop on ES = $500/contract; 0.1% of $100k = $100 -> 0 contracts
        res = simulate_trades_dollars([2.0, -1.0], [10.0, 10.0], START, 0.1,
                                      ES_PV, ES_MARGIN, trades_per_sim=50,
                                      n_sims=100, rng=_rng())
        np.testing.assert_allclose(res["final_balances"], START)
        np.testing.assert_allclose(res["max_drawdowns"], 0.0)

    def test_capped_fraction(self):
        tiny = simulate_trades_dollars([2.0, -1.0], [0.5, 0.5], START, 100.0,
                                       ES_PV, ES_MARGIN, trades_per_sim=50,
                                       n_sims=100, rng=_rng())
        assert tiny["capped_fraction"] == 1.0

        wide = simulate_trades_dollars([2.0, -1.0], [10.0, 10.0], START, 1.0,
                                       ES_PV, ES_MARGIN, trades_per_sim=50,
                                       n_sims=100, rng=_rng())
        assert wide["capped_fraction"] == 0.0

    @pytest.mark.parametrize("pnls,dists", [([], []), ([1.0, -1.0], [0.0, 0.0]),
                                            ([float("nan")], [5.0])])
    def test_empty_or_invalid_pool(self, pnls, dists):
        res = simulate_trades_dollars(pnls, dists, START, 1.0, ES_PV, ES_MARGIN,
                                      trades_per_sim=20, n_sims=10, rng=_rng())
        np.testing.assert_allclose(res["final_balances"], START)
        np.testing.assert_allclose(res["max_drawdowns"], 0.0)
        assert res["capped_fraction"] == 0.0


class TestProfitAtTargetDD:

    def test_tiny_stop_is_margin_capped(self):
        res = mc_profit_at_target_dd_trades([2.0] * 6 + [-1.0] * 4, [0.5] * 10,
                                            START, ES_PV, ES_MARGIN,
                                            trades_per_sim=100, n_sims=500, rng=_rng())
        assert res["margin_capped"] is True
        assert res["risk_pct"] == 100.0
        assert res["avg_profit"] > START

    def test_loss_pool_hits_target_dd(self):
        pnls = [2.0] * 6 + [-1.0] * 4
        dists = [10.0] * 10
        res = mc_profit_at_target_dd_trades(pnls, dists, START, ES_PV, ES_MARGIN,
                                            target_dd=5.0, trades_per_sim=100,
                                            n_sims=1000, rng=_rng(7))
        assert res["margin_capped"] is False
        assert 0.0 < res["risk_pct"] < 100.0
        check = simulate_trades_dollars(pnls, dists, START, res["risk_pct"],
                                        ES_PV, ES_MARGIN, trades_per_sim=100,
                                        n_sims=1000, rng=_rng(7))
        assert float(np.mean(check["max_drawdowns"])) == pytest.approx(5.0, abs=0.5)
        assert res["avg_profit"] > START

    @pytest.mark.parametrize("pnls,dists", [([], []), ([1.0, -1.0], [0.0, 0.0])])
    def test_empty_pool_returns_zeros(self, pnls, dists):
        res = mc_profit_at_target_dd_trades(pnls, dists, START, ES_PV, ES_MARGIN,
                                            rng=_rng())
        assert res == {"avg_profit": 0.0, "risk_pct": 0.0, "margin_capped": False}


class TestEnrichAggsSerial:

    def test_serial_enrichment_sets_both_keys(self):
        from ui.grid_search_tab import _enrich_mc_serial

        with_trades = {"num_trades": 10,
                       "trade_pnls_r": [2.0] * 6 + [-1.0] * 4,
                       "trade_r_distances": [0.5] * 10}
        empty = {"num_trades": 0, "trade_pnls_r": [], "trade_r_distances": []}
        aggs = [with_trades, empty]
        # 100 trades -> one re-assessed block, which runs at full margin
        _enrich_mc_serial(aggs, START, 200, 5.0, ES_PV, ES_MARGIN, 100)

        assert with_trades["mc_avg_profit"] > START
        assert with_trades["mc_margin_capped"] is True
        assert empty["mc_avg_profit"] == 0.0
        assert empty["mc_margin_capped"] is False


class TestSizedParametricSimulation:
    """Monte Carlo tab: _run_simulation with instrument sizing."""

    def test_sized_contracts_capped_by_margin(self):
        from strategies.monte_carlo_core import _run_simulation
        # 100% wins, 0.5 pt stop, 50% risk -> margin cap (3 contracts) binds
        res = _run_simulation(START, 1, 10, 100.0, 2.0, 50.0,
                              point_value=ES_PV, margin_per_contract=ES_MARGIN,
                              stop_pts=0.5)
        np.testing.assert_allclose(res["final_balances"], START + 3 * 0.5 * ES_PV * 2.0)

    def test_unsized_keeps_percent_model(self):
        from strategies.monte_carlo_core import _run_simulation
        res = _run_simulation(START, 1, 10, 100.0, 2.0, 1.0)
        np.testing.assert_allclose(res["final_balances"], START * 1.02)


# ── Phased risk sizing ──────────────────────────────────────────────────

from strategies.monte_carlo_core import (_size_contracts, _contracts_by_risk,
                                         _inner_draws, _phase_blocks,
                                         _inner_avg_dd, simulate_trades_phased,
                                         mc_phased_profit)

PHASED_KW = dict(initial_risk_pct=1.0, initial_band=(0.75, 1.25),
                 initial_trades=30, reassess_every=200, reassess_sims=30,
                 target_dd=5.0)


def _one(equity, risk_pct, stop_pts, band=None):
    return float(_size_contracts(np.array([equity]), risk_pct,
                                 stop_pts * ES_PV, ES_MARGIN, band=band)[0])


class TestSizeContracts:

    @pytest.mark.parametrize("stop_pts,by_risk,sized", [
        (10.0, 2, 2),   # $500/contract -> 2 = 1.00%
        (25.0, 1, 1),   # $1250 -> 1 = 1.25%, inside the band
        (30.0, 0, 0),   # $1500 -> 1 = 1.50%, outside -> skipped
        (3.0, 7, 3),    # $150 -> 7 = 1.05% beats 6 = 0.90%; ES margin caps at 3
    ])
    def test_band_rule(self, stop_pts, by_risk, sized):
        band = (0.75, 1.25)
        assert float(_contracts_by_risk(np.array([START]), 1.0,
                                        stop_pts * ES_PV, band)[0]) == by_risk
        assert _one(START, 1.0, stop_pts, band=band) == sized

    def test_margin_cap_applies(self):
        assert _one(START, 100.0, 0.5) == 3
        assert _one(START, 1.0, 0.5, band=(0.75, 1.25)) == 3

    def test_floor_rule(self):
        assert float(_contracts_by_risk(np.array([START]), 1.0, 150.0)[0]) == 6


class TestPhaseBlocks:

    @pytest.mark.parametrize("trades,expected", [
        (100, [(0, 30), (30, 100)]),
        (500, [(0, 30), (30, 230), (230, 430), (430, 500)]),
        (20, [(0, 20)]),
    ])
    def test_boundaries(self, trades, expected):
        assert _phase_blocks(trades, 30, 200) == expected


class TestSimulateTradesPhased:

    def test_single_block_equals_initial_phase(self):
        pnls = [2.0] * 6 + [-1.0] * 4
        dists = [3.0, 10.0, 25.0, 30.0, 5.0, 3.0, 10.0, 25.0, 30.0, 5.0]
        phased = simulate_trades_phased(pnls, dists, START, ES_PV, ES_MARGIN, 30, 300,
                                        rng=_rng(3), **PHASED_KW)
        plain = simulate_trades_dollars(pnls, dists, START, 1.0, ES_PV, ES_MARGIN,
                                        trades_per_sim=30, n_sims=300, rng=_rng(3),
                                        band=(0.75, 1.25))
        np.testing.assert_array_equal(phased["final_balances"], plain["final_balances"])
        np.testing.assert_array_equal(phased["max_drawdowns"], plain["max_drawdowns"])
        assert phased["block_risks"].shape == (300, 1)
        assert phased["capped_blocks_fraction"] == 0.0

    def test_reassessed_block_hits_target_dd(self):
        pnls = [2.0] * 88 + [-1.0] * 12
        dists = [10.0] * 100
        n_sims, seed = 200, 11
        res = simulate_trades_phased(pnls, dists, START, ES_PV, ES_MARGIN, 100, n_sims,
                                     rng=_rng(seed), **PHASED_KW)
        risks = res["block_risks"]
        assert np.all(risks[:, 0] == 1.0)
        assert not np.all(risks[:, 1] == 1.0)

        assert risks.shape == (n_sims, 2)

        # Rebuild each path's first-30 history, its balance at trade 30 and
        # the boundary seed (draw order documented in simulate_trades_phased).
        # The block starting at trade 30 re-assesses over the remaining 70.
        rng = _rng(seed)
        hist = np.stack([rng.integers(0, 100, size=n_sims) for _ in range(30)], axis=1)
        inner_seed = int(rng.integers(0, 2**63 - 1))
        eq30 = simulate_trades_phased(pnls, dists, START, ES_PV, ES_MARGIN, 30, n_sims,
                                      rng=_rng(seed), **PHASED_KW)["final_balances"]
        rpc = np.asarray(dists)[hist] * ES_PV
        pnl_usd = np.asarray(pnls)[hist] * rpc
        draws = _inner_draws(pnl_usd, rpc, 70, 30, np.random.default_rng(inner_seed))
        avg_dd = _inner_avg_dd(eq30, risks[:, 1], draws, ES_MARGIN)
        capped = risks[:, 1] == 100.0
        ok = capped | (np.abs(avg_dd - 5.0) <= 0.5)
        assert ok.all(), avg_dd[~ok]

    def test_loss_only_pool_collapses_risk(self):
        res = simulate_trades_phased([-1.0] * 10, [10.0] * 10, START, ES_PV, ES_MARGIN,
                                     100, 200, rng=_rng(5), **PHASED_KW)
        # Any whole contract would exceed 5% DD over 70 straight losses, so the
        # search settles at the size where the next block trades ~0 contracts.
        risk1 = res["block_risks"][:, 1]
        eq30 = START - 30 * 2 * 500.0     # 2 contracts x $500 lost 30 times
        assert np.all(risk1 < 1.0)
        assert np.all(np.floor(eq30 * risk1 / 100.0 / 500.0) <= 1)
        assert np.all(res["final_balances"] > 0)
        assert res["capped_blocks_fraction"] == 0.0

    def test_tiny_stop_all_blocks_capped(self):
        pnls, dists = [2.0] * 6 + [-1.0] * 4, [0.5] * 10
        res = simulate_trades_phased(pnls, dists, START, ES_PV, ES_MARGIN, 100, 200,
                                     rng=_rng(), **PHASED_KW)
        assert res["capped_blocks_fraction"] == 1.0
        assert mc_phased_profit(pnls, dists, START, ES_PV, ES_MARGIN, 100, 200,
                                rng=_rng())["margin_capped"] is True

    def test_initial_skipped_fraction(self):
        pnls = [2.0] * 6 + [-1.0] * 4
        wide = simulate_trades_phased(pnls, [30.0] * 10, START, ES_PV, ES_MARGIN, 30, 100,
                                      rng=_rng(), **PHASED_KW)
        assert wide["initial_skipped_fraction"] == 1.0
        np.testing.assert_allclose(wide["final_balances"], START)
        # denominator counts only the initial 30 trades of a longer run
        longer = simulate_trades_phased(pnls, [30.0] * 10, START, ES_PV, ES_MARGIN, 100,
                                        100, rng=_rng(), **PHASED_KW)
        assert longer["initial_skipped_fraction"] == 1.0
        ten = simulate_trades_phased(pnls, [10.0] * 10, START, ES_PV, ES_MARGIN, 50, 100,
                                     rng=_rng(), **PHASED_KW)
        assert ten["initial_skipped_fraction"] == 0.0

    def test_vectorised_speed(self):
        import time
        gen = _rng(9)
        pnls = list(gen.choice([2.0, -1.0], size=100, p=[0.55, 0.45]))
        dists = list(gen.uniform(3.0, 15.0, size=100))
        t0 = time.perf_counter()
        simulate_trades_phased(pnls, dists, START, ES_PV, ES_MARGIN, 100, 2000,
                               rng=_rng(), **PHASED_KW)
        assert time.perf_counter() - t0 < 1.5

    @pytest.mark.parametrize("pnls,dists", [([], []), ([1.0, -1.0], [0.0, 0.0])])
    def test_mc_phased_profit_empty(self, pnls, dists):
        assert mc_phased_profit(pnls, dists, START, ES_PV, ES_MARGIN, 100, 100,
                                rng=_rng()) == {"avg_profit": 0.0, "margin_capped": False,
                                                "initial_skipped_fraction": 0.0,
                                                "avg_max_dd": 0.0}


class TestPhasedEnrichment:

    def test_serial_enrichment_sets_three_keys(self):
        from ui.grid_search_tab import _enrich_mc_serial, format_mc_value

        wide = {"num_trades": 10, "trade_pnls_r": [2.0] * 6 + [-1.0] * 4,
                "trade_r_distances": [30.0] * 10}
        empty = {"num_trades": 0, "trade_pnls_r": [], "trade_r_distances": []}
        _enrich_mc_serial([wide, empty], START, 100, 5.0, ES_PV, ES_MARGIN, 100)

        assert wide["mc_initial_skipped"] == 1.0
        assert isinstance(wide["mc_margin_capped"], bool)
        assert wide["mc_avg_profit"] > 0
        assert "(skip 100%)" in format_mc_value(wide)
        assert (empty["mc_avg_profit"], empty["mc_margin_capped"],
                empty["mc_initial_skipped"]) == (0.0, False, 0.0)

    def test_worker_returns_skipped_fraction(self):
        from strategies import mc_enrichment_worker as w
        w.init_worker(START, 100, 5.0, ES_PV, ES_MARGIN, 100)
        key, value, capped, skipped = w.enrich_one((7, [2.0, -1.0], [10.0, 10.0]))
        assert key == 7 and value > 0 and skipped == 0.0 and isinstance(capped, bool)
        assert w.enrich_one((8, [], [])) == (8, 0.0, False, 0.0)


class TestPhasedParametricSimulation:

    def test_phased_first_block_at_one_percent(self):
        from strategies.monte_carlo_core import _run_simulation
        # 100% wins, RR 2, 10-pt stop: block 0 always 2 contracts at $100k-ish
        res = _run_simulation(START, 100, 50, 100.0, 2.0, 50.0,
                              point_value=ES_PV, margin_per_contract=ES_MARGIN,
                              stop_pts=10.0, phased=True)
        assert res["block_risks"].shape == (50, 2)
        assert np.all(res["block_risks"][:, 0] == 1.0)
        # all wins -> no DD at full margin -> re-assessed blocks capped at 100%
        assert np.all(res["block_risks"][:, 1] == 100.0)
        # trades 1-30 sized at 1% with the band rule, trade 31 at full margin
        eq = res["sampled_equity"]
        for t in range(31):
            expected = _size_contracts(eq[:, t], 1.0 if t < 30 else 100.0, 500.0,
                                       ES_MARGIN, band=(0.75, 1.25) if t < 30 else None)
            np.testing.assert_allclose(eq[:, t + 1] - eq[:, t], expected * 1000.0)
        first = _run_simulation(START, 1, 5, 100.0, 2.0, 50.0,
                                point_value=ES_PV, margin_per_contract=ES_MARGIN,
                                stop_pts=10.0, phased=True)
        np.testing.assert_allclose(first["final_balances"], START + 2 * 500.0 * 2.0)

    def test_phased_ignored_without_sizing(self):
        from strategies.monte_carlo_core import _run_simulation
        res = _run_simulation(START, 1, 10, 100.0, 2.0, 1.0, phased=True)
        np.testing.assert_allclose(res["final_balances"], START * 1.02)
        assert "block_risks" not in res

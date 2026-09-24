"""
Monte Carlo simulation math — pure numpy, no streamlit or plotly.

Extracted from ui/monte_carlo_tab.py so that:
 - Multiprocessing workers (e.g. grid search MC enrichment) can import this
   without pulling in streamlit, which is expensive under spawn mode.
 - UI code (monte_carlo_tab, performance_tab, grid_search_tab) still
   imports from the original location via re-export in monte_carlo_tab.
"""
import numpy as np


def compute_mc_avg_profit_at_dd(win_rate, reward_risk, risk_pct,
                                starting_balance, max_dd_threshold=5.0,
                                trades_per_sim=100, n_sims=5000,
                                skip_threshold=False):
    """Quick Monte Carlo: return avg final balance based on avg max drawdown.

    Uses the average of all per-simulation max drawdowns (not per-sim filtering).
    When avg max DD <= threshold: returns mean of all final balances.
    When avg max DD > threshold: returns None.

    If skip_threshold=True (used by Grid Search for comparability):
      - Always returns a float, never None.
      - For invalid inputs (0% WR, 0 RR), returns deterministic result:
        balance * (1 - risk_pct/100)^trades_per_sim (pure losing account).
    """
    if risk_pct <= 0 or starting_balance <= 0:
        if skip_threshold:
            return float(starting_balance)
        return None
    if win_rate <= 0 or reward_risk <= 0:
        if skip_threshold:
            decay = (1.0 - risk_pct / 100.0) ** trades_per_sim
            return float(starting_balance * decay)
        return None
    results = _run_simulation_streaming(starting_balance, trades_per_sim, n_sims,
                                        win_rate, reward_risk, risk_pct)
    if skip_threshold:
        return float(np.mean(results["final_balances"]))
    avg_max_dd = float(np.mean(results["max_drawdowns"]))
    if avg_max_dd > max_dd_threshold:
        return None
    return float(np.mean(results["final_balances"]))


def compute_mc_avg_profit_at_target_dd(win_rate, reward_risk,
                                       starting_balance, target_dd=5.0,
                                       trades_per_sim=100, n_sims=5000):
    """Find the risk % that produces target avg max DD, return avg profit there.

    Uses binary search to find the risk_pct where avg max drawdown ≈ target_dd%.
    Returns the mean final balance at that risk level, or 0.0 if no valid risk
    can be found (e.g. 0% win rate or 0 RR).
    """
    if win_rate <= 0 or reward_risk <= 0:
        return 0.0

    lo, hi = 0.01, 100.0
    tolerance = 0.05
    max_iterations = 30
    best_profit = 0.0

    for _ in range(max_iterations):
        mid = (lo + hi) / 2.0
        results = _run_simulation_streaming(starting_balance, trades_per_sim, n_sims,
                                            win_rate, reward_risk, mid)
        avg_dd = float(np.mean(results["max_drawdowns"]))
        avg_profit = float(np.mean(results["final_balances"]))

        if abs(avg_dd - target_dd) <= tolerance:
            return avg_profit
        if avg_dd < target_dd:
            best_profit = avg_profit
            lo = mid
        else:
            hi = mid

    mid = (lo + hi) / 2.0
    results = _run_simulation_streaming(starting_balance, trades_per_sim, n_sims,
                                        win_rate, reward_risk, mid)
    return float(np.mean(results["final_balances"]))


def _prepare_trade_pool(pnls_r, r_dists):
    """Return (pnls, dists) arrays filtered to usable trades (finite pnl, r_dist > 0)."""
    pnls = np.asarray(list(pnls_r), dtype=float)
    dists = np.asarray(list(r_dists), dtype=float)
    n = min(len(pnls), len(dists))
    pnls, dists = pnls[:n], dists[:n]
    mask = np.isfinite(pnls) & np.isfinite(dists) & (dists > 0)
    return pnls[mask], dists[mask]


def _contracts_by_risk(equity, risk_pct, risk_per_contract, band=None):
    """Whole contracts the risk rule allows, before the margin cap (floats >= 0).

    band=None -> floor(equity * risk_pct/100 / risk_per_contract).
    band=(lo, hi) -> the integer count whose actual risk
    (contracts * risk_per_contract / equity, in %) is nearest to risk_pct and
    inside [lo, hi]; 0 when no count fits (trade skipped). Ties round down.
    """
    equity = np.asarray(equity, dtype=float)
    target = equity * (np.asarray(risk_pct, dtype=float) / 100.0) / risk_per_contract
    down = np.maximum(np.floor(target), 0.0)
    if band is None:
        return down
    lo, hi = band
    eps = 1e-9
    up = down + 1.0
    positive = equity > 0
    safe_eq = np.where(positive, equity, 1.0)
    pct_down = down * risk_per_contract / safe_eq * 100.0
    pct_up = up * risk_per_contract / safe_eq * 100.0
    ok_down = positive & (down > 0) & (pct_down >= lo - eps) & (pct_down <= hi + eps)
    ok_up = positive & (pct_up >= lo - eps) & (pct_up <= hi + eps)
    prefer_up = (up - target) < (target - down)
    return np.where(ok_up & (prefer_up | ~ok_down), up, np.where(ok_down, down, 0.0))


def _size_contracts(equity, risk_pct, risk_per_contract, margin, band=None):
    """Per-trade contract count shared by the outer and inner MC loops:
    the risk rule (_contracts_by_risk: floor rule, or nearest-within-band when
    `band` is given) capped by margin, min(., floor(equity / margin))."""
    by_risk = _contracts_by_risk(equity, risk_pct, risk_per_contract, band)
    by_margin = np.maximum(np.floor(np.asarray(equity, dtype=float) / margin), 0.0)
    return np.minimum(by_risk, by_margin)


def simulate_trades_dollars(pnls_r, r_dists, starting_balance, risk_pct,
                            point_value, margin_per_contract,
                            trades_per_sim, n_sims, rng=None, band=None):
    """Bootstrap MC in dollars with per-trade contract sizing.

    Each simulated trade samples an index i (with replacement) from the
    candidate's real trades. Sizing from the balance at that moment:
        risk_per_contract = r_dists[i] * point_value
        by_risk   = floor(balance * risk_pct/100 / risk_per_contract)
        by_margin = floor(balance / margin_per_contract)
        contracts = min(by_risk, by_margin)        # 0 -> trade skipped
        pnl$      = contracts * pnls_r[i] * r_dists[i] * point_value
    Balance updates after every trade, so the margin cap rolls forward.
    With band=(lo, hi), by_risk uses the nearest-integer-within-band rule
    of _contracts_by_risk instead of floor.
    Returns {"final_balances", "max_drawdowns", "capped_fraction"} where
    capped_fraction is the share of simulated trades where by_margin < by_risk.
    """
    if rng is None:
        rng = np.random.default_rng()
    n_sims = int(n_sims)
    trades_per_sim = int(trades_per_sim)
    equity = np.full(n_sims, float(starting_balance))
    max_dd = np.zeros(n_sims)

    pnls, dists = _prepare_trade_pool(pnls_r, r_dists)
    if len(pnls) == 0 or n_sims <= 0 or trades_per_sim <= 0:
        return {"final_balances": equity, "max_drawdowns": max_dd,
                "capped_fraction": 0.0}

    risk_per_contract = dists * point_value     # $ lost per contract at -1R
    pnl_per_contract = pnls * risk_per_contract  # $ per contract for this trade

    running_peak = equity.copy()
    n_capped = 0
    for _ in range(trades_per_sim):
        idx = rng.integers(0, len(pnls), size=n_sims)
        by_risk = _contracts_by_risk(equity, risk_pct, risk_per_contract[idx], band)
        by_margin = np.floor(equity / margin_per_contract)
        n_capped += int(np.count_nonzero(by_margin < by_risk))
        contracts = np.maximum(np.minimum(by_risk, by_margin), 0.0)
        equity += contracts * pnl_per_contract[idx]
        np.maximum(running_peak, equity, out=running_peak)
        dd = (running_peak - equity) / running_peak * 100.0
        np.maximum(max_dd, dd, out=max_dd)

    return {"final_balances": equity, "max_drawdowns": max_dd,
            "capped_fraction": n_capped / float(n_sims * trades_per_sim)}


def mc_profit_at_target_dd_trades(pnls_r, r_dists, starting_balance, point_value,
                                  margin_per_contract, target_dd=5.0,
                                  trades_per_sim=100, n_sims=1000, rng=None):
    """Binary-search risk_pct so avg max DD ≈ target_dd (same loop shape and
    tolerance as compute_mc_avg_profit_at_target_dd). First evaluate risk_pct=100:
    if avg DD is still below target, the margin cap binds — return that result with
    margin_capped=True. Returns dict {"avg_profit": float, "risk_pct": float,
    "margin_capped": bool}. Empty/invalid pool -> {0.0, 0.0, False}.

    Every risk level is simulated with the same random draws (one seed per
    call), so avg DD is monotone in risk_pct and the search is stable.
    """
    empty = {"avg_profit": 0.0, "risk_pct": 0.0, "margin_capped": False}
    pnls, dists = _prepare_trade_pool(pnls_r, r_dists)
    if (len(pnls) == 0 or starting_balance <= 0 or point_value <= 0
            or margin_per_contract <= 0):
        return empty
    if rng is None:
        rng = np.random.default_rng()
    seed = int(rng.integers(0, 2**63 - 1))

    def _sim(risk_pct):
        res = simulate_trades_dollars(pnls, dists, starting_balance, risk_pct,
                                      point_value, margin_per_contract,
                                      trades_per_sim, n_sims,
                                      rng=np.random.default_rng(seed))
        return (float(np.mean(res["max_drawdowns"])),
                float(np.mean(res["final_balances"])))

    avg_dd, avg_profit = _sim(100.0)
    if avg_dd < target_dd:
        return {"avg_profit": avg_profit, "risk_pct": 100.0, "margin_capped": True}

    lo, hi = 0.01, 100.0
    tolerance = 0.05
    max_iterations = 30

    for _ in range(max_iterations):
        mid = (lo + hi) / 2.0
        avg_dd, avg_profit = _sim(mid)
        if abs(avg_dd - target_dd) <= tolerance:
            return {"avg_profit": avg_profit, "risk_pct": mid, "margin_capped": False}
        if avg_dd < target_dd:
            lo = mid
        else:
            hi = mid

    mid = (lo + hi) / 2.0
    _avg_dd, avg_profit = _sim(mid)
    return {"avg_profit": avg_profit, "risk_pct": mid, "margin_capped": False}


# ── Phased risk sizing ─────────────────────────────────────────────────
# A live trader has no risk assessment for a new strategy, so trades 1..N run
# at a fixed initial risk; every M trades thereafter the risk % is re-derived
# from the trades observed so far. All outer paths hit a block boundary together, so
# they are re-assessed together (arrays shaped (paths, ...)), never looped.

_REASSESS_CHUNK = 1000   # outer paths per re-assessment batch (bounds memory)
_BISECT_LO = 0.01
_BISECT_HI = 100.0
_BISECT_TOLERANCE = 0.05
_BISECT_MAX_ITERATIONS = 30


def _phase_blocks(trades_per_sim, initial_trades, reassess_every):
    """Phase block boundaries as a list of (start, end) trade indices.

    Block 0 = trades [0, initial_trades); later blocks are `reassess_every`
    long, the last truncated at trades_per_sim. A block's length is also the
    inner MC horizon of its re-assessment.
    """
    trades_per_sim = max(int(trades_per_sim), 0)
    initial_trades = max(int(initial_trades), 1)
    reassess_every = max(int(reassess_every), 1)
    if trades_per_sim == 0:
        return []
    blocks = [(0, min(initial_trades, trades_per_sim))]
    start = initial_trades
    while start < trades_per_sim:
        end = min(start + reassess_every, trades_per_sim)
        blocks.append((start, end))
        start = end
    return blocks


def _inner_draws(hist_pnl_usd, hist_rpc, n_trades, n_inner, rng):
    """Bootstrap each path's own history for the inner MC.

    hist_pnl_usd / hist_rpc: (paths, history) $ P&L and $ risk per contract.
    Returns (pnl_per_contract, risk_per_contract), each (n_trades, paths, n_inner).
    """
    n_paths, n_hist = hist_rpc.shape
    idx = rng.integers(0, n_hist, size=(n_trades, n_paths, n_inner))
    rows = np.arange(n_paths)[None, :, None]
    return hist_pnl_usd[rows, idx], hist_rpc[rows, idx]


def _inner_avg_dd(equity0, risk_pct, draws, margin):
    """Avg max DD (%) per path of the inner MC at a per-path risk %.

    equity0, risk_pct: (paths,). draws: output of _inner_draws. Contracts use
    the floor rule capped by margin, re-sized after every trade.
    """
    pnl_pc, rpc = draws
    n_trades, n_paths, n_inner = rpc.shape
    equity = np.repeat(np.asarray(equity0, dtype=float)[:, None], n_inner, axis=1)
    peak = np.maximum(equity, 1e-12)
    max_dd = np.zeros_like(equity)
    risk = np.asarray(risk_pct, dtype=float)[:, None]
    for t in range(n_trades):
        contracts = _size_contracts(equity, risk, rpc[t], margin)
        equity += contracts * pnl_pc[t]
        np.maximum(peak, equity, out=peak)
        np.maximum(max_dd, (peak - equity) / peak * 100.0, out=max_dd)
    return max_dd.mean(axis=1)


def _bisect_block_risk(equity0, draws, margin, target_dd):
    """Per-path bisection of risk % so the inner avg max DD ≈ target_dd.

    Same loop shape as mc_profit_at_target_dd_trades, elementwise: first
    evaluate 100% (paths still below target are margin-capped and keep 100%),
    then bisect [0.01, 100] up to 30 times with tolerance 0.05; a converged
    path keeps its value, the rest end at the final (lo+hi)/2. The same draws
    are reused for every iteration. Returns (risk_pct, capped), both (paths,).
    """
    n_paths = len(equity0)
    capped = _inner_avg_dd(equity0, np.full(n_paths, _BISECT_HI), draws, margin) < target_dd
    risk = np.full(n_paths, _BISECT_HI)
    lo = np.full(n_paths, _BISECT_LO)
    hi = np.full(n_paths, _BISECT_HI)

    active = np.flatnonzero(~capped)
    cur_draws = (draws[0][:, active], draws[1][:, active])
    for _ in range(_BISECT_MAX_ITERATIONS):
        if active.size == 0:
            break
        mid = (lo[active] + hi[active]) / 2.0
        dd = _inner_avg_dd(equity0[active], mid, cur_draws, margin)
        conv = np.abs(dd - target_dd) <= _BISECT_TOLERANCE
        below = dd < target_dd
        risk[active[conv]] = mid[conv]
        lo[active[below & ~conv]] = mid[below & ~conv]
        hi[active[~below & ~conv]] = mid[~below & ~conv]
        if conv.any():
            keep = ~conv
            active = active[keep]
            cur_draws = (cur_draws[0][:, keep], cur_draws[1][:, keep])
    risk[active] = (lo[active] + hi[active]) / 2.0
    return risk, capped


def _reassess_block_risk(hist_pnl_usd, hist_rpc, equity, margin, n_trades,
                         n_inner, target_dd, seed):
    """Re-assess every path's risk % from its own history (see
    _bisect_block_risk). Paths are processed in chunks of _REASSESS_CHUNK
    drawing from one generator seeded with `seed`. Returns (risk_pct, capped)."""
    n_paths = len(equity)
    risk = np.empty(n_paths)
    capped = np.zeros(n_paths, dtype=bool)
    inner_rng = np.random.default_rng(seed)
    for start in range(0, n_paths, _REASSESS_CHUNK):
        sl = slice(start, start + _REASSESS_CHUNK)
        draws = _inner_draws(hist_pnl_usd[sl], hist_rpc[sl], n_trades, n_inner, inner_rng)
        risk[sl], capped[sl] = _bisect_block_risk(equity[sl], draws, margin, target_dd)
    return risk, capped


def simulate_trades_phased(pnls_r, r_dists, starting_balance, point_value,
                           margin_per_contract, trades_per_sim, n_sims, *,
                           initial_risk_pct, initial_band, initial_trades,
                           reassess_every, reassess_sims, target_dd, rng=None):
    """Bootstrap MC in dollars with phased risk sizing.

    Trades run in blocks from _phase_blocks (initial_trades, then
    reassess_every each):
      - block 0: target initial_risk_pct with the band rule
        (_contracts_by_risk with band=initial_band; 0 contracts = skipped);
      - block k >= 1: at the boundary each path's risk % is re-assessed from
        its own sampled trades so far (_reassess_block_risk: an inner MC of
        `reassess_sims` sims x min(reassess_every, trades remaining) trades
        from the path's current balance, bisected to target_dd avg max DD);
        that block then uses the floor rule at the path's risk %.
    Contracts are always capped by floor(balance / margin) and re-sized after
    every trade.

    Random draws: per outer trade one rng.integers(0, pool, size=n_sims)
    (identical to simulate_trades_dollars, so block 0 alone reproduces it);
    at each boundary one seed = rng.integers(0, 2**63 - 1) for the inner MC.

    Returns {"final_balances", "max_drawdowns", "block_risks" (n_sims, n_blocks),
    "capped_blocks_fraction" (share of re-assessed path-blocks at full margin),
    "initial_skipped_fraction" (share of block-0 trades skipped by the band)}.
    """
    if rng is None:
        rng = np.random.default_rng()
    n_sims = max(int(n_sims), 0)
    trades_per_sim = max(int(trades_per_sim), 0)
    blocks = _phase_blocks(trades_per_sim, initial_trades, reassess_every)
    n_blocks = len(blocks)

    equity = np.full(n_sims, float(starting_balance))
    max_dd = np.zeros(n_sims)
    block_risks = np.zeros((n_sims, n_blocks))
    if n_blocks:
        block_risks[:, 0] = initial_risk_pct
    result = {"final_balances": equity, "max_drawdowns": max_dd,
              "block_risks": block_risks, "capped_blocks_fraction": 0.0,
              "initial_skipped_fraction": 0.0}

    pnls, dists = _prepare_trade_pool(pnls_r, r_dists)
    if len(pnls) == 0 or n_sims == 0 or trades_per_sim == 0:
        return result

    risk_per_contract = dists * point_value
    pnl_per_contract = pnls * risk_per_contract
    hist_idx = np.empty((n_sims, trades_per_sim), dtype=np.int64)
    running_peak = equity.copy()
    n_skipped = 0
    n_capped_blocks = 0

    block = 0
    for t in range(trades_per_sim):
        if block + 1 < n_blocks and t == blocks[block + 1][0]:
            block += 1
            start, end = blocks[block]
            seed = int(rng.integers(0, 2**63 - 1))
            hist = hist_idx[:, :t]
            risk, capped = _reassess_block_risk(
                pnl_per_contract[hist], risk_per_contract[hist], equity,
                margin_per_contract, end - start, reassess_sims, target_dd, seed)
            block_risks[:, block] = risk
            n_capped_blocks += int(np.count_nonzero(capped))

        idx = rng.integers(0, len(pnls), size=n_sims)
        hist_idx[:, t] = idx
        rpc = risk_per_contract[idx]
        if block == 0:
            by_risk = _contracts_by_risk(equity, initial_risk_pct, rpc, initial_band)
            n_skipped += int(np.count_nonzero(by_risk == 0))
            by_margin = np.maximum(np.floor(equity / margin_per_contract), 0.0)
            contracts = np.minimum(by_risk, by_margin)
        else:
            contracts = _size_contracts(equity, block_risks[:, block], rpc,
                                        margin_per_contract)
        equity += contracts * pnl_per_contract[idx]
        np.maximum(running_peak, equity, out=running_peak)
        dd = (running_peak - equity) / running_peak * 100.0
        np.maximum(max_dd, dd, out=max_dd)

    n_initial = n_sims * min(trades_per_sim, max(int(initial_trades), 1))
    result["initial_skipped_fraction"] = n_skipped / float(n_initial)
    if n_blocks > 1:
        result["capped_blocks_fraction"] = n_capped_blocks / float(n_sims * (n_blocks - 1))
    return result


def mc_phased_profit(pnls_r, r_dists, starting_balance, point_value,
                     margin_per_contract, trades_per_sim, n_sims, rng=None):
    """simulate_trades_phased with the MC_* constants from config.constants.

    Returns {"avg_profit" (mean final balance), "margin_capped" (more than half
    of the re-assessed blocks ran at full margin), "initial_skipped_fraction",
    "avg_max_dd"}. Empty/invalid pool -> zeros / False.
    """
    from config.constants import (MC_INITIAL_RISK_PCT, MC_INITIAL_RISK_BAND,
                                  MC_INITIAL_TRADES, MC_REASSESS_EVERY,
                                  MC_REASSESS_SIMS, MC_TARGET_DD)
    empty = {"avg_profit": 0.0, "margin_capped": False,
             "initial_skipped_fraction": 0.0, "avg_max_dd": 0.0}
    pnls, dists = _prepare_trade_pool(pnls_r, r_dists)
    if (len(pnls) == 0 or starting_balance <= 0 or point_value <= 0
            or margin_per_contract <= 0 or int(trades_per_sim) <= 0 or int(n_sims) <= 0):
        return empty
    res = simulate_trades_phased(
        pnls, dists, starting_balance, point_value, margin_per_contract,
        trades_per_sim, n_sims,
        initial_risk_pct=MC_INITIAL_RISK_PCT, initial_band=MC_INITIAL_RISK_BAND,
        initial_trades=MC_INITIAL_TRADES, reassess_every=MC_REASSESS_EVERY,
        reassess_sims=MC_REASSESS_SIMS,
        target_dd=MC_TARGET_DD, rng=rng)
    return {"avg_profit": float(np.mean(res["final_balances"])),
            "margin_capped": bool(res["capped_blocks_fraction"] > 0.5),
            "initial_skipped_fraction": float(res["initial_skipped_fraction"]),
            "avg_max_dd": float(np.mean(res["max_drawdowns"]))}


def _run_simulation_streaming(starting_balance, trades_per_sim, n_simulations,
                              win_rate, reward_risk, risk_pct):
    """Memory-efficient variant used by enrichment (no equity history).

    Tracks only per-sim current equity, running peak, and running max DD.
    Peak memory is O(n_simulations), not O(n_simulations * trades_per_sim).
    Returns only the two arrays the enrichment path consumes.
    """
    rng = np.random.default_rng()

    equity = np.full(n_simulations, float(starting_balance))
    running_peak = equity.copy()
    max_dd = np.zeros(n_simulations)

    win_prob = win_rate / 100.0
    risk_frac = risk_pct / 100.0

    for _ in range(trades_per_sim):
        outcomes = rng.random(n_simulations) < win_prob
        risk_amount = equity * risk_frac
        pnl = np.where(outcomes, risk_amount * reward_risk, -risk_amount)
        equity += pnl
        np.maximum(running_peak, equity, out=running_peak)
        dd = (running_peak - equity) / running_peak * 100.0
        np.maximum(max_dd, dd, out=max_dd)

    return {"final_balances": equity, "max_drawdowns": max_dd}


def _run_simulation(starting_balance, trades_per_sim, n_simulations,
                    win_rate, reward_risk, risk_pct,
                    point_value=None, margin_per_contract=None, stop_pts=None,
                    phased=False):
    """Run Monte Carlo simulation using vectorized NumPy operations.

    When point_value, margin_per_contract and stop_pts are all given, each
    trade is sized in whole contracts from the current equity:
        contracts = min(floor(eq*risk/(stop_pts*pv)), floor(eq/margin))
        pnl       = contracts * stop_pts * pv * (RR if win else -1)
    Otherwise the plain % of equity risk model is used.

    phased=True (sized mode only; risk_pct is then ignored) uses the phased
    scheme of simulate_trades_phased with the MC_* constants: trades
    1..MC_INITIAL_TRADES target MC_INITIAL_RISK_PCT with the band rule, and
    every MC_REASSESS_EVERY trades thereafter each path's risk % is
    re-assessed by an inner MC bootstrapped from that path's realised
    +RR / -1R outcomes.

    Returns dict with final_balances, max_drawdowns, and sampled equity curves
    (plus block_risks when phased).
    """
    rng = np.random.default_rng()

    outcomes = rng.random((n_simulations, trades_per_sim)) < (win_rate / 100.0)

    equity = np.empty((n_simulations, trades_per_sim + 1))
    equity[:, 0] = starting_balance

    sized = (point_value is not None and margin_per_contract is not None
             and stop_pts is not None and point_value > 0
             and margin_per_contract > 0 and stop_pts > 0)

    phased = bool(phased) and sized
    if phased:
        from config.constants import (MC_INITIAL_RISK_PCT, MC_INITIAL_RISK_BAND,
                                      MC_INITIAL_TRADES, MC_REASSESS_EVERY,
                                      MC_REASSESS_SIMS, MC_TARGET_DD)
        rpc_usd = stop_pts * point_value
        blocks = _phase_blocks(trades_per_sim, MC_INITIAL_TRADES, MC_REASSESS_EVERY)
        n_blocks = len(blocks)
        block_risks = np.zeros((n_simulations, n_blocks))
        if n_blocks:
            block_risks[:, 0] = MC_INITIAL_RISK_PCT
        block = 0

    for t in range(trades_per_sim):
        if phased:
            eq = equity[:, t]
            if block + 1 < n_blocks and t == blocks[block + 1][0]:
                block += 1
                start, end = blocks[block]
                hist_pnl = np.where(outcomes[:, :t], reward_risk, -1.0) * rpc_usd
                hist_rpc = np.full(hist_pnl.shape, rpc_usd)
                block_risks[:, block], _capped = _reassess_block_risk(
                    hist_pnl, hist_rpc, eq, margin_per_contract, end - start,
                    MC_REASSESS_SIMS, MC_TARGET_DD, int(rng.integers(0, 2**63 - 1)))
            if block == 0:
                contracts = _size_contracts(eq, MC_INITIAL_RISK_PCT, rpc_usd,
                                            margin_per_contract, band=MC_INITIAL_RISK_BAND)
            else:
                contracts = _size_contracts(eq, block_risks[:, block], rpc_usd,
                                            margin_per_contract)
            risk_amount = contracts * rpc_usd
        elif sized:
            eq = equity[:, t]
            risk_per_contract = stop_pts * point_value
            by_risk = np.floor(eq * (risk_pct / 100.0) / risk_per_contract)
            by_margin = np.floor(eq / margin_per_contract)
            contracts = np.maximum(np.minimum(by_risk, by_margin), 0.0)
            risk_amount = contracts * risk_per_contract
        else:
            risk_amount = equity[:, t] * (risk_pct / 100.0)
        pnl = np.where(outcomes[:, t],
                       risk_amount * reward_risk,
                       -risk_amount)
        equity[:, t + 1] = equity[:, t] + pnl

    final_balances = equity[:, -1]

    running_peak = np.maximum.accumulate(equity, axis=1)
    drawdowns = (running_peak - equity) / running_peak * 100.0
    max_drawdowns = np.max(drawdowns, axis=1)

    max_curves = 500
    if n_simulations <= max_curves:
        sampled_equity = equity
    else:
        indices = rng.choice(n_simulations, max_curves, replace=False)
        sampled_equity = equity[indices]

    out = {
        "final_balances": final_balances,
        "max_drawdowns": max_drawdowns,
        "sampled_equity": sampled_equity,
        "median_curve": np.median(equity, axis=0),
    }
    if phased:
        out["block_risks"] = block_risks
    return out

# NumPy-optimized version of execute_custom_strategy
# Uses VECTORIZED pre-computation: all triggers, conditions, and stop-violation
# checks are computed as boolean arrays over ALL bars at once (true numpy).
# The main loop only handles stateful position management + pre-computed lookups.

import copy
import warnings
import pandas as pd
import numpy as np
from config.constants import get_indicator_map
from indicators.atr_indicator import atr_indicator
from strategies.strategy_validator import validate_strategy
from strategies.first_strategy import _prepare_ichimoku_columns


# ======================================================================
# Vectorized pre-computation helpers (module-level, no closures)
# ======================================================================

def _vec_trigger(config, _arrays, indicator_map, _high, _low, _close, _n):
    """Pre-compute trigger boolean mask AND trigger prices for ALL bars at once.

    Returns (mask, prices) where:
      mask[i]   = True if trigger fires at bar i
      prices[i] = the entry/exit price if trigger fires at bar i
    """
    element1_name = config.get('element1')
    event = config.get('event')
    compare_type = config.get('compare_type', 'Indicator')
    element2_name = config.get('element2')
    fixed_value = config.get('value')

    col1 = indicator_map.get(element1_name)
    if col1 is None or col1 not in _arrays:
        return np.zeros(_n, dtype=bool), _close.copy()

    arr1 = _arrays[col1]
    is_price = (col1 == 'latest')

    # Determine comparison array
    if compare_type == "Fixed Value":
        if fixed_value is None:
            return np.zeros(_n, dtype=bool), _close.copy()
        arr2 = np.full(_n, fixed_value, dtype=np.float64)
    else:
        col2 = indicator_map.get(element2_name)
        if col2 is None or col2 not in _arrays:
            return np.zeros(_n, dtype=bool), _close.copy()
        arr2 = _arrays[col2].astype(np.float64, copy=False)

    # Previous-bar values (index 0 = NaN so cross events at bar 0 are False)
    prev1 = np.empty(_n, dtype=np.float64)
    prev1[0] = np.nan
    prev1[1:] = arr1[:-1]

    prev2 = np.empty(_n, dtype=np.float64)
    prev2[0] = np.nan
    prev2[1:] = arr2[:-1]

    mask = np.zeros(_n, dtype=bool)
    prices = _close.copy()

    if event == "Close Above":
        mask[:] = arr1 > arr2

    elif event == "Close Below":
        mask[:] = arr1 < arr2

    elif event == "Close":
        ca = (arr1 > arr2) & (prev1 <= prev2)
        cb = (arr1 < arr2) & (prev1 >= prev2)
        mask[:] = ca | cb
        mask[0] = False

    elif event == "Cross Above":
        was_below = prev1 <= prev2
        if is_price:
            now_above = _high > arr2
        else:
            now_above = arr1 > arr2
        mask[:] = was_below & now_above
        mask[0] = False
        if is_price:
            prices = arr2.copy()

    elif event == "Cross Below":
        was_above = prev1 >= prev2
        if is_price:
            now_below = _low < arr2
        else:
            now_below = arr1 < arr2
        mask[:] = was_above & now_below
        mask[0] = False
        if is_price:
            prices = arr2.copy()

    elif event == "Cross":
        was_below = prev1 <= prev2
        was_above = prev1 >= prev2
        if is_price:
            ca = was_below & (_high > arr2)
            cb = was_above & (_low < arr2)
        else:
            ca = was_below & (arr1 > arr2)
            cb = was_above & (arr1 < arr2)
        mask[:] = ca | cb
        mask[0] = False
        if is_price:
            prices = arr2.copy()

    elif event == "At Level":
        mask[:] = np.abs(arr1 - arr2) < 0.01

    return mask, prices


def _vec_condition(config, _arrays, indicator_map, _n):
    """Pre-compute condition boolean mask for ALL bars at once."""
    element1_name = config.get('element1')
    operator = config.get('operator')
    compare_type = config.get('compare_type', 'Indicator')
    element2_name = config.get('element2')
    fixed_value = config.get('value')

    col1 = indicator_map.get(element1_name)
    if col1 is None or col1 not in _arrays:
        return np.zeros(_n, dtype=bool)

    arr1 = _arrays[col1]

    if compare_type == "Fixed Value":
        if fixed_value is None:
            return np.zeros(_n, dtype=bool)
        arr2 = np.full(_n, fixed_value, dtype=np.float64)
    else:
        col2 = indicator_map.get(element2_name)
        if col2 is None or col2 not in _arrays:
            return np.zeros(_n, dtype=bool)
        arr2 = _arrays[col2]

    if operator == "Above":
        return arr1 > arr2
    elif operator == "Below":
        return arr1 < arr2
    return np.zeros(_n, dtype=bool)


def _vec_stop_violated(config, _arrays, indicator_map, strategy_direction, _n):
    """Pre-compute stop-already-violated boolean mask for ALL bars."""
    element1_name = config.get('element1')
    if element1_name in ("R Profit", "R Loss", "ATR Target"):
        return np.zeros(_n, dtype=bool)
    if config.get('stop_type') == 'ATR':
        return np.zeros(_n, dtype=bool)

    event = config.get('event')
    compare_type = config.get('compare_type', 'Indicator')
    element2_name = config.get('element2')
    fixed_value = config.get('value')

    col1 = indicator_map.get(element1_name)
    if col1 is None or col1 not in _arrays:
        return np.zeros(_n, dtype=bool)

    arr1 = _arrays[col1]

    if compare_type == "Fixed Value":
        if fixed_value is None:
            return np.zeros(_n, dtype=bool)
        arr2 = np.full(_n, fixed_value, dtype=np.float64)
    else:
        col2 = indicator_map.get(element2_name)
        if col2 is None or col2 not in _arrays:
            return np.zeros(_n, dtype=bool)
        arr2 = _arrays[col2]

    if event in ("Cross Below", "Close Below"):
        return arr1 < arr2
    elif event in ("Cross Above", "Close Above"):
        return arr1 > arr2
    elif event in ("Cross", "Close"):
        if strategy_direction == 'Long':
            return arr1 < arr2
        else:
            return arr1 > arr2
    return np.zeros(_n, dtype=bool)


# ======================================================================
# Main function
# ======================================================================

def execute_custom_strategy_numpy(df: pd.DataFrame, strategy_config: dict,
                                   period_start=None, period_end=None):
    """
    Vectorized NumPy version of execute_custom_strategy.
    Returns (None, stats_df) — stats only, no modified DataFrame.
    """
    df = df.copy()
    strategy_config = copy.deepcopy(strategy_config)

    # Validate strategy before execution
    ema_count = 0
    while f"ema_{ema_count}" in df.columns:
        ema_count += 1

    is_valid, errors = validate_strategy(strategy_config, ema_count=ema_count)
    if not is_valid:
        warnings.warn(f"Invalid strategy skipped: {'; '.join(errors)}")
        from strategies.first_strategy import _empty_stats_df
        return None, _empty_stats_df()

    for col in ["entry_signal", "exit_signal"]:
        if col in df.columns:
            df.drop(columns=col, inplace=True)

    indicator_map = get_indicator_map(ema_count)

    _prepare_ichimoku_columns(df, indicator_map, strategy_config)

    # -------------------------------------------------
    # Extract ALL columns as numpy arrays
    # -------------------------------------------------
    _arrays = {col: df[col].values for col in df.columns}
    _high = _arrays['high']
    _low = _arrays['low']
    _close = _arrays['latest']
    _index_arr = df.index.values
    _n = len(df)

    if _n == 0:
        return None, None

    # -------------------------------------------------
    # Extract configuration
    # -------------------------------------------------
    entry_config = strategy_config.get('entry', {})
    entry_trigger = entry_config.get('trigger', {})
    entry_conditions = entry_config.get('conditions', [])
    entry_r = entry_config.get('position_size', 1.0)

    initial_stop_config = strategy_config.get('initial_stop') or None
    strategy_direction = strategy_config.get('direction', 'Long')
    max_positions = strategy_config.get('max_positions', 1)

    exit_groups = strategy_config.get('exit_groups', [])
    if not exit_groups:
        exit_config = strategy_config.get('exit', {})
        if exit_config:
            exit_groups = [{
                'group_id': 1, 'allocation_pct': 100.0,
                'targets': [{'type': 'Target', 'trigger': exit_config.get('trigger', {}),
                             'conditions': exit_config.get('conditions', [])}],
                'stops': []
            }]
    for g in exit_groups:
        if 'allocation_pct' not in g:
            g['allocation_pct'] = 100.0 / max(len(exit_groups), 1)

    # -------------------------------------------------
    # VECTORIZED PRE-COMPUTATION: entry
    # -------------------------------------------------
    entry_trig_mask, entry_trig_prices = _vec_trigger(
        entry_trigger, _arrays, indicator_map, _high, _low, _close, _n)

    entry_cond_mask = np.ones(_n, dtype=bool)
    for cond in entry_conditions:
        entry_cond_mask &= _vec_condition(cond, _arrays, indicator_map, _n)

    # Date range mask
    date_mask = np.ones(_n, dtype=bool)
    if period_start is not None:
        date_mask &= _index_arr >= period_start
    if period_end is not None:
        date_mask &= _index_arr <= period_end

    # Stop-already-violated mask
    stop_violated_mask = np.zeros(_n, dtype=bool)
    if initial_stop_config:
        stop_violated_mask |= _vec_stop_violated(
            initial_stop_config, _arrays, indicator_map, strategy_direction, _n)
    for group in exit_groups:
        for stop in group.get('stops', []):
            stop_trigger = stop.get('trigger', {})
            if stop_trigger:
                stop_violated_mask |= _vec_stop_violated(
                    stop_trigger, _arrays, indicator_map, strategy_direction, _n)

    # Combined entry-possible mask (everything except position state)
    entry_possible = entry_trig_mask & entry_cond_mask & date_mask & ~stop_violated_mask

    # -------------------------------------------------
    # VECTORIZED PRE-COMPUTATION: exit triggers
    # For indicator-based exits (not R Profit/Loss, not ATR Target),
    # pre-compute trigger + condition masks.
    # -------------------------------------------------
    _exit_masks = {}   # (group_idx, 'target'/'stop', item_idx) -> combined bool mask
    _exit_prices = {}  # (group_idx, 'target'/'stop', item_idx) -> prices array

    for g_idx, group in enumerate(exit_groups):
        for t_idx, item in enumerate(group.get('targets', [])):
            trigger = item.get('trigger', {})
            el1 = trigger.get('element1', '')
            if el1 in ('R Profit', 'R Loss', 'ATR Target'):
                continue
            t_mask, t_prices = _vec_trigger(
                trigger, _arrays, indicator_map, _high, _low, _close, _n)
            for cond in item.get('conditions', []):
                t_mask = t_mask & _vec_condition(cond, _arrays, indicator_map, _n)
            key = (g_idx, 'target', t_idx)
            _exit_masks[key] = t_mask
            _exit_prices[key] = t_prices
        for s_idx, item in enumerate(group.get('stops', [])):
            trigger = item.get('trigger', {})
            el1 = trigger.get('element1', '')
            if el1 in ('R Profit', 'R Loss', 'ATR Target'):
                continue
            t_mask, t_prices = _vec_trigger(
                trigger, _arrays, indicator_map, _high, _low, _close, _n)
            for cond in item.get('conditions', []):
                t_mask = t_mask & _vec_condition(cond, _arrays, indicator_map, _n)
            key = (g_idx, 'stop', s_idx)
            _exit_masks[key] = t_mask
            _exit_prices[key] = t_prices

    # -------------------------------------------------
    # PRE-EXTRACT initial stop arrays for fast per-bar locked-value checks
    # -------------------------------------------------
    atr_stop_arr = None
    istop_arr1 = None
    istop_event = None
    istop_is_price = False

    if initial_stop_config:
        if initial_stop_config.get('stop_type') == 'ATR':
            atr_period = initial_stop_config.get('atr_period', 14)
            atr_stop_arr = atr_indicator(df['high'], df['low'], df['latest'], period=atr_period).values
        else:
            # Indicator-based stop: pre-extract the arrays
            istop_col1 = indicator_map.get(initial_stop_config.get('element1'))
            istop_event = initial_stop_config.get('event')
            if istop_col1 and istop_col1 in _arrays:
                istop_arr1 = _arrays[istop_col1]
                istop_is_price = (istop_col1 == 'latest')

    # Pre-compute static stop price arrays (for indicator-based stops)
    istop_price_arr = None
    if initial_stop_config and initial_stop_config.get('stop_type') != 'ATR':
        compare_type = initial_stop_config.get('compare_type', 'Indicator')
        if compare_type == 'Fixed Value':
            fv = initial_stop_config.get('value')
            if fv is not None:
                istop_price_arr = np.full(_n, fv, dtype=np.float64)
        else:
            el2 = initial_stop_config.get('element2')
            col2 = indicator_map.get(el2)
            if col2 and col2 in _arrays:
                istop_price_arr = _arrays[col2].astype(np.float64, copy=False)

    # ATR target series cache
    atr_target_array_cache = {}

    def _get_atr_target_array(period):
        if period not in atr_target_array_cache:
            atr_target_array_cache[period] = atr_indicator(
                df['high'], df['low'], df['latest'], period=period).values
        return atr_target_array_cache[period]

    # -------------------------------------------------
    # Inline helpers for position-dependent checks (R, ATR Target, locked stop)
    # These are minimal — no dict.get() chains, no function call overhead
    # -------------------------------------------------

    def _check_locked_stop(i, locked_value):
        """Check if initial stop triggers at bar i with a locked indicator value."""
        if locked_value is None or istop_arr1 is None or i == 0:
            return False
        val = istop_arr1[i]
        prev_val = istop_arr1[i - 1]

        if istop_event == "Cross Below":
            if istop_is_price:
                return (prev_val >= locked_value) and (_low[i] < locked_value)
            return (prev_val >= locked_value) and (val < locked_value)
        elif istop_event == "Cross Above":
            if istop_is_price:
                return (prev_val <= locked_value) and (_high[i] > locked_value)
            return (prev_val <= locked_value) and (val > locked_value)
        elif istop_event == "Close Below":
            return val < locked_value
        elif istop_event == "Close Above":
            return val > locked_value
        elif istop_event in ("Cross", "Close"):
            if istop_is_price:
                ca = (prev_val <= locked_value) and (_high[i] > locked_value)
                cb = (prev_val >= locked_value) and (_low[i] < locked_value)
            else:
                ca = (prev_val <= locked_value) and (val > locked_value)
                cb = (prev_val >= locked_value) and (val < locked_value)
            return ca or cb
        return False

    def _check_r_trigger(trigger, i, entry_price, r_distance):
        """Inline R Profit/Loss check — minimal overhead."""
        if entry_price is None or r_distance <= 0:
            return False
        el1 = trigger.get('element1')
        event = trigger.get('event')
        fv = trigger.get('value')
        if fv is None or i == 0:
            return False

        if el1 == "R Profit":
            price = _high[i] if strategy_direction == 'Long' else _low[i]
        else:
            price = _low[i] if strategy_direction == 'Long' else _high[i]

        move = (price - entry_price) if strategy_direction == 'Long' else (entry_price - price)
        r_val = (move / r_distance) if el1 == "R Profit" else (-move / r_distance)

        prev_price = _close[i - 1]
        prev_move = (prev_price - entry_price) if strategy_direction == 'Long' else (entry_price - prev_price)
        prev_r = (prev_move / r_distance) if el1 == "R Profit" else (-prev_move / r_distance)

        if event == "Cross Above":
            return (r_val > fv) and (prev_r <= fv)
        elif event == "Close Above":
            return r_val > fv
        elif event == "Cross Below":
            return (r_val < fv) and (prev_r >= fv)
        elif event == "Close Below":
            return r_val < fv
        elif event in ("Cross", "Close"):
            return ((r_val > fv) and (prev_r <= fv)) or ((r_val < fv) and (prev_r >= fv))
        elif event == "At Level":
            return abs(r_val - fv) < 0.01
        return False

    def _get_r_price(trigger, entry_price, r_distance):
        el1 = trigger.get('element1')
        fv = trigger.get('value', 0)
        if el1 == "R Profit":
            return entry_price + fv * r_distance * (1 if strategy_direction == 'Long' else -1)
        else:
            return entry_price - fv * r_distance * (1 if strategy_direction == 'Long' else -1)

    def _check_atr_target(trigger, i, locked_price):
        if locked_price is None or i == 0:
            return False
        event = trigger.get('event')
        # Default event when None: Long targets fire on Cross Above, Short on Cross Below
        if event is None:
            event = "Cross Above" if strategy_direction == 'Long' else "Cross Below"
        price = _high[i] if strategy_direction == 'Long' else _low[i]
        prev_price = _close[i - 1]

        if event == "Cross Above":
            return (price > locked_price) and (prev_price <= locked_price)
        elif event == "Close Above":
            return price > locked_price
        elif event == "Cross Below":
            return (price < locked_price) and (prev_price >= locked_price)
        elif event == "Close Below":
            return price < locked_price
        elif event == "Cross":
            return ((price > locked_price) and (prev_price <= locked_price)) or \
                   ((price < locked_price) and (prev_price >= locked_price))
        elif event == "Close":
            ca = (price > locked_price) and (prev_price <= locked_price)
            cb = (price < locked_price) and (prev_price >= locked_price)
            return ca or cb
        return False

    def _compute_atr_target_level(trigger, entry_price, bar_idx):
        atr_p = trigger.get('atr_period', 14)
        atr_m = trigger.get('atr_multiplier', 2.0)
        atr_arr = _get_atr_target_array(atr_p)
        v = atr_arr[bar_idx]
        if np.isnan(v):
            return None
        return entry_price + v * atr_m * (1 if strategy_direction == 'Long' else -1)

    # -------------------------------------------------
    # MAIN LOOP — uses pre-computed masks, minimal per-bar work
    # -------------------------------------------------
    all_trades = []
    trade_counter = 0
    open_positions = []

    for i in range(_n):
        # ----- STEP 1: Check exits for ALL open positions -----
        positions_to_remove = []
        for pos_idx, pos in enumerate(open_positions):
            if pos['entry_bar_idx'] == i:
                continue

            pos_exit_triggered = False
            pos_exit_type = None

            # --- Initial stop (locked value — per-bar check) ---
            if initial_stop_config:
                stop_type = initial_stop_config.get('stop_type', 'Indicator')
                if stop_type == 'ATR':
                    # ATR stop: check Price cross locked ATR level using event from config
                    locked = pos['locked_stop_value']
                    if locked is not None and i > 0:
                        prev_close = _close[i - 1]
                        stop_event = initial_stop_config.get('event', '')
                        triggered = False

                        if stop_event == "Cross Above":
                            triggered = (prev_close <= locked) and (_high[i] > locked)
                        elif stop_event == "Cross Below":
                            triggered = (prev_close >= locked) and (_low[i] < locked)
                        elif stop_event == "Close Above":
                            triggered = _close[i] > locked
                        elif stop_event == "Close Below":
                            triggered = _close[i] < locked
                        elif stop_event in ("Cross", "Close"):
                            ca = (prev_close <= locked) and (_high[i] > locked)
                            cb = (prev_close >= locked) and (_low[i] < locked)
                            triggered = ca or cb

                        if triggered:
                            pos_exit_type = "Initial Stop"
                            pos_exit_triggered = True
                            stop_exit_price = locked
                else:
                    # Indicator stop with locked value
                    if _check_locked_stop(i, pos['locked_stop_value']):
                        pos_exit_type = "Initial Stop"
                        pos_exit_triggered = True
                        stop_exit_price = pos['locked_stop_value']

                if pos_exit_triggered:
                    for group_idx in pos['active_exit_groups']:
                        alloc_pct = exit_groups[group_idx].get('allocation_pct', 100.0)
                        all_trades.append({
                            'trade_id': pos['trade_id'],
                            'entry_price': pos['entry_price'],
                            'exit_price': stop_exit_price,
                            'allocation_pct': alloc_pct,
                            'r_distance': pos['r_distance'],
                            'entry_r': pos['entry_r'],
                            'exit_type': 'Initial Stop'
                        })
                    pos['active_exit_groups'] = set()

            # --- Exit groups (use pre-computed masks where possible) ---
            if not pos_exit_triggered:
                groups_to_close = []

                for group_idx in list(pos['active_exit_groups']):
                    group = exit_groups[group_idx]
                    alloc_pct = group.get('allocation_pct', 100.0)

                    # Check targets
                    for t_idx, target in enumerate(group.get('targets', [])):
                        target_key = (group_idx, 'target', t_idx)
                        trigger = target.get('trigger', {})
                        el1 = trigger.get('element1', '')
                        hit = False
                        t_exit_price = _close[i]

                        if el1 in ("R Profit", "R Loss"):
                            if _check_r_trigger(trigger, i, pos['entry_price'], pos['r_distance']):
                                hit = True
                                t_exit_price = _get_r_price(trigger, pos['entry_price'], pos['r_distance'])
                        elif el1 == "ATR Target":
                            locked_p = pos.get('locked_atr_targets', {}).get(target_key)
                            if _check_atr_target(trigger, i, locked_p):
                                hit = True
                                t_exit_price = locked_p
                        elif target_key in _exit_masks:
                            # PRE-COMPUTED: just check the boolean array
                            if _exit_masks[target_key][i]:
                                hit = True
                                t_exit_price = _exit_prices[target_key][i]

                        if hit:
                            pos_exit_triggered = True
                            if pos_exit_type is None:
                                pos_exit_type = "Target"
                            all_trades.append({
                                'trade_id': pos['trade_id'],
                                'entry_price': pos['entry_price'],
                                'exit_price': t_exit_price,
                                'allocation_pct': alloc_pct,
                                'r_distance': pos['r_distance'],
                                'entry_r': pos['entry_r'],
                                'exit_type': 'Target'
                            })
                            groups_to_close.append(group_idx)
                            break

                    # Check stops (only if no target hit for this group)
                    if group_idx not in groups_to_close:
                        for s_idx, stop in enumerate(group.get('stops', [])):
                            stop_key = (group_idx, 'stop', s_idx)
                            trigger = stop.get('trigger', {})
                            el1 = trigger.get('element1', '')
                            hit = False
                            s_exit_price = _close[i]

                            if el1 in ("R Profit", "R Loss"):
                                if _check_r_trigger(trigger, i, pos['entry_price'], pos['r_distance']):
                                    hit = True
                                    s_exit_price = _get_r_price(trigger, pos['entry_price'], pos['r_distance'])
                            elif el1 == "ATR Target":
                                locked_p = pos.get('locked_atr_targets', {}).get(stop_key)
                                if _check_atr_target(trigger, i, locked_p):
                                    hit = True
                                    s_exit_price = locked_p
                            elif stop_key in _exit_masks:
                                if _exit_masks[stop_key][i]:
                                    hit = True
                                    s_exit_price = _exit_prices[stop_key][i]

                            if hit:
                                pos_exit_triggered = True
                                pos_exit_type = "Stop"
                                all_trades.append({
                                    'trade_id': pos['trade_id'],
                                    'entry_price': pos['entry_price'],
                                    'exit_price': s_exit_price,
                                    'allocation_pct': alloc_pct,
                                    'r_distance': pos['r_distance'],
                                    'entry_r': pos['entry_r'],
                                    'exit_type': 'Stop'
                                })
                                groups_to_close.append(group_idx)
                                break

                for group_idx in groups_to_close:
                    pos['active_exit_groups'].discard(group_idx)

            if pos_exit_triggered:
                if not pos['active_exit_groups']:
                    positions_to_remove.append(pos_idx)

        any_position_closed = len(positions_to_remove) > 0
        for pos_idx in sorted(positions_to_remove, reverse=True):
            open_positions.pop(pos_idx)

        # ----- STEP 2: Check entry (mostly pre-computed!) -----
        can_enter = (not any_position_closed
                     and (max_positions is None or len(open_positions) < max_positions))

        if can_enter and entry_possible[i]:
            trade_counter += 1
            new_entry_price = entry_trig_prices[i]

            # Get static stop price at entry
            if initial_stop_config:
                stop_type = initial_stop_config.get('stop_type', 'Indicator')
                if stop_type == 'ATR':
                    if atr_stop_arr is not None:
                        atr_val = atr_stop_arr[i]
                        if not np.isnan(atr_val):
                            atr_mult = initial_stop_config.get('atr_multiplier', 1.5)
                            if strategy_direction == 'Long':
                                new_locked_stop = new_entry_price - atr_val * atr_mult
                            else:
                                new_locked_stop = new_entry_price + atr_val * atr_mult
                        else:
                            new_locked_stop = None
                    else:
                        new_locked_stop = None
                else:
                    new_locked_stop = istop_price_arr[i] if istop_price_arr is not None else None
            else:
                new_locked_stop = None

            new_r_distance = abs(new_entry_price - new_locked_stop) if new_locked_stop is not None else 0.0

            # Guard: reject entries with degenerate r_distance.
            # If stop is too close to entry (< 0.01% of price), the R calculation
            # produces absurd multiples from normal price moves. Skip the entry.
            _MIN_R_FRAC = 1e-4  # 0.01% of entry price
            if new_entry_price > 0 and new_r_distance < new_entry_price * _MIN_R_FRAC:
                trade_counter -= 1  # undo counter increment
                continue

            # Compute locked ATR target prices
            locked_atr_targets = {}
            for g_idx, group in enumerate(exit_groups):
                for t_idx, target in enumerate(group.get('targets', [])):
                    t_trigger = target.get('trigger', {})
                    if t_trigger.get('element1') == 'ATR Target':
                        locked_atr_targets[(g_idx, 'target', t_idx)] = _compute_atr_target_level(t_trigger, new_entry_price, i)
                for s_idx, stop in enumerate(group.get('stops', [])):
                    s_trigger = stop.get('trigger', {})
                    if s_trigger.get('element1') == 'ATR Target':
                        locked_atr_targets[(g_idx, 'stop', s_idx)] = _compute_atr_target_level(s_trigger, new_entry_price, i)

            open_positions.append({
                'trade_id': trade_counter,
                'entry_price': new_entry_price,
                'r_distance': new_r_distance,
                'locked_stop_value': new_locked_stop,
                'entry_r': entry_r,
                'active_exit_groups': set(range(len(exit_groups))),
                'entry_bar_idx': i,
                'locked_atr_targets': locked_atr_targets,
            })

    # -------------------------------------------------
    # Handle open positions at end of data
    # -------------------------------------------------
    last_price = _close[-1] if _n > 0 else 0.0
    for pos in open_positions:
        for group_idx in pos['active_exit_groups']:
            alloc_pct = exit_groups[group_idx].get('allocation_pct', 100.0)
            all_trades.append({
                'trade_id': pos['trade_id'],
                'entry_price': pos['entry_price'],
                'exit_price': last_price,
                'allocation_pct': alloc_pct,
                'r_distance': pos['r_distance'],
                'entry_r': pos['entry_r'],
                'exit_type': 'End of Data'
            })

    # -------------------------------------------------
    # Statistics (identical to original)
    # -------------------------------------------------
    for trade in all_trades:
        entry_price = trade['entry_price']
        exit_price = trade['exit_price']
        alloc_pct = trade['allocation_pct']
        r_distance = trade['r_distance']
        t_entry_r = trade['entry_r']

        # Guard: skip trades with non-finite prices
        if not (np.isfinite(entry_price) and np.isfinite(exit_price)):
            trade['pnl_r'] = 0.0
            continue

        if strategy_direction == 'Long':
            price_move = exit_price - entry_price
        else:
            price_move = entry_price - exit_price

        if r_distance > 0:
            group_r = t_entry_r * (alloc_pct / 100.0)
            pnl_r = (price_move / r_distance) * group_r
            # Guard: cap non-finite results from near-zero r_distance
            trade['pnl_r'] = pnl_r if np.isfinite(pnl_r) else 0.0
        else:
            trade['pnl_r'] = 0.0

    exit_type_priority = {'Initial Stop': 0, 'Stop': 1, 'End of Data': 2, 'Target': 3}

    from collections import OrderedDict
    entry_trades = OrderedDict()
    for trade in all_trades:
        tid = trade['trade_id']
        if tid not in entry_trades:
            entry_trades[tid] = {
                'pnl_r': 0.0,
                'exit_type': trade['exit_type'],
                'static_alloc': 0.0,
                'dynamic_alloc': 0.0,
                'target_alloc': 0.0,
                'eod_alloc': 0.0,
            }
        entry = entry_trades[tid]
        entry['pnl_r'] += trade['pnl_r']
        alloc = trade.get('allocation_pct', 100.0)
        if trade['exit_type'] == 'Initial Stop':
            entry['static_alloc'] += alloc
        elif trade['exit_type'] == 'Stop':
            entry['dynamic_alloc'] += alloc
        elif trade['exit_type'] == 'Target':
            entry['target_alloc'] += alloc
        elif trade['exit_type'] == 'End of Data':
            entry['eod_alloc'] += alloc
        if exit_type_priority.get(trade['exit_type'], 99) < exit_type_priority.get(entry['exit_type'], 99):
            entry['exit_type'] = trade['exit_type']

    trade_pnls_r = [t['pnl_r'] for t in entry_trades.values()]
    num_trades = len(entry_trades)

    if num_trades > 0:
        wins = sum(pnl > 0 for pnl in trade_pnls_r)
        flat = sum(pnl == 0 for pnl in trade_pnls_r)
        losses = num_trades - wins - flat
        # Win/loss rates exclude flat (0R) trades — they had no meaningful outcome
        meaningful = wins + losses
        win_rate = (wins / meaningful * 100) if meaningful > 0 else 0.0
        loss_rate = (losses / meaningful * 100) if meaningful > 0 else 0.0
        total_pnl = sum(trade_pnls_r)
        avg_pnl_per_trade = total_pnl / num_trades
        winning_trades_pnl = sum(pnl for pnl in trade_pnls_r if pnl > 0)
        losing_trades_pnl = sum(pnl for pnl in trade_pnls_r if pnl < 0)

        target_exit_pct = sum(t['target_alloc'] for t in entry_trades.values()) / num_trades
        static_exit_pct = sum(t['static_alloc'] for t in entry_trades.values()) / num_trades
        dynamic_exit_pct = sum(t['dynamic_alloc'] for t in entry_trades.values()) / num_trades
        eod_exit_pct = sum(t['eod_alloc'] for t in entry_trades.values()) / num_trades

        avg_win_pnl = winning_trades_pnl / wins if wins > 0 else 0.0
        avg_lose_pnl = losing_trades_pnl / losses if losses > 0 else 0.0
        rr_ratio = abs(avg_win_pnl / avg_lose_pnl) if avg_lose_pnl != 0 else 0.0

        cumulative = np.cumsum(trade_pnls_r)
        peak = np.maximum.accumulate(cumulative)
        drawdowns = cumulative - peak
        max_drawdown = abs(drawdowns.min()) if len(drawdowns) > 0 else 0.0

        if num_trades >= 2:
            pnl_std = np.std(trade_pnls_r, ddof=1)
            sqn = (np.mean(trade_pnls_r) / pnl_std) * np.sqrt(num_trades) if pnl_std > 0 else 0.0
        else:
            sqn = 0.0
    else:
        win_rate = loss_rate = total_pnl = avg_pnl_per_trade = 0.0
        winning_trades_pnl = losing_trades_pnl = 0.0
        target_exit_pct = static_exit_pct = dynamic_exit_pct = eod_exit_pct = 0.0
        rr_ratio = max_drawdown = sqn = 0.0

    stats_df = pd.DataFrame(
        {"value": [
            num_trades, win_rate, loss_rate, 0.0, total_pnl,
            avg_pnl_per_trade, winning_trades_pnl, losing_trades_pnl,
            target_exit_pct, static_exit_pct, dynamic_exit_pct, eod_exit_pct,
            rr_ratio, max_drawdown, sqn,
        ]},
        index=[
            "Number of trades", "Win rate (%)", "Loss rate (%)",
            "Total return (%)", "Total P&L (R)", "Avg P&L per trade (R)",
            "Winning trades P&L (R)", "Losing trades P&L (R)",
            "Target exit (%)", "Static exit (%)", "Dynamic exit (%)", "EOD exit (%)",
            "RR Ratio", "Max Drawdown (R)", "SQN",
        ],
    )

    stats_df.attrs['trade_pnls_r'] = list(trade_pnls_r)
    stats_df.attrs['total_static_alloc'] = sum(t['static_alloc'] for t in entry_trades.values()) if num_trades > 0 else 0.0
    stats_df.attrs['total_dynamic_alloc'] = sum(t['dynamic_alloc'] for t in entry_trades.values()) if num_trades > 0 else 0.0
    stats_df.attrs['total_target_alloc'] = sum(t['target_alloc'] for t in entry_trades.values()) if num_trades > 0 else 0.0
    stats_df.attrs['total_eod_alloc'] = sum(t['eod_alloc'] for t in entry_trades.values()) if num_trades > 0 else 0.0

    return None, stats_df

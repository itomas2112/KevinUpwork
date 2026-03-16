# win %, loss %, number of trades, return

import copy
import pandas as pd
import numpy as np
from config.constants import get_indicator_map
from indicators.atr_indicator import atr_indicator


# ---------------------------------------------------------------------------
# Ichimoku special comparison preprocessing
# ---------------------------------------------------------------------------
_ICHIMOKU_DISPLACEMENT = 26
_SENKOU_ELEMENTS = frozenset({"Senkou A", "Senkou B"})


def _prepare_ichimoku_columns(df, indicator_map, strategy_config):
    """
    Handle Ichimoku special comparison rules by creating shifted columns
    and remapping element names in the strategy config (already deep-copied).

    Rules:
      1. Chikou vs X  →  latest.shift(26) vs X_col.shift(26)
         (compare price 26 bars ago with indicator 26 bars ago)
      2. Senkou A/B vs non-Senkou indicator  →  no change (compare at time t)
      3. Senkou A vs Senkou B  →  use senkou_a_current vs senkou_b_current
         (raw / unshifted values = cloud 26 bars ahead)
    """
    created = set()

    def _ensure_shifted(orig_col, shift):
        """Create a shifted column once; return its name."""
        new_col = f'_ichi_{orig_col}_s{shift}'
        if new_col not in created and orig_col in df.columns:
            df[new_col] = df[orig_col].shift(shift)
            created.add(new_col)
            indicator_map[new_col] = new_col
        return new_col

    def _remap_pair(config, key1='element1', key2='element2'):
        """Remap element names for a single trigger / condition dict."""
        e1 = config.get(key1)
        e2 = config.get(key2)
        compare_type = config.get('compare_type', 'Indicator')

        if not e1:
            return

        # Chikou with Fixed Value — Chikou becomes current price, fixed value stays
        if e1 == 'Chikou' and compare_type == 'Fixed Value':
            config[key1] = 'Price'
            return

        if not e2 or compare_type == 'Fixed Value':
            return

        chikou_involved = (e1 == 'Chikou' or e2 == 'Chikou')
        senkou_vs_senkou = (e1 in _SENKOU_ELEMENTS and e2 in _SENKOU_ELEMENTS)

        if chikou_involved:
            # Rule 1: Price[t] vs Indicator[t-26]
            # Chikou → current price (latest), other → shifted back by 26
            for key, elem in [(key1, e1), (key2, e2)]:
                if elem == 'Chikou':
                    config[key] = 'Price'
                else:
                    orig_col = indicator_map.get(elem)
                    if orig_col and orig_col in df.columns:
                        config[key] = _ensure_shifted(orig_col, _ICHIMOKU_DISPLACEMENT)

        elif senkou_vs_senkou:
            # Rule 3: use raw / unshifted Senkou columns
            remap = {
                'Senkou A': 'senkou_a_current',
                'Senkou B': 'senkou_b_current',
            }
            for key, elem in [(key1, e1), (key2, e2)]:
                raw_col = remap[elem]
                syn = f'_raw_{raw_col}'
                if syn not in indicator_map:
                    indicator_map[syn] = raw_col
                config[key] = syn

        # Rule 2 (Senkou vs non-Senkou, non-Chikou): no changes needed

    def _process_trigger(trigger):
        if trigger:
            _remap_pair(trigger)

    def _process_conditions(conditions):
        for cond in (conditions or []):
            _remap_pair(cond)

    # --- Entry ---
    entry = strategy_config.get('entry', {})
    _process_trigger(entry.get('trigger'))
    _process_conditions(entry.get('conditions'))

    # --- Initial stop ---
    initial_stop = strategy_config.get('initial_stop')
    if initial_stop:
        _remap_pair(initial_stop)

    # --- Exit groups ---
    for group in strategy_config.get('exit_groups', []):
        for target in group.get('targets', []):
            _process_trigger(target.get('trigger'))
            _process_conditions(target.get('conditions'))
        for stop in group.get('stops', []):
            _process_trigger(stop.get('trigger'))
            _process_conditions(stop.get('conditions'))

    # --- Old-style single exit (backward compat) ---
    old_exit = strategy_config.get('exit', {})
    if old_exit:
        _process_trigger(old_exit.get('trigger'))
        _process_conditions(old_exit.get('conditions'))

def ichimoku_tenkan_kijun_strategy(df: pd.DataFrame):
    df = df.copy()

    # -------------------------------------------------
    # Clean existing signals
    # -------------------------------------------------
    for col in ["entry_signal", "exit_signal"]:
        if col in df.columns:
            df.drop(columns=col, inplace=True)

    # -------------------------------------------------
    # Cross conditions
    # -------------------------------------------------
    tenkan = df["tenkan"]
    kijun = df["kijun"]

    cross_up = (tenkan > kijun) & (tenkan.shift(1) <= kijun.shift(1))
    cross_down = (tenkan < kijun) & (tenkan.shift(1) >= kijun.shift(1))

    # -------------------------------------------------
    # Signal generation (entry first, exit after)
    # -------------------------------------------------
    in_trade = False
    entry_signal = []
    exit_signal = []

    entry_prices = []
    trade_returns = []

    current_entry_price = None

    for i, (up, down) in enumerate(zip(cross_up, cross_down)):

        price = df["latest"].iloc[i]

        if not in_trade and up:
            # ---- Entry
            entry_signal.append(True)
            exit_signal.append(False)

            in_trade = True
            current_entry_price = price
            entry_prices.append(price)

        elif in_trade and down:
            # ---- Exit
            entry_signal.append(False)
            exit_signal.append(True)

            trade_return = price / current_entry_price
            trade_returns.append(trade_return)

            in_trade = False
            current_entry_price = None

        else:
            entry_signal.append(False)
            exit_signal.append(False)

    # -------------------------------------------------
    # Handle open trade at the end
    # -------------------------------------------------
    if in_trade and current_entry_price is not None:
        last_price = df["latest"].iloc[-1]
        trade_return = last_price / current_entry_price
        trade_returns.append(trade_return)

    # -------------------------------------------------
    # Attach signals
    # -------------------------------------------------
    df["entry_signal"] = entry_signal
    df["exit_signal"] = exit_signal

    # -------------------------------------------------
    # Statistics
    # -------------------------------------------------
    num_trades = len(trade_returns)

    if num_trades > 0:
        wins = sum(r > 1 for r in trade_returns)
        losses = num_trades - wins

        win_rate = wins / num_trades * 100
        loss_rate = losses / num_trades * 100

        total_return = 1.0
        for r in trade_returns:
            total_return *= r
    else:
        win_rate = 0.0
        loss_rate = 0.0
        total_return = 1.0

    stats_df = pd.DataFrame(
        {
            "value": [
                num_trades,
                win_rate,
                loss_rate,
                (total_return - 1) * 100,
            ]
        },
        index=[
            "Number of trades",
            "Win rate (%)",
            "Loss rate (%)",
            "Total return (%)",
        ],
    )

    return df, stats_df


def execute_custom_strategy(df: pd.DataFrame, strategy_config: dict, period_start=None, period_end=None):
    """
    Execute a custom strategy based on the saved strategy configuration.
    Supports multiple exit groups with OCO (One-Cancels-Other) targets and stops.
    Uses R-based position sizing: R = distance between entry price and static stop.

    Parameters:
    -----------
    df : pd.DataFrame
        DataFrame with OHLC data and all calculated indicators
    strategy_config : dict
        The strategy configuration from saved_strategies
    period_start : pd.Timestamp, optional
        Start of the DRM date range — entries only allowed from this point
    period_end : pd.Timestamp, optional
        End of the DRM date range — entries only allowed up to this point

    Returns:
    --------
    df : pd.DataFrame
        DataFrame with entry_signal and exit_signal columns added
    stats_df : pd.DataFrame
        Statistics DataFrame with win rate, loss rate, number of trades, total return, P&L in R
    """

    df = df.copy()
    strategy_config = copy.deepcopy(strategy_config)  # avoid mutating caller's data

    # -------------------------------------------------
    # Clean existing signals
    # -------------------------------------------------
    for col in ["entry_signal", "exit_signal"]:
        if col in df.columns:
            df.drop(columns=col, inplace=True)

    # -------------------------------------------------
    # Use global indicator mapping
    # -------------------------------------------------
    # Count EMA columns in the DataFrame to build the full indicator map
    ema_count = 0
    while f"ema_{ema_count}" in df.columns:
        ema_count += 1
    indicator_map = get_indicator_map(ema_count)

    # -------------------------------------------------
    # Ichimoku special comparison preprocessing
    # -------------------------------------------------
    _prepare_ichimoku_columns(df, indicator_map, strategy_config)

    # -------------------------------------------------
    # Helper function to check if condition is met
    # -------------------------------------------------
    def check_condition(condition_config, current_idx):
        """Check if a single condition is met at given index"""
        element1_name = condition_config.get('element1')
        operator = condition_config.get('operator')
        compare_type = condition_config.get('compare_type', 'Indicator')
        element2_name = condition_config.get('element2')
        fixed_value = condition_config.get('value')

        # Get the column for element1
        col1 = indicator_map.get(element1_name)

        if col1 is None or col1 not in df.columns:
            return False

        series1 = df[col1]
        value1 = series1.iloc[current_idx]

        # Determine what to compare against
        if compare_type == "Fixed Value":
            if fixed_value is None:
                return False
            value2 = fixed_value
        else:
            col2 = indicator_map.get(element2_name)
            if col2 is None or col2 not in df.columns:
                return False
            series2 = df[col2]
            value2 = series2.iloc[current_idx]

        # Check the operator
        if operator == "Above":
            return value1 > value2
        elif operator == "Below":
            return value1 < value2

        return False

    # -------------------------------------------------
    # Helper: check R Profit / R Loss trigger
    # -------------------------------------------------
    def check_r_trigger(trigger_config, current_idx, pos_entry_price, pos_r_distance):
        """
        Check if an R Profit or R Loss trigger event occurred.
        R Profit = unrealized profit in R terms (positive when profitable).
        R Loss = unrealized loss in R terms (positive when losing).
        Only valid while in a position.

        Uses high/low prices for current bar detection (like regular Cross events)
        and close price for previous bar state.
        """
        if pos_entry_price is None or pos_r_distance <= 0:
            return False

        element1_name = trigger_config.get('element1')
        event = trigger_config.get('event')
        fixed_value = trigger_config.get('value')

        if fixed_value is None:
            return False

        # Use high/low for current bar R value depending on what we're checking:
        # R Profit (Long) → high gives best-case profit → use high
        # R Profit (Short) → low gives best-case profit → use low
        # R Loss (Long) → low gives worst-case loss → use low
        # R Loss (Short) → high gives worst-case loss → use high
        if element1_name == "R Profit":
            if strategy_direction == 'Long':
                price = df['high'].iloc[current_idx]
            else:
                price = df['low'].iloc[current_idx]
        else:  # R Loss
            if strategy_direction == 'Long':
                price = df['low'].iloc[current_idx]
            else:
                price = df['high'].iloc[current_idx]

        # Calculate directional price move for current bar
        if strategy_direction == 'Long':
            price_move = price - pos_entry_price
        else:
            price_move = pos_entry_price - price

        # R Profit: positive when in profit, negative when in loss
        # R Loss: positive when in loss, negative when in profit
        if element1_name == "R Profit":
            r_value = price_move / pos_r_distance
        else:  # R Loss
            r_value = -price_move / pos_r_distance

        # Previous bar uses close price (same pattern as regular Cross events)
        if current_idx > 0:
            prev_price = df['latest'].iloc[current_idx - 1]
            if strategy_direction == 'Long':
                prev_move = prev_price - pos_entry_price
            else:
                prev_move = pos_entry_price - prev_price

            if element1_name == "R Profit":
                prev_r_value = prev_move / pos_r_distance
            else:
                prev_r_value = -prev_move / pos_r_distance
        else:
            return False

        # Check event type
        if event in ("Cross Above", "Close Above"):
            return (r_value > fixed_value) and (prev_r_value <= fixed_value)
        elif event in ("Cross Below", "Close Below"):
            return (r_value < fixed_value) and (prev_r_value >= fixed_value)
        elif event in ("Cross", "Close"):
            cross_above = (r_value > fixed_value) and (prev_r_value <= fixed_value)
            cross_below = (r_value < fixed_value) and (prev_r_value >= fixed_value)
            return cross_above or cross_below
        elif event == "At Level":
            return abs(r_value - fixed_value) < 0.01

        return False

    def get_r_trigger_price(trigger_config, current_idx, pos_entry_price, pos_r_distance):
        """
        Compute the exact price at the target R level rather than using
        the bar's close (which may overshoot).
        """
        if pos_entry_price is None or pos_r_distance <= 0:
            return df['latest'].iloc[current_idx]

        element1_name = trigger_config.get('element1')
        fixed_value = trigger_config.get('value')
        if fixed_value is None:
            return df['latest'].iloc[current_idx]

        if element1_name == "R Profit":
            if strategy_direction == 'Long':
                return pos_entry_price + fixed_value * pos_r_distance
            else:
                return pos_entry_price - fixed_value * pos_r_distance
        else:  # R Loss
            if strategy_direction == 'Long':
                return pos_entry_price - fixed_value * pos_r_distance
            else:
                return pos_entry_price + fixed_value * pos_r_distance

    # -------------------------------------------------
    # Helper function to check trigger events
    # -------------------------------------------------
    def check_trigger(trigger_config, current_idx, locked_value=None):
        """
        Check if a trigger event occurred at given index.

        Parameters:
        -----------
        locked_value : float, optional
            If provided, overrides the element2 value with this fixed price.
            Used by the initial (static) stop to lock the indicator value at entry time.
        """
        element1_name = trigger_config.get('element1')
        event = trigger_config.get('event')
        compare_type = trigger_config.get('compare_type', 'Indicator')
        element2_name = trigger_config.get('element2')
        fixed_value = trigger_config.get('value')

        # Get the series for element1
        col1 = indicator_map.get(element1_name)

        if col1 is None or col1 not in df.columns:
            return False

        series1 = df[col1]

        # Determine col2
        col2 = None
        if locked_value is None and compare_type != "Fixed Value":
            col2 = indicator_map.get(element2_name)
            if col2 is None or col2 not in df.columns:
                return False

        value1 = series1.iloc[current_idx]

        # Determine what to compare against
        if locked_value is not None:
            value2 = locked_value

            if current_idx > 0:
                value1_prev = series1.iloc[current_idx - 1]
                value2_prev = locked_value
        elif compare_type == "Fixed Value":
            if fixed_value is None:
                return False

            value2 = fixed_value

            if current_idx > 0:
                value1_prev = series1.iloc[current_idx - 1]
                value2_prev = fixed_value
        else:
            series2 = df[col2]
            value2 = series2.iloc[current_idx]

            if current_idx > 0:
                value1_prev = series1.iloc[current_idx - 1]
                value2_prev = series2.iloc[current_idx - 1]

        # Check the event type
        if event == "Close Above":
            # Simply check if current close is above the value
            return value1 > value2

        elif event == "Close Below":
            # Simply check if current close is below the value
            return value1 < value2

        elif event == "Close":
            # Close in either direction (previous bar on other side)
            if current_idx == 0:
                return False
            cross_above = (value1 > value2) and (value1_prev <= value2_prev)
            cross_below = (value1 < value2) and (value1_prev >= value2_prev)
            return cross_above or cross_below

        elif event == "Cross Above":
            # Previous bar must have been below, current bar's high exceeds the value
            if current_idx == 0:
                return False
            if value1_prev > value2_prev:
                return False
            # Use high price for element1 if it's Price, otherwise use the indicator value
            if col1 == "latest":
                high_val = df["high"].iloc[current_idx]
                return high_val > value2
            else:
                return value1 > value2

        elif event == "Cross Below":
            # Previous bar must have been above, current bar's low goes below the value
            if current_idx == 0:
                return False
            if value1_prev < value2_prev:
                return False
            # Use low price for element1 if it's Price, otherwise use the indicator value
            if col1 == "latest":
                low_val = df["low"].iloc[current_idx]
                return low_val < value2
            else:
                return value1 < value2

        elif event == "Cross":
            # Cross in either direction
            if current_idx == 0:
                return False
            if col1 == "latest":
                high_val = df["high"].iloc[current_idx]
                low_val = df["low"].iloc[current_idx]
                cross_above = (value1_prev <= value2_prev) and (high_val > value2)
                cross_below = (value1_prev >= value2_prev) and (low_val < value2)
            else:
                cross_above = (value1 > value2) and (value1_prev <= value2_prev)
                cross_below = (value1 < value2) and (value1_prev >= value2_prev)
            return cross_above or cross_below

        elif event == "At Level":
            return abs(value1 - value2) < 0.01

        return False

    def get_trigger_price(trigger_config, current_idx, locked_value=None):
        """
        Determine the trade price for a triggered event.
        - Cross events with Price as element1: use the crossed indicator's value +/- $0.01
        - Close events or non-Price triggers: use bar close price
        - If locked_value is provided (static stop), use that as the crossed level
        """
        event = trigger_config.get('event', '')
        is_cross = event in ('Cross', 'Cross Above', 'Cross Below')

        if is_cross:
            element1 = trigger_config.get('element1')

            if element1 == 'Price':
                # Static stop: use locked price
                if locked_value is not None:
                    return locked_value

                compare_type = trigger_config.get('compare_type', 'Indicator')
                if compare_type == 'Fixed Value':
                    fixed_val = trigger_config.get('value')
                    if fixed_val is not None:
                        # $0.01 offset based on cross direction
                        if event == 'Cross Above':
                            return fixed_val + 0.01
                        elif event == 'Cross Below':
                            return fixed_val - 0.01
                        return fixed_val
                else:
                    element2 = trigger_config.get('element2')
                    col2 = indicator_map.get(element2)
                    if col2 and col2 in df.columns:
                        crossed_val = df[col2].iloc[current_idx]
                        # $0.01 offset based on cross direction
                        if event == 'Cross Above':
                            return crossed_val + 0.01
                        elif event == 'Cross Below':
                            return crossed_val - 0.01
                        return crossed_val

        return df['latest'].iloc[current_idx]

    # -------------------------------------------------
    # Helper: get the static stop indicator value at a bar
    # -------------------------------------------------
    def get_static_stop_price(stop_config, current_idx, entry_price=None):
        """
        Return the stop price used by the initial (static) stop at a given bar.
        For Indicator mode: returns the indicator's value (locked at entry time).
        For ATR mode: returns entry_price ± ATR × multiplier based on direction.
        """
        if not stop_config:
            return None

        stop_type = stop_config.get('stop_type', 'Indicator')

        if stop_type == 'ATR':
            if entry_price is None:
                return None
            atr_mult = stop_config.get('atr_multiplier', 1.5)
            # Use pre-computed ATR series
            atr_val = atr_stop_series.iloc[current_idx] if atr_stop_series is not None else None
            if pd.isna(atr_val):
                return None
            # Long: stop below entry, Short: stop above entry
            if strategy_direction == 'Long':
                return entry_price - atr_val * atr_mult
            else:
                return entry_price + atr_val * atr_mult

        # Indicator mode
        compare_type = stop_config.get('compare_type', 'Indicator')
        if compare_type == 'Fixed Value':
            return stop_config.get('value')

        element2 = stop_config.get('element2')
        col2 = indicator_map.get(element2)
        if col2 and col2 in df.columns:
            return df[col2].iloc[current_idx]
        return None

    # -------------------------------------------------
    # Helper function to check trigger with conditions
    # -------------------------------------------------
    def check_trigger_and_conditions(trigger_config, conditions, current_idx):
        """Check if trigger is activated AND all conditions are met"""

        # First check the trigger
        if not check_trigger(trigger_config, current_idx):
            return False

        # If trigger is activated, check all conditions
        for condition in conditions:
            if not check_condition(condition, current_idx):
                return False  # If any condition fails, return False

        # All conditions met
        return True

    # -------------------------------------------------
    # Helper function to check exit (target or stop)
    # -------------------------------------------------
    def check_exit_signal(exit_config, current_idx, pos_entry_price=None, pos_r_distance=0.0):
        """Check if an exit signal (target or stop) is triggered"""
        trigger = exit_config.get('trigger', {})
        conditions = exit_config.get('conditions', [])

        # R Profit / R Loss triggers use special handler
        element1 = trigger.get('element1', '')
        if element1 in ("R Profit", "R Loss"):
            if not check_r_trigger(trigger, current_idx, pos_entry_price, pos_r_distance):
                return False
            # Still check conditions
            for condition in conditions:
                if not check_condition(condition, current_idx):
                    return False
            return True

        return check_trigger_and_conditions(trigger, conditions, current_idx)

    # -------------------------------------------------
    # Helper function to check if stop is already violated
    # -------------------------------------------------
    def is_stop_already_violated(trigger_config, current_idx, strategy_direction):
        """
        Check if a stop condition is already in a violated state at entry time.

        For Long trades:
        - "Cross Below" stop: check if element1 is ALREADY below element2
        - "Cross Above" stop: check if element1 is ALREADY above element2

        For Short trades:
        - "Cross Above" stop: check if element1 is ALREADY above element2
        - "Cross Below" stop: check if element1 is ALREADY below element2

        This prevents entries when the stop would be immediately triggered.
        """
        element1_name = trigger_config.get('element1')

        # R Profit/R Loss can't be evaluated before entry (no position exists yet)
        if element1_name in ("R Profit", "R Loss"):
            return False

        # ATR stops can't be evaluated before entry (stop level depends on entry price)
        if trigger_config.get('stop_type') == 'ATR':
            return False

        event = trigger_config.get('event')
        compare_type = trigger_config.get('compare_type', 'Indicator')
        element2_name = trigger_config.get('element2')
        fixed_value = trigger_config.get('value')

        # Get the series for element1
        col1 = indicator_map.get(element1_name)

        if col1 is None or col1 not in df.columns:
            return False

        series1 = df[col1]
        value1 = series1.iloc[current_idx]

        # Determine what to compare against
        if compare_type == "Fixed Value":
            if fixed_value is None:
                return False
            value2 = fixed_value
        else:
            col2 = indicator_map.get(element2_name)
            if col2 is None or col2 not in df.columns:
                return False
            series2 = df[col2]
            value2 = series2.iloc[current_idx]

        # Check if the stop condition is already violated
        # A "Cross Below" stop means we exit when element1 goes below element2
        # So if element1 is ALREADY below element2, we should NOT enter
        if event in ("Cross Below", "Close Below"):
            # Already below = stop already violated
            return value1 < value2
        elif event in ("Cross Above", "Close Above"):
            # Already above = stop already violated
            return value1 > value2
        elif event in ("Cross", "Close"):
            # For generic "Cross"/"Close", we need to determine which direction matters
            # based on strategy direction
            if strategy_direction == 'Long':
                return value1 < value2
            else:
                return value1 > value2

        return False

    def are_any_stops_already_violated(initial_stop_config, exit_groups, current_idx, strategy_direction):
        """
        Check if ANY stop condition (initial stop or group stops) is already violated.
        Returns True if entry should be blocked.
        """
        # Check initial stop
        if initial_stop_config:
            if is_stop_already_violated(initial_stop_config, current_idx, strategy_direction):
                return True

        # Check all stop triggers in all exit groups
        for group in exit_groups:
            for stop in group.get('stops', []):
                stop_trigger = stop.get('trigger', {})
                if stop_trigger:
                    if is_stop_already_violated(stop_trigger, current_idx, strategy_direction):
                        return True

        return False

    # -------------------------------------------------
    # Extract entry configuration
    # -------------------------------------------------
    entry_config = strategy_config.get('entry', {})
    entry_trigger = entry_config.get('trigger', {})
    entry_conditions = entry_config.get('conditions', [])
    entry_r = entry_config.get('position_size', 1.0)  # R multiples

    # -------------------------------------------------
    # Extract initial stop (shared across all exit groups)
    # -------------------------------------------------
    initial_stop_config = strategy_config.get('initial_stop') or None

    # Pre-compute ATR series if using ATR-based static stop (avoids recalc per entry)
    atr_stop_series = None
    if initial_stop_config and initial_stop_config.get('stop_type') == 'ATR':
        atr_period = initial_stop_config.get('atr_period', 14)
        atr_stop_series = atr_indicator(df['high'], df['low'], df['latest'], period=atr_period)

    # -------------------------------------------------
    # Extract exit groups
    # -------------------------------------------------
    exit_groups = strategy_config.get('exit_groups', [])

    # Fallback to old single exit structure for backward compatibility
    if not exit_groups:
        exit_config = strategy_config.get('exit', {})
        if exit_config:
            exit_groups = [{
                'group_id': 1,
                'allocation_pct': 100.0,
                'targets': [{
                    'type': 'Target',
                    'trigger': exit_config.get('trigger', {}),
                    'conditions': exit_config.get('conditions', [])
                }],
                'stops': []
            }]

    # Backward compat: if exit groups have position_size but no allocation_pct, split equally
    for g in exit_groups:
        if 'allocation_pct' not in g:
            g['allocation_pct'] = 100.0 / max(len(exit_groups), 1)

    # Get strategy direction
    strategy_direction = strategy_config.get('direction', 'Long')

    # Max positions: None = unlimited, integer >= 1 (default 1 for backward compat)
    max_positions = strategy_config.get('max_positions', 1)

    # -------------------------------------------------
    # Signal generation with multiple positions + OCO exits
    # -------------------------------------------------
    entry_signal = []
    exit_signal = []
    exit_type_series = []  # Track exit type per bar: None, "Target", "Stop", "Initial Stop"

    # Trade tracking for P&L (R-based)
    all_trades = []  # List of dicts: {trade_id, entry_price, exit_price, allocation_pct, r_distance, entry_r, exit_type}
    trade_counter = 0  # Unique ID per entry

    # Multi-position tracking: list of open position dicts
    # Each: {trade_id, entry_price, r_distance, locked_stop_value, entry_r, active_exit_groups, entry_bar_idx}
    open_positions = []

    for i in range(len(df)):
        bar_entry = False
        bar_exit = False
        bar_exit_type = None

        # ----- STEP 1: Check exits for ALL open positions -----
        positions_to_remove = []
        for pos_idx, pos in enumerate(open_positions):
            # Skip exit checks on the entry bar itself
            if pos['entry_bar_idx'] == i:
                continue

            pos_exit_triggered = False
            pos_exit_type = None

            # Check initial stop (applies to ALL active groups in this position)
            if initial_stop_config and check_trigger(initial_stop_config, i, locked_value=pos['locked_stop_value']):
                pos_exit_type = "Initial Stop"
                pos_exit_triggered = True
                stop_exit_price = get_trigger_price(initial_stop_config, i, locked_value=pos['locked_stop_value'])

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

            # If initial stop didn't trigger, check each active exit group
            if not pos_exit_triggered:
                groups_to_close = []

                for group_idx in list(pos['active_exit_groups']):
                    group = exit_groups[group_idx]
                    alloc_pct = group.get('allocation_pct', 100.0)

                    # Check targets
                    for target in group.get('targets', []):
                        if check_exit_signal(target, i, pos['entry_price'], pos['r_distance']):
                            pos_exit_triggered = True
                            if pos_exit_type is None:
                                pos_exit_type = "Target"

                            target_trigger = target.get('trigger', {})
                            t_el1 = target_trigger.get('element1', '')
                            t_exit_price = (get_r_trigger_price(target_trigger, i, pos['entry_price'], pos['r_distance'])
                                            if t_el1 in ("R Profit", "R Loss")
                                            else get_trigger_price(target_trigger, i))
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

                    # If no target hit, check stops
                    if group_idx not in groups_to_close:
                        for stop in group.get('stops', []):
                            if check_exit_signal(stop, i, pos['entry_price'], pos['r_distance']):
                                pos_exit_triggered = True
                                pos_exit_type = "Stop"

                                stop_trigger = stop.get('trigger', {})
                                s_el1 = stop_trigger.get('element1', '')
                                s_exit_price = (get_r_trigger_price(stop_trigger, i, pos['entry_price'], pos['r_distance'])
                                                if s_el1 in ("R Profit", "R Loss")
                                                else get_trigger_price(stop_trigger, i))
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

            # If any exit happened for this position
            if pos_exit_triggered:
                bar_exit = True
                # Keep highest-priority exit type across all positions on this bar
                exit_type_priority = {'Initial Stop': 0, 'Stop': 1, 'Target': 2}
                if bar_exit_type is None or exit_type_priority.get(pos_exit_type, 99) < exit_type_priority.get(bar_exit_type, 99):
                    bar_exit_type = pos_exit_type

                # If all groups closed, mark position for removal
                if not pos['active_exit_groups']:
                    positions_to_remove.append(pos_idx)

        # Remove fully closed positions (reverse order to preserve indices)
        for pos_idx in sorted(positions_to_remove, reverse=True):
            open_positions.pop(pos_idx)

        # ----- STEP 2: Check entry -----
        can_enter = max_positions is None or len(open_positions) < max_positions

        if can_enter:
            bar_time = df.index[i]
            in_date_range = True
            if period_start is not None and bar_time < period_start:
                in_date_range = False
            if period_end is not None and bar_time > period_end:
                in_date_range = False

            entry_triggered = in_date_range and check_trigger_and_conditions(entry_trigger, entry_conditions, i)

            if entry_triggered:
                stops_violated = are_any_stops_already_violated(
                    initial_stop_config, exit_groups, i, strategy_direction
                )
                if stops_violated:
                    entry_triggered = False

            if entry_triggered:
                bar_entry = True
                trade_counter += 1
                new_entry_price = get_trigger_price(entry_trigger, i)
                new_locked_stop = get_static_stop_price(initial_stop_config, i, entry_price=new_entry_price)

                if new_locked_stop is not None:
                    new_r_distance = abs(new_entry_price - new_locked_stop)
                else:
                    new_r_distance = 0.0

                open_positions.append({
                    'trade_id': trade_counter,
                    'entry_price': new_entry_price,
                    'r_distance': new_r_distance,
                    'locked_stop_value': new_locked_stop,
                    'entry_r': entry_r,
                    'active_exit_groups': set(range(len(exit_groups))),
                    'entry_bar_idx': i,
                })

        # ----- STEP 3: Record bar signals -----
        entry_signal.append(bar_entry)
        exit_signal.append(bar_exit)
        exit_type_series.append(bar_exit_type)

    # -------------------------------------------------
    # Handle open positions at the end (close all remaining)
    # -------------------------------------------------
    last_price = df["latest"].iloc[-1] if len(df) > 0 else 0.0
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
    # Attach signals
    # -------------------------------------------------
    df["entry_signal"] = entry_signal
    df["exit_signal"] = exit_signal
    df["exit_type"] = exit_type_series

    # -------------------------------------------------
    # Statistics with R-based P&L calculation
    # Consolidate exit group records into one trade per entry
    # -------------------------------------------------

    # Step 1: Calculate P&L in R for each group exit record
    for trade in all_trades:
        entry_price = trade['entry_price']
        exit_price = trade['exit_price']
        alloc_pct = trade['allocation_pct']
        r_distance = trade['r_distance']
        t_entry_r = trade['entry_r']

        if strategy_direction == 'Long':
            price_move = exit_price - entry_price
        else:
            price_move = entry_price - exit_price

        if r_distance > 0:
            group_r = t_entry_r * (alloc_pct / 100.0)
            trade['pnl_r'] = (price_move / r_distance) * group_r
        else:
            trade['pnl_r'] = 0.0

    # Step 2: Group by trade_id to consolidate exit groups into one trade per entry
    # Exit type priority: Initial Stop > Stop > End of Data > Target
    exit_type_priority = {'Initial Stop': 0, 'Stop': 1, 'End of Data': 2, 'Target': 3}

    from collections import OrderedDict
    entry_trades = OrderedDict()
    for trade in all_trades:
        tid = trade['trade_id']
        if tid not in entry_trades:
            entry_trades[tid] = {
                'pnl_r': 0.0,
                'exit_type': trade['exit_type'],
            }
        entry = entry_trades[tid]
        entry['pnl_r'] += trade['pnl_r']
        # Keep the highest-priority (lowest number) exit type
        if exit_type_priority.get(trade['exit_type'], 99) < exit_type_priority.get(entry['exit_type'], 99):
            entry['exit_type'] = trade['exit_type']

    # Step 3: Build per-trade lists from consolidated entries
    trade_pnls_r = [t['pnl_r'] for t in entry_trades.values()]
    consolidated_exit_types = [t['exit_type'] for t in entry_trades.values()]

    num_trades = len(entry_trades)

    if num_trades > 0:
        wins = sum(pnl > 0 for pnl in trade_pnls_r)
        losses = num_trades - wins

        win_rate = wins / num_trades * 100
        loss_rate = losses / num_trades * 100

        total_pnl = sum(trade_pnls_r)
        avg_pnl_per_trade = total_pnl / num_trades
        winning_trades_pnl = sum(pnl for pnl in trade_pnls_r if pnl > 0)
        losing_trades_pnl = sum(pnl for pnl in trade_pnls_r if pnl < 0)

        target_exits = sum(1 for t in consolidated_exit_types if t == 'Target')
        stop_exits = sum(1 for t in consolidated_exit_types if t in ('Stop', 'Initial Stop'))
        target_exit_pct = target_exits / num_trades * 100
        stop_exit_pct = stop_exits / num_trades * 100

        # Sharpe Ratio: mean(R P&Ls) / stdev(R P&Ls), risk-free rate = 0
        if num_trades >= 2:
            pnl_std = np.std(trade_pnls_r, ddof=1)
            sharpe_ratio = (np.mean(trade_pnls_r) / pnl_std) if pnl_std > 0 else 0.0
        else:
            sharpe_ratio = 0.0

        # Maximum Drawdown: largest peak-to-trough decline in cumulative R equity curve
        cumulative = np.cumsum(trade_pnls_r)
        peak = np.maximum.accumulate(cumulative)
        drawdowns = cumulative - peak  # always <= 0
        max_drawdown = abs(drawdowns.min()) if len(drawdowns) > 0 else 0.0

        # MAR Ratio: Total P&L (R) / Max Drawdown (R)
        mar_ratio = (total_pnl / max_drawdown) if max_drawdown > 0 else 0.0

        # SQN (System Quality Number): sqrt(N) * mean(R P&Ls) / stdev(R P&Ls)
        if num_trades >= 2 and pnl_std > 0:
            sqn = (np.mean(trade_pnls_r) / pnl_std) * np.sqrt(num_trades)
        else:
            sqn = 0.0
    else:
        win_rate = 0.0
        loss_rate = 0.0
        total_pnl = 0.0
        avg_pnl_per_trade = 0.0
        winning_trades_pnl = 0.0
        losing_trades_pnl = 0.0
        target_exit_pct = 0.0
        stop_exit_pct = 0.0
        sharpe_ratio = 0.0
        max_drawdown = 0.0
        mar_ratio = 0.0
        sqn = 0.0

    stats_df = pd.DataFrame(
        {
            "value": [
                num_trades,
                win_rate,
                loss_rate,
                0.0,  # Total return (%) — deprecated, use Total P&L (R)
                total_pnl,
                avg_pnl_per_trade,
                winning_trades_pnl,
                losing_trades_pnl,
                target_exit_pct,
                stop_exit_pct,
                sharpe_ratio,
                max_drawdown,
                mar_ratio,
                sqn,
            ]
        },
        index=[
            "Number of trades",
            "Win rate (%)",
            "Loss rate (%)",
            "Total return (%)",
            "Total P&L (R)",
            "Avg P&L per trade (R)",
            "Winning trades P&L (R)",
            "Losing trades P&L (R)",
            "Target exit (%)",
            "Stop exit (%)",
            "Sharpe Ratio",
            "Max Drawdown (R)",
            "MAR Ratio",
            "SQN",
        ],
    )

    # Store individual trade R P&Ls for aggregation across periods
    stats_df.attrs['trade_pnls_r'] = list(trade_pnls_r)

    return df, stats_df
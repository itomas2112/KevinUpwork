# win %, loss %, number of trades, return

import pandas as pd
import numpy as np
from config.constants import get_indicator_map

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
    def check_r_trigger(trigger_config, current_idx):
        """
        Check if an R Profit or R Loss trigger event occurred.
        R Profit = unrealized profit in R terms (positive when profitable).
        R Loss = unrealized loss in R terms (positive when losing).
        Only valid while in a position.
        """
        if current_entry_price is None or current_r_distance <= 0:
            return False

        element1_name = trigger_config.get('element1')
        event = trigger_config.get('event')
        fixed_value = trigger_config.get('value')

        if fixed_value is None:
            return False

        price = df['latest'].iloc[current_idx]

        # Calculate directional price move
        if strategy_direction == 'Long':
            price_move = price - current_entry_price
        else:
            price_move = current_entry_price - price

        # R Profit: positive when in profit, negative when in loss
        # R Loss: positive when in loss, negative when in profit
        if element1_name == "R Profit":
            r_value = price_move / current_r_distance
        else:  # R Loss
            r_value = -price_move / current_r_distance

        # Need previous bar for cross detection
        if current_idx > 0:
            prev_price = df['latest'].iloc[current_idx - 1]
            if strategy_direction == 'Long':
                prev_move = prev_price - current_entry_price
            else:
                prev_move = current_entry_price - prev_price

            if element1_name == "R Profit":
                prev_r_value = prev_move / current_r_distance
            else:
                prev_r_value = -prev_move / current_r_distance
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

    def get_r_trigger_price(current_idx):
        """For R Profit/R Loss triggers, exit at the bar's close price."""
        return df['latest'].iloc[current_idx]

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
    def get_static_stop_price(stop_config, current_idx):
        """
        Return the indicator value used by the initial (static) stop at a given bar.
        The static stop is always Price × Indicator, so we return the indicator's value.
        """
        if not stop_config:
            return None

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
    def check_exit_signal(exit_config, current_idx):
        """Check if an exit signal (target or stop) is triggered"""
        trigger = exit_config.get('trigger', {})
        conditions = exit_config.get('conditions', [])

        # R Profit / R Loss triggers use special handler
        element1 = trigger.get('element1', '')
        if element1 in ("R Profit", "R Loss"):
            if not check_r_trigger(trigger, current_idx):
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

    # -------------------------------------------------
    # Signal generation with multiple exits (OCO logic)
    # -------------------------------------------------
    in_position = False
    entry_signal = []
    exit_signal = []
    exit_type_series = []  # Track exit type per bar: None, "Target", "Stop", "Initial Stop"

    # Track which exit groups are still active (not yet closed)
    active_exit_groups = set()

    # Trade tracking for P&L (R-based)
    all_trades = []  # List of dicts: {entry_price, exit_price, allocation_pct, r_distance, entry_r, exit_type}
    current_entry_price = None
    current_r_distance = 0.0  # abs(entry_price - static_stop_value)
    locked_stop_value = None  # Static stop price locked at entry time

    for i in range(len(df)):
        price = df["latest"].iloc[i]

        # Check entry conditions (only if no position)
        if not in_position:
            # Block entries outside the DRM date range
            bar_time = df.index[i]
            in_date_range = True
            if period_start is not None and bar_time < period_start:
                in_date_range = False
            if period_end is not None and bar_time > period_end:
                in_date_range = False

            # First check if entry trigger and conditions are met
            entry_triggered = in_date_range and check_trigger_and_conditions(entry_trigger, entry_conditions, i)

            # If entry would trigger, also check that stops are not already violated
            if entry_triggered:
                stops_violated = are_any_stops_already_violated(
                    initial_stop_config, exit_groups, i, strategy_direction
                )
                if stops_violated:
                    # Block entry - stop condition already violated
                    entry_triggered = False

            if entry_triggered:
                # ---- Entry
                entry_signal.append(True)
                exit_signal.append(False)
                exit_type_series.append(None)

                in_position = True
                current_entry_price = get_trigger_price(entry_trigger, i)

                # Lock the static stop price at entry time
                locked_stop_value = get_static_stop_price(initial_stop_config, i)

                # Calculate R-distance from static stop
                if locked_stop_value is not None:
                    current_r_distance = abs(current_entry_price - locked_stop_value)
                else:
                    current_r_distance = 0.0

                # Activate all exit groups
                active_exit_groups = set(range(len(exit_groups)))
            else:
                entry_signal.append(False)
                exit_signal.append(False)
                exit_type_series.append(None)

        # Check exit conditions (only if in position)
        elif in_position:
            exit_triggered = False
            bar_exit_type = None  # Track the exit type for THIS bar

            # First, check initial stop (applies to ALL active groups)
            # Uses locked_stop_value so the stop level stays fixed from entry time
            if initial_stop_config and check_trigger(initial_stop_config, i, locked_value=locked_stop_value):
                # Initial stop closes ALL remaining position
                bar_exit_type = "Initial Stop"
                exit_triggered = True
                stop_exit_price = get_trigger_price(initial_stop_config, i, locked_value=locked_stop_value)

                # Record trade for all active groups
                for group_idx in active_exit_groups:
                    alloc_pct = exit_groups[group_idx].get('allocation_pct', 100.0)
                    all_trades.append({
                        'entry_price': current_entry_price,
                        'exit_price': stop_exit_price,
                        'allocation_pct': alloc_pct,
                        'r_distance': current_r_distance,
                        'entry_r': entry_r,
                        'exit_type': 'Initial Stop'
                    })

                # Close all groups
                active_exit_groups.clear()

            # If initial stop didn't trigger, check each active exit group
            if not exit_triggered:
                groups_to_close = []

                for group_idx in list(active_exit_groups):
                    group = exit_groups[group_idx]
                    alloc_pct = group.get('allocation_pct', 100.0)

                    # Check all targets in this group
                    for target in group.get('targets', []):
                        if check_exit_signal(target, i):
                            exit_triggered = True

                            # Mark bar exit type — Target, unless a stop also triggers
                            if bar_exit_type is None:
                                bar_exit_type = "Target"

                            # Record trade
                            target_trigger = target.get('trigger', {})
                            t_el1 = target_trigger.get('element1', '')
                            t_exit_price = (get_r_trigger_price(i)
                                            if t_el1 in ("R Profit", "R Loss")
                                            else get_trigger_price(target_trigger, i))
                            all_trades.append({
                                'entry_price': current_entry_price,
                                'exit_price': t_exit_price,
                                'allocation_pct': alloc_pct,
                                'r_distance': current_r_distance,
                                'entry_r': entry_r,
                                'exit_type': 'Target'
                            })

                            # OCO: This group is closed, cancel all other targets/stops in group
                            groups_to_close.append(group_idx)
                            break  # Exit target loop, this group is done

                    # If no target hit, check stops in this group
                    if group_idx not in groups_to_close:
                        for stop in group.get('stops', []):
                            if check_exit_signal(stop, i):
                                exit_triggered = True

                                # Stop takes priority over target for bar marker
                                bar_exit_type = "Stop"

                                # Record trade
                                stop_trigger = stop.get('trigger', {})
                                s_el1 = stop_trigger.get('element1', '')
                                s_exit_price = (get_r_trigger_price(i)
                                                if s_el1 in ("R Profit", "R Loss")
                                                else get_trigger_price(stop_trigger, i))
                                all_trades.append({
                                    'entry_price': current_entry_price,
                                    'exit_price': s_exit_price,
                                    'allocation_pct': alloc_pct,
                                    'r_distance': current_r_distance,
                                    'entry_r': entry_r,
                                    'exit_type': 'Stop'
                                })

                                # OCO: This group is closed
                                groups_to_close.append(group_idx)
                                break  # Exit stop loop, this group is done

                # Remove closed groups
                for group_idx in groups_to_close:
                    active_exit_groups.discard(group_idx)

            # Update signals
            if exit_triggered:
                entry_signal.append(False)
                exit_signal.append(True)
                exit_type_series.append(bar_exit_type)

                # If all groups closed, position is done
                if not active_exit_groups:
                    in_position = False
                    current_entry_price = None
                    locked_stop_value = None
            else:
                entry_signal.append(False)
                exit_signal.append(False)
                exit_type_series.append(None)

        else:
            entry_signal.append(False)
            exit_signal.append(False)
            exit_type_series.append(None)

    # -------------------------------------------------
    # Handle open position at the end (close remaining groups)
    # -------------------------------------------------
    if in_position and current_entry_price is not None:
        last_price = df["latest"].iloc[-1]

        # Close all remaining active groups at last price
        for group_idx in active_exit_groups:
            alloc_pct = exit_groups[group_idx].get('allocation_pct', 100.0)
            all_trades.append({
                'entry_price': current_entry_price,
                'exit_price': last_price,
                'allocation_pct': alloc_pct,
                'r_distance': current_r_distance,
                'entry_r': entry_r,
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
    # -------------------------------------------------
    num_trades = len(all_trades)

    # Calculate P&L in R for each trade
    trade_returns = []
    trade_pnls_r = []

    for trade in all_trades:
        entry_price = trade['entry_price']
        exit_price = trade['exit_price']
        alloc_pct = trade['allocation_pct']
        r_distance = trade['r_distance']
        t_entry_r = trade['entry_r']

        # Calculate return ratio
        if strategy_direction == 'Long':
            trade_return = exit_price / entry_price
            price_move = exit_price - entry_price
        else:  # Short
            trade_return = entry_price / exit_price
            price_move = entry_price - exit_price

        trade_returns.append(trade_return)

        # Calculate P&L in R
        # group_R = entry_R * (allocation_pct / 100)
        # pnl_R = (price_move / r_distance) * group_R
        if r_distance > 0:
            group_r = t_entry_r * (alloc_pct / 100.0)
            pnl_r = (price_move / r_distance) * group_r
        else:
            pnl_r = 0.0

        trade_pnls_r.append(pnl_r)

    if num_trades > 0:
        wins = sum(r > 1 for r in trade_returns)
        losses = num_trades - wins

        win_rate = wins / num_trades * 100
        loss_rate = losses / num_trades * 100

        total_return = 1.0
        for r in trade_returns:
            total_return *= r

        total_pnl = sum(trade_pnls_r)
        avg_pnl_per_trade = total_pnl / num_trades if num_trades > 0 else 0
        winning_trades_pnl = sum(pnl for pnl in trade_pnls_r if pnl > 0)
        losing_trades_pnl = sum(pnl for pnl in trade_pnls_r if pnl < 0)

        target_exits = sum(1 for t in all_trades if t['exit_type'] == 'Target')
        stop_exits = sum(1 for t in all_trades if t['exit_type'] in ('Stop', 'Initial Stop'))
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
        total_return = 1.0
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
                (total_return - 1) * 100,
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
# win %, loss %, number of trades, return

import pandas as pd
import numpy as np
from config.constants import INDICATOR_MAP

# Chikou Span is displaced 26 periods back on the chart.
# When used in strategy conditions/triggers, both elements must be
# evaluated at the displaced index to avoid forward-looking bias.
CHIKOU_DISPLACEMENT = 26


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
    indicator_map = INDICATOR_MAP

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

        # Determine col2 early for Chikou displacement check
        col2 = None
        if compare_type != "Fixed Value":
            col2 = indicator_map.get(element2_name)
            if col2 is None or col2 not in df.columns:
                return False

        # Chikou displacement: evaluate 26 bars back to avoid forward-looking bias
        effective_idx = current_idx
        if col1 == "chikou" or (col2 is not None and col2 == "chikou"):
            effective_idx = current_idx - CHIKOU_DISPLACEMENT
            if effective_idx < 0:
                return False

        series1 = df[col1]
        value1 = series1.iloc[effective_idx]

        # Determine what to compare against
        if compare_type == "Fixed Value":
            if fixed_value is None:
                return False
            value2 = fixed_value
        else:
            series2 = df[col2]
            value2 = series2.iloc[effective_idx]

        # Check the operator
        if operator == "Above":
            return value1 > value2
        elif operator == "Below":
            return value1 < value2

        return False

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

        # Determine col2 for Chikou displacement check
        col2 = None
        if locked_value is None and compare_type != "Fixed Value":
            col2 = indicator_map.get(element2_name)
            if col2 is None or col2 not in df.columns:
                return False

        # Chikou displacement: shift evaluation back 26 bars to avoid forward-looking bias
        # Not applied when locked_value is used (static stop with fixed price level)
        effective_idx = current_idx
        if locked_value is None and (col1 == "chikou" or (col2 is not None and col2 == "chikou")):
            effective_idx = current_idx - CHIKOU_DISPLACEMENT
            if effective_idx < 0:
                return False

        value1 = series1.iloc[effective_idx]

        # Determine what to compare against
        if locked_value is not None:
            # Static stop: use locked price from entry time
            value2 = locked_value

            if effective_idx > 0:
                value1_prev = series1.iloc[effective_idx - 1]
                value2_prev = locked_value
        elif compare_type == "Fixed Value":
            if fixed_value is None:
                return False

            value2 = fixed_value

            if effective_idx > 0:
                value1_prev = series1.iloc[effective_idx - 1]
                value2_prev = fixed_value
        else:
            # Compare against another indicator
            series2 = df[col2]
            value2 = series2.iloc[effective_idx]

            if effective_idx > 0:
                value1_prev = series1.iloc[effective_idx - 1]
                value2_prev = series2.iloc[effective_idx - 1]

        # Check the event type
        if event in ("Cross Above", "Close Above"):
            if effective_idx == 0:
                return False
            return (value1 > value2) and (value1_prev <= value2_prev)

        elif event in ("Cross Below", "Close Below"):
            if effective_idx == 0:
                return False
            return (value1 < value2) and (value1_prev >= value2_prev)

        elif event in ("Cross", "Close"):
            if effective_idx == 0:
                return False
            cross_above = (value1 > value2) and (value1_prev <= value2_prev)
            cross_below = (value1 < value2) and (value1_prev >= value2_prev)
            return cross_above or cross_below

        elif event == "At Level":
            return abs(value1 - value2) < 0.01

        return False

    def get_trigger_price(trigger_config, current_idx, locked_value=None):
        """
        Determine the trade price for a triggered event.
        - Cross events with Price as element1: use the crossed indicator's value
        - Close events or non-Price triggers: use bar close price
        - If locked_value is provided (static stop), use that as the crossed level
        - Chikou-involved triggers: use current bar close (displaced evaluation
          means there's no meaningful crossed level at the current bar)
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
                        return fixed_val
                else:
                    element2 = trigger_config.get('element2')
                    col2 = indicator_map.get(element2)
                    # Chikou: use current close (no meaningful crossed level)
                    if col2 == "chikou":
                        return df['latest'].iloc[current_idx]
                    if col2 and col2 in df.columns:
                        return df[col2].iloc[current_idx]

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
            # Chikou: read from displaced index to avoid forward-looking bias
            if col2 == "chikou":
                effective_idx = current_idx - CHIKOU_DISPLACEMENT
                if effective_idx < 0:
                    return None
                return df[col2].iloc[effective_idx]
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
        event = trigger_config.get('event')
        compare_type = trigger_config.get('compare_type', 'Indicator')
        element2_name = trigger_config.get('element2')
        fixed_value = trigger_config.get('value')

        # Get the series for element1
        col1 = indicator_map.get(element1_name)

        if col1 is None or col1 not in df.columns:
            return False

        # Determine col2 for Chikou displacement check
        col2 = None
        if compare_type != "Fixed Value":
            col2 = indicator_map.get(element2_name)
            if col2 is None or col2 not in df.columns:
                return False

        # Chikou displacement: evaluate 26 bars back
        effective_idx = current_idx
        if col1 == "chikou" or (col2 is not None and col2 == "chikou"):
            effective_idx = current_idx - CHIKOU_DISPLACEMENT
            if effective_idx < 0:
                return False

        series1 = df[col1]
        value1 = series1.iloc[effective_idx]

        # Determine what to compare against
        if compare_type == "Fixed Value":
            if fixed_value is None:
                return False
            value2 = fixed_value
        else:
            series2 = df[col2]
            value2 = series2.iloc[effective_idx]

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
                            all_trades.append({
                                'entry_price': current_entry_price,
                                'exit_price': get_trigger_price(target_trigger, i),
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
                                all_trades.append({
                                    'entry_price': current_entry_price,
                                    'exit_price': get_trigger_price(stop_trigger, i),
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
        ],
    )

    return df, stats_df
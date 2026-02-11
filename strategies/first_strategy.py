# win %, loss %, number of trades, return

import pandas as pd
import numpy as np
from config.constants import INDICATOR_MAP


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
                (total_return - 1)*100,
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


def execute_custom_strategy(df: pd.DataFrame, strategy_config: dict):
    """
    Execute a custom strategy based on the saved strategy configuration.
    Supports multiple exit groups with OCO (One-Cancels-Other) targets and stops.

    Parameters:
    -----------
    df : pd.DataFrame
        DataFrame with OHLC data and all calculated indicators
    strategy_config : dict
        The strategy configuration from saved_strategies

    Returns:
    --------
    df : pd.DataFrame
        DataFrame with entry_signal and exit_signal columns added
    stats_df : pd.DataFrame
        Statistics DataFrame with win rate, loss rate, number of trades, total return, P&L
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
            # Compare against another indicator
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
    # Helper function to check trigger events
    # -------------------------------------------------
    def check_trigger(trigger_config, current_idx):
        """Check if a trigger event occurred at given index"""
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
        value1 = series1.iloc[current_idx]

        # Determine what to compare against
        if compare_type == "Fixed Value":
            if fixed_value is None:
                return False

            value2 = fixed_value

            if current_idx > 0:
                value1_prev = series1.iloc[current_idx - 1]
                value2_prev = fixed_value
        else:
            # Compare against another indicator
            col2 = indicator_map.get(element2_name)

            if col2 is None or col2 not in df.columns:
                return False

            series2 = df[col2]
            value2 = series2.iloc[current_idx]

            if current_idx > 0:
                value1_prev = series1.iloc[current_idx - 1]
                value2_prev = series2.iloc[current_idx - 1]

        # Check the event type
        if event == "Cross Above":
            if current_idx == 0:
                return False
            return (value1 > value2) and (value1_prev <= value2_prev)

        elif event == "Cross Below":
            if current_idx == 0:
                return False
            return (value1 < value2) and (value1_prev >= value2_prev)

        elif event == "Cross":
            if current_idx == 0:
                return False
            cross_above = (value1 > value2) and (value1_prev <= value2_prev)
            cross_below = (value1 < value2) and (value1_prev >= value2_prev)
            return cross_above or cross_below

        elif event == "At Level":
            return abs(value1 - value2) < 0.01

        return False

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
    # Extract entry configuration
    # -------------------------------------------------
    entry_config = strategy_config.get('entry', {})
    entry_trigger = entry_config.get('trigger', {})
    entry_conditions = entry_config.get('conditions', [])
    entry_position_size = entry_config.get('position_size', 1.0)

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
                'position_size': entry_position_size,
                'targets': [{
                    'type': 'Target',
                    'trigger': exit_config.get('trigger', {}),
                    'conditions': exit_config.get('conditions', [])
                }],
                'stops': []
            }]

    # Get strategy direction
    strategy_direction = strategy_config.get('direction', 'Long')

    # -------------------------------------------------
    # Signal generation with multiple exits (OCO logic)
    # -------------------------------------------------
    current_position_size = 0
    entry_signal = []
    exit_signal = []

    # Track which exit groups are still active (not yet closed)
    active_exit_groups = set()

    # Trade tracking for P&L
    all_trades = []  # List of dicts: {entry_price, exit_price, position_size, exit_type}
    current_entry_price = None

    for i in range(len(df)):
        price = df["latest"].iloc[i]

        # Check entry conditions (only if no position)
        if current_position_size == 0:
            if check_trigger_and_conditions(entry_trigger, entry_conditions, i):
                # ---- Entry
                entry_signal.append(True)
                exit_signal.append(False)

                current_position_size = entry_position_size
                current_entry_price = price

                # Activate all exit groups
                active_exit_groups = set(range(len(exit_groups)))
            else:
                entry_signal.append(False)
                exit_signal.append(False)

        # Check exit conditions (only if in position)
        elif current_position_size > 0:
            exit_triggered = False
            exit_size = 0
            exit_type_label = None

            # First, check initial stop (applies to ALL active groups)
            if initial_stop_config and check_trigger(initial_stop_config, i):
                # Initial stop closes ALL remaining position
                exit_size = current_position_size
                exit_type_label = "Initial Stop"
                exit_triggered = True

                # Record trade for all active groups
                for group_idx in active_exit_groups:
                    group_size = exit_groups[group_idx].get('position_size', 0)
                    all_trades.append({
                        'entry_price': current_entry_price,
                        'exit_price': price,
                        'position_size': group_size,
                        'exit_type': 'Initial Stop'
                    })

                # Close all groups
                active_exit_groups.clear()

            # If initial stop didn't trigger, check each active exit group
            if not exit_triggered:
                groups_to_close = []

                for group_idx in list(active_exit_groups):
                    group = exit_groups[group_idx]
                    group_size = group.get('position_size', 0)

                    # Check all targets in this group
                    for target in group.get('targets', []):
                        if check_exit_signal(target, i):
                            exit_size += group_size
                            exit_triggered = True
                            exit_type_label = "Target"

                            # Record trade
                            all_trades.append({
                                'entry_price': current_entry_price,
                                'exit_price': price,
                                'position_size': group_size,
                                'exit_type': 'Target'
                            })

                            # OCO: This group is closed, cancel all other targets/stops in group
                            groups_to_close.append(group_idx)
                            break  # Exit target loop, this group is done

                    # If no target hit, check stops in this group
                    if group_idx not in groups_to_close:
                        for stop in group.get('stops', []):
                            if check_exit_signal(stop, i):
                                exit_size += group_size
                                exit_triggered = True
                                exit_type_label = "Stop"

                                # Record trade
                                all_trades.append({
                                    'entry_price': current_entry_price,
                                    'exit_price': price,
                                    'position_size': group_size,
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
                current_position_size -= exit_size

                # Ensure we don't go negative due to floating point errors
                if current_position_size < 0.001:
                    current_position_size = 0
                    current_entry_price = None
            else:
                entry_signal.append(False)
                exit_signal.append(False)

        else:
            entry_signal.append(False)
            exit_signal.append(False)

    # -------------------------------------------------
    # Handle open position at the end (close remaining groups)
    # -------------------------------------------------
    if current_position_size > 0 and current_entry_price is not None:
        last_price = df["latest"].iloc[-1]

        # Close all remaining active groups at last price
        for group_idx in active_exit_groups:
            group_size = exit_groups[group_idx].get('position_size', 0)
            all_trades.append({
                'entry_price': current_entry_price,
                'exit_price': last_price,
                'position_size': group_size,
                'exit_type': 'End of Data'
            })

    # -------------------------------------------------
    # Attach signals
    # -------------------------------------------------
    df["entry_signal"] = entry_signal
    df["exit_signal"] = exit_signal

    # -------------------------------------------------
    # Statistics with P&L calculation
    # -------------------------------------------------
    num_trades = len(all_trades)

    # Get market parameters
    tick_size = strategy_config.get('tick_size', 0.25)
    minimal_change = strategy_config.get('minimal_change', 1.0)

    # Calculate P&L for each trade
    trade_returns = []
    trade_pnls = []

    for trade in all_trades:
        entry_price = trade['entry_price']
        exit_price = trade['exit_price']
        position_size = trade['position_size']

        # Calculate return
        if strategy_direction == 'Long':
            trade_return = exit_price / entry_price
            price_change = exit_price - entry_price
        else:  # Short
            trade_return = entry_price / exit_price
            price_change = entry_price - exit_price

        trade_returns.append(trade_return)

        # Calculate P&L
        ticks_moved = price_change / minimal_change
        pnl = position_size * ticks_moved * tick_size
        trade_pnls.append(pnl)

    if num_trades > 0:
        wins = sum(r > 1 for r in trade_returns)
        losses = num_trades - wins

        win_rate = wins / num_trades * 100
        loss_rate = losses / num_trades * 100

        total_return = 1.0
        for r in trade_returns:
            total_return *= r

        total_pnl = sum(trade_pnls)
        avg_pnl_per_trade = total_pnl / num_trades if num_trades > 0 else 0
        winning_trades_pnl = sum(pnl for pnl in trade_pnls if pnl > 0)
        losing_trades_pnl = sum(pnl for pnl in trade_pnls if pnl < 0)
    else:
        win_rate = 0.0
        loss_rate = 0.0
        total_return = 1.0
        total_pnl = 0.0
        avg_pnl_per_trade = 0.0
        winning_trades_pnl = 0.0
        losing_trades_pnl = 0.0

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
            ]
        },
        index=[
            "Number of trades",
            "Win rate (%)",
            "Loss rate (%)",
            "Total return (%)",
            "Total P&L ($)",
            "Avg P&L per trade ($)",
            "Winning trades P&L ($)",
            "Losing trades P&L ($)",
        ],
    )

    return df, stats_df


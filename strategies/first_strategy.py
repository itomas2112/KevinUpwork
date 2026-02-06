# win %, loss %, number of trades, return

import pandas as pd
import numpy as np


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
        Statistics DataFrame with win rate, loss rate, number of trades, total return
    """

    df = df.copy()

    # -------------------------------------------------
    # Clean existing signals
    # -------------------------------------------------
    for col in ["entry_signal", "exit_signal"]:
        if col in df.columns:
            df.drop(columns=col, inplace=True)

    # -------------------------------------------------
    # Map indicator names to DataFrame columns
    # -------------------------------------------------
    indicator_map = {
        "Price": "latest",
        "BB Upper Band": "bb_upper",
        "BB Middle Band": "bb_mid",  # Changed from "bb_middle"
        "BB Lower Band": "bb_lower",
        "KC Upper Band": "kc_upper",
        "KC Middle Band": "kc_mid",  # Changed from "kc_middle"
        "KC Lower Band": "kc_lower",
        "Tenkan": "tenkan",
        "Kijun": "kijun",
        "Senkou A": "senkou_a",
        "Senkou B": "senkou_b",
        "RSI": "rsi",
        "RSI 13 SMA": "ci_13",  # Based on your columns, this maps to ci_13
        "RSI 33 SMA": "ci_33",  # Based on your columns, this maps to ci_33
        "CMB": "cmb",  # You'll need to add this column if you want to use CMB
        "CMB 13 SMA": "cmb_13_sma",  # You'll need to add this column if you want to use CMB
        "CMB 33 SMA": "cmb_33_sma",  # You'll need to add this column if you want to use CMB
    }

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
    # Extract entry and exit configurations
    # -------------------------------------------------
    entry_config = strategy_config.get('entry', {})
    exit_config = strategy_config.get('exit', {})

    entry_trigger = entry_config.get('trigger', {})
    entry_conditions = entry_config.get('conditions', [])

    exit_trigger = exit_config.get('trigger', {})
    exit_conditions = exit_config.get('conditions', [])

    # -------------------------------------------------
    # Signal generation (entry first, exit after)
    # -------------------------------------------------
    in_trade = False
    entry_signal = []
    exit_signal = []

    entry_prices = []
    trade_returns = []

    current_entry_price = None

    for i in range(len(df)):
        price = df["latest"].iloc[i]

        # Check entry conditions (only if not in trade)
        if not in_trade:
            if check_trigger_and_conditions(entry_trigger, entry_conditions, i):
                # ---- Entry
                entry_signal.append(True)
                exit_signal.append(False)

                in_trade = True
                current_entry_price = price
                entry_prices.append(price)
            else:
                entry_signal.append(False)
                exit_signal.append(False)

        # Check exit conditions (only if in trade)
        else:  # in_trade
            if check_trigger_and_conditions(exit_trigger, exit_conditions, i):
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


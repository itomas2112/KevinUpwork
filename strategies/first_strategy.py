# win %, loss %, number of trades, return

import copy
import pandas as pd
import numpy as np
from config.constants import get_indicator_map
from indicators.atr_indicator import atr_indicator
from strategies.strategy_validator import validate_strategy
from strategies.risk_validation import validate_risk_distance as _validate_risk


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

        else:
            # Rule 2: One side is Senkou A/B, other is not Senkou and not Chikou.
            # senkou_a / senkou_b columns are shifted forward 26 bars for cloud display.
            # For trading comparisons (e.g. Price vs Senkou A), use the unshifted
            # senkou_a_current / senkou_b_current columns to get the current bar's value.
            _senkou_remap = {
                'Senkou A': 'senkou_a_current',
                'Senkou B': 'senkou_b_current',
            }
            for key, elem in [(key1, e1), (key2, e2)]:
                if elem in _SENKOU_ELEMENTS:
                    raw_col = _senkou_remap[elem]
                    syn = f'_raw_{raw_col}'
                    if syn not in indicator_map:
                        indicator_map[syn] = raw_col
                    config[key] = syn

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


def _empty_stats_df():
    """Return an empty stats DataFrame with the standard structure."""
    stats_df = pd.DataFrame(
        {"value": [0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]},
        index=[
            "Number of trades", "Win rate (%)", "Loss rate (%)", "Total return (%)",
            "Total P&L (R)", "Avg P&L per trade (R)", "Winning trades P&L (R)",
            "Losing trades P&L (R)", "Target exit (%)", "Static exit (%)",
            "Dynamic exit (%)", "EOD exit (%)", "RR Ratio", "Max Drawdown (R)", "SQN",
        ],
    )
    stats_df.attrs['trade_pnls_r'] = []
    stats_df.attrs['total_static_alloc'] = 0.0
    stats_df.attrs['total_dynamic_alloc'] = 0.0
    stats_df.attrs['total_target_alloc'] = 0.0
    stats_df.attrs['total_eod_alloc'] = 0.0
    return stats_df


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
    # Validate strategy before execution
    # -------------------------------------------------
    ema_count = 0
    while f"ema_{ema_count}" in df.columns:
        ema_count += 1

    is_valid, errors = validate_strategy(strategy_config, ema_count=ema_count)
    if not is_valid:
        import warnings
        warnings.warn(f"Invalid strategy skipped: {'; '.join(errors)}")
        return df, _empty_stats_df()

    # -------------------------------------------------
    # Clean existing signals
    # -------------------------------------------------
    for col in ["entry_signal", "exit_signal"]:
        if col in df.columns:
            df.drop(columns=col, inplace=True)

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
        if event == "Cross Above":
            return (r_value > fixed_value) and (prev_r_value <= fixed_value)
        elif event == "Close Above":
            return r_value > fixed_value
        elif event == "Cross Below":
            return (r_value < fixed_value) and (prev_r_value >= fixed_value)
        elif event == "Close Below":
            return r_value < fixed_value
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
    # Helper: ATR Target trigger (fixed at entry, like static stop)
    # -------------------------------------------------
    def check_atr_target_trigger(trigger_config, current_idx, locked_atr_price):
        """
        Check if an ATR Target trigger event occurred.
        The locked_atr_price was computed at entry time and never changes.
        Uses high price for Long targets, low price for Short targets
        (same logic as R Profit triggers).
        """
        if locked_atr_price is None:
            return False

        event = trigger_config.get('event')
        # Default event when None: Long targets fire on Cross Above, Short on Cross Below
        if event is None:
            event = "Cross Above" if strategy_direction == 'Long' else "Cross Below"

        # Current bar: use high for Long (best-case profit), low for Short
        if strategy_direction == 'Long':
            price = df['high'].iloc[current_idx]
        else:
            price = df['low'].iloc[current_idx]

        # Previous bar: use close for cross detection
        if current_idx > 0:
            prev_price = df['latest'].iloc[current_idx - 1]
        else:
            return False

        if event == "Cross Above":
            return (price > locked_atr_price) and (prev_price <= locked_atr_price)
        elif event == "Close Above":
            return price > locked_atr_price
        elif event == "Cross Below":
            return (price < locked_atr_price) and (prev_price >= locked_atr_price)
        elif event == "Close Below":
            return price < locked_atr_price
        elif event == "Cross":
            cross_above = (price > locked_atr_price) and (prev_price <= locked_atr_price)
            cross_below = (price < locked_atr_price) and (prev_price >= locked_atr_price)
            return cross_above or cross_below
        elif event == "Close":
            close_above = (price > locked_atr_price) and (prev_price <= locked_atr_price)
            close_below = (price < locked_atr_price) and (prev_price >= locked_atr_price)
            return close_above or close_below

        return False

    def get_atr_target_price(locked_atr_price):
        """Return the locked ATR target price (exact level for exit price)."""
        return locked_atr_price

    # -------------------------------------------------
    # Helper: ATR Trailing Stop (Chandelier Exit)
    # Level is recomputed each bar:
    #   Long: max(prev_level, highest_high_since_entry - ATR(N) * mult)
    #   Short: min(prev_level, lowest_low_since_entry  + ATR(N) * mult)
    # The level only ratchets in the favourable direction.
    # -------------------------------------------------
    def init_trailing_state(trigger_config, bar_idx):
        """Compute initial trailing state on the entry bar."""
        atr_period = trigger_config.get('atr_period', 14)
        atr_series = _get_atr_target_series(atr_period)
        atr_val = atr_series.iloc[bar_idx]
        if strategy_direction == 'Long':
            extreme = df['high'].iloc[bar_idx]
        else:
            extreme = df['low'].iloc[bar_idx]
        if pd.isna(atr_val):
            return {'extreme': extreme, 'level': None, 'prev_level': None}
        atr_mult = trigger_config.get('atr_multiplier', 3.0)
        if strategy_direction == 'Long':
            level = extreme - atr_val * atr_mult
        else:
            level = extreme + atr_val * atr_mult
        return {'extreme': extreme, 'level': level, 'prev_level': level}

    def update_trailing_state(state, trigger_config, bar_idx):
        """Update trailing state for one ATR Trailing stop after a new bar."""
        if state is None:
            return state
        # Ratchet: remember the level we crossed against last bar
        state['prev_level'] = state['level']
        atr_period = trigger_config.get('atr_period', 14)
        atr_series = _get_atr_target_series(atr_period)
        atr_val = atr_series.iloc[bar_idx]
        if strategy_direction == 'Long':
            state['extreme'] = max(state['extreme'], df['high'].iloc[bar_idx])
        else:
            state['extreme'] = min(state['extreme'], df['low'].iloc[bar_idx])
        if pd.isna(atr_val):
            return state
        atr_mult = trigger_config.get('atr_multiplier', 3.0)
        if strategy_direction == 'Long':
            candidate = state['extreme'] - atr_val * atr_mult
            if state['level'] is None or candidate > state['level']:
                state['level'] = candidate
        else:
            candidate = state['extreme'] + atr_val * atr_mult
            if state['level'] is None or candidate < state['level']:
                state['level'] = candidate
        return state

    def check_atr_trailing_trigger(trigger_config, current_idx, state):
        """Check if an ATR Trailing stop fires this bar based on its event type.

        Default event when None: Long fires on Cross Below, Short on Cross Above.
        """
        if state is None or state.get('level') is None:
            return False

        event = trigger_config.get('event')
        if event is None:
            event = "Cross Below" if strategy_direction == 'Long' else "Cross Above"

        level = state['level']
        prev_level = state.get('prev_level', level)

        # Use intra-bar low/high for cross detection, close for "close" events.
        if strategy_direction == 'Long':
            low_p = df['low'].iloc[current_idx]
            close_p = df['latest'].iloc[current_idx]
            prev_close = df['latest'].iloc[current_idx - 1] if current_idx > 0 else close_p
            if event == "Cross Below":
                return (prev_close >= prev_level) and (low_p < level)
            elif event == "Close Below":
                return close_p < level
            elif event == "Cross":
                return (prev_close >= prev_level) and (low_p < level)
            elif event == "Close":
                return close_p < level
        else:  # Short
            high_p = df['high'].iloc[current_idx]
            close_p = df['latest'].iloc[current_idx]
            prev_close = df['latest'].iloc[current_idx - 1] if current_idx > 0 else close_p
            if event == "Cross Above":
                return (prev_close <= prev_level) and (high_p > level)
            elif event == "Close Above":
                return close_p > level
            elif event == "Cross":
                return (prev_close <= prev_level) and (high_p > level)
            elif event == "Close":
                return close_p > level
        return False

    def compute_atr_target_level(trigger_config, entry_price, bar_idx):
        """
        Compute the locked ATR target price at entry time.
        Long: entry + ATR × multiplier (target above)
        Short: entry - ATR × multiplier (target below)
        """
        atr_period = trigger_config.get('atr_period', 14)
        atr_mult = trigger_config.get('atr_multiplier', 2.0)
        atr_series = _get_atr_target_series(atr_period)
        atr_val = atr_series.iloc[bar_idx]
        if pd.isna(atr_val):
            return None
        if strategy_direction == 'Long':
            return entry_price + atr_val * atr_mult
        else:
            return entry_price - atr_val * atr_mult

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

        # Guard: if current values are NaN, no event can fire
        if pd.isna(value1) or pd.isna(value2):
            return False

        # Check the event type
        if event == "Close Above":
            # State: current bar's close is above the value
            return value1 > value2

        elif event == "Close Below":
            # State: current bar's close is below the value
            return value1 < value2

        elif event == "Close":
            # Close in either direction (previous bar on other side)
            if current_idx == 0:
                return False
            if pd.isna(value1_prev) or pd.isna(value2_prev):
                return False
            cross_above = (value1 > value2) and (value1_prev <= value2_prev)
            cross_below = (value1 < value2) and (value1_prev >= value2_prev)
            return cross_above or cross_below

        elif event == "Cross Above":
            # Standard cross: prev1 <= prev2 (both at t-1) AND now1 > now2 (both at t+0).
            # Price exception: use bar high at t+0 against element2 at t-1 (the line
            # that price actually crossed during this bar, not a t+0 reference that
            # may have moved).
            if current_idx == 0:
                return False
            if pd.isna(value1_prev) or pd.isna(value2_prev):
                return False
            if value1_prev > value2_prev:
                return False
            if col1 == "latest":
                high_val = df["high"].iloc[current_idx]
                return high_val > value2_prev
            else:
                return value1 > value2

        elif event == "Cross Below":
            if current_idx == 0:
                return False
            if pd.isna(value1_prev) or pd.isna(value2_prev):
                return False
            if value1_prev < value2_prev:
                return False
            if col1 == "latest":
                low_val = df["low"].iloc[current_idx]
                return low_val < value2_prev
            else:
                return value1 < value2

        elif event == "Cross":
            # Cross in either direction
            if current_idx == 0:
                return False
            if pd.isna(value1_prev) or pd.isna(value2_prev):
                return False
            if col1 == "latest":
                high_val = df["high"].iloc[current_idx]
                low_val = df["low"].iloc[current_idx]
                cross_above = (value1_prev <= value2_prev) and (high_val > value2_prev)
                cross_below = (value1_prev >= value2_prev) and (low_val < value2_prev)
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
        - Cross events with Price as element1: use the crossed indicator/fixed value
          as the fill price (the level that was actually crossed).
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
                        return fixed_val
                else:
                    element2 = trigger_config.get('element2')
                    col2 = indicator_map.get(element2)
                    if col2 and col2 in df.columns and current_idx > 0:
                        # Price crossed element2's t-1 value during bar t+0;
                        # use that as the fill level (matches the cross-detection logic).
                        return df[col2].iloc[current_idx - 1]

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
    def check_exit_signal(exit_config, current_idx, pos_entry_price=None, pos_r_distance=0.0,
                          locked_atr_targets=None, exit_key=None,
                          trailing_states=None):
        """Check if an exit signal (target or stop) is triggered"""
        trigger = exit_config.get('trigger', {})
        conditions = exit_config.get('conditions', [])

        element1 = trigger.get('element1', '')

        # R Profit / R Loss triggers use special handler
        if element1 in ("R Profit", "R Loss"):
            if not check_r_trigger(trigger, current_idx, pos_entry_price, pos_r_distance):
                return False
            # Still check conditions
            for condition in conditions:
                if not check_condition(condition, current_idx):
                    return False
            return True

        # ATR Target triggers use locked price from entry
        if element1 == "ATR Target" and locked_atr_targets is not None:
            # Look up the locked price for this specific exit config
            locked_price = locked_atr_targets.get(exit_key)
            if not check_atr_target_trigger(trigger, current_idx, locked_price):
                return False
            for condition in conditions:
                if not check_condition(condition, current_idx):
                    return False
            return True

        # ATR Trailing (Chandelier) — uses per-position trailing state
        if element1 == "ATR Trailing" and trailing_states is not None:
            state = trailing_states.get(exit_key)
            if not check_atr_trailing_trigger(trigger, current_idx, state):
                return False
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

        # ATR Target can't be evaluated before entry (level depends on entry price)
        if element1_name == "ATR Target":
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

    # Pre-compute ATR series for any ATR Target exits (keyed by period to avoid duplicates)
    atr_target_series_cache = {}  # {period: pd.Series}

    def _get_atr_target_series(period):
        """Get or compute ATR series for a given period (cached)."""
        if period not in atr_target_series_cache:
            atr_target_series_cache[period] = atr_indicator(
                df['high'], df['low'], df['latest'], period=period
            )
        return atr_target_series_cache[period]

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
    # Each: {trade_id, entry_price, r_distance, locked_stop_value, entry_r, active_exit_groups, entry_bar_idx, locked_atr_targets}
    open_positions = []

    for i in range(len(df)):
        bar_entry = False
        bar_exit = False
        bar_exit_type = None

        # ----- STEP 0: Update ATR Trailing state for every open position -----
        # (must happen before the exit check so the level is current for this bar)
        for pos in open_positions:
            if pos['entry_bar_idx'] == i:
                continue  # entry bar already initialised below
            trailing = pos.get('trailing_states')
            if not trailing:
                continue
            for stop_key, state in trailing.items():
                _, _, s_idx = stop_key
                g_idx = stop_key[0]
                stop_cfg = exit_groups[g_idx]['stops'][s_idx]
                update_trailing_state(state, stop_cfg.get('trigger', {}), i)

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
                        'exit_type': 'Initial Stop',
                        'entry_bar_idx': pos['entry_bar_idx'],
                        'exit_bar_idx': i,
                    })
                pos['active_exit_groups'] = set()

            # If initial stop didn't trigger, check each active exit group
            if not pos_exit_triggered:
                groups_to_close = []

                for group_idx in list(pos['active_exit_groups']):
                    group = exit_groups[group_idx]
                    alloc_pct = group.get('allocation_pct', 100.0)

                    # Check targets
                    for t_idx, target in enumerate(group.get('targets', [])):
                        target_key = (group_idx, 'target', t_idx)
                        if check_exit_signal(target, i, pos['entry_price'], pos['r_distance'],
                                             locked_atr_targets=pos.get('locked_atr_targets'),
                                             exit_key=target_key):
                            pos_exit_triggered = True
                            if pos_exit_type is None:
                                pos_exit_type = "Target"

                            target_trigger = target.get('trigger', {})
                            t_el1 = target_trigger.get('element1', '')
                            if t_el1 in ("R Profit", "R Loss"):
                                t_exit_price = get_r_trigger_price(target_trigger, i, pos['entry_price'], pos['r_distance'])
                            elif t_el1 == "ATR Target":
                                t_exit_price = get_atr_target_price(pos.get('locked_atr_targets', {}).get(target_key))
                            else:
                                t_exit_price = get_trigger_price(target_trigger, i)
                            all_trades.append({
                                'trade_id': pos['trade_id'],
                                'entry_price': pos['entry_price'],
                                'exit_price': t_exit_price,
                                'allocation_pct': alloc_pct,
                                'r_distance': pos['r_distance'],
                                'entry_r': pos['entry_r'],
                                'exit_type': 'Target',
                                'entry_bar_idx': pos['entry_bar_idx'],
                                'exit_bar_idx': i,
                            })
                            groups_to_close.append(group_idx)
                            break

                    # If no target hit, check stops
                    if group_idx not in groups_to_close:
                        for s_idx, stop in enumerate(group.get('stops', [])):
                            stop_key = (group_idx, 'stop', s_idx)
                            if check_exit_signal(stop, i, pos['entry_price'], pos['r_distance'],
                                                 locked_atr_targets=pos.get('locked_atr_targets'),
                                                 exit_key=stop_key,
                                                 trailing_states=pos.get('trailing_states')):
                                pos_exit_triggered = True
                                pos_exit_type = "Stop"

                                stop_trigger = stop.get('trigger', {})
                                s_el1 = stop_trigger.get('element1', '')
                                if s_el1 in ("R Profit", "R Loss"):
                                    s_exit_price = get_r_trigger_price(stop_trigger, i, pos['entry_price'], pos['r_distance'])
                                elif s_el1 == "ATR Target":
                                    s_exit_price = get_atr_target_price(pos.get('locked_atr_targets', {}).get(stop_key))
                                elif s_el1 == "ATR Trailing":
                                    trailing = pos.get('trailing_states', {}).get(stop_key, {})
                                    s_exit_price = trailing.get('level', get_trigger_price(stop_trigger, i))
                                else:
                                    s_exit_price = get_trigger_price(stop_trigger, i)
                                all_trades.append({
                                    'trade_id': pos['trade_id'],
                                    'entry_price': pos['entry_price'],
                                    'exit_price': s_exit_price,
                                    'allocation_pct': alloc_pct,
                                    'r_distance': pos['r_distance'],
                                    'entry_r': pos['entry_r'],
                                    'exit_type': 'Stop',
                                    'entry_bar_idx': pos['entry_bar_idx'],
                                    'exit_bar_idx': i,
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
        any_position_closed = len(positions_to_remove) > 0
        for pos_idx in sorted(positions_to_remove, reverse=True):
            open_positions.pop(pos_idx)

        # ----- STEP 2: Check entry -----
        # Block entry on any bar where a position just fully closed
        # (matches original if/elif behavior: exit and entry can't happen on the same bar)
        can_enter = (not any_position_closed
                     and (max_positions is None or len(open_positions) < max_positions))

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

                # Risk validation: reject entries with degenerate stop distance
                _atr_at_entry = df['atr'].iloc[i] if 'atr' in df.columns else None
                _rv_valid, _rv_reason = _validate_risk(
                    new_entry_price, new_locked_stop, atr=_atr_at_entry)
                if not _rv_valid:
                    trade_counter -= 1
                    bar_entry = False
                else:
                    # Compute locked ATR target prices for any ATR Target exits
                    # and initialise trailing state for any ATR Trailing stops.
                    locked_atr_targets = {}
                    trailing_states = {}
                    for g_idx, group in enumerate(exit_groups):
                        for t_idx, target in enumerate(group.get('targets', [])):
                            t_trigger = target.get('trigger', {})
                            if t_trigger.get('element1') == 'ATR Target':
                                level = compute_atr_target_level(t_trigger, new_entry_price, i)
                                locked_atr_targets[(g_idx, 'target', t_idx)] = level
                        for s_idx, stop in enumerate(group.get('stops', [])):
                            s_trigger = stop.get('trigger', {})
                            if s_trigger.get('element1') == 'ATR Target':
                                level = compute_atr_target_level(s_trigger, new_entry_price, i)
                                locked_atr_targets[(g_idx, 'stop', s_idx)] = level
                            elif s_trigger.get('element1') == 'ATR Trailing':
                                trailing_states[(g_idx, 'stop', s_idx)] = init_trailing_state(s_trigger, i)

                    open_positions.append({
                        'trade_id': trade_counter,
                        'entry_price': new_entry_price,
                        'r_distance': new_r_distance,
                        'locked_stop_value': new_locked_stop,
                        'entry_r': entry_r,
                        'active_exit_groups': set(range(len(exit_groups))),
                        'entry_bar_idx': i,
                        'locked_atr_targets': locked_atr_targets,
                        'trailing_states': trailing_states,
                    })

        # ----- STEP 3: Record bar signals -----
        entry_signal.append(bar_entry)
        exit_signal.append(bar_exit)
        exit_type_series.append(bar_exit_type)

    # -------------------------------------------------
    # Handle open positions at the end (close all remaining)
    # -------------------------------------------------
    last_price = df["latest"].iloc[-1] if len(df) > 0 else 0.0
    last_bar_idx = len(df) - 1 if len(df) > 0 else 0
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
                'exit_type': 'End of Data',
                'entry_bar_idx': pos['entry_bar_idx'],
                'exit_bar_idx': last_bar_idx,
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
                'static_alloc': 0.0,
                'dynamic_alloc': 0.0,
                'target_alloc': 0.0,
                'eod_alloc': 0.0,
                'entry_bar_idx': trade.get('entry_bar_idx', 0),
                'exit_bar_idx': trade.get('exit_bar_idx', trade.get('entry_bar_idx', 0)),
            }
        entry = entry_trades[tid]
        entry['pnl_r'] += trade['pnl_r']
        # Track allocation-weighted exit types
        alloc = trade.get('allocation_pct', 100.0)
        if trade['exit_type'] == 'Initial Stop':
            entry['static_alloc'] += alloc
        elif trade['exit_type'] == 'Stop':
            entry['dynamic_alloc'] += alloc
        elif trade['exit_type'] == 'Target':
            entry['target_alloc'] += alloc
        elif trade['exit_type'] == 'End of Data':
            entry['eod_alloc'] += alloc
        # Keep the highest-priority (lowest number) exit type
        if exit_type_priority.get(trade['exit_type'], 99) < exit_type_priority.get(entry['exit_type'], 99):
            entry['exit_type'] = trade['exit_type']
        # Holding period spans entry to last partial exit
        t_exit_idx = trade.get('exit_bar_idx')
        if t_exit_idx is not None and t_exit_idx > entry['exit_bar_idx']:
            entry['exit_bar_idx'] = t_exit_idx

    # Step 3: Build per-trade lists from consolidated entries
    trade_pnls_r = [t['pnl_r'] for t in entry_trades.values()]
    consolidated_exit_types = [t['exit_type'] for t in entry_trades.values()]
    trade_holding_periods = [max(0, t['exit_bar_idx'] - t['entry_bar_idx']) for t in entry_trades.values()]

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

        # Allocation-weighted exit type percentages
        target_exit_pct = sum(t['target_alloc'] for t in entry_trades.values()) / num_trades
        static_exit_pct = sum(t['static_alloc'] for t in entry_trades.values()) / num_trades
        dynamic_exit_pct = sum(t['dynamic_alloc'] for t in entry_trades.values()) / num_trades
        eod_exit_pct = sum(t['eod_alloc'] for t in entry_trades.values()) / num_trades

        avg_win_pnl = winning_trades_pnl / wins if wins > 0 else 0.0
        avg_lose_pnl = losing_trades_pnl / losses if losses > 0 else 0.0
        rr_ratio = abs(avg_win_pnl / avg_lose_pnl) if avg_lose_pnl != 0 else 0.0

        # Maximum Drawdown: largest peak-to-trough decline in cumulative R equity curve
        cumulative = np.cumsum(trade_pnls_r)
        peak = np.maximum.accumulate(cumulative)
        drawdowns = cumulative - peak  # always <= 0
        max_drawdown = abs(drawdowns.min()) if len(drawdowns) > 0 else 0.0

        # SQN (System Quality Number): sqrt(N) * mean(R P&Ls) / stdev(R P&Ls)
        if num_trades >= 2:
            pnl_std = np.std(trade_pnls_r, ddof=1)
            sqn = (np.mean(trade_pnls_r) / pnl_std) * np.sqrt(num_trades) if pnl_std > 0 else 0.0
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
        static_exit_pct = 0.0
        dynamic_exit_pct = 0.0
        eod_exit_pct = 0.0
        rr_ratio = 0.0
        max_drawdown = 0.0
        sqn = 0.0

    stats_df = pd.DataFrame(
        {
            "value": [
                num_trades,
                win_rate,
                loss_rate,
                0.0,  # Total return (%) -- deprecated, use Total P&L (R)
                total_pnl,
                avg_pnl_per_trade,
                winning_trades_pnl,
                losing_trades_pnl,
                target_exit_pct,
                static_exit_pct,
                dynamic_exit_pct,
                eod_exit_pct,
                rr_ratio,
                max_drawdown,
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
            "Static exit (%)",
            "Dynamic exit (%)",
            "EOD exit (%)",
            "RR Ratio",
            "Max Drawdown (R)",
            "SQN",
        ],
    )

    # Store individual trade R P&Ls and allocation totals for aggregation across periods
    stats_df.attrs['trade_pnls_r'] = list(trade_pnls_r)
    stats_df.attrs['trade_holding_periods'] = list(trade_holding_periods)
    stats_df.attrs['total_static_alloc'] = sum(t['static_alloc'] for t in entry_trades.values()) if num_trades > 0 else 0.0
    stats_df.attrs['total_dynamic_alloc'] = sum(t['dynamic_alloc'] for t in entry_trades.values()) if num_trades > 0 else 0.0
    stats_df.attrs['total_target_alloc'] = sum(t['target_alloc'] for t in entry_trades.values()) if num_trades > 0 else 0.0
    stats_df.attrs['total_eod_alloc'] = sum(t['eod_alloc'] for t in entry_trades.values()) if num_trades > 0 else 0.0

    return df, stats_df
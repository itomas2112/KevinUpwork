"""
Risk distance validation for trade entry.

Determines whether the stop distance for a candidate trade is economically
tradable.  Called by both pandas and numpy engines before admitting a trade.

Policy is separated from engine: this module decides whether a stop is
tradable; the engine computes raw stop distances and calls this to filter.
"""
import numpy as np


# Default policy thresholds (can be overridden via strategy config)
DEFAULT_K_ATR = 0.25        # min stop = 0.25 × ATR
DEFAULT_K_PRICE = 1e-4      # min stop = 0.01% of entry price
DEFAULT_N_TICKS = 0         # disabled by default (no tick_size info yet)


def validate_risk_distance(entry_price, stop_price, atr=None,
                           tick_size=None,
                           k_atr=DEFAULT_K_ATR,
                           k_price=DEFAULT_K_PRICE,
                           n_ticks=DEFAULT_N_TICKS):
    """Check whether the stop distance is economically tradable.

    Returns (is_valid, reason) where:
      is_valid: True if the trade should be admitted
      reason:   None if valid, otherwise one of:
                'no_stop'          — stop_price is None or NaN
                'too_close_vs_atr' — r_distance < k_atr * ATR
                'too_close_vs_price' — r_distance < k_price * entry_price
                'too_close_vs_tick'  — r_distance < n_ticks * tick_size
                'zero_distance'    — stop equals entry exactly
    """
    # No stop at all
    if stop_price is None or not np.isfinite(stop_price):
        return False, 'no_stop'

    if not np.isfinite(entry_price) or entry_price <= 0:
        return False, 'no_stop'

    r_distance = abs(entry_price - stop_price)

    # Exact zero
    if r_distance == 0:
        return False, 'zero_distance'

    # ATR floor (primary guard — instrument-adaptive)
    if atr is not None and np.isfinite(atr) and atr > 0:
        if r_distance < k_atr * atr:
            return False, 'too_close_vs_atr'

    # Price floor (fallback — always available)
    if r_distance < k_price * entry_price:
        return False, 'too_close_vs_price'

    # Tick floor (future: when tick_size is available per instrument)
    if tick_size is not None and tick_size > 0 and n_ticks > 0:
        if r_distance < n_ticks * tick_size:
            return False, 'too_close_vs_tick'

    return True, None

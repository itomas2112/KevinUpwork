# data/loader.py
import pandas as pd
from datetime import datetime

def load_ohlc(file):
    if not file.name.lower().endswith(".csv"):
        raise ValueError("Invalid file format. Please upload a CSV file.")

    df = pd.read_csv(file)

    df.columns = [c.lower().strip() for c in df.columns]
    df["time"] = pd.to_datetime(df["time"])
    df = df.dropna(subset=['high'])

    return df.set_index("time").sort_index()


def resample_ohlc(df, timeframe, base_timeframe="15m"):
    """Resample OHLC data to a higher timeframe.

    Args:
        df: DataFrame with OHLC data indexed by datetime (from load_ohlc)
        timeframe: "15m", "1H", "4H", "1D", "1W", or "1M"
        base_timeframe: the native timeframe of the uploaded data

    Returns:
        Resampled DataFrame in the same format. If timeframe matches base, returns a copy unchanged.
    """
    if timeframe == base_timeframe:
        return df.copy()

    # "MS" rather than "ME" for the month: pandas 3 dropped the bare "M" alias
    # outright, and "ME" bins on month *ends*, so with the left-labelling below
    # a bar would run from the 31st of one month to the 31st of the next and
    # carry the wrong month's name. "MS" bins on month starts, which is the
    # calendar month. "W" is Sunday-anchored, which is where the trading week
    # opens, so a Sunday-evening bar starts its week instead of closing the
    # previous one.
    tf_map = {"1H": "1h", "4H": "4h", "1D": "1D", "1W": "W", "1M": "MS"}
    rule = tf_map.get(timeframe)
    if rule is None:
        raise ValueError(f"Unsupported timeframe: {timeframe}")

    agg = {
        'open': 'first',
        'high': 'max',
        'low': 'min',
    }

    # Handle both 'close' and 'latest' column names
    if 'latest' in df.columns:
        agg['latest'] = 'last'
    if 'close' in df.columns:
        agg['close'] = 'last'
    if 'volume' in df.columns:
        agg['volume'] = 'sum'

    # Include any other columns not covered above (take last value)
    for col in df.columns:
        if col not in agg:
            agg[col] = 'last'

    # label/closed spelled out for every rule rather than only the new ones.
    # pandas defaults them per-frequency -- left for hours and days, right for
    # weeks and months -- so left alone a weekly bar would be stamped with the
    # *end* of its period while an hourly one is stamped with the start. Two
    # labelling conventions in one system do not announce themselves; they just
    # put wave markings on the wrong bar, because the projection engine matches
    # base bars to display bars through a pd.Grouper that has to agree with this
    # index exactly. For hours and days this is a no-op that makes the
    # assumption explicit; for weeks and months it is what makes them agree.
    resampled = df.resample(rule, label="left", closed="left").agg(agg).dropna(subset=['high'])
    return resampled


def load_drm(file, sheet_name):
    if not file.name.lower().endswith(".xlsx"):
        raise ValueError("Invalid file format. Please upload a XLSX file.")

    df = pd.read_excel(file, sheet_name=sheet_name)
    df[sheet_name] = df[sheet_name].ffill().copy()

    return df


def parse_drm_periods(drm_df_input, sheet_name, primary_choice, secondary_choice):
    """
    Converts DRM rows like:
    '28.09.2025_17:00, 30.09.2025_19:00'
    into a list of (start_ts, end_ts) tuples.
    """
    periods = []

    drm_df = drm_df_input[(drm_df_input[sheet_name] == primary_choice) & (drm_df_input.iloc[:,1] == secondary_choice)].iloc[:,2:].values

    for row in drm_df.flatten():
        if not isinstance(row, str):
            continue

        try:
            start_str, end_str = [x.strip() for x in row.split(",")]

            start_dt = pd.to_datetime(
                start_str,
                format="%d.%m.%Y_%H:%M"
            )
            end_dt = pd.to_datetime(
                end_str,
                format="%d.%m.%Y_%H:%M"
            )

            periods.append((start_dt, end_dt))
        except Exception:
            # Skip malformed rows silently
            continue

    return periods

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

def load_drm(file, sheet_name):
    if not file.name.lower().endswith(".xlsx"):
        raise ValueError("Invalid file format. Please upload a XLSX file.")

    df = pd.read_excel(file, usecols="C:V", skiprows=0, nrows=41, sheet_name=sheet_name)

    return df.values[~df.isna()]


def parse_drm_periods(drm_df):
    """
    Converts DRM rows like:
    '28.09.2025_17:00, 30.09.2025_19:00'
    into a list of (start_ts, end_ts) tuples.
    """
    periods = []

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

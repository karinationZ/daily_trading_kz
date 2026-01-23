from pathlib import Path
from typing import List

import pandas as pd

from daily_trading.settings import logger


def load_data(file_path, start_ts=None, end_ts=None):
    logger.info(f"Loading mgc from {file_path}...")

    df = pd.read_csv(file_path)

    # normalize volume column
    if "Volume" in df.columns:
        df.rename(columns={"Volume": "volume"}, inplace=True)

    # Convert Unix 'time' column to datetime in UTC
    df["datetime"] = pd.to_datetime(df["time"], unit="s", utc=True)
    df.set_index("datetime", inplace=True)

    # Keep relevant columns
    keep_cols = ["open", "high", "low", "close", "volume"]
    if "EMA" in df.columns:
        keep_cols.append("EMA")
    df = df[keep_cols]

    # Ensure numeric
    df = df.apply(pd.to_numeric, errors="coerce")
    df.dropna(subset=["open", "high", "low", "close"], inplace=True)

    # Filter by timeframe if provided
    if start_ts is not None:
        start_ts = pd.Timestamp(start_ts, tz="UTC") if not isinstance(start_ts, pd.Timestamp) else start_ts
        df = df[df.index >= start_ts]
    if end_ts is not None:
        end_ts = pd.Timestamp(end_ts, tz="UTC") if not isinstance(end_ts, pd.Timestamp) else end_ts
        df = df[df.index < end_ts]

    logger.info(f"Loaded {len(df)} candles.")
    logger.info(f"Timezone: {df.index.tz}")
    return df


def save_ohlc_to_csv(df: pd.DataFrame, symbol: str, timeframe: str = "5m", out_dir: str = "mgc"):
    """
    Save OHLC dataframe to CSV for backtesting.

    Args:
        df (pd.DataFrame): OHLC dataframe with datetime index
        symbol (str): Instrument symbol (e.g. MNQ, MES)
        timeframe (str): Timeframe string (default: "5m")
        out_dir (str): Output directory
    """
    Path(out_dir).mkdir(parents=True, exist_ok=True)

    # Ensure datetime index
    if not isinstance(df.index, pd.DatetimeIndex):
        raise ValueError("DataFrame index must be DatetimeIndex")

    # Ensure UTC (important for session logic)
    if df.index.tz is None:
        df.index = df.index.tz_localize("UTC")

    file_path = Path(out_dir) / f"{symbol}_{timeframe}.csv"

    export_df = df.copy()
    export_df["timestamp"] = export_df.index

    export_df.to_csv(file_path, index=False, float_format="%.5f")

    print(f"Saved {len(df)} candles to {file_path}")


def merge_csv_no_dups(files: List[str], out_file: str):
    if not files:
        raise ValueError("No CSV files provided")

    dfs = []

    for file in files:
        df = pd.read_csv(file)

        if "time" not in df.columns:
            raise ValueError(f"'time' column missing in {file}")

        # Ensure UNIX timestamp consistency
        df["time"] = df["time"].astype("int64")
        dfs.append(df)

    # Merge all files
    df_merged = pd.concat(dfs, ignore_index=True)

    # Drop duplicates by timestamp
    df_merged = df_merged.drop_duplicates(subset=["time"], keep="last")

    # Sort: earliest on top
    df_merged = df_merged.sort_values("time", ascending=True).reset_index(drop=True)

    # Save
    df_merged.to_csv(out_file, index=False)

    print(f"✅ Merged CSV saved to: {out_file}")
    print(f"Files merged: {len(files)}")
    print(f"Total rows: {len(df_merged)}")

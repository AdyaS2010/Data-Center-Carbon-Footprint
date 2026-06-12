"""
01_prepare_data.py - Clean, resample, and merge the NREL ESIF datasets.

Uses deterministic sampling (every Nth row by position index) so every run
produces the exact same output. No randomness in data preparation.
"""

import pandas as pd
import os
import sys

sys.path.insert(0, os.path.dirname(__file__))
from config import (
    POWER_CSV, WEATHER_CSV, MERGED_CSV, DATA_DIR,
    SAMPLE_STEP, PUE_MIN, PUE_MAX, TEMP_MIN_F, TEMP_MAX_F,
)


def parse_timestamp(ts_str):
    """Parse NREL's timestamp format: 'Mon Nov 09 2015 20:00:01 GMT-0700 (MST)'"""
    try:
        date_part = ts_str.split(" GMT")[0]
        return pd.to_datetime(date_part, format="%a %b %d %Y %H:%M:%S")
    except (ValueError, AttributeError):
        return pd.NaT


def load_deterministic(filepath, usecols, step):
    """Load CSV keeping every `step`-th row (deterministic, no randomness)."""
    chunks = pd.read_csv(filepath, usecols=usecols, chunksize=100_000)
    sampled = []
    for i, chunk in enumerate(chunks):
        sampled.append(chunk.iloc[::step])
        if i % 10 == 0:
            print(f"    chunk {i}...")
    return pd.concat(sampled, ignore_index=True)


def main():
    print("=" * 60)
    print("  STEP 1: Prepare Data (Deterministic Sampling)")
    print("=" * 60)

    # ── Load ─────────────────────────────────────────────────────────────
    print(f"\nLoading power metrics (every {SAMPLE_STEP}th row)...")
    power_cols = ["ts", "cooling_kw", "hvac_kw", "it_power_kw", "pue", "pump_kw"]
    df_power = load_deterministic(POWER_CSV, power_cols, SAMPLE_STEP)
    print(f"  {len(df_power):,} rows")

    print(f"\nLoading weather data (every {SAMPLE_STEP}th row)...")
    weather_cols = ["ts", "outdoor_air_humidity", "outdoor_air_temp"]
    df_weather = load_deterministic(WEATHER_CSV, weather_cols, SAMPLE_STEP)
    print(f"  {len(df_weather):,} rows")

    # ── Parse timestamps ─────────────────────────────────────────────────
    print("\nParsing timestamps...")
    df_power["datetime"] = df_power["ts"].apply(parse_timestamp)
    df_weather["datetime"] = df_weather["ts"].apply(parse_timestamp)
    df_power = df_power.dropna(subset=["datetime"]).sort_values("datetime")
    df_weather = df_weather.dropna(subset=["datetime"]).sort_values("datetime")
    print(f"  Power:   {df_power['datetime'].min()} -> {df_power['datetime'].max()}")
    print(f"  Weather: {df_weather['datetime'].min()} -> {df_weather['datetime'].max()}")

    # ── Remove invalid readings ──────────────────────────────────────────
    print("\nRemoving invalid weather readings...")
    before = len(df_weather)
    df_weather = df_weather[
        (df_weather["outdoor_air_temp"] > TEMP_MIN_F)
        & (df_weather["outdoor_air_temp"] < TEMP_MAX_F)
        & (df_weather["outdoor_air_humidity"] >= 0)
        & (df_weather["outdoor_air_humidity"] <= 100)
    ]
    print(f"  Removed {before - len(df_weather):,} invalid rows")

    # ── Resample to hourly ───────────────────────────────────────────────
    print("\nResampling to hourly averages...")
    df_power.set_index("datetime", inplace=True)
    df_weather.set_index("datetime", inplace=True)

    power_hourly = df_power[
        ["cooling_kw", "hvac_kw", "it_power_kw", "pue", "pump_kw"]
    ].resample("h").mean()

    weather_hourly = df_weather[
        ["outdoor_air_temp", "outdoor_air_humidity"]
    ].resample("h").mean()

    # Inner join on aligned timestamps
    df = power_hourly.join(weather_hourly, how="inner")
    df = df.dropna(subset=["pue", "outdoor_air_temp"])

    # Remove PUE outliers
    before = len(df)
    df = df[(df["pue"] >= PUE_MIN) & (df["pue"] <= PUE_MAX)]
    print(f"  Removed {before - len(df):,} PUE outliers")

    # ── Save ─────────────────────────────────────────────────────────────
    os.makedirs(DATA_DIR, exist_ok=True)
    df.to_csv(MERGED_CSV)
    size_mb = os.path.getsize(MERGED_CSV) / 1024 / 1024
    print(f"\nSaved: {MERGED_CSV}")
    print(f"  Rows:  {len(df):,}")
    print(f"  Size:  {size_mb:.1f} MB")
    print(f"  Range: {df.index.min()} -> {df.index.max()}")

    print("\nDescriptive Statistics:")
    print(df.describe().round(3).to_string())


if __name__ == "__main__":
    main()

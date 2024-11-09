# temperature_utils.py
"""Temperature calculation utilities."""

import pandas as pd
from decimal import Decimal, ROUND_HALF_UP
from typing import Optional
from datetime import datetime

def calculate_temperature_stats(df: pd.DataFrame) -> None:
    """Calculates and prints temperature statistics."""
    print("\nDebug: Temperature Calculation Details")
    print("-" * 50)

    regular_temp_max = calculate_regular_temp_max(df)
    six_hr_max = calculate_six_hour_high(df)

    true_high = max(regular_temp_max, six_hr_max)
    if true_high == float('-inf'):
        print("Error: No valid temperature readings available")
        return

    df["Todays_High"] = true_high
    df["Todays_High_Rounded"] = int(Decimal(str(true_high)).quantize(0, ROUND_HALF_UP))

    print_temperature_summary(regular_temp_max, six_hr_max, true_high, df)

def calculate_regular_temp_max(df: pd.DataFrame) -> float:
    """Calculates regular temperature maximum for current day only."""
    today = datetime.utcnow().date()

    # Filter for today's readings only
    today_data = df[pd.to_datetime(df['Date']).dt.date == today]

    regular_temp_max = float('-inf')
    if "Temperature" in today_data.columns:
        regular_temp_max = today_data["Temperature"].max()
        print(f"Regular temperature max for today: {regular_temp_max:.2f}°F")
        if not today_data.empty:
            print(f"Time of regular max: {today_data.loc[today_data['Temperature'].idxmax(), 'Date']}")
    return regular_temp_max

def calculate_six_hour_high(df: pd.DataFrame) -> float:
    """Calculates the 6-hour high temperature for current day only."""
    today = datetime.utcnow().date()

    # Filter for today's readings only
    today_data = df[pd.to_datetime(df['Date']).dt.date == today]

    if "Temp_High_6hr" not in today_data.columns:
        print("Warning: 6-hour high temperature not available in data")
        if "Temperature" in today_data.columns:
            rolling_max = today_data["Temperature"].rolling(window="6H", center=True).max()
            if not rolling_max.empty:
                six_hr_max = rolling_max.max()
                max_time = today_data.loc[rolling_max.idxmax(), 'Date']
                print(f"Calculated 6-hour rolling maximum: {six_hr_max:.2f}°F")
                print(f"Time of rolling maximum: {max_time}")
                return six_hr_max
        return float('-inf')

    six_hr_series = pd.to_numeric(today_data["Temp_High_6hr"].copy(), errors='coerce')
    six_hr_series = six_hr_series[(six_hr_series > -100) & (six_hr_series < 150)]

    if not six_hr_series.empty:
        six_hr_max = six_hr_series.max()
        max_time = today_data.loc[six_hr_series.idxmax(), 'Date']
        print(f"6-hour high max from data: {six_hr_max:.2f}°F")
        print(f"Time of 6-hour max: {max_time}")
        return six_hr_max

    return float('-inf')

def print_temperature_summary(regular_max: float, six_hr_max: float,
                            true_high: float, df: pd.DataFrame) -> None:
    """Prints a summary of temperature calculations."""
    print("\nSummary:")
    print(f"Regular Temperature Maximum: {regular_max:.2f}°F")
    print(f"6-hour High Maximum: {six_hr_max:.2f}°F")
    print(f"Final High Temperature: {true_high:.2f}°F "
          f"(Rounded: {df['Todays_High_Rounded'].iloc[0]}°F)")
    print("-" * 50)
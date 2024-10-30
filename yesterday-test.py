# fetch_yesterdays_high.py

import datetime
from weather_utils import get_weather_data
from decimal import Decimal, ROUND_HALF_UP

# Prompt the user to enter the station ID
station_id = input("Enter your station ID (e.g., KMIA): ").strip().upper()

# Validate the station ID
if len(station_id) != 4:
    raise ValueError("Station ID must be exactly 4 characters long.")

def fetch_yesterdays_high(station_id):
    # Calculate yesterday's date
    yesterday = datetime.datetime.now() - datetime.timedelta(days=1)
    formatted_date = yesterday.strftime('%Y-%m-%d')
    print(f"Fetching high temperature for {formatted_date} at station {station_id}...")

    # Set the start and end times to cover the entire day
    start_time = yesterday.strftime('%Y%m%d0000')
    end_time = yesterday.strftime('%Y%m%d2359')

    # Fetch weather data using get_weather_data
    weather_data = get_weather_data(station_id, start_time=start_time, end_time=end_time)

    if weather_data is None or weather_data.empty:
        print("Unable to fetch weather data for yesterday.")
        return

    # Use 'Temp_High_6hr' to find the high temperature
    if 'Temp_High_6hr' in weather_data.columns:
        high_temperature = weather_data['Temp_High_6hr'].max()
    else:
        # Fallback to using 'Temperature' if 'Temp_High_6hr' is not available
        high_temperature = weather_data['Temperature'].max()

    # Use Decimal to round the temperature
    rounded_temp = int(Decimal(high_temperature).quantize(0, rounding=ROUND_HALF_UP))

    # Print the original and rounded temperatures
    print(f"Yesterday's high temperature at station {station_id} was: {high_temperature:.2f} °F ({rounded_temp})")

# Run the function
fetch_yesterdays_high(station_id)

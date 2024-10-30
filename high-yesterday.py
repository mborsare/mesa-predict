# fetch_yesterdays_high.py

import datetime
from weather_utils import get_weather_data, get_station_metadata
from decimal import Decimal, ROUND_HALF_UP
import pytz  # Ensure pytz is installed: pip install pytz

# Prompt the user to enter the station ID
station_id = input("Enter your station ID to retrieve yesterday's high (e.g., KMIA): ").strip().upper()

# Validate the station ID
if len(station_id) != 4:
    raise ValueError("Station ID must be exactly 4 characters long.")

def fetch_yesterdays_high(station_id):
    # Fetch station metadata to determine timezone
    metadata = get_station_metadata(station_id)
    if metadata is None:
        print("Cannot proceed without station metadata.")
        return

    timezone_str = metadata['timezone']

    try:
        station_tz = pytz.timezone(timezone_str)
    except pytz.UnknownTimeZoneError:
        print(f"Unknown timezone '{timezone_str}' for station ID '{station_id}'.")
        return

    # Get current time in station's timezone
    now = datetime.datetime.now(station_tz)

    # Calculate yesterday's date in station's local timezone
    yesterday = now - datetime.timedelta(days=1)
    formatted_date = yesterday.strftime('%Y-%m-%d')
    print(f"Fetching high temperature for {formatted_date} at station {station_id}...")

    # Set the start and end times to cover the entire day in station's local timezone
    yesterday_start = yesterday.replace(hour=0, minute=0, second=0, microsecond=0)
    yesterday_end = yesterday.replace(hour=23, minute=59, second=59, microsecond=999999)

    # Convert start_time and end_time to UTC for API request
    yesterday_start_utc = yesterday_start.astimezone(pytz.utc)
    yesterday_end_utc = yesterday_end.astimezone(pytz.utc)

    # Format start_time and end_time in 'YYYYMMDDHHMM'
    start_time = yesterday_start_utc.strftime('%Y%m%d%H%M')
    end_time = yesterday_end_utc.strftime('%Y%m%d%H%M')

    # Fetch weather data using get_weather_data
    weather_data = get_weather_data(station_id, start_time=start_time, end_time=end_time)

    if weather_data is None or weather_data.empty:
        print("Unable to fetch weather data for yesterday.")
        return

    # Convert weather_data['Date'] from UTC to station's local timezone
    weather_data['Date'] = weather_data['Date'].dt.tz_convert(station_tz)

    # Rename columns to match those used during training (if necessary)
    rename_columns = {
        'Humidity': 'Humidity_Avg',
        'Wind_Speed': 'Wind_Speed_Avg',
        'Temp_High_6hr': 'Temp_High_6hr_Avg'
    }
    # Only rename existing columns to avoid KeyError
    rename_columns = {k: v for k, v in rename_columns.items() if k in weather_data.columns}
    weather_data.rename(columns=rename_columns, inplace=True)

    # Determine which temperature column to use for high temperature
    if 'Temp_High_6hr_Avg' in weather_data.columns:
        high_temperature = weather_data['Temp_High_6hr_Avg'].max()
        temp_column_used = 'Temp_High_6hr_Avg'
    elif 'Temperature' in weather_data.columns:
        high_temperature = weather_data['Temperature'].max()
        temp_column_used = 'Temperature'
    else:
        print("No temperature data available to determine yesterday's high.")
        return

    # Use Decimal to round the temperature
    rounded_temp = int(Decimal(float(high_temperature)).quantize(0, rounding=ROUND_HALF_UP))

    # Fetch the timestamp of the last data point
    last_timestamp = weather_data['Date'].max()
    human_readable_timestamp = last_timestamp.strftime('%Y-%m-%d %H:%M:%S %Z')

    # Print the original and rounded temperatures
    print(f"Yesterday's high temperature at station {station_id} was: {high_temperature:.2f} °F ({rounded_temp})")
    print(f"Last Data Point Timestamp: {human_readable_timestamp}")

# Run the function
fetch_yesterdays_high(station_id)

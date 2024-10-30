# weather_utils.py

import requests
import pandas as pd
import datetime
import json
import pytz

# Mesowest API URLs
timeseries_url = "https://api.mesowest.net/v2/stations/timeseries"
station_info_url = "https://api.mesowest.net/v2/stations/metadata"

# Mesowest token provided by you
token = "d8c6aee36a994f90857925cea26934be"


def get_station_metadata(station_id):
    """
    Fetches station metadata, including TIMEZONE.

    Parameters:
    - station_id: The station ID to fetch metadata for.

    Returns:
    - A dictionary containing station metadata or None if not found.
    """
    params = {"STID": station_id, "token": token}

    try:
        response = requests.get(station_info_url, params=params)
        response.raise_for_status()
        data = response.json()

        if "STATION" not in data or not data["STATION"]:
            print(f"No metadata found for station ID '{station_id}'.")
            return None

        station_data = data["STATION"][0]
        timezone = station_data.get("TIMEZONE")

        if timezone is None:
            print(f"Timezone not available for station ID '{station_id}'.")
            return None

        return {"timezone": timezone}

    except requests.exceptions.RequestException as e:
        print(f"Error fetching station metadata: {e}")
        return None


def get_weather_data(station_id, start_time=None, end_time=None, recent=None):
    """
    Fetches weather data from the Mesowest API.

    Parameters:
    - station_id: The station ID to fetch data for.
    - start_time: Start time in 'YYYYMMDDHHMM' format (for historical data).
    - end_time: End time in 'YYYYMMDDHHMM' format (for historical data).
    - recent: Number of minutes to look back from now (for recent data).

    Returns:
    - A pandas DataFrame containing the processed weather data.
    """
    params = {
        "STID": station_id,
        "units": "temp|F,speed|mph",
        "token": token,
        "vars": "air_temp,relative_humidity,wind_speed,air_temp_high_6_hour,sea_level_pressure",
        "obtimezone": "UTC",  # Set to 'UTC' for consistency
        "complete": "1",  # Ensure complete data within the window
    }

    if recent:
        params["recent"] = recent
    elif start_time and end_time:
        params["start"] = start_time
        params["end"] = end_time
    else:
        raise ValueError(
            "Either 'recent' or both 'start_time' and 'end_time' must be provided."
        )

    try:
        response = requests.get(timeseries_url, params=params)
        response.raise_for_status()
        data = response.json()

        if "STATION" not in data or not data["STATION"]:
            print("Unexpected response format or no station data available.")
            print("Raw API Response:")
            # Pretty-print the JSON for better readability
            print(json.dumps(data, indent=4))
            return None

        station_data = data["STATION"][0]["OBSERVATIONS"]

        date_time = station_data.get("date_time", [])
        df_data = {"Date": date_time}

        # Extract variables
        if "air_temp_set_1" in station_data:
            df_data["Temperature"] = station_data["air_temp_set_1"]
        if "air_temp_high_6_hour_set_1" in station_data:
            df_data["Temp_High_6hr"] = station_data["air_temp_high_6_hour_set_1"]
        if "relative_humidity_set_1" in station_data:
            df_data["Humidity"] = station_data["relative_humidity_set_1"]
        if "wind_speed_set_1" in station_data:
            df_data["Wind_Speed"] = station_data["wind_speed_set_1"]
        if "sea_level_pressure_set_1" in station_data:
            df_data["Sea_Level_Pressure"] = station_data["sea_level_pressure_set_1"]

        df = pd.DataFrame(df_data)

        # Dynamically determine subset based on available columns
        subset = ["Temperature", "Temp_High_6hr", "Humidity", "Wind_Speed"]
        if "Sea_Level_Pressure" in df.columns:
            subset.append("Sea_Level_Pressure")

        df.dropna(how="all", subset=subset, inplace=True)

        if df.empty:
            print("No complete data available after removing null values.")
            return None

        # Parse 'Date' as datetime in UTC
        df["Date"] = pd.to_datetime(df["Date"], utc=True)

        print("Data processed successfully.")
        return df

    except requests.exceptions.RequestException as e:
        print(f"Error fetching data: {e}")
        return None

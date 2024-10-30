# weather_utils.py

import requests
import pandas as pd
import datetime

# Mesowest API URL for timeseries data
timeseries_url = "https://api.mesowest.net/v2/stations/timeseries"

# Mesowest token provided by you
token = "d8c6aee36a994f90857925cea26934be"

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
        'STID': station_id,
        'units': 'temp|F,speed|mph',
        'token': token,
        'vars': 'air_temp,relative_humidity,wind_speed,air_temp_high_6_hour',
        'obtimezone': 'local'
    }

    if recent:
        params['recent'] = recent
    elif start_time and end_time:
        params['start'] = start_time
        params['end'] = end_time
    else:
        raise ValueError("Either 'recent' or both 'start_time' and 'end_time' must be provided.")

    try:
        response = requests.get(timeseries_url, params=params)
        response.raise_for_status()
        data = response.json()

        if 'STATION' not in data or not data['STATION']:
            print("Unexpected response format or no station data available.")
            return None

        station_data = data['STATION'][0]['OBSERVATIONS']

        date_time = station_data.get('date_time', [])
        df_data = {'Date': date_time}

        # Extract variables, including 'air_temp_high_6_hour_set_1'
        if 'air_temp_set_1' in station_data:
            df_data['Temperature'] = station_data['air_temp_set_1']
        if 'air_temp_high_6_hour_set_1' in station_data:
            df_data['Temp_High_6hr'] = station_data['air_temp_high_6_hour_set_1']
        if 'relative_humidity_set_1' in station_data:
            df_data['Humidity'] = station_data['relative_humidity_set_1']
        if 'wind_speed_set_1' in station_data:
            df_data['Wind_Speed'] = station_data['wind_speed_set_1']

        df = pd.DataFrame(df_data)
        df.dropna(how='all', subset=['Temperature', 'Temp_High_6hr', 'Humidity', 'Wind_Speed'], inplace=True)

        if df.empty:
            print("No complete data available after removing null values.")
            return None

        df['Date'] = pd.to_datetime(df['Date'])

        print("Data processed successfully.")
        return df

    except requests.exceptions.RequestException as e:
        print(f"Error fetching data: {e}")
        return None

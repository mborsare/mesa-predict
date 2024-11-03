# weather_utils.py

import requests
import pandas as pd
import datetime
import json
import pytz
from typing import Optional, Dict, Tuple
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler
from decimal import Decimal, ROUND_HALF_UP

# API configuration
API_CONFIG = {
    "timeseries_url": "https://api.mesowest.net/v2/stations/timeseries",
    "station_info_url": "https://api.mesowest.net/v2/stations/metadata",
    "token": "d8c6aee36a994f90857925cea26934be",
    "tomorrow_io_token": "bZkgwxlNihNKB0dlPnD55rMi3EYAEaew"
}

class WeatherDataProcessor:
    def __init__(self, station_id: str):
        self.station_id = station_id.strip().upper()
        if len(self.station_id) != 4:
            raise ValueError("Station ID must be exactly 4 characters long.")

        # Fetch station metadata including timezone and coordinates
        self.station_metadata = self._get_station_metadata()
        if not self.station_metadata:
            raise ValueError(f"Cannot fetch metadata for station {station_id}")

        self.timezone = self._get_station_timezone()
        if not self.timezone:
            raise ValueError(f"Cannot determine timezone for station {station_id}")

    def _get_station_metadata(self) -> Optional[Dict]:
        """Fetches complete station metadata."""
        params = {'STID': self.station_id, 'token': API_CONFIG['token']}

        try:
            response = requests.get(API_CONFIG['station_info_url'], params=params)
            response.raise_for_status()
            data = response.json()

            if 'STATION' not in data or not data['STATION']:
                print(f"No metadata found for station ID '{self.station_id}'.")
                return None

            return data['STATION'][0]

        except requests.exceptions.RequestException as e:
            print(f"Error fetching station metadata: {e}")
            return None

    def _get_station_timezone(self) -> Optional[pytz.timezone]:
        """Fetches and validates station timezone."""
        if self.station_metadata:
            timezone_str = self.station_metadata.get('TIMEZONE')
            try:
                return pytz.timezone(timezone_str) if timezone_str else None
            except pytz.UnknownTimeZoneError:
                print(f"Unknown timezone: {timezone_str}")
                return None
        return None

    def get_coordinates(self) -> Tuple[float, float]:
        """Returns the latitude and longitude of the station."""
        if self.station_metadata and 'LATITUDE' in self.station_metadata and 'LONGITUDE' in self.station_metadata:
            return float(self.station_metadata['LATITUDE']), float(self.station_metadata['LONGITUDE'])
        else:
            raise ValueError("Coordinates not available in station metadata")

    def get_weather_data(
        self,
        start_time: Optional[str] = None,
        end_time: Optional[str] = None,
        recent: Optional[int] = None
    ) -> Optional[pd.DataFrame]:
        """
        Fetches weather data from Mesowest API.

        Args:
            start_time: Start time for data fetch (format: YYYYMMDDHHMM)
            end_time: End time for data fetch (format: YYYYMMDDHHMM)
            recent: Number of most recent hours of data to fetch

        Returns:
            DataFrame with weather data or None if fetch fails
        """
        params = {
            "STID": self.station_id,
            "units": "temp|F,speed|mph",
            "token": API_CONFIG["token"],
            "vars": "air_temp,relative_humidity,wind_speed,air_temp_high_6_hour,sea_level_pressure",
            "obtimezone": "UTC",
            "complete": "0"
        }

        if recent:
            params["recent"] = recent
        elif start_time and end_time:
            params["start"] = start_time
            params["end"] = end_time
        else:
            raise ValueError("Either 'recent' or both 'start_time' and 'end_time' must be provided.")

        try:
            response = requests.get(API_CONFIG["timeseries_url"], params=params)
            response.raise_for_status()
            return self._process_weather_data(response.json())
        except requests.exceptions.RequestException as e:
            print(f"Error fetching data: {e}")
            return None

    def get_recent_data(self, hours: int = 24) -> Optional[pd.DataFrame]:
        """
        Fetches recent weather data for the station.

        Args:
            hours: Number of hours of historical data to fetch (default: 24)

        Returns:
            DataFrame containing recent weather observations or None if fetch fails
        """
        return self.get_weather_data(recent=hours)

    def _process_weather_data(self, data: Dict) -> Optional[pd.DataFrame]:
        """Processes raw weather data into a DataFrame."""
        if "STATION" not in data or not data["STATION"]:
            print("Unexpected response format or no station data available.")
            return None

        station_data = data["STATION"][0]["OBSERVATIONS"]

        # Create initial DataFrame
        df_data = {"Date": station_data.get("date_time", [])}

        # Map API fields to DataFrame columns
        variables = {
            "Temperature": "air_temp_set_1",
            "Temp_High_6hr": "air_temp_high_6_hour_set_1",
            "Humidity": "relative_humidity_set_1",
            "Wind_Speed": "wind_speed_set_1",
            "Sea_Level_Pressure": "sea_level_pressure_set_1"
        }

        for column, key in variables.items():
            if key in station_data:
                df_data[column] = station_data[key]

        df = pd.DataFrame(df_data)

        # Clean and process the DataFrame
        subset = [col for col in variables.keys() if col in df.columns]
        df.dropna(how="all", subset=subset, inplace=True)

        if df.empty:
            print("No complete data available after removing null values.")
            return None

        df["Date"] = pd.to_datetime(df["Date"], utc=True)
        df["Date"] = df["Date"].dt.tz_convert(self.timezone)

        # Calculate today's high
        self._calculate_todays_high(df)

        return df

    def _calculate_todays_high(self, df: pd.DataFrame) -> None:
        """Calculates and adds today's high temperature to the DataFrame."""
        print("\nDebug: Temperature Calculation Details")
        print("-" * 50)

        if "Temperature" not in df.columns:
            print("Warning: Regular temperature readings not available")
            regular_temp_max = float('-inf')
        else:
            regular_temp_max = df["Temperature"].max()
            print(f"Regular temperature max: {regular_temp_max:.2f}°F")
            print(f"Time of regular max: {df.loc[df['Temperature'].idxmax(), 'Date']}")

        if "Temp_High_6hr" not in df.columns:
            print("Warning: 6-hour high temperature readings not available")
            six_hr_max = float('-inf')
        else:
            # Remove NaN values for accurate max calculation
            six_hr_series = df["Temp_High_6hr"].dropna()
            if not six_hr_series.empty:
                six_hr_max = six_hr_series.max()
                print(f"6-hour high max: {six_hr_max:.2f}°F")
                print(f"Time of 6-hour max: {df.loc[df['Temp_High_6hr'].idxmax(), 'Date']}")
            else:
                print("Warning: All 6-hour high temperatures are NaN")
                six_hr_max = float('-inf')

        # Calculate true high by comparing both series
        true_high = max(regular_temp_max, six_hr_max)
        print(f"True high temperature: {true_high:.2f}°F")

        # Update DataFrame with calculated values
        df["Todays_High"] = true_high
        df["Todays_High_Rounded"] = int(Decimal(true_high).quantize(0, ROUND_HALF_UP))

        print("\nSummary:")
        print(f"Regular Temperature Maximum: {regular_temp_max:.2f}°F")
        print(f"6-hour High Maximum: {six_hr_max:.2f}°F")
        print(f"Final High Temperature: {true_high:.2f}°F (Rounded: {df['Todays_High_Rounded'].iloc[0]}°F)")
        print("-" * 50)

class TemperaturePredictor:
    def __init__(self, weather_processor: WeatherDataProcessor):
        self.weather_processor = weather_processor
        self.model = RandomForestRegressor(n_estimators=100, random_state=42)
        self.scaler = StandardScaler()

    def _prepare_features(self, data: pd.DataFrame) -> Optional[pd.DataFrame]:
        """Prepares features for model training."""
        try:
            data["Hours_Since_Midnight"] = (
                data["Date"].dt.hour + data["Date"].dt.minute / 60
            )
            data["DateOnly"] = data["Date"].dt.date

            grouped = data.groupby("DateOnly")

            features = pd.DataFrame({
                "Max_Temp_So_Far": grouped["Temperature"].max(),
                "Humidity_Avg": grouped["Humidity"].mean(),
                "Wind_Speed_Avg": grouped["Wind_Speed"].mean(),
                "Hours_Since_Midnight": grouped["Hours_Since_Midnight"].mean(),
                "Actual_High_Temp": grouped["Temp_High_6hr"].max(),
                # Add placeholder columns for Tomorrow.io features
                "Tomorrow_Max_Temp": 0,
                "Tomorrow_Avg_Humidity": 0,
                "Tomorrow_Avg_WindSpeed": 0
            }).dropna()

            # Remove outliers
            return features[
                (features["Humidity_Avg"] >= 0)
                & (features["Humidity_Avg"] <= 100)
                & (features["Wind_Speed_Avg"] >= 0)
                & (features["Wind_Speed_Avg"] <= 100)
                & (features["Actual_High_Temp"] >= -50)
                & (features["Actual_High_Temp"] <= 150)
            ]

        except Exception as e:
            print(f"Error preparing features: {e}")
            return None

    def _prepare_prediction_features(self, data: pd.DataFrame) -> Optional[pd.DataFrame]:
        """Prepares features for making predictions."""
        try:
            features = pd.DataFrame({
                "Max_Temp_So_Far": [data["Temperature"].max()],
                "Humidity_Avg": [data["Humidity"].mean()],
                "Wind_Speed_Avg": [data["Wind_Speed"].mean()],
                "Hours_Since_Midnight": [
                    data["Date"].dt.hour.iloc[-1] + data["Date"].dt.minute.iloc[-1] / 60
                ],
                # Add placeholder columns for Tomorrow.io features
                "Tomorrow_Max_Temp": [0],
                "Tomorrow_Avg_Humidity": [0],
                "Tomorrow_Avg_WindSpeed": [0]
            })
            return features
        except Exception as e:
            print(f"Error preparing prediction features: {e}")
            return None

    def _train_model(self, training_data: pd.DataFrame) -> bool:
        """Trains the prediction model."""
        # Prepare features
        features = self._prepare_features(training_data)
        if features is None:
            return False

        X_train = features[[
            "Max_Temp_So_Far",
            "Humidity_Avg",
            "Wind_Speed_Avg",
            "Hours_Since_Midnight",
            "Tomorrow_Max_Temp",
            "Tomorrow_Avg_Humidity",
            "Tomorrow_Avg_WindSpeed"
        ]]
        y_train = features["Actual_High_Temp"]

        # Train model
        X_train_scaled = self.scaler.fit_transform(X_train)
        self.model.fit(X_train_scaled, y_train)
        return True
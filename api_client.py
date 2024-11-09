# api_client.py
"""Weather API client for fetching weather data."""

import requests
from typing import Optional, Dict
from config import API_CONFIG

class WeatherAPIClient:
    """Handles API communication with weather services"""

    def __init__(self, station_id: str):
        self.station_id = station_id.strip().upper()
        if len(self.station_id) != 4:
            raise ValueError("Station ID must be exactly 4 characters long.")

    def get_station_metadata(self) -> Optional[Dict]:
        """Fetches station metadata from the API."""
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

    def get_weather_data(self, **params) -> Optional[Dict]:
        """Fetches weather data from the API."""
        try:
            response = requests.get(API_CONFIG["timeseries_url"], params=params)
            response.raise_for_status()
            print(f"\nDebug: API URL: {response.url}\n")
            return response.json()
        except requests.exceptions.RequestException as e:
            print(f"Error fetching weather data: {e}")
            return None
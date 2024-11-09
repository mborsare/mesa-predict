# data_processor.py
import requests
from datetime import datetime, timedelta
from typing import Optional, Dict, List, Tuple, Any
import logging
from requests.exceptions import RequestException
import pandas as pd

class WeatherDataProcessor:
    """Process weather data from MesoWest API."""

    def __init__(self, station_id: str):
        self.station_id = station_id.upper()
        self.base_url = "https://api.mesowest.net/v2/stations"
        self.token = "d8c6aee36a994f90857925cea26934be"
        self.logger = logging.getLogger(__name__)
        self._cached_data = None
        self._cached_coordinates = None

    def get_coordinates(self) -> Optional[Tuple[float, float]]:
        """Get station coordinates."""
        if self._cached_coordinates is not None:
            return self._cached_coordinates

        try:
            response = requests.get(
                f"{self.base_url}/metadata",
                params={'stid': self.station_id, 'token': self.token},
                timeout=10
            )
            response.raise_for_status()
            data = response.json()

            if not data.get('STATION'):
                self.logger.error(f"No station data found for {self.station_id}")
                return None

            station = data['STATION'][0]
            self._cached_coordinates = (float(station['LATITUDE']), float(station['LONGITUDE']))
            self.logger.info(f"Retrieved coordinates for {self.station_id}: {self._cached_coordinates}")
            return self._cached_coordinates

        except RequestException as e:
            self.logger.error(f"Failed to get coordinates: {str(e)}")
            return None
        except (KeyError, IndexError) as e:
            self.logger.error(f"Invalid station data format: {str(e)}")
            return None

    def _fetch_data(self) -> Optional[Dict[str, Any]]:
        """Fetch last 7 days of weather data with caching."""
        if self._cached_data is not None:
            return self._cached_data

        try:
            end = datetime.utcnow()
            start = end - timedelta(days=7)

            response = requests.get(
                f"{self.base_url}/timeseries",
                params={
                    'stid': self.station_id,
                    'start': start.strftime('%Y%m%d%H%M'),
                    'end': end.strftime('%Y%m%d%H%M'),
                    'vars': 'air_temp,air_temp_high_6_hour',
                    'token': self.token,
                    'units': 'temp|F'
                },
                timeout=10
            )
            response.raise_for_status()
            self._cached_data = response.json()
            return self._cached_data

        except RequestException as e:
            self.logger.error(f"Failed to fetch weather data: {str(e)}")
            return None

    def get_recent_temperatures(self, days: int = 7) -> List[Dict[str, Any]]:
        """Get daily high temperatures for recent days."""
        data = self._fetch_data()
        if not data or 'STATION' not in data:
            return []

        try:
            observations = data['STATION'][0].get('OBSERVATIONS', {})
            dates = observations.get('date_time', [])
            temps = observations.get('air_temp_set_1', [])

            # Create dictionary of date to temperatures
            daily_temps = {}
            for date_str, temp in zip(dates, temps):
                if temp is None:
                    continue
                date = datetime.strptime(date_str, '%Y-%m-%dT%H:%M:%SZ').date()
                if date not in daily_temps:
                    daily_temps[date] = []
                daily_temps[date].append(temp)

            # Calculate daily highs
            result = []
            today = datetime.utcnow().date()
            for i in range(days):
                date = today - timedelta(days=i)
                if date in daily_temps and daily_temps[date]:
                    result.append({
                        'date': date,
                        'high': max(daily_temps[date])
                    })
                    self.logger.debug(f"High temperature for {date}: {max(daily_temps[date])}°F")

            return result

        except (KeyError, IndexError) as e:
            self.logger.error(f"Error processing recent temperature data: {str(e)}")
            return []

    def get_regular_temperature_max(self) -> Optional[float]:
        """Get maximum regular temperature for current day only."""
        data = self._fetch_data()
        if not data:
            return None

        try:
            today = datetime.utcnow().date()
            observations = data['STATION'][0].get('OBSERVATIONS', {})
            dates = observations.get('date_time', [])
            temps = observations.get('air_temp_set_1', [])

            # Filter for today's temperatures only
            today_temps = [
                temp for date_str, temp in zip(dates, temps)
                if temp is not None and
                datetime.strptime(date_str, '%Y-%m-%dT%H:%M:%SZ').date() == today
            ]

            if not today_temps:
                return None

            max_temp = max(today_temps)
            self.logger.info(f"Maximum regular temperature: {max_temp}°F")
            return max_temp

        except (KeyError, IndexError) as e:
            self.logger.error(f"Error processing temperature data: {str(e)}")
            return None

    def get_6hr_high_temperature_max(self) -> Optional[float]:
        """Get maximum 6-hour high temperature for current day only."""
        data = self._fetch_data()
        if not data:
            return None

        try:
            today = datetime.utcnow().date()
            observations = data['STATION'][0].get('OBSERVATIONS', {})
            dates = observations.get('date_time', [])
            temps = observations.get('air_temp_high_6_hour_set_1', [])

            # Filter for today's temperatures only
            today_temps = [
                temp for date_str, temp in zip(dates, temps)
                if temp is not None and
                datetime.strptime(date_str, '%Y-%m-%dT%H:%M:%SZ').date() == today
            ]

            if not today_temps:
                self.logger.warning("No valid 6-hour high temperature readings found for today")
                return None

            max_temp = max(today_temps)
            self.logger.info(f"Maximum 6-hour high temperature: {max_temp}°F")
            return max_temp

        except (KeyError, IndexError) as e:
            self.logger.error(f"Error processing 6-hour temperature data: {str(e)}")
            return None
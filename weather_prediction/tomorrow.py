"""Tomorrow.io API client with async support and proper error handling."""
from datetime import datetime, timedelta, date
import logging
from typing import Optional, Sequence, Dict, Any
import aiohttp
from decimal import Decimal
import pytz

from .models import (
    TemperatureReading, Temperature, StationMetadata,
    DataSource, TemperatureUnit, DataSourceError, ValidationError
)
from .config import APIConfig

class TomorrowClient:
    """Async client for Tomorrow.io API with proper error handling."""

    def __init__(self, config: APIConfig, logger: Optional[logging.Logger] = None):
        self.config = config
        self.logger = logger or logging.getLogger(__name__)
        self._session: Optional[aiohttp.ClientSession] = None

    async def __aenter__(self) -> 'TomorrowClient':
        timeout = aiohttp.ClientTimeout(total=self.config.timeout)
        self._session = aiohttp.ClientSession(timeout=timeout)
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb) -> None:
        if self._session:
            await self._session.close()
            self._session = None

    async def get_temperature_forecast(
        self,
        station: StationMetadata,
        start: datetime,
        end: datetime
    ) -> Sequence[TemperatureReading]:
        """Fetch temperature forecast for location and time range."""
        if not self._session:
            raise DataSourceError("Client session not initialized")

        if not start.tzinfo or not end.tzinfo:
            raise ValidationError("Timestamps must be timezone-aware")

        if end - start > timedelta(days=7):
            raise ValidationError("Forecast range cannot exceed 7 days")

        try:
            async with self._session.get(
                f"{self.config.base_url}/timelines",
                params={
                    "location": f"{station.latitude},{station.longitude}",
                    "fields": ["temperature"],
                    "timesteps": "1h",
                    "units": "imperial",
                    "startTime": start.isoformat(),
                    "endTime": end.isoformat(),
                    "apikey": self.config.token
                }
            ) as response:
                response.raise_for_status()
                data = await response.json()
                self.logger.debug(f"Raw forecast response: {data}")
                return self._parse_forecast_data(data, station)

        except aiohttp.ClientError as e:
            raise DataSourceError(f"Failed to fetch forecast: {e}")

    def _parse_forecast_data(
        self,
        data: Dict[str, Any],
        station: StationMetadata
    ) -> Sequence[TemperatureReading]:
        """Parse API response into temperature readings."""
        try:
            readings = []
            timelines = data.get('data', {}).get('timelines', [])

            for timeline in timelines:
                for interval in timeline.get('intervals', []):
                    timestamp = datetime.fromisoformat(
                        interval['startTime'].replace('Z', '+00:00')
                    )
                    temp = interval.get('values', {}).get('temperature')

                    if temp is not None:
                        readings.append(TemperatureReading(
                            value=Temperature(Decimal(str(temp))),
                            timestamp=timestamp,
                            source=DataSource.TOMORROW_IO,
                            unit=TemperatureUnit.FAHRENHEIT,
                            station_id=station.station_id
                        ))

            self.logger.debug(f"Parsed {len(readings)} forecast readings")
            return readings

        except (KeyError, ValueError) as e:
            self.logger.error(f"Error parsing forecast data: {e}")
            self.logger.debug(f"Problematic data: {data}")
            raise ValidationError(f"Error parsing forecast data: {e}")

    async def get_historical_data(
        self,
        station: StationMetadata,
        target_date: datetime,
    ) -> Optional[Dict[str, Any]]:
        """Get historical weather data from Tomorrow.io."""
        if not self._session:
            raise DataSourceError("Client session not initialized")

        try:
            # Calculate 6-hour window
            end_time = target_date
            start_time = end_time - timedelta(hours=6)

            params = {
                "location": f"{station.latitude},{station.longitude}",
                "fields": ["temperature", "humidity", "windSpeed", "windDirection"],
                "timesteps": "1h",
                "startTime": start_time.strftime('%Y-%m-%dT%H:%M:%SZ'),
                "endTime": end_time.strftime('%Y-%m-%dT%H:%M:%SZ'),
                "apikey": self.config.token,
                "units": "imperial"
            }

            self.logger.debug(f"Requesting historical data with params: {params}")

            async with self._session.get(
                f"{self.config.base_url}/timelines",
                params=params
            ) as response:
                response.raise_for_status()
                return await response.json()

        except Exception as e:
            self.logger.error(f"Failed to get Tomorrow.io historical data: {e}")
            return None

    async def get_daily_high_prediction(
        self,
        station: StationMetadata,
        date: datetime
    ) -> Optional[TemperatureReading]:
        """Get historical high temperature for backtesting."""
        try:
            # Target afternoon hours for high temperature
            target_time = date.replace(hour=15, minute=0, second=0, microsecond=0)
            data = await self.get_historical_data(station, target_time)

            if not data or 'data' not in data:
                return None

            # Get max temperature from available data
            temps = []
            for interval in data['data']['timelines'][0]['intervals']:
                if 'temperature' in interval['values']:
                    temps.append(interval['values']['temperature'])

            if not temps:
                return None

            max_temp = max(temps)
            return TemperatureReading(
                value=Temperature(Decimal(str(max_temp))),
                timestamp=target_time,
                source=DataSource.TOMORROW_IO,
                unit=TemperatureUnit.FAHRENHEIT,
                station_id=station.station_id
            )

        except Exception as e:
            self.logger.error(f"Failed to get historical high: {e}")
            return None
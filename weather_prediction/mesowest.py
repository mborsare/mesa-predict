"""MesoWest API client with proper error handling and typing."""
from datetime import datetime
import logging
from typing import Optional, Sequence, Dict, Any
import aiohttp
from decimal import Decimal
import pytz

from .models import (
    TemperatureReading, StationMetadata, DataSource,
    Temperature, StationId, Latitude, Longitude,
    TemperatureUnit, DataSourceError, ValidationError
)
from .config import APIConfig

class MesoWestClient:
    """Async client for MesoWest API."""

    def __init__(self, config: APIConfig, logger: Optional[logging.Logger] = None):
        self.config = config
        self.logger = logger or logging.getLogger(__name__)
        self._session: Optional[aiohttp.ClientSession] = None

    async def __aenter__(self) -> 'MesoWestClient':
        timeout = aiohttp.ClientTimeout(total=self.config.timeout)
        self._session = aiohttp.ClientSession(timeout=timeout)
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb) -> None:
        if self._session:
            await self._session.close()
            self._session = None

    async def get_station_metadata(
        self,
        station_id: StationId
    ) -> StationMetadata:
        """Fetch station metadata."""
        if not self._session:
            raise DataSourceError("Client session not initialized")

        try:
            async with self._session.get(
                f"{self.config.base_url}/metadata",
                params={'stid': station_id, 'token': self.config.token}
            ) as response:
                response.raise_for_status()
                data = await response.json()

                if not data.get('STATION'):
                    raise ValidationError(f"No data found for station {station_id}")

                station = data['STATION'][0]
                return StationMetadata(
                    station_id=station_id,
                    latitude=Latitude(Decimal(str(station['LATITUDE']))),
                    longitude=Longitude(Decimal(str(station['LONGITUDE']))),
                    timezone=station.get('TIMEZONE', 'UTC')
                )

        except aiohttp.ClientError as e:
            raise DataSourceError(f"Failed to fetch station metadata: {e}")
        except (KeyError, ValueError) as e:
            raise ValidationError(f"Invalid station data format: {e}")

    async def get_temperature_readings(
        self,
        station_id: StationId,
        start: datetime,
        end: datetime
    ) -> Sequence[TemperatureReading]:
        """Fetch temperature readings for time range."""
        if not self._session:
            raise DataSourceError("Client session not initialized")

        if not start.tzinfo or not end.tzinfo:
            raise ValidationError("Timestamps must be timezone-aware")

        try:
            async with self._session.get(
                f"{self.config.base_url}/timeseries",
                params={
                    'stid': station_id,
                    'start': start.strftime('%Y%m%d%H%M'),
                    'end': end.strftime('%Y%m%d%H%M'),
                    'vars': 'air_temp,air_temp_high_6_hour',
                    'token': self.config.token,
                    'units': 'temp|F'
                }
            ) as response:
                response.raise_for_status()
                data = await response.json()
                self.logger.debug(f"Raw API response: {data}")  # Debug line
                return self._parse_temperature_data(data, station_id)

        except aiohttp.ClientError as e:
            raise DataSourceError(f"Failed to fetch temperature data: {e}")

    def _parse_temperature_data(
        self,
        data: Dict[str, Any],
        station_id: StationId
    ) -> Sequence[TemperatureReading]:
        """Parse API response into temperature readings."""
        try:
            if not data.get('STATION'):
                raise ValidationError("No station data in response")

            observations = data['STATION'][0].get('OBSERVATIONS', {})
            readings = []

            # Debug print
            self.logger.debug(f"Processing observations: {observations}")

            # Get all available timestamps
            timestamps = observations.get('date_time', [])
            regular_temps = observations.get('air_temp_set_1', [])
            six_hour_temps = observations.get('air_temp_high_6_hour_set_1', [])

            for i, dt in enumerate(timestamps):
                timestamp = datetime.strptime(
                    dt,
                    '%Y-%m-%dT%H:%M:%SZ'
                ).replace(tzinfo=pytz.UTC)

                # Regular temperature reading
                if i < len(regular_temps) and regular_temps[i] is not None:
                    readings.append(TemperatureReading(
                        value=Temperature(Decimal(str(regular_temps[i]))),
                        timestamp=timestamp,
                        source=DataSource.REGULAR,
                        unit=TemperatureUnit.FAHRENHEIT,
                        station_id=station_id
                    ))

                # 6-hour high temperature reading
                if i < len(six_hour_temps) and six_hour_temps[i] is not None:
                    readings.append(TemperatureReading(
                        value=Temperature(Decimal(str(six_hour_temps[i]))),
                        timestamp=timestamp,
                        source=DataSource.SIX_HOUR,
                        unit=TemperatureUnit.FAHRENHEIT,
                        station_id=station_id
                    ))

            self.logger.debug(f"Parsed {len(readings)} temperature readings")
            return readings

        except (KeyError, ValueError, IndexError) as e:
            self.logger.error(f"Error parsing temperature data: {e}")
            self.logger.debug(f"Problematic data: {data}")
            raise ValidationError(f"Error parsing temperature data: {e}")
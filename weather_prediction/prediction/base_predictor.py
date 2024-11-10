"""Base predictor class with core prediction functionality."""
from datetime import datetime, timedelta
import logging
from typing import Optional, Sequence, Dict, Tuple, Any
from decimal import Decimal
import pytz

from ..models import (
    TemperatureReading, StationMetadata, DataSource,
    Temperature, WeatherError, TemperaturePrediction, TemperatureUnit
)
from ..mesowest import MesoWestClient
from ..tomorrow import TomorrowClient
from ..analyzer import TemperatureAnalyzer

class BasePredictor:
    """Base class for temperature prediction."""

    def __init__(
        self,
        mesowest: MesoWestClient,
        tomorrow: TomorrowClient,
        analyzer: TemperatureAnalyzer,
        logger: Optional[logging.Logger] = None
    ):
        self.mesowest = mesowest
        self.tomorrow = tomorrow
        self.analyzer = analyzer
        self.logger = logger or logging.getLogger(__name__)

    async def _get_historical_data(
        self,
        station: StationMetadata,
        target_date: Optional[datetime] = None
    ) -> Tuple[Sequence[TemperatureReading], Optional[Dict[str, Any]]]:
        """Fetch historical temperature data from both sources."""
        try:
            tz = pytz.timezone(station.timezone)

            # For MesoWest, we can get 24 hours of data
            if target_date:
                end = target_date
                start = end - timedelta(days=1)
            else:
                end = datetime.now(tz)
                start = end - timedelta(days=1)

            mesowest_readings = await self.mesowest.get_temperature_readings(
                station.station_id,
                start,
                end
            )

            # For Tomorrow.io, we're limited to 6 hours of historical data
            tomorrow_data = None
            if hasattr(self.tomorrow, 'get_historical_data'):
                # Use afternoon hours for better high temperature capture
                query_time = end.replace(hour=15, minute=0, second=0, microsecond=0)
                tomorrow_data = await self.tomorrow.get_historical_data(
                    station,
                    query_time
                )

            self.logger.debug(
                f"Retrieved {len(mesowest_readings)} MesoWest readings and "
                f"{'6-hour historical' if tomorrow_data else 'no'} Tomorrow.io data "
                f"for {station.station_id}"
            )

            return mesowest_readings, tomorrow_data

        except Exception as e:
            self.logger.error(f"Failed to get historical data: {e}")
            return [], None

    async def _get_current_high(
        self,
        station: StationMetadata,
        target_date: Optional[datetime] = None
    ) -> Optional[TemperatureReading]:
        """Get current day's high temperature."""
        try:
            tz = pytz.timezone(station.timezone)
            date = target_date if target_date else datetime.now(tz)
            start = date.replace(hour=0, minute=0, second=0, microsecond=0)
            end = date + timedelta(days=1) if target_date else datetime.now(tz)

            readings = await self.mesowest.get_temperature_readings(
                station.station_id,
                start,
                end
            )

            return self.analyzer.get_todays_high(readings)

        except Exception as e:
            self.logger.error(f"Failed to get current high: {e}")
            return None

    def _log_prediction_details(self, prediction: TemperaturePrediction) -> None:
        """Log detailed prediction information."""
        self.logger.debug(
            f"\nTemperature Prediction:"
            f"\nStation: {prediction.station.station_id}"
            f"\nPredicted High: {prediction.temperature.value}°F"
            f"\nConfidence: {prediction.confidence}%"
            f"\nData Sources: {', '.join(s.name for s in prediction.data_sources)}"
        )
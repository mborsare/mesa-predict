"""Main script for temperature prediction."""
import asyncio
import logging
import sys
from typing import Optional, Tuple
from datetime import datetime, time
import pytz
from decimal import Decimal

from .models import (
    StationId, StationMetadata, Latitude, Longitude,
    WeatherError, ValidationError, DataSource
)
from .config import Config, setup_logging
from .mesowest import MesoWestClient
from .tomorrow import TomorrowClient
from .analyzer import TemperatureAnalyzer
from .predictor import TemperaturePredictor

def format_temperature_output(
    prediction,
    station_id: str,
    current_high_data: Optional[Tuple[float, datetime, DataSource]] = None
) -> str:
    """Format the temperature prediction output with current observations."""
    now = datetime.now(pytz.timezone('America/New_York'))

    # Access the value attribute of the TemperatureReading
    predicted_temp = float(prediction.temperature.value)

    # Convert DataSource enums to strings
    data_sources_str = [str(source) for source in prediction.data_sources]

    output = f"""
Temperature Prediction:
Station: {station_id}
Predicted High: {predicted_temp:.1f}°F
Confidence: {prediction.confidence:.1f}%
Data Sources: {', '.join(data_sources_str)}"""

    if current_high_data is not None:
        current_high, high_timestamp, data_source = current_high_data
        # Convert timestamp to local time
        local_high_time = high_timestamp.astimezone(pytz.timezone('America/New_York'))
        output += f"""

Today's high as of {now.strftime('%I:%M %p')}:
Observed high: {current_high:.1f}°F ({round(current_high)}°F rounded) at {local_high_time.strftime('%I:%M %p')} ({str(data_source)})"""

    return output

async def get_station_metadata(
    station_id: str,
    mesowest: MesoWestClient
) -> StationMetadata:
    """Validate and fetch station metadata."""
    try:
        if len(station_id) != 4:
            raise ValidationError("Station ID must be 4 characters")
        return await mesowest.get_station_metadata(StationId(station_id.upper()))
    except Exception as e:
        logging.error(f"Failed to get station metadata: {e}")
        raise

async def get_todays_high(
    mesowest: MesoWestClient,
    station_id: StationId
) -> Optional[Tuple[float, datetime, DataSource]]:
    """Get today's highest observed temperature and when it occurred."""
    try:
        # Get today's start and end times in UTC
        local_tz = pytz.timezone('America/New_York')  # Adjust as needed
        now = datetime.now(local_tz)
        start = datetime.combine(now.date(), time.min).astimezone(pytz.UTC)
        end = datetime.now(pytz.UTC)

        # Fetch temperature readings
        readings = await mesowest.get_temperature_readings(station_id, start, end)

        # Separate regular and 6-hour readings
        regular_readings = [r for r in readings if r.source == DataSource.REGULAR]
        six_hour_readings = [r for r in readings if r.source == DataSource.SIX_HOUR]

        # Find highest from regular readings
        max_regular = max(
            regular_readings,
            key=lambda x: float(x.value),
            default=None
        )

        # Find highest from 6-hour readings
        max_six_hour = max(
            six_hour_readings,
            key=lambda x: float(x.value),
            default=None
        )

        # Compare the two maximums
        if max_regular and max_six_hour:
            regular_value = float(max_regular.value)
            six_hour_value = float(max_six_hour.value)

            if six_hour_value > regular_value:
                return (six_hour_value, max_six_hour.timestamp, DataSource.SIX_HOUR)
            return (regular_value, max_regular.timestamp, DataSource.REGULAR)
        elif max_regular:
            return (float(max_regular.value), max_regular.timestamp, DataSource.REGULAR)
        elif max_six_hour:
            return (float(max_six_hour.value), max_six_hour.timestamp, DataSource.SIX_HOUR)

        return None

    except Exception as e:
        logging.warning(f"Failed to get current high temperature: {e}")
        return None

async def predict_temperature(station_id: str) -> None:
    """Main prediction routine with proper error handling."""
    config = Config.load()
    logger = setup_logging(config)
    try:
        logger.info(f"Processing weather data for station: {station_id}")
        async with MesoWestClient(config.mesowest, logger) as mesowest:
            async with TomorrowClient(config.tomorrow_io, logger) as tomorrow:
                # Get and validate station metadata
                station = await get_station_metadata(station_id, mesowest)
                logger.info(
                    f"Station coordinates: ({station.latitude}, {station.longitude})"
                )

                # Initialize analysis components
                analyzer = TemperatureAnalyzer(logger)
                predictor = TemperaturePredictor(
                    mesowest=mesowest,
                    tomorrow=tomorrow,
                    analyzer=analyzer,
                    logger=logger
                )

                # Get prediction and current high
                prediction = await predictor.predict_todays_high(station)
                current_high_data = await get_todays_high(mesowest, station.station_id)

                # Log the formatted output
                logger.info(format_temperature_output(
                    prediction=prediction,
                    station_id=station_id.upper(),
                    current_high_data=current_high_data
                ))

                if prediction.confidence < 70:
                    logger.warning(
                        "Low confidence prediction! "
                        f"Confidence score: {prediction.confidence}%"
                    )
    except WeatherError as e:
        logger.error(f"Weather prediction error: {e}")
        sys.exit(1)
    except Exception as e:
        logger.exception("Unexpected error occurred")
        sys.exit(1)

def main() -> None:
    """Entry point with async support."""
    try:
        station_id = input("Enter your station ID (e.g., KMIA): ").strip()
        asyncio.run(predict_temperature(station_id))
    except KeyboardInterrupt:
        print("\nOperation cancelled by user")
        sys.exit(0)

if __name__ == "__main__":
    main()
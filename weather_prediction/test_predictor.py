"""Test suite for enhanced temperature predictor."""
import asyncio
import logging
from datetime import datetime, timedelta
from decimal import Decimal
import pytz
from pathlib import Path

from weather_prediction.models import (
    StationMetadata, StationId, Latitude, Longitude,
    TemperatureReading, Temperature, TemperatureUnit,
    DataSource
)
from weather_prediction.config import Config, setup_logging
from weather_prediction.mesowest import MesoWestClient
from weather_prediction.tomorrow import TomorrowClient
from weather_prediction.analyzer import TemperatureAnalyzer
from weather_prediction.predictor import TemperaturePredictor

async def test_prediction(station_id: str = "KNYC") -> None:
    """Test the enhanced predictor with real data."""
    config = Config.load()
    logger = setup_logging(config)
    logger.setLevel(logging.INFO)

    try:
        async with MesoWestClient(config.mesowest, logger) as mesowest:
            async with TomorrowClient(config.tomorrow_io, logger) as tomorrow:
                # Get station metadata
                station = await mesowest.get_station_metadata(
                    StationId(station_id.upper())
                )

                logger.info(f"Testing predictor for station {station_id}")
                logger.info(f"Location: {station.latitude}, {station.longitude}")

                # Initialize components
                analyzer = TemperatureAnalyzer(logger)
                predictor = TemperaturePredictor(
                    mesowest=mesowest,
                    tomorrow=tomorrow,
                    analyzer=analyzer,
                    logger=logger
                )

                # Get historical data for verification
                end = datetime.now(pytz.UTC)
                start = end - timedelta(days=1)
                actual_readings = await mesowest.get_temperature_readings(
                    station.station_id, start, end
                )
                if actual_readings:
                    actual_high = max(float(r.value) for r in actual_readings)
                    logger.info(f"Actual high from past 24 hours: {actual_high:.1f}°F")

                # Make prediction
                prediction = await predictor.predict_todays_high(station)

                # Analyze results
                logger.info("\nPrediction Analysis:")
                logger.info(f"Predicted High: {prediction.temperature.value}°F")
                logger.info(f"Confidence: {prediction.confidence}%")
                logger.info(f"Data Sources Used: {', '.join(s.name for s in prediction.data_sources)}")

                if actual_readings:
                    error = float(prediction.temperature.value) - actual_high
                    logger.info(f"Error: {error:+.1f}°F")

                # If ML model was used, show feature importance
                if (DataSource.ML_MODEL in prediction.data_sources and
                    hasattr(predictor.model, 'feature_importances_')):
                    importance = dict(zip(predictor.FEATURE_COLUMNS,
                                       predictor.model.feature_importances_))
                    logger.info("\nML Model Feature Importance:")
                    for feature, score in sorted(importance.items(),
                                              key=lambda x: x[1], reverse=True):
                        logger.info(f"{feature}: {score:.3f}")

    except Exception as e:
        logger.exception("Test failed")
        raise

def main():
    """CLI entry point."""
    try:
        station_id = input("Enter station ID (e.g., KNYC): ").strip().upper()
        asyncio.run(test_prediction(station_id))

    except KeyboardInterrupt:
        print("\nTest cancelled by user")
    except Exception as e:
        print(f"\nTest failed: {e}")

if __name__ == "__main__":
    main()
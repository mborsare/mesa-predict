"""Backtesting framework with enhanced data sources."""
import asyncio
import logging
from datetime import datetime, timedelta
from decimal import Decimal
from typing import Optional, Sequence, List, Dict, Tuple, Any
import pandas as pd
import numpy as np
from dataclasses import dataclass
import pytz
from pathlib import Path
import aiohttp

from .models import (
    StationId, StationMetadata, TemperatureReading,
    TemperaturePrediction, WeatherError, DataSource,
    Temperature, TemperatureUnit, DataSourceError
)
from .config import Config, setup_logging, APIConfig
from .mesowest import MesoWestClient
from .prediction import TemperaturePredictor
from .analyzer import TemperatureAnalyzer

@dataclass
class BacktestResult:
    """Single backtest result."""
    date: datetime
    actual_high: Optional[Tuple[float, datetime, DataSource]]
    prediction: Optional[float]
    confidence: Optional[float]
    error: Optional[float]
    data_sources: List[DataSource]

@dataclass
class BacktestSummary:
    """Summary of backtest results."""
    station_id: StationId
    start_date: datetime
    end_date: datetime
    total_days: int
    valid_days: int
    mae: Decimal
    rmse: Decimal
    bias: Decimal
    confidence_correlation: Decimal
    ml_contribution: Decimal
    tomorrow_contribution: Decimal
    results: List[BacktestResult]

class MockTomorrowClient:
    """Mock client for backtesting that doesn't make real API calls."""

    def __init__(self, config: APIConfig, logger: Optional[logging.Logger] = None):
        self.config = config
        self.logger = logger or logging.getLogger(__name__)

    async def __aenter__(self) -> 'MockTomorrowClient':
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb) -> None:
        pass

    async def get_historical_data(
        self,
        station: StationMetadata,
        target_date: datetime
    ) -> Optional[Dict[str, Any]]:
        """Return None for historical data during backtesting."""
        return None

    async def get_daily_high_prediction(
        self,
        station: StationMetadata,
        date: datetime
    ) -> Optional[TemperatureReading]:
        """Return None for predictions during backtesting."""
        return None

    async def get_forecast(self, *args, **kwargs):
        """Return empty list during backtesting."""
        return []

async def get_daily_high(
    mesowest: MesoWestClient,
    station_id: StationId,
    date: datetime
) -> Optional[Tuple[float, datetime, DataSource]]:
    """Get highest temperature and when it occurred for a specific date."""
    try:
        start = date.replace(hour=0, minute=0, second=0, microsecond=0)
        end = start + timedelta(days=1)

        readings = await mesowest.get_temperature_readings(station_id, start, end)

        regular_readings = [r for r in readings if r.source == DataSource.REGULAR]
        six_hour_readings = [r for r in readings if r.source == DataSource.SIX_HOUR]

        max_regular = max(regular_readings, key=lambda x: float(x.value), default=None)
        max_six_hour = max(six_hour_readings, key=lambda x: float(x.value), default=None)

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
        logging.warning(f"Failed to get high temperature for {date.date()}: {e}")
        return None

async def run_backtest(station_id: str, days: int = 21) -> None:
    """Main backtest routine."""
    config = Config.load()
    logger = setup_logging(config)

    try:
        async with MesoWestClient(config.mesowest, logger) as mesowest:
            async with MockTomorrowClient(config.tomorrow_io, logger) as tomorrow:
                # Get station metadata
                station = await mesowest.get_station_metadata(StationId(station_id.upper()))
                logger.info(f"Starting backtest for {station_id} over {days} days")

                # Initialize components
                analyzer = TemperatureAnalyzer(logger)
                predictor = TemperaturePredictor(mesowest, tomorrow, analyzer, logger)

                # Process each day
                results = []
                end_date = datetime.now(pytz.UTC) - timedelta(days=1)

                for day_offset in range(days):
                    target_date = end_date - timedelta(days=day_offset)

                    try:
                        # Get actual high
                        actual_high = await get_daily_high(mesowest, station.station_id, target_date)

                        # Get prediction for specific target date
                        prediction = await predictor.predict_todays_high(station, target_date=target_date)

                        # Calculate error
                        error = None
                        if actual_high and prediction:
                            error = float(prediction.temperature.value) - actual_high[0]

                        # Store result
                        result = BacktestResult(
                            date=target_date,
                            actual_high=actual_high,
                            prediction=float(prediction.temperature.value) if prediction else None,
                            confidence=float(prediction.confidence) if prediction else None,
                            error=error,
                            data_sources=prediction.data_sources if prediction else []
                        )
                        results.append(result)

                        # Log result
                        if actual_high and prediction:
                            logger.info(
                                f"\nDate: {target_date.date()}"
                                f"\nActual: {actual_high[0]:.1f}°F at {actual_high[1].astimezone(pytz.timezone('America/New_York')):%I:%M %p} ({actual_high[2]})"
                                f"\nPredicted: {prediction.temperature.value:.1f}°F"
                                f"\nError: {error:+.1f}°F"
                                f"\nConfidence: {prediction.confidence:.1f}%"
                                f"\nData Sources: {', '.join(str(s) for s in prediction.data_sources)}"
                            )

                    except Exception as e:
                        logger.error(f"Failed to backtest {target_date.date()}: {e}")

                # Calculate summary statistics if we have valid results
                valid_results = [r for r in results if r.error is not None]

                if valid_results:
                    errors = [r.error for r in valid_results]
                    confidences = [r.confidence for r in valid_results]
                    accuracies = [1 - abs(e)/100 for e in errors]

                    # Calculate ML and Tomorrow.io usage
                    total = len(valid_results)
                    ml_used = sum(1 for r in valid_results if DataSource.ML_MODEL in r.data_sources)
                    tomorrow_used = sum(1 for r in valid_results if DataSource.TOMORROW_IO in r.data_sources)

                    # Create summary
                    summary = BacktestSummary(
                        station_id=station.station_id,
                        start_date=end_date - timedelta(days=days),
                        end_date=end_date,
                        total_days=len(results),
                        valid_days=len(valid_results),
                        mae=Decimal(str(np.mean(np.abs(errors)))).quantize(Decimal('0.1')),
                        rmse=Decimal(str(np.sqrt(np.mean(np.square(errors))))).quantize(Decimal('0.1')),
                        bias=Decimal(str(np.mean(errors))).quantize(Decimal('0.1')),
                        confidence_correlation=Decimal(str(np.corrcoef(confidences, accuracies)[0,1])).quantize(Decimal('0.01')),
                        ml_contribution=Decimal(str(ml_used / total * 100)).quantize(Decimal('0.1')),
                        tomorrow_contribution=Decimal(str(tomorrow_used / total * 100)).quantize(Decimal('0.1')),
                        results=results
                    )

                    # Save results
                    output_dir = Path('backtest_results')
                    output_dir.mkdir(exist_ok=True)

                    # Create DataFrame for CSV output
                    df = pd.DataFrame([
                        {
                            'Date': r.date.date(),
                            'Actual': r.actual_high[0] if r.actual_high else None,
                            'Actual_Time': r.actual_high[1] if r.actual_high else None,
                            'Actual_Source': str(r.actual_high[2]) if r.actual_high else None,
                            'Predicted': r.prediction,
                            'Error': r.error,
                            'Confidence': r.confidence,
                            'Data_Sources': ', '.join(str(s) for s in r.data_sources),
                            'Used_Tomorrow': DataSource.TOMORROW_IO in r.data_sources,
                            'Used_ML': DataSource.ML_MODEL in r.data_sources
                        }
                        for r in results
                    ])

                    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
                    df.to_csv(output_dir / f'backtest_{station_id}_{timestamp}.csv', index=False)

                    # Log summary
                    logger.info(f"""
\nBacktest Summary for {station_id}
{'='*50}
Period: {summary.start_date.date()} to {summary.end_date.date()}
Total Days: {summary.total_days}
Valid Days: {summary.valid_days}
Mean Absolute Error: {summary.mae}°F
Root Mean Square Error: {summary.rmse}°F
Bias: {summary.bias}°F
Confidence Correlation: {summary.confidence_correlation}
Tomorrow.io Used: {summary.tomorrow_contribution}% of predictions
ML Model Used: {summary.ml_contribution}% of predictions
                    """)

    except Exception as e:
        logger.exception("Backtest failed")
        raise

def main():
    """CLI entry point."""
    try:
        station_id = input("Enter station ID (e.g., KMIA): ").strip().upper()
        days = int(input("Enter number of days to backtest (default 21): ") or 21)

        asyncio.run(run_backtest(station_id, days))

    except KeyboardInterrupt:
        print("\nBacktest cancelled by user")
    except ValueError as e:
        print(f"\nInvalid input: {e}")
    except Exception as e:
        print(f"\nBacktest failed: {e}")

if __name__ == "__main__":
    main()
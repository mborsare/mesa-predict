"""Main temperature predictor implementation."""
from datetime import datetime
from typing import Optional, Dict, List
from decimal import Decimal
import pytz

from ..models import (
    StationMetadata, DataSource, Temperature,
    TemperatureReading, TemperaturePrediction,
    WeatherError, TemperatureUnit
)
from .base_predictor import BasePredictor
from .trend_analyzer import TrendAnalyzer
from .prediction_weights import PredictionWeights

class TemperaturePredictor(BasePredictor):
    """Predicts temperatures using multiple data sources with weighted analysis."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.trend_analyzer = TrendAnalyzer(self.logger)
        self.weights_handler = PredictionWeights()

    async def predict_todays_high(
        self,
        station: StationMetadata,
        target_date: Optional[datetime] = None
    ) -> TemperaturePrediction:
        """Predict today's high temperature using all available sources."""
        try:
            # Gather historical data (respects 6-hour Tomorrow.io constraint)
            historical_readings, tomorrow_data = await self._get_historical_data(
                station, target_date
            )

            # Analyze temperature trends
            trend_change, trend_confidence = self.trend_analyzer.analyze_trends(
                historical_readings, tomorrow_data
            )

            # Get predictions from various sources
            current_high = await self._get_current_high(station, target_date)

            # Only get forecast for real-time predictions
            tomorrow_high = None
            if not target_date:
                tomorrow_high = await self.tomorrow.get_daily_high_prediction(
                    station, datetime.now(pytz.UTC)
                )

            if not any([historical_readings, current_high, tomorrow_high]):
                raise WeatherError("No data available from any source")

            # Calculate historical pattern from MesoWest data
            historical_mean = None
            if historical_readings:
                historical_mean, _ = self.analyzer.analyze_temperature_pattern(
                    historical_readings
                )

            # Get Tomorrow.io historical high if in backtest mode
            tomorrow_historical = None
            if target_date:
                # Use afternoon target time for better high temp capture
                target_time = target_date.replace(hour=15, minute=0, second=0, microsecond=0)
                tomorrow_historical = await self.tomorrow.get_daily_high_prediction(
                    station, target_time
                )

            # Combine all available predictions
            prediction = self._combine_predictions(
                station=station,
                current_high=current_high,
                forecast=tomorrow_high,
                historical_mean=historical_mean,
                historical_readings=historical_readings,
                tomorrow_historical=tomorrow_historical,
                trend_change=trend_change,
                trend_confidence=trend_confidence,
                is_backtest=target_date is not None
            )

            self._log_prediction_details(prediction)
            return prediction

        except Exception as e:
            self.logger.exception("Failed to predict temperature")
            raise WeatherError(f"Prediction failed: {str(e)}") from e

    def _combine_predictions(
        self,
        station: StationMetadata,
        current_high: Optional[TemperatureReading],
        forecast: Optional[TemperatureReading],
        historical_mean: Optional[Decimal],
        historical_readings: List[TemperatureReading],
        tomorrow_historical: Optional[TemperatureReading],
        trend_change: float,
        trend_confidence: float,
        is_backtest: bool = False
    ) -> TemperaturePrediction:
        """Combine predictions using weighted analysis."""
        valid_sources: Dict[DataSource, Decimal] = {}
        weights = self.weights_handler.get_weights(is_backtest, trend_change, trend_confidence)

        # Collect valid sources based on context
        if is_backtest:
            if tomorrow_historical:
                valid_sources[DataSource.TOMORROW_IO] = tomorrow_historical.value
            if historical_mean is not None:
                valid_sources[DataSource.HISTORICAL] = historical_mean
            if current_high and current_high.source == DataSource.SIX_HOUR:
                valid_sources[DataSource.SIX_HOUR] = current_high.value
        else:
            if forecast:
                valid_sources[DataSource.TOMORROW_IO] = forecast.value
            if current_high:
                valid_sources[current_high.source] = current_high.value
            if historical_mean is not None:
                valid_sources[DataSource.HISTORICAL] = historical_mean

        if not valid_sources:
            raise WeatherError("No valid temperature sources available")

        # Calculate weighted prediction
        total_weight = sum(weights.get(source, Decimal('0'))
                         for source in valid_sources.keys())
        weighted_sum = sum(value * weights.get(source, Decimal('0'))
                         for source, value in valid_sources.items())
        predicted_temp = weighted_sum / total_weight

        # Create prediction object
        prediction = TemperatureReading(
            value=Temperature(predicted_temp.quantize(Decimal('0.1'))),
            timestamp=datetime.now(pytz.UTC),
            source=max(valid_sources.keys(), key=lambda s: weights.get(s, Decimal('0'))),
            unit=TemperatureUnit.FAHRENHEIT,
            station_id=station.station_id
        )

        # Calculate confidence
        confidence = self.trend_analyzer.calculate_confidence(
            prediction,
            historical_readings,
            trend_confidence,
            temp_change=abs(float(predicted_temp) - float(historical_mean) if historical_mean else 0)
        )

        # Create and return final prediction
        return TemperaturePrediction(
            temperature=prediction,
            confidence=confidence,
            data_sources=list(valid_sources.keys()),
            station=station
        )

    def _log_prediction_details(self, prediction: TemperaturePrediction) -> None:
        """Log detailed prediction information."""
        self.logger.debug(
            f"\nTemperature Prediction:"
            f"\nStation: {prediction.station.station_id}"
            f"\nPredicted High: {prediction.temperature.value}°F"
            f"\nConfidence: {prediction.confidence}%"
            f"\nData Sources: {', '.join(s.name for s in prediction.data_sources)}"
        )
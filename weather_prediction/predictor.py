"""Temperature prediction with weighted multi-source analysis and machine learning."""
from datetime import datetime, timedelta
import logging
from typing import Optional, Sequence, Dict, List, Tuple, Any
from decimal import Decimal
import pytz
import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.preprocessing import StandardScaler

from .models import (
    TemperatureReading, StationMetadata, DataSource,
    Temperature, WeatherError, ValidationError,
    TemperaturePrediction, TemperatureUnit
)
from .mesowest import MesoWestClient
from .tomorrow import TomorrowClient
from .analyzer import TemperatureAnalyzer

class TemperaturePredictor:
    """Predicts temperatures using multiple data sources with weighted analysis."""

    FEATURE_COLUMNS = [
        'air_temp', 'relative_humidity', 'dew_point_temperature',
        'wind_speed', 'wind_direction', 'wind_gust',
        'sea_level_pressure', 'visibility',
        'air_temp_high_6_hour', 'air_temp_low_6_hour',
        'hour', 'month'
    ]

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

        # Regular prediction weights
        self._source_weights = {
            DataSource.TOMORROW_IO: Decimal('0.3'),    # Future forecast
            DataSource.REGULAR: Decimal('0.2'),        # Current conditions
            DataSource.SIX_HOUR: Decimal('0.2'),       # Recent history
            DataSource.HISTORICAL: Decimal('0.2'),     # Long-term pattern
            DataSource.ML_MODEL: Decimal('0.1')        # Machine learning prediction
        }

        # Backtesting weights
        self._backtest_weights = {
            DataSource.TOMORROW_IO: Decimal('0.2'),    # Limited historical data
            DataSource.HISTORICAL: Decimal('0.5'),     # More weight on historical
            DataSource.SIX_HOUR: Decimal('0.3')        # Recent observations
        }

        # Initialize ML components
        self.model = xgb.XGBRegressor(
            n_estimators=100,
            max_depth=6,
            learning_rate=0.1,
            objective='reg:squarederror',
            random_state=42
        )
        self.scaler = StandardScaler()
        self._model_trained = False

    async def _get_historical_data(
        self,
        station: StationMetadata,
        target_date: Optional[datetime] = None
    ) -> Tuple[Sequence[TemperatureReading], Optional[Dict[str, Any]]]:
        """Fetch historical temperature data from both sources."""
        try:
            tz = pytz.timezone(station.timezone)
            if target_date:
                end = target_date
                start = end - timedelta(days=1)  # Get 24 hours for MesoWest
            else:
                end = datetime.now(tz)
                start = end - timedelta(days=1)

            # Get MesoWest readings
            mesowest_readings = await self.mesowest.get_temperature_readings(
                station.station_id,
                start,
                end
            )

            # Get Tomorrow.io historical data if available
            tomorrow_data = None
            if hasattr(self.tomorrow, 'get_historical_data'):
                tomorrow_data = await self.tomorrow.get_historical_data(
                    station,
                    end if target_date else datetime.now(pytz.UTC)
                )

            self.logger.debug(
                f"Retrieved {len(mesowest_readings)} MesoWest readings and "
                f"{'historical' if tomorrow_data else 'no'} Tomorrow.io data "
                f"for {station.station_id}"
            )

            return mesowest_readings, tomorrow_data

        except Exception as e:
            self.logger.error(f"Failed to get historical data: {e}")
            return [], None

    async def _get_ml_prediction(
        self,
        station: StationMetadata,
        target_date: Optional[datetime] = None
    ) -> Optional[TemperatureReading]:
        """Get prediction from trained ML model."""
        if not self._model_trained:
            return None

        try:
            # Get recent data for features
            readings, _ = await self._get_historical_data(station, target_date)
            if not readings:
                return None

            # Prepare features
            features = self._prepare_ml_features(readings)
            if features.empty:
                return None

            # Make prediction
            X = features[self.FEATURE_COLUMNS].values
            X_scaled = self.scaler.transform(X)
            prediction = self.model.predict(X_scaled)[-1]  # Get last prediction

            return TemperatureReading(
                value=Temperature(Decimal(str(prediction))),
                timestamp=target_date or datetime.now(pytz.UTC),
                source=DataSource.ML_MODEL,
                unit=TemperatureUnit.FAHRENHEIT,
                station_id=station.station_id
            )

        except Exception as e:
            self.logger.error(f"ML prediction failed: {e}")
            return None

    def _prepare_ml_features(self, readings: Sequence[TemperatureReading]) -> pd.DataFrame:
        """Prepare features for ML model."""
        try:
            data = []
            for reading in readings:
                row = {
                    'timestamp': reading.timestamp,
                    'air_temp': float(reading.value),
                    'hour': reading.timestamp.hour,
                    'month': reading.timestamp.month
                }
                data.append(row)

            df = pd.DataFrame(data)
            df.set_index('timestamp', inplace=True)
            df.sort_index(inplace=True)

            # Add missing columns with defaults
            for col in self.FEATURE_COLUMNS:
                if col not in df.columns:
                    df[col] = 0

            return df

        except Exception as e:
            self.logger.error(f"Failed to prepare ML features: {e}")
            return pd.DataFrame()
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

    def _analyze_temperature_trend(
        self,
        mesowest_readings: Sequence[TemperatureReading],
        tomorrow_data: Optional[Dict[str, Any]] = None
    ) -> Tuple[float, float]:
        """Analyze temperature trends from both data sources."""
        try:
            # Analyze MesoWest trends using most recent data
            mesowest_temps = [float(r.value) for r in mesowest_readings[-6:]]  # Last 6 hours
            if len(mesowest_temps) >= 3:
                recent_mesowest = np.mean(mesowest_temps[-2:])
                older_mesowest = np.mean(mesowest_temps[:-2])
                mesowest_change = recent_mesowest - older_mesowest
            else:
                mesowest_change = 0

            # Analyze Tomorrow.io trends if available
            tomorrow_change = 0
            if tomorrow_data and 'data' in tomorrow_data:
                temps = []
                for interval in tomorrow_data['data']['timelines'][0]['intervals']:
                    if 'temperature' in interval['values']:
                        temps.append(interval['values']['temperature'])
                if len(temps) >= 3:
                    recent_tomorrow = np.mean(temps[-2:])
                    older_tomorrow = np.mean(temps[:-2])
                    tomorrow_change = recent_tomorrow - older_tomorrow

            # Combine trends with adjusted weights
            combined_change = (mesowest_change * 0.7) + (tomorrow_change * 0.3)
            trend_confidence = min(
                90,
                40 + (50 * (1 - abs(mesowest_change - tomorrow_change) / 5))
            ) if tomorrow_data else 40

            return combined_change, trend_confidence

        except Exception as e:
            self.logger.warning(f"Error analyzing temperature trend: {e}")
            return 0, 40

    async def predict_todays_high(
        self,
        station: StationMetadata,
        target_date: Optional[datetime] = None
    ) -> TemperaturePrediction:
        """Predict today's high temperature using all available sources."""
        try:
            # Gather data from all sources
            historical_readings, tomorrow_data = await self._get_historical_data(
                station, target_date
            )

            # Analyze temperature trends
            trend_change, trend_confidence = self._analyze_temperature_trend(
                historical_readings, tomorrow_data
            )

            # Get current conditions and tomorrow forecasts
            current_high = await self._get_current_high(station, target_date)
            tomorrow_high = None
            if not target_date:  # Only get forecast for real-time predictions
                tomorrow_high = await self.tomorrow.get_daily_high_prediction(station, datetime.now(pytz.UTC))
            ml_prediction = await self._get_ml_prediction(station, target_date)

            if not any([historical_readings, current_high, tomorrow_high, ml_prediction]):
                raise WeatherError("No data available from any source")

            # Calculate historical pattern
            historical_mean = None
            if historical_readings:
                historical_mean, _ = self.analyzer.analyze_temperature_pattern(
                    historical_readings
                )

            # Get Tomorrow.io historical high if in backtest mode
            tomorrow_historical = None
            if target_date:
                tomorrow_historical = await self.tomorrow.get_daily_high_prediction(station, target_date)

            # Combine predictions with weights
            prediction = self._combine_predictions(
                station=station,
                current_high=current_high,
                forecast=tomorrow_high,
                historical_mean=historical_mean,
                historical_readings=historical_readings,
                ml_prediction=ml_prediction,
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
        historical_readings: Sequence[TemperatureReading],
        ml_prediction: Optional[TemperatureReading],
        tomorrow_historical: Optional[TemperatureReading],
        trend_change: float,
        trend_confidence: float,
        is_backtest: bool = False
    ) -> TemperaturePrediction:
        """Combine predictions using dynamic weights based on trends."""
        valid_sources: Dict[DataSource, Decimal] = {}
        weights = self._backtest_weights if is_backtest else self._source_weights

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
            if ml_prediction:
                valid_sources[DataSource.ML_MODEL] = ml_prediction.value

        if not valid_sources:
            raise WeatherError("No valid temperature sources available")

        # Calculate weighted average
        total_weight = sum(weights.get(source, Decimal('0'))
                         for source in valid_sources.keys())

        weighted_sum = sum(
            value * weights.get(source, Decimal('0'))
            for source, value in valid_sources.items()
        )

        predicted_temp = weighted_sum / total_weight

        # Create prediction reading
        prediction = TemperatureReading(
            value=Temperature(predicted_temp.quantize(Decimal('0.1'))),
            timestamp=datetime.now(pytz.UTC),
            source=max(valid_sources.keys(), key=lambda s: weights.get(s, Decimal('0'))),
            unit=TemperatureUnit.FAHRENHEIT,
            station_id=station.station_id
        )

        # Calculate confidence
        confidence = self._calculate_confidence(
            prediction,
            historical_readings,
            trend_confidence,
            temp_change=abs(float(predicted_temp) - float(historical_mean) if historical_mean else 0)
        )

        return TemperaturePrediction(
            temperature=prediction,
            confidence=confidence,
            data_sources=list(valid_sources.keys()),
            station=station
        )

    def _calculate_confidence(
        self,
        prediction: TemperatureReading,
        historical_readings: Sequence[TemperatureReading],
        trend_confidence: float,
        temp_change: float = 0
    ) -> Decimal:
        """Calculate confidence score based on all available data."""
        if not historical_readings:
            return Decimal('50.0')

        try:
            # Calculate temperature variability
            temps = [float(r.value) for r in historical_readings]
            temp_std = np.std(temps)

            # Base confidence starts at 85% and decreases with variability
            base_confidence = 85 - (temp_std * 1.5)

            # Adjust for temperature change rate
            if temp_change > 0:
                base_confidence -= (temp_change * 1.2)

            # Incorporate trend confidence
            base_confidence = (base_confidence * 0.7) + (trend_confidence * 0.3)

            # Adjust for data recency
            recent_readings = [r for r in historical_readings
                             if (datetime.now(r.timestamp.tzinfo) - r.timestamp).days <= 2]
            if not recent_readings:
                base_confidence -= 10

            # Ensure confidence stays within reasonable bounds
            confidence = max(min(base_confidence, 90), 50)

            return Decimal(str(confidence)).quantize(Decimal('0.1'))

        except Exception as e:
            self.logger.warning(f"Error calculating confidence: {e}")
            return Decimal('50.0')

    def _log_prediction_details(self, prediction: TemperaturePrediction) -> None:
        """Log detailed prediction information at debug level."""
        self.logger.debug(
            f"\nTemperature Prediction:"
            f"\nStation: {prediction.station.station_id}"
            f"\nPredicted High: {prediction.temperature.value}°F"
            f"\nConfidence: {prediction.confidence}%"
            f"\nData Sources: {', '.join(s.name for s in prediction.data_sources)}"
        )
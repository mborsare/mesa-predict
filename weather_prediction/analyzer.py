"""Temperature analysis with statistical validation."""
from datetime import datetime, timedelta
import logging
from typing import Optional, Sequence, Tuple, Dict
from decimal import Decimal
import numpy as np
from dataclasses import dataclass

from .models import (
    TemperatureReading, DataSource, ValidationError,
    StationMetadata, WeatherError
)

@dataclass(frozen=True)
class AnalysisWindow:
    """Time window for temperature analysis."""
    start: datetime
    end: datetime
    readings: Sequence[TemperatureReading]

    def __post_init__(self) -> None:
        """Validate analysis window."""
        if not self.start.tzinfo or not self.end.tzinfo:
            raise ValidationError("Timestamps must be timezone-aware")
        if self.end <= self.start:
            raise ValidationError("End time must be after start time")

class TemperatureAnalyzer:
    """Analyzes temperature data with statistical validation."""

    def __init__(self, logger: Optional[logging.Logger] = None):
        self.logger = logger or logging.getLogger(__name__)

    def get_todays_high(
        self,
        readings: Sequence[TemperatureReading]
    ) -> Optional[TemperatureReading]:
        """Calculate today's high temperature from all sources."""
        if not readings:
            return None

        today = datetime.now(readings[0].timestamp.tzinfo).date()

        # Filter for today's readings
        today_readings = [
            r for r in readings
            if r.timestamp.date() == today
        ]

        if not today_readings:
            return None

        # Get highest reading from any source
        max_reading = max(today_readings, key=lambda r: r.value)

        self.logger.debug(
            f"Today's high: {max_reading.value}°F at {max_reading.timestamp} "
            f"from {max_reading.source.name}"
        )

        return max_reading

    def analyze_temperature_pattern(
        self,
        readings: Sequence[TemperatureReading],
        window_days: int = 7
    ) -> Tuple[Decimal, Decimal]:
        """Analyze temperature pattern with statistical validation."""
        if not readings:
            raise ValidationError("No temperature readings provided")

        # Group readings by date
        daily_highs: Dict[datetime.date, Decimal] = {}
        for reading in readings:
            date = reading.timestamp.date()
            if date not in daily_highs or reading.value > daily_highs[date]:
                daily_highs[date] = reading.value

        if not daily_highs:
            raise WeatherError("No valid daily highs found")

        # Convert to numpy array for statistics
        highs = np.array([float(temp) for temp in daily_highs.values()])

        mean_high = Decimal(str(np.mean(highs)))
        std_dev = Decimal(str(np.std(highs)))

        self.logger.debug(
            f"Temperature pattern analysis over {len(daily_highs)} days:\n"
            f"Mean high: {mean_high:.1f}°F\n"
            f"Standard deviation: {std_dev:.1f}°F"
        )

        return mean_high, std_dev

    def validate_reading(
        self,
        reading: TemperatureReading,
        historical_readings: Sequence[TemperatureReading]
    ) -> bool:
        """Validate a temperature reading against historical data."""
        mean_high, std_dev = self.analyze_temperature_pattern(historical_readings)

        # Convert to float for numpy calculations
        value = float(reading.value)
        mean = float(mean_high)
        std = float(std_dev)

        # Check if reading is within 3 standard deviations
        z_score = abs(value - mean) / std if std > 0 else float('inf')

        is_valid = z_score <= 3

        if not is_valid:
            self.logger.warning(
                f"Suspicious temperature reading: {reading.value}°F\n"
                f"Z-score: {z_score:.2f}\n"
                f"Mean: {mean_high:.1f}°F\n"
                f"Std Dev: {std_dev:.1f}°F"
            )

        return is_valid

    def get_confidence_score(
        self,
        reading: TemperatureReading,
        historical_readings: Sequence[TemperatureReading]
    ) -> Decimal:
        """Calculate confidence score for a temperature reading."""
        try:
            mean_high, std_dev = self.analyze_temperature_pattern(historical_readings)

            # Convert to float for numpy calculations
            value = float(reading.value)
            mean = float(mean_high)
            std = float(std_dev)

            # Calculate z-score
            z_score = abs(value - mean) / std if std > 0 else float('inf')

            # Convert z-score to confidence percentage
            # Higher z-scores = lower confidence
            confidence = max(0, min(100, 100 * (1 - z_score / 6)))

            return Decimal(str(confidence)).quantize(Decimal('0.1'))

        except WeatherError:
            # If we can't calculate statistics, return lower confidence
            return Decimal('50.0')
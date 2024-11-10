"""Domain models and type definitions for weather prediction system."""
from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from decimal import Decimal
from enum import Enum, auto
from typing import NewType, Protocol, Optional, Sequence
import pytz

# Custom types for domain concepts
Temperature = NewType('Temperature', Decimal)
StationId = NewType('StationId', str)
Latitude = NewType('Latitude', Decimal)
Longitude = NewType('Longitude', Decimal)

class WeatherError(Exception):
    """Base exception for weather prediction system."""
    pass

class ValidationError(WeatherError):
    """Raised when data validation fails."""
    pass

class DataSourceError(WeatherError):
    """Raised when data source operations fail."""
    pass

class TemperatureUnit(Enum):
    """Temperature measurement units."""
    FAHRENHEIT = auto()
    CELSIUS = auto()

class DataSource(Enum):
    """Temperature data sources."""
    REGULAR = auto()
    SIX_HOUR = auto()
    HISTORICAL = auto()
    TOMORROW_IO = auto()
    ML_MODEL = auto()
    TOMORROW_HISTORICAL = auto()

@dataclass(frozen=True)
class StationMetadata:
    """Weather station metadata."""
    station_id: StationId
    latitude: Latitude
    longitude: Longitude
    timezone: str

    def __post_init__(self) -> None:
        """Validate station metadata."""
        if len(self.station_id) != 4:
            raise ValidationError("Station ID must be exactly 4 characters")
        if not -90 <= float(self.latitude) <= 90:
            raise ValidationError(f"Invalid latitude: {self.latitude}")
        if not -180 <= float(self.longitude) <= 180:
            raise ValidationError(f"Invalid longitude: {self.longitude}")
        try:
            pytz.timezone(self.timezone)  # Validate timezone string
        except pytz.exceptions.UnknownTimeZoneError:
            raise ValidationError(f"Invalid timezone: {self.timezone}")

    def get_tzinfo(self) -> datetime.tzinfo:
        """Get timezone object for datetime operations."""
        return pytz.timezone(self.timezone)

@dataclass(frozen=True)
class TemperatureReading:
    """Immutable temperature reading with metadata."""
    value: Temperature
    timestamp: datetime
    source: DataSource
    unit: TemperatureUnit
    station_id: StationId

    def __post_init__(self) -> None:
        """Validate temperature reading."""
        if not isinstance(self.value, Decimal):
            object.__setattr__(self, 'value',
                             Decimal(str(self.value)).quantize(Decimal('0.01')))
        if not self.timestamp.tzinfo:
            raise ValidationError("Timestamp must be timezone-aware")
        if not -100 <= float(self.value) <= 150:
            raise ValidationError(f"Temperature {self.value}°F out of valid range")

    @property
    def fahrenheit(self) -> Decimal:
        """Get temperature in Fahrenheit."""
        if self.unit == TemperatureUnit.FAHRENHEIT:
            return self.value
        return (self.value * Decimal('1.8')) + Decimal('32')

    @property
    def celsius(self) -> Decimal:
        """Get temperature in Celsius."""
        if self.unit == TemperatureUnit.CELSIUS:
            return self.value
        return (self.value - Decimal('32')) / Decimal('1.8')

@dataclass(frozen=True)
class TemperaturePrediction:
    """Temperature prediction with confidence metrics."""
    temperature: TemperatureReading
    confidence: Decimal
    data_sources: Sequence[DataSource]
    station: StationMetadata

    def __post_init__(self) -> None:
        """Validate prediction attributes."""
        if not 0 <= self.confidence <= 100:
            raise ValidationError("Confidence must be between 0 and 100")
        if not self.data_sources:
            raise ValidationError("At least one data source required")

class DataSource(Enum):
    """Temperature data sources."""
    REGULAR = auto()
    SIX_HOUR = auto()
    HISTORICAL = auto()
    TOMORROW_IO = auto()
    ML_MODEL = auto()  # Added new source for machine learning predictions

@dataclass(frozen=True)
class WeatherReading:
    """Complete weather observation with all available metrics."""
    temperature: TemperatureReading
    relative_humidity: Optional[Decimal]
    dew_point: Optional[Decimal]
    wind_speed: Optional[Decimal]
    wind_direction: Optional[Decimal]
    wind_gust: Optional[Decimal]
    pressure: Optional[Decimal]
    visibility: Optional[Decimal]
    timestamp: datetime
    station_id: StationId

    def to_feature_dict(self) -> Dict[str, float]:
        """Convert reading to feature dictionary for ML model."""
        return {
            'air_temp': float(self.temperature.value),
            'relative_humidity': float(self.relative_humidity) if self.relative_humidity else None,
            'dew_point_temperature': float(self.dew_point) if self.dew_point else None,
            'wind_speed': float(self.wind_speed) if self.wind_speed else None,
            'wind_direction': float(self.wind_direction) if self.wind_direction else None,
            'wind_gust': float(self.wind_gust) if self.wind_gust else None,
            'sea_level_pressure': float(self.pressure) if self.pressure else None,
            'visibility': float(self.visibility) if self.visibility else None,
            'hour': self.timestamp.hour,
            'month': self.timestamp.month
        }
"""Temperature trend analysis functionality."""
from typing import Dict, Any, Tuple, Sequence, Optional
import numpy as np
import logging
from decimal import Decimal

from ..models import TemperatureReading

class TrendAnalyzer:
    """Analyzes temperature trends from multiple data sources."""

    def __init__(self, logger: logging.Logger):
        self.logger = logger

    def analyze_trends(
        self,
        mesowest_readings: Sequence[TemperatureReading],
        tomorrow_data: Optional[Dict[str, Any]] = None
    ) -> Tuple[float, float]:
        """Analyze temperature trends from both data sources."""
        try:
            # Analyze MesoWest trends using most recent data (6-hour window)
            mesowest_temps = [float(r.value) for r in mesowest_readings[-6:]]
            if len(mesowest_temps) >= 3:
                recent_mesowest = np.mean(mesowest_temps[-2:])
                older_mesowest = np.mean(mesowest_temps[:-2])
                mesowest_change = recent_mesowest - older_mesowest
            else:
                mesowest_change = 0

            # Analyze Tomorrow.io trends if available (6-hour window)
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

    def calculate_confidence(
        self,
        prediction: TemperatureReading,
        historical_readings: Sequence[TemperatureReading],
        trend_confidence: float,
        temp_change: float = 0
    ) -> Decimal:
        """Calculate confidence score based on available data."""
        if not historical_readings:
            return Decimal('50.0')

        try:
            # Calculate temperature variability (using 6-hour window)
            temps = [float(r.value) for r in historical_readings[-6:]]
            temp_std = np.std(temps)

            # Base confidence starts at 85% and decreases with variability
            base_confidence = 85 - (temp_std * 1.5)

            # Adjust for temperature change rate
            if temp_change > 0:
                base_confidence -= (temp_change * 1.2)

            # Incorporate trend confidence
            base_confidence = (base_confidence * 0.7) + (trend_confidence * 0.3)

            # Ensure confidence stays within reasonable bounds
            confidence = max(min(base_confidence, 90), 50)

            return Decimal(str(confidence)).quantize(Decimal('0.1'))

        except Exception as e:
            self.logger.warning(f"Error calculating confidence: {e}")
            return Decimal('50.0')
# tomorrow_utils.py
class EnhancedTemperaturePredictor:
    """Enhanced temperature predictor with improved validation and debugging."""

    def __init__(self, weather_processor, tomorrow_client, debug_mode: bool = False):
        self.weather_processor = weather_processor
        self.tomorrow_client = tomorrow_client
        self.debug_mode = debug_mode
        self.logger = logging.getLogger(__name__)
        self.analyzer = TemperatureAnalyzer(debug_mode=debug_mode)

    def predict_todays_high(self) -> Optional[float]:
        """Predict today's high temperature with enhanced validation."""
        data_sources = self._gather_data_sources()

        # Log all data sources
        self._log_data_sources(data_sources)

        # Filter out None values and validate temperatures
        valid_data = {
            k: v for k, v in data_sources.items()
            if v is not None and self.analyzer._validate_temperature(v)
        }

        if not valid_data:
            self.logger.warning("No valid temperature estimates available")
            return None

        weights = self._get_source_weights()

        # Calculate weighted average
        total_weight = sum(weights[k] for k in valid_data.keys())
        weighted_sum = sum(valid_data[k] * weights[k] for k in valid_data.keys())

        final_high_temp = round(weighted_sum / total_weight, 1)

        # Validate final prediction
        if not self.analyzer._validate_temperature(final_high_temp):
            self.logger.error(f"Invalid final prediction: {final_high_temp}°F")
            return None

        self._log_prediction_details(final_high_temp, valid_data)
        return final_high_temp

    def _gather_data_sources(self) -> Dict[str, Optional[float]]:
        """Gather temperature data from all sources with validation."""
        return {
            "Regular Max": self.weather_processor.get_regular_temperature_max(),
            "6hr High Max": self.weather_processor.get_6hr_high_temperature_max(),
            "Historical Mean": self._get_historical_pattern()[0],
            "Tomorrow.io": self._get_tomorrow_forecast()
        }

    def _log_data_sources(self, sources: Dict[str, Optional[float]]):
        """Log temperature data sources with validation info."""
        for source, value in sources.items():
            if value is not None:
                valid = self.analyzer._validate_temperature(value)
                self.logger.debug(
                    f"{source}: {value}°F "
                    f"({'valid' if valid else 'invalid temperature'})"
                )
            else:
                self.logger.debug(f"{source}: No data")

    def _log_prediction_details(self, final_temp: float, valid_data: Dict[str, float]):
        """Log detailed prediction information."""
        self.logger.debug(
            f"Prediction based on: {', '.join(valid_data.keys())}\n"
            f"Final prediction: {final_temp}°F"
        )
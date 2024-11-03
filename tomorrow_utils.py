# tomorrow_utils.py

import requests
from datetime import datetime, timedelta
import pandas as pd
import pytz
from typing import Optional, Dict, Tuple
from weather_utils import WeatherDataProcessor, TemperaturePredictor

class TomorrowioClient:
    def __init__(self, api_key: str):
        self.api_key = api_key
        self.base_url = "https://api.tomorrow.io/v4/timelines"

    def get_forecast_data(self, lat: float, lon: float) -> Optional[pd.DataFrame]:
        """Fetches forecast data from Tomorrow.io API."""
        params = {
            "location": f"{lat},{lon}",
            "fields": [
                "temperature",
                "humidity",
                "windSpeed",
                "pressureSurfaceLevel"
            ],
            "units": "imperial",
            "timesteps": ["1h"],
            "apikey": self.api_key
        }

        try:
            response = requests.get(self.base_url, params=params)
            response.raise_for_status()
            data = response.json()

            # Extract timeline data
            if "data" in data and "timelines" in data["data"]:
                timeline = data["data"]["timelines"][0]

                # Convert to DataFrame
                records = []
                for interval in timeline["intervals"]:
                    record = {
                        "Date": pd.to_datetime(interval["startTime"]),
                        "Tomorrow_Temperature": interval["values"]["temperature"],
                        "Tomorrow_Humidity": interval["values"]["humidity"],
                        "Tomorrow_WindSpeed": interval["values"]["windSpeed"],
                        "Tomorrow_Pressure": interval["values"]["pressureSurfaceLevel"]
                    }
                    records.append(record)

                return pd.DataFrame(records)

            return None

        except requests.exceptions.RequestException as e:
            print(f"Error fetching Tomorrow.io data: {e}")
            return None

class EnhancedTemperaturePredictor(TemperaturePredictor):
    def __init__(self, weather_processor: WeatherDataProcessor, tomorrow_client: TomorrowioClient):
        super().__init__(weather_processor)
        self.tomorrow_client = tomorrow_client
        # Get coordinates automatically
        self.station_lat, self.station_lon = weather_processor.get_coordinates()
        print(f"Using station coordinates: {self.station_lat}, {self.station_lon}")

    def _prepare_prediction_features(self, data: pd.DataFrame) -> Optional[pd.DataFrame]:
        """Enhanced feature preparation including Tomorrow.io data."""
        try:
            # Get base features
            base_features = super()._prepare_prediction_features(data)
            if base_features is None:
                return None

            # Get Tomorrow.io forecast
            tomorrow_data = self.tomorrow_client.get_forecast_data(
                self.station_lat,
                self.station_lon
            )

            if tomorrow_data is None:
                print("Warning: Could not fetch Tomorrow.io data. Using base features only.")
                return base_features

            # Get the next few hours of forecast
            current_time = data["Date"].max()
            next_6hrs = tomorrow_data[
                (tomorrow_data["Date"] > current_time) &
                (tomorrow_data["Date"] <= current_time + pd.Timedelta(hours=6))
            ]

            if not next_6hrs.empty:
                # Add Tomorrow.io features
                base_features["Tomorrow_Max_Temp"] = next_6hrs["Tomorrow_Temperature"].max()
                base_features["Tomorrow_Avg_Humidity"] = next_6hrs["Tomorrow_Humidity"].mean()
                base_features["Tomorrow_Avg_WindSpeed"] = next_6hrs["Tomorrow_WindSpeed"].mean()
            else:
                print("Warning: No Tomorrow.io forecast data available for the next 6 hours.")

            return base_features

        except Exception as e:
            print(f"Error preparing enhanced prediction features: {e}")
            return None

    def train_with_historical_data(self, days: int = 30) -> bool:
        """
        Trains the model with historical data.

        Args:
            days: Number of days of historical data to use for training

        Returns:
            bool: True if training was successful, False otherwise
        """
        try:
            print(f"Training model with {days} days of historical data...")

            # Calculate date range for training
            end_time = datetime.now(self.weather_processor.timezone)
            start_time = end_time - timedelta(days=days)

            # Format dates for API
            start_str = start_time.strftime("%Y%m%d%H%M")
            end_str = end_time.strftime("%Y%m%d%H%M")

            # Get historical data
            training_data = self.weather_processor.get_weather_data(
                start_time=start_str,
                end_time=end_str
            )

            if training_data is None or training_data.empty:
                print("Error: No historical data available for training")
                return False

            print(f"Retrieved {len(training_data)} historical observations")

            # Train the model
            return self._train_model(training_data)

        except Exception as e:
            print(f"Error training model: {e}")
            return False

    def predict_todays_high(self) -> Optional[float]:
        """
        Predicts today's high temperature using both historical data and Tomorrow.io forecast.

        Returns:
            Predicted high temperature for today or None if prediction fails
        """
        try:
            # First, ensure the model is trained
            if not self.train_with_historical_data():
                print("Error: Failed to train the model")
                return None

            # Get current data from weather processor
            current_data = self.weather_processor.get_recent_data()
            if current_data is None:
                print("Error: Could not fetch recent weather data")
                return None

            # Prepare features including Tomorrow.io data
            features = self._prepare_prediction_features(current_data)
            if features is None:
                return None

            # Scale features
            features_scaled = self.scaler.transform(features)

            # Make prediction
            base_prediction = self.model.predict(features_scaled)[0]

            # If we have Tomorrow.io data, blend it with our model prediction
            if "Tomorrow_Max_Temp" in features.columns:
                tomorrow_max = features["Tomorrow_Max_Temp"].iloc[0]
                # Blend the predictions (70% weight to our model, 30% to Tomorrow.io)
                blended_prediction = (0.7 * base_prediction) + (0.3 * tomorrow_max)
                final_prediction = round(blended_prediction, 1)
            else:
                final_prediction = round(base_prediction, 1)

            print(f"\nPredicted high temperature for today: {final_prediction}°F")
            return final_prediction

        except Exception as e:
            print(f"Error predicting today's high temperature: {e}")
            return None
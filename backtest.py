import sys
import logging
import traceback
from decimal import Decimal, ROUND_HALF_UP
from datetime import datetime, timedelta

import pandas as pd
import numpy as np
import pytz

from weather_utils import WeatherDataProcessor, API_CONFIG
from tomorrow_utils import TomorrowioClient, EnhancedTemperaturePredictor

# Configure logging
logging.basicConfig(
    filename='backtest.log',
    level=logging.INFO,
    format='%(asctime)s:%(levelname)s:%(message)s'
)

def get_station_id():
    """Prompt user for station ID and validate it."""
    station_id = input("Enter your station ID (e.g., KMIA): ").strip().upper()
    if len(station_id) != 4:
        logging.error("Station ID must be exactly 4 characters long.")
        print("Error: Station ID must be exactly 4 characters long.")
        sys.exit(1)
    return station_id

def fetch_day_data(weather_processor, tomorrow_client, date):
    """
    Fetch and prepare data for a specific day including Tomorrow.io forecast.

    Args:
        weather_processor (WeatherDataProcessor): Instance of WeatherDataProcessor.
        tomorrow_client (TomorrowioClient): Instance of TomorrowioClient.
        date (datetime): Date for which to fetch data.

    Returns:
        tuple: (historical_data, tomorrow_data) or (None, None) if fetch fails
    """
    start = date.replace(hour=0, minute=0, second=0, microsecond=0)
    end = date.replace(hour=23, minute=59, second=59, microsecond=999999)

    # Convert to UTC and format as strings
    start_utc = start.astimezone(pytz.utc).strftime('%Y%m%d%H%M')
    end_utc = end.astimezone(pytz.utc).strftime('%Y%m%d%H%M')

    # Get historical data
    historical_data = weather_processor.get_weather_data(start_time=start_utc, end_time=end_utc)
    if historical_data is None or historical_data.empty:
        logging.warning(f"No historical data available for {date.date()}.")
        return None, None

    # Get Tomorrow.io forecast data
    lat, lon = weather_processor.get_coordinates()
    tomorrow_data = tomorrow_client.get_forecast_data(lat, lon)

    return historical_data, tomorrow_data

def fetch_actual_temp(weather_processor, date):
    """Fetch the actual high temperature for a specific day."""
    start = date.replace(hour=0, minute=0, second=0, microsecond=0)
    end = date.replace(hour=23, minute=59, second=59, microsecond=999999)

    start_utc = start.astimezone(pytz.utc).strftime('%Y%m%d%H%M')
    end_utc = end.astimezone(pytz.utc).strftime('%Y%m%d%H%M')

    actual_data = weather_processor.get_weather_data(start_time=start_utc, end_time=end_utc)
    if actual_data is None or actual_data.empty:
        logging.warning(f"No actual data available for {date.date()}.")
        return None

    actual_high = actual_data['Temp_High_6hr'].max()
    return int(Decimal(actual_high).quantize(0, ROUND_HALF_UP)) if pd.notnull(actual_high) else None

def process_day(predictor, weather_processor, tomorrow_client, date, actual_temps, predicted_temps, dates):
    """
    Process a single day's data using the enhanced predictor.

    Args:
        predictor (EnhancedTemperaturePredictor): Instance of EnhancedTemperaturePredictor.
        weather_processor (WeatherDataProcessor): Instance of WeatherDataProcessor.
        tomorrow_client (TomorrowioClient): Instance of TomorrowioClient.
        date (datetime): Date being processed.
        actual_temps (list): List to append actual temperatures.
        predicted_temps (list): List to append predicted temperatures.
        dates (list): List to append dates.
    """
    historical_data, tomorrow_data = fetch_day_data(weather_processor, tomorrow_client, date)

    if historical_data is None or historical_data.empty:
        logging.warning(f"No data to process for {date.date()}.")
        actual_temps.append(None)
        predicted_temps.append(None)
        dates.append(date.date())
        return

    # Make prediction using enhanced predictor
    try:
        # Temporarily set the data for prediction
        predictor.historical_data = historical_data
        predictor.tomorrow_data = tomorrow_data
        prediction = predictor.predict_todays_high()

        if prediction is not None:
            rounded_pred = int(Decimal(prediction).quantize(0, ROUND_HALF_UP))
        else:
            rounded_pred = None

    except Exception as e:
        logging.error(f"Error making prediction for {date.date()}: {e}")
        rounded_pred = None

    actual = fetch_actual_temp(weather_processor, date)

    actual_temps.append(actual)
    predicted_temps.append(rounded_pred)
    dates.append(date.date())

def display_results(actual_temps, predicted_temps, dates):
    """Display and save backtest results."""
    results = pd.DataFrame({
        'Date': dates,
        'Actual High': actual_temps,
        'Predicted High': predicted_temps
    })

    print("\nBacktest Results:")
    print(results)
    results.to_csv('backtest_results.csv', index=False)

    # Calculate metrics excluding None values
    valid_indices = [i for i, a in enumerate(actual_temps) if a is not None and predicted_temps[i] is not None]
    filtered_actual = [actual_temps[i] for i in valid_indices]
    filtered_predicted = [predicted_temps[i] for i in valid_indices]

    if filtered_actual:
        mae = np.mean(np.abs(np.array(filtered_actual) - np.array(filtered_predicted)))
        rmse = np.sqrt(np.mean((np.array(filtered_actual) - np.array(filtered_predicted)) ** 2))
        logging.info(f"MAE: {mae:.2f}, RMSE: {rmse:.2f}")
        print(f"\nMetrics:")
        print(f"Mean Absolute Error: {mae:.2f}°F")
        print(f"Root Mean Square Error: {rmse:.2f}°F")
    else:
        logging.warning("No valid data to calculate MAE and RMSE.")
        print("\nNo valid data to calculate metrics.")

def backtest_predictions(station_id, backtest_days=90, training_days=90):
    """
    Main function to perform backtesting of temperature predictions.

    Args:
        station_id (str): Station identifier.
        backtest_days (int, optional): Number of days to backtest.
        training_days (int, optional): Number of days to use for training.
    """
    try:
        logging.info(f"Starting backtest for the past {backtest_days} days.")
        print(f"Starting backtest for the past {backtest_days} days...")

        actual_temps, predicted_temps, dates = [], [], []

        # Initialize clients
        weather_processor = WeatherDataProcessor(station_id)
        tomorrow_client = TomorrowioClient(API_CONFIG["tomorrow_io_token"])

        # Initialize enhanced predictor
        predictor = EnhancedTemperaturePredictor(weather_processor, tomorrow_client)

        station_tz = weather_processor.timezone
        now = datetime.now(station_tz)
        backtest_end_date = now - timedelta(days=1)

        # Perform backtesting for each day
        for day_offset in range(backtest_days):
            target_date = backtest_end_date - timedelta(days=day_offset)
            print(f"\nProcessing {target_date.date()}...")

            process_day(predictor, weather_processor, tomorrow_client,
                       target_date, actual_temps, predicted_temps, dates)

        # Display and log the results
        display_results(actual_temps, predicted_temps, dates)

    except Exception as e:
        logging.error(f"Unexpected error: {e}")
        traceback.print_exc()
        print(f"An unexpected error occurred: {e}")

def main():
    """Entry point of the script."""
    station_id = get_station_id()
    backtest_predictions(station_id)

if __name__ == "__main__":
    main()
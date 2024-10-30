import sys
import logging
import traceback
from decimal import Decimal, ROUND_HALF_UP
from datetime import datetime, timedelta

import pandas as pd
import numpy as np
import pytz
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler

from weather_utils import get_weather_data, get_station_metadata

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

def prepare_features(data, required_columns, tz):
    """
    Prepare features for model training or prediction.

    Args:
        data (pd.DataFrame): Raw weather data.
        required_columns (list): Columns required for processing.
        tz (pytz.timezone): Timezone of the station.

    Returns:
        pd.DataFrame: Processed data with engineered features.
    """
    # Ensure required columns are present
    missing_cols = [col for col in required_columns if col not in data.columns]
    if missing_cols:
        logging.error(f"Missing columns: {missing_cols}")
        return pd.DataFrame()

    # Rename columns for consistency
    rename_map = {
        'Humidity': 'Humidity_Avg',
        'Wind_Speed': 'Wind_Speed_Avg',
        'Temp_High_6hr': 'Temp_High_6hr_Avg'
    }
    data = data.rename(columns=rename_map)

    # Engineer additional features
    data['Hours_Since_Midnight'] = data['Date'].dt.hour + data['Date'].dt.minute / 60

    # Drop rows with missing essential data
    data = data.dropna(subset=['Temperature', 'Humidity_Avg', 'Wind_Speed_Avg', 'Temp_High_6hr_Avg'])

    return data

def train_model(data):
    """
    Train the Random Forest model.

    Args:
        data (pd.DataFrame): Processed training data.

    Returns:
        tuple: Trained model and fitted scaler.
    """
    X_train = data[['Temperature', 'Humidity_Avg', 'Wind_Speed_Avg', 'Hours_Since_Midnight']]
    y_train = data['Temp_High_6hr_Avg']

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)

    model = RandomForestRegressor(n_estimators=100, random_state=42)
    model.fit(X_train_scaled, y_train)

    logging.info("Model training completed.")
    return model, scaler

def fetch_day_data(station_id, date, tz):
    """
    Fetch and prepare data for a specific day.

    Args:
        station_id (str): Station identifier.
        date (datetime): Date for which to fetch data.
        tz (pytz.timezone): Timezone of the station.

    Returns:
        pd.DataFrame: Prepared data for the day.
    """
    start = date.replace(hour=0, minute=0, second=0, microsecond=0)
    end = date.replace(hour=23, minute=59, second=59, microsecond=999999)

    # Convert to UTC and format as strings
    start_utc = start.astimezone(pytz.utc).strftime('%Y%m%d%H%M')
    end_utc = end.astimezone(pytz.utc).strftime('%Y%m%d%H%M')

    historical_data = get_weather_data(station_id, start_time=start_utc, end_time=end_utc)
    if historical_data is None or historical_data.empty:
        logging.warning(f"No historical data available for {date.date()}.")
        return pd.DataFrame()

    # Convert 'Date' to station timezone
    historical_data['Date'] = pd.to_datetime(historical_data['Date']).dt.tz_convert(tz)

    # Prepare features
    prepared_data = prepare_features(historical_data, ['Temperature', 'Humidity', 'Wind_Speed', 'Temp_High_6hr'], tz)
    return prepared_data

def fetch_actual_temp(station_id, date, tz):
    """
    Fetch the actual high temperature for a specific day.

    Args:
        station_id (str): Station identifier.
        date (datetime): Date for which to fetch actual temperature.
        tz (pytz.timezone): Timezone of the station.

    Returns:
        int or None: Actual high temperature or None if unavailable.
    """
    start = date.replace(hour=0, minute=0, second=0, microsecond=0)
    end = date.replace(hour=23, minute=59, second=59, microsecond=999999)

    start_utc = start.astimezone(pytz.utc).strftime('%Y%m%d%H%M')
    end_utc = end.astimezone(pytz.utc).strftime('%Y%m%d%H%M')

    actual_data = get_weather_data(station_id, start_time=start_utc, end_time=end_utc)
    if actual_data is None or actual_data.empty:
        logging.warning(f"No actual data available for {date.date()}.")
        return None

    actual_high = actual_data['Temp_High_6hr'].max()
    return int(actual_high) if pd.notnull(actual_high) else None

def process_day(model, scaler, station_id, day_data, date, actual_temps, predicted_temps, dates):
    """
    Process a single day's data: predict and compare with actual.

    Args:
        model: Trained Random Forest model.
        scaler: Fitted StandardScaler.
        station_id (str): Station identifier.
        day_data (pd.DataFrame): Prepared data for the day.
        date (datetime): Date being processed.
        actual_temps (list): List to append actual temperatures.
        predicted_temps (list): List to append predicted temperatures.
        dates (list): List to append dates.
    """
    if day_data.empty:
        logging.warning(f"No data to process for {date.date()}.")
        actual_temps.append(None)
        predicted_temps.append(None)
        dates.append(date.date())
        return

    X_input = day_data[['Temperature', 'Humidity_Avg', 'Wind_Speed_Avg', 'Hours_Since_Midnight']]
    X_scaled = scaler.transform(X_input)
    prediction = model.predict(X_scaled).mean()
    rounded_pred = int(Decimal(prediction).quantize(0, rounding=ROUND_HALF_UP))

    # Fetch the actual high temperature without calling .mean()
    actual = fetch_actual_temp(station_id, date, day_data['Date'].dt.tz) if 'Date' in day_data.columns else None

    actual_temps.append(actual)
    predicted_temps.append(rounded_pred)
    dates.append(date.date())

def display_results(actual_temps, predicted_temps, dates):
    """
    Display and save backtest results.

    Args:
        actual_temps (list): List of actual temperatures.
        predicted_temps (list): List of predicted temperatures.
        dates (list): List of dates.
    """
    results = pd.DataFrame({
        'Date': dates,
        'Actual High': actual_temps,
        'Predicted High': predicted_temps
    })

    print(results)
    results.to_csv('backtest_results.csv', index=False)

    # Calculate metrics excluding None values
    valid_indices = [i for i, a in enumerate(actual_temps) if a is not None]
    filtered_actual = [actual_temps[i] for i in valid_indices]
    filtered_predicted = [predicted_temps[i] for i in valid_indices]

    if filtered_actual:
        mae = np.mean(np.abs(np.array(filtered_actual) - np.array(filtered_predicted)))
        rmse = np.sqrt(np.mean((np.array(filtered_actual) - np.array(predicted_temps)[valid_indices]) ** 2))
        logging.info(f"MAE: {mae:.2f}, RMSE: {rmse:.2f}")
        print(f"\nMAE: {mae:.2f}, RMSE: {rmse:.2f}")
    else:
        logging.warning("No valid data to calculate MAE and RMSE.")
        print("\nNo valid data to calculate MAE and RMSE.")

def backtest_predictions(station_id, backtest_days=30):
    """
    Main function to perform backtesting of temperature predictions.

    Args:
        station_id (str): Station identifier.
        backtest_days (int, optional): Number of days to backtest. Defaults to 30.
    """
    try:
        logging.info(f"Starting backtest for the past {backtest_days} days.")
        print(f"Starting backtest for the past {backtest_days} days...\n")

        actual_temps, predicted_temps, dates = [], [], []

        # Fetch station metadata
        logging.info("Fetching station metadata.")
        metadata = get_station_metadata(station_id)
        if metadata is None:
            logging.error("Cannot proceed without station metadata.")
            print("Error: Cannot retrieve station metadata.")
            return

        timezone_str = metadata.get('timezone', 'UTC')
        station_tz = pytz.timezone(timezone_str)
        logging.info(f"Station timezone: {timezone_str}")

        # Define training period: last 60 days up to yesterday
        now = datetime.now(station_tz)
        backtest_end_date = now - timedelta(days=1)
        backtest_start_date = backtest_end_date - timedelta(days=60)

        # Format dates for data fetching
        start_time_str = backtest_start_date.astimezone(pytz.utc).strftime('%Y%m%d%H%M')
        end_time_str = backtest_end_date.astimezone(pytz.utc).strftime('%Y%m%d%H%M')

        # Fetch and prepare training data
        logging.info("Fetching training data.")
        training_data = get_weather_data(station_id, start_time=start_time_str, end_time=end_time_str)
        if training_data is None or training_data.empty:
            logging.error("No training data available.")
            print("Error: No training data available.")
            return

        training_data['Date'] = pd.to_datetime(training_data['Date']).dt.tz_convert(station_tz)
        training_data = prepare_features(training_data, ['Temperature', 'Humidity', 'Wind_Speed', 'Temp_High_6hr'], station_tz)

        if training_data.empty or len(training_data) < 10:
            logging.error("Insufficient data for model training.")
            print("Error: Insufficient data for model training.")
            return

        # Train the model
        model, scaler = train_model(training_data)

        # Perform backtesting for each day
        for day_offset in range(backtest_days):
            target_date = backtest_end_date - timedelta(days=day_offset)
            day_data = fetch_day_data(station_id, target_date, station_tz)
            process_day(model, scaler, station_id, day_data, target_date, actual_temps, predicted_temps, dates)
            print("Data processed successfully.")

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

# predict_today.py

import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler
import datetime
from weather_utils import get_weather_data, get_station_metadata
from decimal import Decimal, ROUND_HALF_UP
import numpy as np
import pytz  # Ensure pytz is installed: pip install pytz

# Prompt the user to enter the station ID
station_id = input("Enter your station ID (e.g., KMIA): ").strip().upper()

# Validate the station ID
if len(station_id) != 4:
    raise ValueError("Station ID must be exactly 4 characters long.")


def predict_todays_high(station_id):
    print("Training the Random Forest regression model...")

    # Fetch station metadata to determine timezone
    metadata = get_station_metadata(station_id)
    if metadata is None:
        print("Cannot proceed without station metadata.")
        return

    timezone_str = metadata["timezone"]

    try:
        station_tz = pytz.timezone(timezone_str)
    except pytz.UnknownTimeZoneError:
        print(f"Unknown timezone '{timezone_str}' for station ID '{station_id}'.")
        return

    # Get current time in station's timezone
    now = datetime.datetime.now(station_tz)

    # Fetch extended training data (last 60 days)
    training_start_date = now - datetime.timedelta(days=60)
    training_end_date = now - datetime.timedelta(days=1)

    # Set training_start_date to 00:00 and training_end_date to 23:59:59 to cover full days
    training_start_date = training_start_date.replace(
        hour=0, minute=0, second=0, microsecond=0
    )
    training_end_date = training_end_date.replace(
        hour=23, minute=59, second=59, microsecond=999999
    )

    # Convert training_start_date and training_end_date to UTC for API request
    training_start_date_utc = training_start_date.astimezone(pytz.utc)
    training_end_date_utc = training_end_date.astimezone(pytz.utc)

    # Format start_time and end_time in 'YYYYMMDDHHMM'
    training_start_time = training_start_date_utc.strftime("%Y%m%d%H%M")
    training_end_time = training_end_date_utc.strftime("%Y%m%d%H%M")

    # Fetch training data using get_weather_data
    training_data = get_weather_data(
        station_id, start_time=training_start_time, end_time=training_end_time
    )

    if training_data is None or training_data.empty:
        print("Unable to fetch training data.")
        return

    # Convert training_data['Date'] from UTC to station's local timezone
    training_data["Date"] = training_data["Date"].dt.tz_convert(station_tz)

    # Debug: Print the range of training data timestamps
    print(
        f"Training data spans from {training_data['Date'].min()} to {training_data['Date'].max()}"
    )

    # Rename columns to match those used during training
    rename_columns = {
        "Humidity": "Humidity_Avg",
        "Wind_Speed": "Wind_Speed_Avg",
        "Temp_High_6hr": "Temp_High_6hr_Avg",
    }
    # Only rename existing columns to avoid KeyError
    rename_columns = {
        k: v for k, v in rename_columns.items() if k in training_data.columns
    }
    training_data.rename(columns=rename_columns, inplace=True)

    # Ensure necessary columns are present
    required_columns = [
        "Humidity_Avg",
        "Wind_Speed_Avg",
        "Temp_High_6hr_Avg",
        "Temperature",
    ]
    for col in required_columns:
        if col not in training_data.columns:
            print(f"Missing required column in training data: {col}")
            return

    training_data.dropna(subset=required_columns, inplace=True)

    # Add 'Hours_Since_Midnight' feature
    training_data["Hours_Since_Midnight"] = (
        training_data["Date"].dt.hour + training_data["Date"].dt.minute / 60
    )
    training_data["DateOnly"] = training_data["Date"].dt.date

    # Group by date and compute features
    grouped = training_data.groupby("DateOnly")

    # Compute maximum observed temperature for each day up to the last available time
    max_temp_so_far = grouped["Temperature"].max()
    humidity_avg = grouped["Humidity_Avg"].mean()
    wind_speed_avg = grouped["Wind_Speed_Avg"].mean()
    avg_time = grouped["Hours_Since_Midnight"].mean()
    actual_high_temp = grouped["Temp_High_6hr_Avg"].max()

    # Create a new DataFrame for model training
    model_training_data = pd.DataFrame(
        {
            "Max_Temp_So_Far": max_temp_so_far,
            "Humidity_Avg": humidity_avg,
            "Wind_Speed_Avg": wind_speed_avg,
            "Hours_Since_Midnight": avg_time,
            "Actual_High_Temp": actual_high_temp,
        }
    ).dropna()

    # Remove outliers from training data
    model_training_data = model_training_data[
        (model_training_data["Humidity_Avg"] >= 0)
        & (model_training_data["Humidity_Avg"] <= 100)
        & (model_training_data["Wind_Speed_Avg"] >= 0)
        & (model_training_data["Wind_Speed_Avg"] <= 100)
        & (model_training_data["Actual_High_Temp"] >= -50)
        & (model_training_data["Actual_High_Temp"] <= 150)
    ]

    # Check if there are enough samples after cleaning
    if len(model_training_data) < 10:
        print(
            "Not enough training data after cleaning. Consider extending the training period."
        )
        return

    # Prepare training features and target
    X_train = model_training_data[
        ["Max_Temp_So_Far", "Humidity_Avg", "Wind_Speed_Avg", "Hours_Since_Midnight"]
    ]
    y_train = model_training_data["Actual_High_Temp"]

    # Initialize the scaler
    scaler = StandardScaler()

    # Fit the scaler on training data and transform
    X_train_scaled = scaler.fit_transform(X_train)

    # Train the Random Forest model on scaled data
    model = RandomForestRegressor(n_estimators=100, random_state=42)
    model.fit(X_train_scaled, y_train)

    print("Model training completed.")
    print("Starting prediction for today's high temperature...")

    # Fetch today's weather data up to the current time minus 5 minutes to account for data lag
    today_start = now.replace(hour=0, minute=0, second=0, microsecond=0)
    current_time = now - datetime.timedelta(minutes=5)

    # Convert to UTC for API request
    today_start_utc = today_start.astimezone(pytz.utc)
    current_time_utc = current_time.astimezone(pytz.utc)

    # Format start_time and end_time in 'YYYYMMDDHHMM'
    current_start_time = today_start_utc.strftime("%Y%m%d%H%M")
    current_end_time = current_time_utc.strftime("%Y%m%d%H%M")

    # Fetch current weather data
    current_data = get_weather_data(
        station_id, start_time=current_start_time, end_time=current_end_time
    )

    if current_data is None or current_data.empty:
        print("Unable to fetch current weather data.")
        return

    # Convert current_data['Date'] from UTC to station's local timezone
    current_data["Date"] = current_data["Date"].dt.tz_convert(station_tz)

    # Debug: Print the range of current data timestamps
    print(
        f"Current data spans from {current_data['Date'].min()} to {current_data['Date'].max()}"
    )

    # Inspect the latest data points
    print("Latest data points:")
    print(current_data[["Date", "Temperature"]].tail())

    # Rename columns to match training data
    rename_columns = {
        "Humidity": "Humidity_Avg",
        "Wind_Speed": "Wind_Speed_Avg",
        "Temp_High_6hr": "Temp_High_6hr_Avg",
    }
    # Only rename existing columns to avoid KeyError
    rename_columns = {
        k: v for k, v in rename_columns.items() if k in current_data.columns
    }
    current_data.rename(columns=rename_columns, inplace=True)

    # Ensure necessary columns are present
    required_columns = ["Humidity_Avg", "Wind_Speed_Avg", "Temperature"]
    for col in required_columns:
        if col not in current_data.columns:
            print(f"Missing required column in current data: {col}")
            return

    # Compute features for prediction
    max_temp_so_far_today = current_data["Temperature"].max()
    humidity_avg_today = current_data["Humidity_Avg"].mean()
    wind_speed_avg_today = current_data["Wind_Speed_Avg"].mean()
    current_data["Hours_Since_Midnight"] = (
        current_data["Date"].dt.hour + current_data["Date"].dt.minute / 60
    )
    current_time_hours = current_data["Hours_Since_Midnight"].iloc[-1]

    # Prepare data for prediction
    X_input = pd.DataFrame(
        {
            "Max_Temp_So_Far": [max_temp_so_far_today],
            "Humidity_Avg": [humidity_avg_today],
            "Wind_Speed_Avg": [wind_speed_avg_today],
            "Hours_Since_Midnight": [current_time_hours],
        }
    )

    # Check for NaN values
    if X_input.isnull().values.any():
        print("NaN values found in current features:")
        print(X_input)
        X_input = X_input.dropna()
        if X_input.empty:
            print("No valid data available after dropping NaNs.")
            return

    try:
        if X_input.empty:
            print("No valid data to make a prediction.")
            return

        # Scale the input features
        X_input_scaled = scaler.transform(X_input)

        # Predict
        y_predicted = model.predict(X_input_scaled)
        predicted_high_temp = y_predicted[0]
        predicted_high_temp_rounded = int(
            Decimal(float(predicted_high_temp)).quantize(0, rounding=ROUND_HALF_UP)
        )

        # Round the current high temperature
        current_high_temp_rounded = int(
            Decimal(float(max_temp_so_far_today)).quantize(0, rounding=ROUND_HALF_UP)
        )

        # Fetch the timestamp of the last data point
        last_timestamp = current_data["Date"].max()
        human_readable_timestamp = last_timestamp.strftime("%Y-%m-%d %H:%M:%S")

        # Print the current day's high temperature and timestamp for manual QC
        print(
            f"\nToday's Current High Temperature So Far: {max_temp_so_far_today:.2f} °F ({current_high_temp_rounded})"
        )
        print(f"Last Data Point Timestamp: {human_readable_timestamp}")
        print(
            f"Predicted High Temperature for Today: {predicted_high_temp:.2f} °F ({predicted_high_temp_rounded})"
        )

    except Exception as e:
        print(f"An error occurred during prediction: {e}")


# Run the prediction
predict_todays_high(station_id)

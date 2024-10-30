# predict_today.py

import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler
import datetime
from weather_utils import get_weather_data
from decimal import Decimal, ROUND_HALF_UP
import numpy as np

# Prompt the user to enter the station ID
station_id = input("Enter your station ID (e.g., KMIA): ").strip().upper()

# Validate the station ID
if len(station_id) != 4:
    raise ValueError("Station ID must be exactly 4 characters long.")

def predict_todays_high(station_id):
    print("Training the linear regression model...")

    # Fetch extended training data
    training_start_date = datetime.datetime.now() - datetime.timedelta(days=60)  # Use last 60 days
    training_end_date = datetime.datetime.now() - datetime.timedelta(days=1)

    # Fetch training data using get_weather_data
    training_data = get_weather_data(
        station_id,
        start_time=training_start_date.strftime('%Y%m%d0000'),
        end_time=training_end_date.strftime('%Y%m%d2359')
    )

    if training_data is None or training_data.empty:
        print("Unable to fetch training data.")
        return

    # Rename columns to match those used during training
    training_data.rename(columns={
        'Humidity': 'Humidity_Avg',
        'Wind_Speed': 'Wind_Speed_Avg'
    }, inplace=True)

    # Ensure necessary columns are present
    required_columns = ['Humidity_Avg', 'Wind_Speed_Avg', 'Temp_High_6hr']
    training_data.dropna(subset=required_columns, inplace=True)

    # Remove outliers from training data
    training_data = training_data[
        (training_data['Humidity_Avg'] >= 0) & (training_data['Humidity_Avg'] <= 100) &
        (training_data['Wind_Speed_Avg'] >= 0) & (training_data['Wind_Speed_Avg'] <= 100) &
        (training_data['Temp_High_6hr'] >= -50) & (training_data['Temp_High_6hr'] <= 150)
    ]

    # Prepare training features and target
    X_train = training_data[['Humidity_Avg', 'Wind_Speed_Avg']]
    y_train = training_data['Temp_High_6hr']

    # Initialize the scaler
    scaler = StandardScaler()

    # Fit the scaler on training data and transform
    X_train_scaled = scaler.fit_transform(X_train)

    # Train the model on scaled data
    model = LinearRegression()
    model.fit(X_train_scaled, y_train)

    print("Model training completed.")
    print("Starting prediction for today's high temperature...")

    # Fetch current weather data
    current_time = datetime.datetime.now()
    start_time = current_time.strftime('%Y%m%d0000')
    end_time = current_time.strftime('%Y%m%d%H%M')

    # Fetch current weather data
    current_data = get_weather_data(
        station_id,
        start_time=start_time,
        end_time=end_time
    )

    if current_data is None or current_data.empty:
        print("Unable to fetch current weather data.")
        return

    # Rename columns to match training data
    current_data.rename(columns={
        'Humidity': 'Humidity_Avg',
        'Wind_Speed': 'Wind_Speed_Avg'
    }, inplace=True)

    # Ensure necessary columns are present
    required_columns = ['Humidity_Avg', 'Wind_Speed_Avg']
    missing_columns = [col for col in required_columns if col not in current_data.columns]
    if missing_columns:
        print(f"Missing required data for prediction: {missing_columns}")
        return

    # Prepare data for prediction
    X_real_time = current_data[['Humidity_Avg', 'Wind_Speed_Avg']]

    # Check for NaN values
    if X_real_time.isnull().values.any():
        print("NaN values found in current features:")
        print(X_real_time)
        X_real_time = X_real_time.dropna()
        if X_real_time.empty:
            print("No valid data available after dropping NaNs.")
            return

    try:
        if X_real_time.empty:
            print("No valid data to make a prediction.")
            return

        # Take the last valid row
        X_input = X_real_time.iloc[[-1]]

        # Scale the input features
        X_input_scaled = scaler.transform(X_input)

        # Predict
        y_predicted = model.predict(X_input_scaled)
        predicted_high_temp = y_predicted[0]
        predicted_high_temp_rounded = int(Decimal(float(predicted_high_temp)).quantize(0, rounding=ROUND_HALF_UP))

        # Print the predicted temperature
        print(f"Predicted High Temperature for Today: {predicted_high_temp:.2f} °F ({predicted_high_temp_rounded})")

    except Exception as e:
        print(f"An error occurred during prediction: {e}")

# Run the prediction
predict_todays_high(station_id)

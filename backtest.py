# backtest.py

import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler
import datetime
from weather_utils import get_weather_data
from decimal import Decimal, ROUND_HALF_UP
import numpy as np

# Include the official data in a dictionary
official_data = {
    'Date': [
        'Oct 1', 'Oct 2', 'Oct 3', 'Oct 4', 'Oct 5', 'Oct 6', 'Oct 7', 'Oct 8', 'Oct 9', 'Oct 10',
        'Oct 11', 'Oct 12', 'Oct 13', 'Oct 14', 'Oct 15', 'Oct 16', 'Oct 17', 'Oct 18', 'Oct 19', 'Oct 20',
        'Oct 21', 'Oct 22', 'Oct 23', 'Oct 24', 'Oct 25', 'Oct 26', 'Oct 27', 'Oct 28', 'Oct 29', 'Oct 30'
    ],
    'High_Temperature': [
        88, 87, 89, 89, 90, 89, 88, 87, 86, 85,
        84, 85, 86, 86, 87, 87, 86, 85, 84, 85,
        85, 84, 85, 85, 85, 86, 83, 83, 85, 84
    ]
}

# Convert official data to a DataFrame
official_df = pd.DataFrame(official_data)
official_df['Date'] = pd.to_datetime('2024 ' + official_df['Date'])

# Prompt the user to enter the station ID
station_id = input("Enter your station ID (e.g., KMIA): ").strip().upper()

# Validate the station ID
if len(station_id) != 4:
    raise ValueError("Station ID must be exactly 4 characters long.")

# Function to perform backtesting
def backtest_predictions(station_id, backtest_days=30):
    print(f"Starting backtest for the past {backtest_days} days...")

    # Initialize lists for actual and predicted temperatures
    actual_temps = []
    predicted_temps = []
    dates = []

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

    for day_offset in range(backtest_days):
        # Calculate target date
        target_date = datetime.datetime.now() - datetime.timedelta(days=day_offset + 1)
        formatted_date = target_date.strftime('* %a %m/%d/%y')
        print(f"\nProcessing data for {formatted_date}")

        # Fetch historical weather data
        start_time = target_date.strftime('%Y%m%d0000')
        end_time = target_date.strftime('%Y%m%d2359')
        historical_data = get_weather_data(station_id, start_time=start_time, end_time=end_time)

        if historical_data is None or historical_data.empty:
            print("Unable to fetch historical data for backtesting.")
            continue

        # Rename columns to match training data
        historical_data.rename(columns={
            'Humidity': 'Humidity_Avg',
            'Wind_Speed': 'Wind_Speed_Avg'
        }, inplace=True)

        # Ensure necessary columns are present
        required_columns = ['Humidity_Avg', 'Wind_Speed_Avg', 'Temp_High_6hr', 'Temperature']
        missing_columns = [col for col in required_columns if col not in historical_data.columns]
        if missing_columns:
            print(f"Missing required data for backtesting: {missing_columns}")
            continue

        # Prepare data for prediction
        X_real_time = historical_data[['Humidity_Avg', 'Wind_Speed_Avg']]

        # Check for NaN values
        if X_real_time.isnull().values.any():
            print("NaN values found in features:")
            print(X_real_time)
            X_real_time = X_real_time.dropna()
            if X_real_time.empty:
                print("No valid data available after dropping NaNs.")
                continue

        # Use 'Temp_High_6hr' to find the actual high temperature
        y_actual = historical_data['Temp_High_6hr'].max()

        # Fetch official high temperature for comparison
        official_high = official_df[official_df['Date'].dt.date == target_date.date()]['High_Temperature']
        if not official_high.empty:
            official_high_temp = official_high.values[0]
        else:
            official_high_temp = None

        # Ensure that official_high_temp is a native Python type
        if isinstance(official_high_temp, np.generic):
            official_high_temp = official_high_temp.item()

        # Use Decimal to round temperatures
        y_actual_rounded = int(Decimal(float(y_actual)).quantize(0, rounding=ROUND_HALF_UP))
        if official_high_temp is not None:
            official_high_temp_rounded = int(Decimal(float(official_high_temp)).quantize(0, rounding=ROUND_HALF_UP))
        else:
            official_high_temp_rounded = None

        try:
            if X_real_time.empty:
                print("No valid data to make a prediction for this date.")
                continue

            # Ensure the input to predict has the correct feature names
            # Take the last valid row
            X_input = X_real_time.iloc[[-1]]

            # Scale the input features
            X_input_scaled = scaler.transform(X_input)

            # Predict
            y_predicted = model.predict(X_input_scaled)
            predicted_high_temp = y_predicted[0]
            predicted_high_temp_rounded = int(Decimal(float(predicted_high_temp)).quantize(0, rounding=ROUND_HALF_UP))

            # Append results to lists
            actual_temps.append(y_actual)
            predicted_temps.append(predicted_high_temp)
            dates.append(target_date.date())

            # Print the original and rounded temperatures
            print(f"Fetched High Temperature: {y_actual:.2f} °F ({y_actual_rounded})")
            print(f"Official High Temperature: {official_high_temp} °F ({official_high_temp_rounded})")
            print(f"Predicted High Temperature: {predicted_high_temp:.2f} °F ({predicted_high_temp_rounded})")
        except Exception as e:
            print(f"An error occurred during prediction: {e}")
            continue

    # Create a DataFrame to compare results
    results_df = pd.DataFrame({
        'Date': dates,
        'Fetched_High_Temp': [f"{a:.2f} ({int(Decimal(float(a)).quantize(0, rounding=ROUND_HALF_UP))})" for a in actual_temps],
        'Official_High_Temp': [
            f"{official_df[official_df['Date'].dt.date == date]['High_Temperature'].values[0]} ({int(Decimal(float(official_df[official_df['Date'].dt.date == date]['High_Temperature'].values[0])).quantize(0, rounding=ROUND_HALF_UP))})"
            if not official_df[official_df['Date'].dt.date == date].empty else None for date in dates
        ],
        'Predicted_High_Temp': [f"{p:.2f} ({int(Decimal(float(p)).quantize(0, rounding=ROUND_HALF_UP))})" for p in predicted_temps]
    })

    print("\nComparison of Fetched and Official High Temperatures:")
    print(results_df)

    # Calculate Mean Absolute Error (MAE) between fetched and predicted temperatures
    if actual_temps and predicted_temps:
        mae = sum(abs(a - p) for a, p in zip(actual_temps, predicted_temps)) / len(actual_temps)
        print(f"\nBacktesting completed. Mean Absolute Error over {len(actual_temps)} days: {mae:.2f} °F")
    else:
        print("No predictions were made during backtesting.")

# Run the backtest
backtest_predictions(station_id, backtest_days=30)

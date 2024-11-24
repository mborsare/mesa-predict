"""Enhanced ML model training with sophisticated feature engineering."""
import logging
from datetime import datetime, timedelta
from typing import Dict, Any, Tuple
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
import xgboost as xgb
import ephem  # for solar calculations

def calculate_solar_features(date: datetime, lat: float, lon: float) -> Dict[str, float]:
    """Calculate solar position features for temperature prediction."""
    obs = ephem.Observer()
    obs.lat = str(lat)
    obs.lon = str(lon)
    obs.date = date
    sun = ephem.Sun()
    sun.compute(obs)

    # Calculate solar noon
    next_sunrise = ephem.localtime(obs.next_rising(sun))
    next_sunset = ephem.localtime(obs.next_setting(sun))
    solar_noon = next_sunrise + (next_sunset - next_sunrise) / 2

    # Time to/from solar noon in hours
    time_to_noon = (solar_noon - date).total_seconds() / 3600

    # Solar elevation and azimuth
    solar_elevation = float(sun.alt) * 180 / np.pi
    solar_azimuth = float(sun.az) * 180 / np.pi

    return {
        'time_to_solar_noon': time_to_noon,
        'solar_elevation': solar_elevation,
        'solar_azimuth': solar_azimuth,
        'day_length': (next_sunset - next_sunrise).total_seconds() / 3600
    }

def find_similar_days(df: pd.DataFrame, target_date: datetime, n_days: int = 5) -> pd.DataFrame:
    """Find historically similar days based on multiple criteria."""
    target_doy = target_date.timetuple().tm_yday

    # Calculate similarity scores
    scores = []
    for date in df['Date'].unique():
        doy = date.timetuple().tm_yday

        # Circular day of year difference
        doy_diff = min(abs(doy - target_doy), 365 - abs(doy - target_doy))

        # Temperature pattern similarity (if available)
        temp_pattern = 0
        if len(df[df['Date'] == date]) > 0:
            pattern_diff = abs(df[df['Date'] == date]['Actual'].mean() -
                             df[df['Date'] == target_date]['Actual'].mean())
            temp_pattern = 1 / (1 + pattern_diff)

        # Combined similarity score
        similarity = (1 / (1 + doy_diff)) + temp_pattern
        scores.append((date, similarity))

    # Get top N similar days
    similar_days = sorted(scores, key=lambda x: x[1], reverse=True)[:n_days]
    return df[df['Date'].isin([day[0] for day in similar_days])]

def prepare_features(df: pd.DataFrame, station_metadata: Dict = None) -> pd.DataFrame:
    """Prepare enhanced features for ML model."""
    # Convert date to datetime if it isn't already
    df['Date'] = pd.to_datetime(df['Date'])

    features = pd.DataFrame()

    # Basic time features with cyclical encoding
    features['hour'] = df['Actual_Time'].dt.hour
    features['day_of_year'] = df['Date'].dt.dayofyear
    features['month'] = df['Date'].dt.month
    features['day_of_week'] = df['Date'].dt.dayofweek

    # Cyclical encoding
    for col, max_val in [('hour', 24), ('day_of_year', 365), ('month', 12)]:
        features[f'{col}_sin'] = np.sin(2 * np.pi * features[col] / max_val)
        features[f'{col}_cos'] = np.cos(2 * np.pi * features[col] / max_val)

    # Temperature features
    features['prev_day_high'] = df['Actual'].shift(1)
    features['prev_day_pred'] = df['Predicted'].shift(1)
    features['temp_change'] = df['Actual'].diff()

    # Multiple rolling windows for temperature patterns
    for window in [3, 5, 7, 14]:
        features[f'rolling_mean_{window}d'] = df['Actual'].rolling(window, min_periods=1).mean()
        features[f'rolling_std_{window}d'] = df['Actual'].rolling(window, min_periods=1).std()
        features[f'rolling_max_{window}d'] = df['Actual'].rolling(window, min_periods=1).max()
        features[f'rolling_min_{window}d'] = df['Actual'].rolling(window, min_periods=1).min()
        features[f'rolling_range_{window}d'] = (
            features[f'rolling_max_{window}d'] - features[f'rolling_min_{window}d']
        )

    # Temperature change features
    features['temp_acceleration'] = features['temp_change'].diff()
    features['temp_jerk'] = features['temp_acceleration'].diff()  # Rate of change of acceleration

    # Daily temperature swing
    features['daily_swing'] = features['rolling_max_1d'] - features['rolling_min_1d']
    features['swing_change'] = features['daily_swing'].diff()

    # Pattern matching features
    for row_idx in range(len(df)):
        target_date = df.iloc[row_idx]['Actual_Time']
        similar_days = find_similar_days(df, target_date)
        features.at[row_idx, 'similar_days_mean'] = similar_days['Actual'].mean()
        features.at[row_idx, 'similar_days_std'] = similar_days['Actual'].std()

    # Add solar features if station metadata available
    if station_metadata and 'latitude' in station_metadata and 'longitude' in station_metadata:
        solar_features = df['Actual_Time'].apply(
            lambda x: calculate_solar_features(x, station_metadata['latitude'], station_metadata['longitude'])
        ).to_list()

        for feature in ['time_to_solar_noon', 'solar_elevation', 'solar_azimuth', 'day_length']:
            features[feature] = [d[feature] for d in solar_features]

    # Seasonal decomposition features
    if len(df) > 14:  # Need enough data for decomposition
        decomposition = pd.DataFrame(index=df.index)
        decomposition['Actual'] = df['Actual']
        decomposition['MA7'] = decomposition['Actual'].rolling(window=7, center=True).mean()
        decomposition['MA14'] = decomposition['Actual'].rolling(window=14, center=True).mean()
        features['seasonal_residual'] = decomposition['Actual'] - decomposition['MA7']
        features['trend_residual'] = decomposition['MA7'] - decomposition['MA14']

    # Fill NaN values using forward fill then backward fill
    features = features.fillna(method='ffill').fillna(method='bfill')

    return features

def train_evaluate_model(
    df: pd.DataFrame,
    logger: logging.Logger,
    station_metadata: Dict = None,
    train_ratio: float = 0.7
) -> Tuple[xgb.XGBRegressor, StandardScaler, Dict[str, float]]:
    """Train and evaluate ML model with enhanced features."""
    # Prepare features
    features_df = prepare_features(df, station_metadata)

    # Sort by date and split train/validation
    n_train = int(len(df) * train_ratio)
    train_idx = df.index[:n_train]
    val_idx = df.index[n_train:]

    # Prepare X and y
    X = features_df
    y = df['Actual']

    # Split into train and validation sets
    X_train, X_val = X.loc[train_idx], X.loc[val_idx]
    y_train, y_val = y.loc[train_idx], y.loc[val_idx]

    # Scale features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_val_scaled = scaler.transform(X_val)

    # Initialize model with more careful parameters
    model = xgb.XGBRegressor(
        n_estimators=200,
        max_depth=6,
        learning_rate=0.05,
        min_child_weight=2,
        subsample=0.8,
        colsample_bytree=0.8,
        objective='reg:squarederror',
        random_state=42
    )

    # Train model
    logger.info("Training enhanced ML model...")
    model.fit(X_train_scaled, y_train)

    # Make predictions
    train_pred = model.predict(X_train_scaled)
    val_pred = model.predict(X_val_scaled)

    # Calculate metrics
    metrics = {
        'train_mae': np.mean(np.abs(y_train - train_pred)),
        'train_rmse': np.sqrt(np.mean((y_train - train_pred) ** 2)),
        'val_mae': np.mean(np.abs(y_val - val_pred)),
        'val_rmse': np.sqrt(np.mean((y_val - val_pred) ** 2)),
        'feature_importance': dict(zip(X_train.columns, model.feature_importances_))
    }

    # Log detailed results
    logger.info("\nEnhanced ML Model Performance:")
    logger.info(f"Training MAE: {metrics['train_mae']:.2f}°F")
    logger.info(f"Training RMSE: {metrics['train_rmse']:.2f}°F")
    logger.info(f"Validation MAE: {metrics['val_mae']:.2f}°F")
    logger.info(f"Validation RMSE: {metrics['val_rmse']:.2f}°F")
    logger.info("\nTop 10 Important Features:")
    importance_sorted = sorted(
        metrics['feature_importance'].items(),
        key=lambda x: x[1],
        reverse=True
    )[:10]
    for feature, importance in importance_sorted:
        logger.info(f"{feature}: {importance:.3f}")

    return model, scaler, metrics
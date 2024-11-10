"""Prediction module containing core prediction functionality."""
from .temperature_predictor import TemperaturePredictor
from .base_predictor import BasePredictor
from .trend_analyzer import TrendAnalyzer
from .prediction_weights import PredictionWeights

__all__ = [
    'TemperaturePredictor',
    'BasePredictor',
    'TrendAnalyzer',
    'PredictionWeights'
]
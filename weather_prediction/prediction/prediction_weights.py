"""Handles weight calculations for different prediction sources."""
from decimal import Decimal
from typing import Dict

from ..models import DataSource

class PredictionWeights:
    """Manages weights for different prediction sources."""

    def __init__(self):
        # Regular prediction weights
        self._source_weights = {
            DataSource.TOMORROW_IO: Decimal('0.3'),    # Future forecast
            DataSource.REGULAR: Decimal('0.2'),        # Current conditions
            DataSource.SIX_HOUR: Decimal('0.2'),       # Recent history
            DataSource.HISTORICAL: Decimal('0.2'),     # Long-term pattern
            DataSource.ML_MODEL: Decimal('0.1')        # Machine learning prediction
        }

        # Backtesting weights (adjusted for 6-hour constraint)
        self._backtest_weights = {
            DataSource.TOMORROW_IO: Decimal('0.2'),    # Limited historical data
            DataSource.HISTORICAL: Decimal('0.5'),     # More weight on historical
            DataSource.SIX_HOUR: Decimal('0.3')        # Recent observations
        }

    def get_weights(
        self,
        is_backtest: bool,
        trend_change: float,
        trend_confidence: float
    ) -> Dict[DataSource, Decimal]:
        """Get appropriate weights based on context and trends."""
        weights = self._backtest_weights if is_backtest else self._source_weights

        # Adjust weights based on trend magnitude and confidence
        if abs(trend_change) > 5 and trend_confidence > 70:
            if is_backtest:
                return {
                    DataSource.TOMORROW_IO: Decimal('0.3'),    # Increase weight for strong trends
                    DataSource.HISTORICAL: Decimal('0.4'),
                    DataSource.SIX_HOUR: Decimal('0.3')
                }
            else:
                return {
                    DataSource.TOMORROW_IO: Decimal('0.35'),
                    DataSource.REGULAR: Decimal('0.2'),
                    DataSource.SIX_HOUR: Decimal('0.2'),
                    DataSource.HISTORICAL: Decimal('0.15'),
                    DataSource.ML_MODEL: Decimal('0.1')
                }

        return weights
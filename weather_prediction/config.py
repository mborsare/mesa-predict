"""Configuration management for weather prediction system."""
from dataclasses import dataclass
from decimal import Decimal
from functools import lru_cache
import logging
import os
from pathlib import Path
from typing import Optional

@dataclass
class APIConfig:
    """API configuration settings."""
    base_url: str
    token: str
    timeout: int = 30
    historical_enabled: bool = True  # New flag for historical data
    cache_duration: int = 3600      # Cache historical data for 1 hour

@dataclass(frozen=True)
class Config:
    """Application configuration."""
    mesowest: APIConfig
    tomorrow_io: APIConfig
    log_level: str = "INFO"
    log_format: str = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    log_file: Optional[Path] = None
    timezone: str = "UTC"
    temperature_min: Decimal = Decimal("-100")
    temperature_max: Decimal = Decimal("150")
    cache_ttl: int = 300  # seconds

    @classmethod
    @lru_cache(maxsize=1)
    def load(cls) -> 'Config':
        """Load configuration from environment and defaults."""
        return cls(
            mesowest=APIConfig(
                base_url="https://api.mesowest.net/v2/stations",
                token=os.getenv(
                    "MESOWEST_TOKEN",
                    "d8c6aee36a994f90857925cea26934be"
                ),
            ),
            tomorrow_io=APIConfig(
                base_url="https://api.tomorrow.io/v4",
                token=os.getenv(
                    "TOMORROW_IO_TOKEN",
                    "bZkgwxlNihNKB0dlPnD55rMi3EYAEaew"
                ),
            ),
            log_level=os.getenv("LOG_LEVEL", "INFO"),
            timezone=os.getenv("TIMEZONE", "UTC"),
            log_file=Path(os.getenv("LOG_FILE")) if os.getenv("LOG_FILE") else None
        )

def setup_logging(config: Config) -> logging.Logger:
    """Configure application logging."""
    logger = logging.getLogger("weather_prediction")
    logger.setLevel(config.log_level)

    formatter = logging.Formatter(config.log_format)

    # Always log to stdout
    console_handler = logging.StreamHandler()
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)

    # Optionally log to file
    if config.log_file:
        file_handler = logging.handlers.RotatingFileHandler(
            config.log_file,
            maxBytes=10_485_760,  # 10MB
            backupCount=5
        )
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)

    return logger
# predict-today.py
from data_processor import WeatherDataProcessor
from config import API_CONFIG
from tomorrow_utils import TomorrowioClient, EnhancedTemperaturePredictor
import sys
import logging
from datetime import datetime

def setup_logging():
    """Configure logging for the application"""
    logging.basicConfig(
        level=logging.DEBUG,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.StreamHandler(sys.stdout)
        ]
    )
    return logging.getLogger(__name__)

def main():
    logger = setup_logging()

    try:
        station_id = input("Enter your station ID (e.g., KMIA): ").strip().upper()
        logger.info(f"Processing weather data for station: {station_id}")

        weather_processor = WeatherDataProcessor(station_id)

        # Debug block to examine raw MesoWest data
        print("\n=== DEBUG: Raw MesoWest Data ===")
        data = weather_processor._fetch_data()
        if data and 'STATION' in data:
            obs = data['STATION'][0].get('OBSERVATIONS', {})
            dates = obs.get('date_time', [])
            temps = obs.get('air_temp_set_1', [])
            six_hr = obs.get('air_temp_high_6_hour_set_1', [])

            print("\nToday's readings:")
            today = datetime.utcnow().date()
            for date_str, temp, high6 in zip(dates, temps, six_hr):
                date = datetime.strptime(date_str, '%Y-%m-%dT%H:%M:%SZ').date()
                if date == today:
                    print(f"{date_str}: Temp={temp}°F, 6hr_high={high6}°F")
        print("\n=== End Debug ===\n")

        coordinates = weather_processor.get_coordinates()
        if not coordinates:
            logger.error("Failed to get station coordinates")
            return

        logger.info(f"Station coordinates: {coordinates}")

        tomorrow_client = TomorrowioClient(API_CONFIG["tomorrow_io_token"])
        predictor = EnhancedTemperaturePredictor(
            weather_processor,
            tomorrow_client,
            debug_mode=True
        )

        high_temp = predictor.predict_todays_high()

        if high_temp is not None:
            logger.info("Final prediction complete")
        else:
            logger.error("Failed to generate prediction")

    except Exception as e:
        logger.error(f"An error occurred: {e}", exc_info=True)

if __name__ == "__main__":
    main()
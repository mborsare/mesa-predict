from weather_utils import WeatherDataProcessor, API_CONFIG
from tomorrow_utils import TomorrowioClient, EnhancedTemperaturePredictor

def main():
    try:
        # Get station ID
        station_id = input("Enter your station ID (e.g., KMIA): ").strip().upper()

        # Initialize clients
        weather_processor = WeatherDataProcessor(station_id)
        tomorrow_client = TomorrowioClient(API_CONFIG["tomorrow_io_token"])

        # Get station coordinates
        coordinates = weather_processor.get_coordinates()
        print(f"Station coordinates are: {coordinates}")

        # Initialize enhanced predictor
        predictor = EnhancedTemperaturePredictor(
            weather_processor,
            tomorrow_client
        )

        # Make prediction
        high_temp = predictor.predict_todays_high()
        if high_temp is not None:
            print(f"\nFinal prediction for today's high temperature: {high_temp}°F")
        else:
            print("\nFailed to generate prediction")

    except Exception as e:
        print(f"\nAn error occurred: {e}")

if __name__ == "__main__":
    main()
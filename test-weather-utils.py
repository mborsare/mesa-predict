# test_weather_utils.py

from weather_utils import get_station_metadata, get_weather_data

station_id = 'KMIA'  # Replace with your desired station ID

# Test fetching station metadata
metadata = get_station_metadata(station_id)
print("Station Metadata:", metadata)

# Test fetching weather data
# Define start and end times (e.g., October 1, 2024)
start_time = '202410010000'  # YYYYMMDDHHMM
end_time = '202410012359'    # YYYYMMDDHHMM

weather_data = get_weather_data(station_id, start_time=start_time, end_time=end_time)
print("Weather Data:")
print(weather_data.head())

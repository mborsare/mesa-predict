from setuptools import setup, find_packages

setup(
    name="weather_prediction",
    version="0.1",
    packages=find_packages(),
    install_requires=[
        "aiohttp>=3.8.0",
        "pytz>=2023.3",
        "numpy>=1.24.0",
        "pyyaml>=6.0.0",
        "python-dotenv>=1.0.0"
    ],
)
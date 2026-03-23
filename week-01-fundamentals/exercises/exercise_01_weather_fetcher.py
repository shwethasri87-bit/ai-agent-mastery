"""
Exercise 1: Weather Fetcher
=============================
Difficulty: Beginner | Time: 1.5 hours

Task:
Create a function that takes a city name, calls the wttr.in API,
and returns a dictionary with temperature and conditions.
Handle errors gracefully.

Instructions:
1. Complete the fetch_weather() function below
2. Handle edge cases: empty city, API timeout, invalid city
3. Test with at least 3 different cities
4. Bonus: Add wind speed and humidity to the output

Run: python exercise_01_weather_fetcher.py
"""

import requests


def fetch_weather(city: str) -> dict:
    """Fetch weather data for a given city.

    Args:
        city: Name of the city

    Returns:
        Dictionary with keys: city, temperature_c, conditions

    Raises:
        ValueError: If city is empty or API call fails
    """
    if not city or not city.strip():
        raise ValueError("City name cannot be empty")

    url = f"https://wttr.in/{city.strip()}?format=j1"
    try:
        response = requests.get(url, timeout=10)
        
        if response.status_code != 200:
            raise ValueError(f"API returned status {response.status_code} for city '{city}'")
            
        try:
            data = response.json()
        except ValueError:
            raise ValueError(f"Invalid city or API error for '{city}'")
            
        current = data.get("current_condition", [{}])[0]
        
        return {
            "city": city,
            "temperature_c": current.get("temp_C"),
            "conditions": current.get("weatherDesc", [{}])[0].get("value", "Unknown"),
            "wind_speed_kmph": current.get("windspeedKmph"),
            "humidity": current.get("humidity")
        }
    except requests.Timeout:
        raise ValueError(f"API call for '{city}' timed out")
    except requests.RequestException as e:
        raise ValueError(f"API call for '{city}' failed: {e}")


# === Test your implementation ===
if __name__ == "__main__":
    # Test 1: Valid city
    print("Test 1: London")
    result = fetch_weather("London")
    print(result)

    # Test 2: Another valid city
    print("Test 2: Tokyo")
    result = fetch_weather("Tokyo")
    print(result)

      # Test 3: Another valid city
    print("Test 3: India")
    result = fetch_weather("India")
    print(result)

    # Test 4: Error handling - empty city
    print("Test 4: Empty city (should raise ValueError)")
    try:
        result = fetch_weather("")
    except ValueError as e:
        print(f"Caught error: {e}")

"""
Weather tool implementation for Intuit.
"""
import os
import aiohttp
from typing import Dict, Any, Optional, Tuple
from pydantic import Field, PrivateAttr

from .base import BaseTool

class WeatherTool(BaseTool):
    """Tool for getting weather information."""
    
    name: str = Field(default="weather")
    description: str = Field(
        default="Get current weather and forecast information for a location."
    )
    
    _api_key: str = PrivateAttr(default=None)
    _client: Optional[aiohttp.ClientSession] = PrivateAttr(default=None)
    
    def __init__(self, **data):
        super().__init__(**data)
        self._api_key = os.getenv("WEATHER_API_KEY")
        if not self._api_key:
            raise ValueError("WEATHER_API_KEY environment variable is not set")
        self._client = aiohttp.ClientSession()
    
    async def _get_location_coords(self, location: str) -> Optional[Tuple[float, float]]:
        """Get coordinates for a location."""
        url = f"http://api.openweathermap.org/geo/1.0/direct"
        params = {
            "q": location,
            "limit": 1,
            "appid": self._api_key
        }
        
        async with self._client.get(url, params=params) as response:
            data = await response.json()
            if not data:
                return None
            return (data[0]["lat"], data[0]["lon"])
    
    async def _get_weather_data(self, lat: float, lon: float) -> Dict[str, Any]:
        """Get current weather data."""
        url = "http://api.openweathermap.org/data/2.5/weather"
        params = {
            "lat": lat,
            "lon": lon,
            "appid": self._api_key,
            "units": "metric"
        }
        
        async with self._client.get(url, params=params) as response:
            return await response.json()
    
    async def _get_forecast(self, lat: float, lon: float) -> Dict[str, Any]:
        """Get weather forecast."""
        url = "http://api.openweathermap.org/data/2.5/forecast"
        params = {
            "lat": lat,
            "lon": lon,
            "appid": self._api_key,
            "units": "metric"
        }
        
        async with self._client.get(url, params=params) as response:
            return await response.json()
    
    async def run(self, location: str) -> Dict[str, Any]:
        """
        Get weather information for a location.
        
        Args:
            location: Name of the location (e.g., "London, UK")
            
        Returns:
            Dict containing current weather and forecast
        """
        try:
            # Get coordinates
            coords = await self._get_location_coords(location)
            if not coords:
                return {"error": f"Location not found: {location}"}
            
            lat, lon = coords
            
            # Get current weather and forecast
            current = await self._get_weather_data(lat, lon)
            forecast = await self._get_forecast(lat, lon)
            
            return {
                "location": location,
                "current": {
                    "temperature": current["main"]["temp"],
                    "humidity": current["main"]["humidity"],
                    "description": current["weather"][0]["description"],
                    "wind_speed": current["wind"]["speed"]
                },
                "forecast": [
                    {
                        "time": item["dt"],
                        "temperature": item["main"]["temp"],
                        "description": item["weather"][0]["description"]
                    }
                    for item in forecast["list"][:5]  # Next 5 time slots
                ]
            }
        except Exception as e:
            return {"error": str(e)}
    
    async def __aenter__(self):
        """Async context manager entry."""
        return self
    
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Async context manager exit."""
        if self._client:
            await self._client.close() 
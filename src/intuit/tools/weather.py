"""
Weather tool implementation for Intuit.
"""
import os
import logging
import aiohttp
from typing import Dict, Any, Optional, Tuple
from pydantic import Field, PrivateAttr

from .basetool import BaseTool

logger = logging.getLogger(__name__)

class WeatherTool(BaseTool):
    """Tool for getting weather information."""
    
    name: str = Field(default="weather")
    description: str = Field(
        default="Get current weather and forecast information for a location. Input should be a location name (e.g., 'London, UK' or 'New York, NY')."
    )
    
    _api_key: str = PrivateAttr(default=None)
    _client: Optional[aiohttp.ClientSession] = PrivateAttr(default=None)
    
    def __init__(self, **data):
        super().__init__(**data)
        self._api_key = os.getenv("WEATHER_API_KEY")
        if not self._api_key:
            raise ValueError("WEATHER_API_KEY environment variable is not set")
    
    async def _ensure_client(self):
        """Ensure the aiohttp client session exists."""
        if self._client is None or self._client.closed:
            self._client = aiohttp.ClientSession()
    
    async def _get_location_coords(
        self, location: str
    ) -> Optional[Tuple[float, float]]:
        """Get coordinates for a location."""
        logger.debug(f"Getting coordinates for location: {location}")
        await self._ensure_client()
        url = f"http://api.openweathermap.org/geo/1.0/direct"
        params = {
            "q": location,
            "limit": 1,
            "appid": self._api_key
        }
        
        async with self._client.get(url, params=params) as response:
            data = await response.json()
            if not data:
                logger.warning(f"No coordinates found for location: {location}")
                return None
            coords = (data[0]["lat"], data[0]["lon"])
            logger.debug(f"Found coordinates: {coords}")
            return coords
    
    async def _get_weather_data(
        self, lat: float, lon: float
    ) -> Dict[str, Any]:
        """Get current weather data."""
        logger.debug(f"Fetching current weather for coords: ({lat}, {lon})")
        await self._ensure_client()
        url = "http://api.openweathermap.org/data/2.5/weather"
        params = {
            "lat": lat,
            "lon": lon,
            "appid": self._api_key,
            "units": "metric"
        }
        
        async with self._client.get(url, params=params) as response:
            data = await response.json()
            logger.debug(
                f"Current weather: {data.get('main', {}).get('temp')}°C, "
                f"{data.get('weather', [{}])[0].get('description', 'N/A')}"
            )
            return data
    
    async def _get_forecast(self, lat: float, lon: float) -> Dict[str, Any]:
        """Get weather forecast."""
        await self._ensure_client()
        url = "http://api.openweathermap.org/data/2.5/forecast"
        params = {
            "lat": lat,
            "lon": lon,
            "appid": self._api_key,
            "units": "metric"
        }
        
        async with self._client.get(url, params=params) as response:
            return await response.json()
    
    async def _arun(self, location: str) -> Dict[str, Any]:
        """
        Get weather information for a location.
        
        Args:
            location: Name of the location (e.g., "London, UK")
            
        Returns:
            Dict containing current weather and forecast
        """
        logger.info(f"WeatherTool called with location: {location}")
        try:
            # Get coordinates
            coords = await self._get_location_coords(location)
            if not coords:
                error_msg = f"Location not found: {location}"
                logger.error(error_msg)
                return {"error": error_msg}
            
            lat, lon = coords
            logger.debug(f"Using coordinates: lat={lat}, lon={lon}")
            
            # Get current weather and forecast
            current = await self._get_weather_data(lat, lon)
            forecast = await self._get_forecast(lat, lon)
            
            # Close the client session after we're done
            if self._client and not self._client.closed:
                await self._client.close()
            
            result = {
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
            
            logger.info(
                f"WeatherTool response: {result['current']['temperature']}°C, "
                f"{result['current']['description']}"
            )
            logger.debug(f"Full weather response: {result}")
            
            return result
        except Exception as e:
            # Ensure client is closed even if there's an error
            if self._client and not self._client.closed:
                await self._client.close()
            error_msg = str(e)
            logger.error(f"WeatherTool error: {error_msg}", exc_info=True)
            return {"error": error_msg}
    
    async def __aenter__(self):
        """Async context manager entry."""
        await self._ensure_client()
        return self
    
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Async context manager exit."""
        if self._client and not self._client.closed:
            await self._client.close()
    
    async def get_weather_async(self, location: str) -> str:
        """
        Get weather information for a location (async version).
        Voice-optimized: Returns only current temperature and conditions.
        
        Args:
            location: Name of the location (e.g., "London, UK")
            
        Returns:
            Concise weather information (current conditions only)
        """
        logger.debug(f"get_weather_async called with location: {location}")
        try:
            result = await self._arun(location)
            
            if "error" in result:
                error_msg = f"Error: {result['error']}"
                logger.error(error_msg)
                return error_msg
            
            # Voice-friendly format: Just current temp and conditions
            current = result["current"]
            temp = current['temperature']
            conditions = current['description']
            
            # Simple, concise output for voice
            output = (
                f"{result['location']}: {temp}°C and {conditions}."
            )
            
            logger.info(f"Voice-optimized weather response for {location}")
            logger.debug(f"Full data available but returning concise: {output}")
            return output
        except Exception as e:
            error_msg = f"Error getting weather information: {str(e)}"
            logger.error(error_msg, exc_info=True)
            return error_msg
    
    def get_weather(self, location: str) -> str:
        """
        Get weather information for a location (sync wrapper).
        
        Args:
            location: Name of the location (e.g., "London, UK")
            
        Returns:
            String representation of weather information
        """
        import asyncio
        
        # Get the current event loop or create a new one if needed
        try:
            loop = asyncio.get_event_loop()
            if loop.is_running():
                # If we're already in an event loop, create a new task
                return "Please use the async version of this method when calling from an async context."
            else:
                # If we have a loop but it's not running, use it
                return loop.run_until_complete(self.get_weather_async(location))
        except RuntimeError:
            # If there is no event loop, create one
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            try:
                return loop.run_until_complete(self.get_weather_async(location))
            finally:
                loop.close()
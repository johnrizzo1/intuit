"""
Weather tool implementation for Intuit.
"""

import os
import aiohttp
from typing import Dict, Any, Optional, Tuple
from pydantic import Field, PrivateAttr

from .basetool import BaseTool


class WeatherTool(BaseTool):
    """Tool for getting weather information."""

    name: str = Field(default="weather")
    description: str = Field(
        default="Get current weather and forecast information for a location. Input should be a location name (e.g., 'London, UK' or 'New York, NY')."
    )

    _api_key: Optional[str] = PrivateAttr(default=None)
    _client: Optional[aiohttp.ClientSession] = PrivateAttr(default=None)

    def __init__(self, **data):
        super().__init__(**data)
        self._api_key = os.getenv("WEATHER_API_KEY")
        if not self._api_key:
            raise ValueError("WEATHER_API_KEY environment variable is not set")

    def _run(self, location: str, **kwargs: Any) -> Dict[str, Any]:
        """
        Execute the tool's functionality synchronously.
        Uses asyncio to run the async version.
        """
        import asyncio
        import concurrent.futures

        # Check if we're in an async context
        try:
            current_loop = asyncio.get_running_loop()
            # We're in an async context, use a thread pool to run the async code
            with concurrent.futures.ThreadPoolExecutor() as executor:
                future = executor.submit(asyncio.run, self._arun(location))
                return future.result()
        except RuntimeError:
            # No running loop, we can use asyncio.run directly
            return asyncio.run(self._arun(location))

    async def _ensure_client(self):
        """Ensure the aiohttp client session exists."""
        if self._client is None or self._client.closed:
            self._client = aiohttp.ClientSession()

    async def _get_location_coords(
        self, location: str
    ) -> Optional[Tuple[float, float]]:
        """Get coordinates for a location."""
        await self._ensure_client()
        url = f"http://api.openweathermap.org/geo/1.0/direct"
        params = {"q": location, "limit": 1, "appid": self._api_key or ""}

        assert self._client is not None, "Client session not initialized"
        async with self._client.get(url, params=params) as response:
            data = await response.json()
            if not data:
                return None
            return (data[0]["lat"], data[0]["lon"])

    async def _get_weather_data(self, lat: float, lon: float) -> Dict[str, Any]:
        """Get current weather data."""
        await self._ensure_client()
        url = "http://api.openweathermap.org/data/2.5/weather"
        params = {
            "lat": lat,
            "lon": lon,
            "appid": self._api_key or "",
            "units": "metric",
        }

        assert self._client is not None, "Client session not initialized"
        async with self._client.get(url, params=params) as response:
            return await response.json()

    async def _get_forecast(self, lat: float, lon: float) -> Dict[str, Any]:
        """Get weather forecast."""
        await self._ensure_client()
        url = "http://api.openweathermap.org/data/2.5/forecast"
        params = {
            "lat": lat,
            "lon": lon,
            "appid": self._api_key or "",
            "units": "metric",
        }

        assert self._client is not None, "Client session not initialized"
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
        try:
            # Get coordinates
            coords = await self._get_location_coords(location)
            if not coords:
                return {"error": f"Location not found: {location}"}

            lat, lon = coords

            # Get current weather and forecast
            current = await self._get_weather_data(lat, lon)
            forecast = await self._get_forecast(lat, lon)

            # Close the client session after we're done
            if self._client and not self._client.closed:
                await self._client.close()

            return {
                "location": location,
                "current": {
                    "temperature": current["main"]["temp"],
                    "humidity": current["main"]["humidity"],
                    "description": current["weather"][0]["description"],
                    "wind_speed": current["wind"]["speed"],
                },
                "forecast": [
                    {
                        "time": item["dt"],
                        "temperature": item["main"]["temp"],
                        "description": item["weather"][0]["description"],
                    }
                    for item in forecast["list"][:5]  # Next 5 time slots
                ],
            }
        except Exception as e:
            # Ensure client is closed even if there's an error
            if self._client and not self._client.closed:
                await self._client.close()
            return {"error": str(e)}

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

        Args:
            location: Name of the location (e.g., "London, UK")

        Returns:
            String representation of weather information
        """
        try:
            result = await self._arun(location)

            if "error" in result:
                return f"Error: {result['error']}"

            # Format the result
            output = f"Weather for {result['location']}:\n\n"

            # Current weather
            current = result["current"]
            output += f"Current Conditions:\n"
            output += f"- Temperature: {current['temperature']}°C\n"
            output += f"- Conditions: {current['description']}\n"
            output += f"- Humidity: {current['humidity']}%\n"
            output += f"- Wind Speed: {current['wind_speed']} m/s\n\n"

            # Forecast
            output += "Forecast:\n"
            for item in result["forecast"]:
                # Convert timestamp to readable format
                from datetime import datetime

                time_str = datetime.fromtimestamp(item["time"]).strftime(
                    "%Y-%m-%d %H:%M"
                )
                output += (
                    f"- {time_str}: {item['temperature']}°C, {item['description']}\n"
                )

            return output
        except Exception as e:
            return f"Error getting weather information: {str(e)}"

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

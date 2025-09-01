"""
Tests for the Weather tool.
"""

import pytest
from unittest.mock import AsyncMock, MagicMock, patch
import aiohttp

from intuit.tools.weather import WeatherTool


@pytest.fixture
async def mock_weather_tool():
    """Create a mock weather tool."""
    with patch("aiohttp.ClientSession") as mock_session:
        session_instance = MagicMock()
        session_instance.close = AsyncMock()
        mock_session.return_value = session_instance

        tool = WeatherTool()
        tool._client = session_instance
        yield tool
        await tool._client.close()


@pytest.mark.asyncio
async def test_weather_tool_initialization(mock_env_vars):
    """Test that the Weather tool initializes correctly."""
    with patch("aiohttp.ClientSession") as mock_session:
        tool = WeatherTool()
        assert tool.name == "weather"
        assert "weather" in tool.description.lower()
        assert tool._api_key == "test_weather_key"
        # Client is initialized lazily, so it should be None initially
        assert tool._client is None


@pytest.mark.asyncio
async def test_get_location_coords(mock_weather_tool):
    """Test the _get_location_coords method."""
    # Mock geocoding response
    mock_response = [
        {"name": "London", "lat": 51.5074, "lon": -0.1278, "country": "GB"}
    ]

    mock_context = MagicMock()
    mock_context.__aenter__ = AsyncMock(
        return_value=MagicMock(json=AsyncMock(return_value=mock_response))
    )
    mock_context.__aexit__ = AsyncMock()
    mock_weather_tool._client.get.return_value = mock_context

    # Test getting coordinates
    coords = await mock_weather_tool._get_location_coords("London")
    assert coords == (51.5074, -0.1278)


@pytest.mark.asyncio
async def test_get_weather_data(mock_weather_tool):
    """Test the _get_weather_data method."""
    # Mock weather response
    mock_response = {
        "main": {"temp": 20.5, "humidity": 65, "pressure": 1013},
        "weather": [{"description": "clear sky", "icon": "01d"}],
        "wind": {"speed": 5.2, "deg": 180},
    }

    mock_context = MagicMock()
    mock_context.__aenter__ = AsyncMock(
        return_value=MagicMock(json=AsyncMock(return_value=mock_response))
    )
    mock_context.__aexit__ = AsyncMock()
    mock_weather_tool._client.get.return_value = mock_context

    # Test getting weather data
    data = await mock_weather_tool._get_weather_data(51.5074, -0.1278)
    assert data["main"]["temp"] == 20.5
    assert data["weather"][0]["description"] == "clear sky"


@pytest.mark.asyncio
async def test_get_forecast(mock_weather_tool):
    """Test the _get_forecast method."""
    # Mock forecast response
    mock_response = {
        "list": [
            {
                "dt": 1704067200,  # 2024-01-01 12:00:00
                "main": {"temp": 20.5, "humidity": 65},
                "weather": [{"description": "clear sky", "icon": "01d"}],
            },
            {
                "dt": 1704153600,  # 2024-01-02 12:00:00
                "main": {"temp": 21.5, "humidity": 70},
                "weather": [{"description": "few clouds", "icon": "02d"}],
            },
        ]
    }

    mock_context = MagicMock()
    mock_context.__aenter__ = AsyncMock(
        return_value=MagicMock(json=AsyncMock(return_value=mock_response))
    )
    mock_context.__aexit__ = AsyncMock()
    mock_weather_tool._client.get.return_value = mock_context

    # Test getting forecast
    data = await mock_weather_tool._get_forecast(51.5074, -0.1278)
    assert len(data["list"]) == 2
    assert data["list"][0]["main"]["temp"] == 20.5
    assert data["list"][1]["weather"][0]["description"] == "few clouds"


@pytest.mark.asyncio
async def test_weather_tool_run(mock_weather_tool):
    """Test the run method of the Weather tool."""
    # Mock responses for all API calls
    mock_responses = [
        # Geocoding response
        [{"name": "London", "lat": 51.5074, "lon": -0.1278}],
        # Current weather response
        {
            "main": {"temp": 20.5, "humidity": 65},
            "weather": [{"description": "clear sky"}],
            "wind": {"speed": 5.2},
        },
        # Forecast response
        {
            "list": [
                {
                    "dt": 1704067200,
                    "main": {"temp": 20.5},
                    "weather": [{"description": "clear sky"}],
                }
            ]
        },
    ]

    mock_contexts = []
    for response in mock_responses:
        mock_context = MagicMock()
        mock_context.__aenter__ = AsyncMock(
            return_value=MagicMock(json=AsyncMock(return_value=response))
        )
        mock_context.__aexit__ = AsyncMock()
        mock_contexts.append(mock_context)

    mock_weather_tool._client.get.side_effect = mock_contexts

    # Test the run method
    result = await mock_weather_tool.arun("London")

    assert result["location"] == "London"
    assert result["current"]["temperature"] == 20.5
    assert result["current"]["description"] == "clear sky"
    assert len(result["forecast"]) > 0


@pytest.mark.asyncio
async def test_weather_tool_error_handling(mock_weather_tool):
    """Test error handling in the Weather tool."""
    # Mock an error in the API call
    mock_context = MagicMock()
    mock_context.__aenter__ = AsyncMock(side_effect=Exception("API Error"))
    mock_context.__aexit__ = AsyncMock()
    mock_weather_tool._client.get.return_value = mock_context

    # Test error handling
    result = await mock_weather_tool.arun("Invalid Location")
    assert "error" in result
    assert "API Error" in result["error"]


@pytest.mark.asyncio
async def test_weather_tool_location_not_found(mock_weather_tool):
    """Test behavior when location is not found."""
    # Mock empty geocoding response
    mock_context = MagicMock()
    mock_context.__aenter__ = AsyncMock(
        return_value=MagicMock(json=AsyncMock(return_value=[]))
    )
    mock_context.__aexit__ = AsyncMock()
    mock_weather_tool._client.get.return_value = mock_context

    # Test location not found
    result = await mock_weather_tool.arun("Invalid Location")
    assert "error" in result
    assert "Location not found" in result["error"]

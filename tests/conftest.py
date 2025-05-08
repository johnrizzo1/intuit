"""
Test fixtures for Intuit.
"""
import os
import pytest
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch
from typing import Dict, Any

from intuit.agent import Agent, AgentConfig
from intuit.tools.web_search import WebSearchTool
from intuit.tools.gmail import GmailTool
from intuit.tools.weather import WeatherTool
from intuit.tools.filesystem import FilesystemTool
from intuit.vector_store.indexer import VectorStore

@pytest.fixture(autouse=True)
def mock_env_vars(monkeypatch):
    """Set up mock environment variables."""
    monkeypatch.setenv("OPENAI_API_KEY", "test_openai_key")
    monkeypatch.setenv("CHROMA_OPENAI_API_KEY", "test_chroma_key")
    monkeypatch.setenv("WEATHER_API_KEY", "test_weather_key")
    monkeypatch.setenv("GOOGLE_APPLICATION_CREDENTIALS", str(Path(__file__).parent / "data" / "credentials.json"))

@pytest.fixture
def mock_web_search_tool():
    """Create a mock web search tool."""
    tool = WebSearchTool()
    tool._search = AsyncMock(return_value=[
        {
            'title': 'Test Result 1',
            'url': 'https://example.com/1',
            'snippet': 'This is a test result 1'
        },
        {
            'title': 'Test Result 2',
            'url': 'https://example.com/2',
            'snippet': 'This is a test result 2'
        }
    ])
    return tool

@pytest.fixture
def mock_gmail_tool():
    """Create a mock Gmail tool."""
    with patch('google.oauth2.credentials.Credentials') as mock_creds, \
         patch('google_auth_oauthlib.flow.InstalledAppFlow') as mock_flow, \
         patch('googleapiclient.discovery.build') as mock_build:
        # Set up mock credentials
        mock_creds.from_authorized_user_info.return_value = MagicMock()
        mock_flow.from_client_secrets_file.return_value.run_local_server.return_value = MagicMock()
        
        # Set up mock Gmail service
        mock_service = MagicMock()
        mock_build.return_value = mock_service
        
        # Set up mock responses
        mock_service.users().messages().list().execute.return_value = {
            'messages': [{'id': 'msg1'}, {'id': 'msg2'}]
        }
        mock_service.users().messages().get().execute.return_value = {
            'payload': {
                'headers': [
                    {'name': 'Subject', 'value': 'Test Subject'},
                    {'name': 'From', 'value': 'test@example.com'},
                    {'name': 'Date', 'value': 'Wed, 1 Jan 2024 12:00:00 GMT'}
                ],
                'body': {'data': 'Test body content'}
            }
        }
        
        tool = GmailTool()
        tool._service = mock_service
        return tool

@pytest.fixture
def mock_weather_tool():
    """Create a mock weather tool."""
    with patch('aiohttp.ClientSession') as mock_session:
        tool = WeatherTool()
        tool._client = mock_session()
        return tool

@pytest.fixture
def mock_vector_store():
    """Create a mock vector store."""
    store = MagicMock()
    store.add_document = AsyncMock()
    store.search = AsyncMock(return_value=[
        MagicMock(metadata={'path': 'test1.txt'}),
        MagicMock(metadata={'path': 'test2.txt'})
    ])
    return store

@pytest.fixture
def mock_filesystem_tool(mock_vector_store):
    """Create a mock filesystem tool."""
    return FilesystemTool(vector_store=mock_vector_store)

@pytest.fixture
def mock_agent():
    """Create a mock agent."""
    agent = AsyncMock()
    agent.run.return_value = "Test response"
    return agent

@pytest.fixture
def test_data_dir(tmp_path):
    """Create a temporary directory with test files."""
    data_dir = tmp_path / "test_data"
    data_dir.mkdir()
    
    # Create test files
    (data_dir / "test1.txt").write_text("Test content 1")
    (data_dir / "test2.txt").write_text("Test content 2")
    (data_dir / "test3.md").write_text("Test content 3")
    
    # Create subdirectory
    subdir = data_dir / "subdir"
    subdir.mkdir()
    (subdir / "test4.txt").write_text("Test content 4")
    
    return data_dir 
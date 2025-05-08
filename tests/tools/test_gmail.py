"""
Tests for the Gmail tool.
"""
import pytest
from unittest.mock import AsyncMock, MagicMock, patch
import json
import os
from pathlib import Path

from intuit.tools.gmail import GmailTool

@pytest.fixture
def mock_credentials_file(tmp_path):
    """Create a mock credentials file."""
    creds = {
        "installed": {
            "client_id": "test_client_id",
            "project_id": "test_project_id",
            "auth_uri": "https://accounts.google.com/o/oauth2/auth",
            "token_uri": "https://oauth2.googleapis.com/token",
            "auth_provider_x509_cert_url": "https://www.googleapis.com/oauth2/v1/certs",
            "client_secret": "test_client_secret",
            "redirect_uris": ["urn:ietf:wg:oauth:2.0:oob", "http://localhost"]
        }
    }
    creds_file = tmp_path / "test_credentials.json"
    creds_file.write_text(json.dumps(creds))
    return str(creds_file)

@pytest.fixture
def mock_token_file(tmp_path):
    """Create a mock token file."""
    token = {
        "token": "test_token",
        "refresh_token": "test_refresh_token",
        "token_uri": "https://oauth2.googleapis.com/token",
        "client_id": "test_client_id",
        "client_secret": "test_client_secret",
        "scopes": ["https://www.googleapis.com/auth/gmail.readonly"]
    }
    token_file = tmp_path / "test_token.json"
    token_file.write_text(json.dumps(token))
    return str(token_file)

@pytest.fixture
def mock_gmail_service():
    """Create a mock Gmail service."""
    service = MagicMock()
    
    # Mock messages.list
    messages_list = MagicMock()
    messages_list.execute.return_value = {
        'messages': [{'id': 'msg1'}, {'id': 'msg2'}]
    }
    service.users().messages().list.return_value = messages_list
    
    # Mock messages.get
    message_get = MagicMock()
    message_get.execute.return_value = {
        'payload': {
            'headers': [
                {'name': 'Subject', 'value': 'Test Subject'},
                {'name': 'From', 'value': 'test@example.com'},
                {'name': 'Date', 'value': 'Wed, 1 Jan 2024 12:00:00 GMT'}
            ],
            'body': {'data': 'VGVzdCBib2R5IGNvbnRlbnQ='}  # base64 encoded "Test body content"
        }
    }
    service.users().messages().get.return_value = message_get
    
    return service

@pytest.fixture
def gmail_tool(mock_credentials_file, mock_token_file, mock_gmail_service):
    """Create a Gmail tool with mock files."""
    with patch('google.oauth2.credentials.Credentials') as mock_creds, \
         patch('google_auth_oauthlib.flow.InstalledAppFlow') as mock_flow, \
         patch('googleapiclient.discovery.build') as mock_build:
        
        # Set up mock credentials
        mock_creds_instance = MagicMock()
        mock_creds.from_authorized_user_file.return_value = mock_creds_instance
        mock_flow.from_client_secrets_file.return_value.run_local_server.return_value = mock_creds_instance
        
        # Set up mock service
        mock_build.return_value = mock_gmail_service
        
        # Create tool with mocked service
        return GmailTool(
            credentials_file=mock_credentials_file,
            token_file=mock_token_file,
            service=mock_gmail_service
        )

@pytest.mark.asyncio
async def test_gmail_tool_initialization(gmail_tool):
    """Test that the Gmail tool initializes correctly."""
    assert gmail_tool.name == "gmail"
    assert "gmail" in gmail_tool.description.lower()
    assert gmail_tool._service is not None

@pytest.mark.asyncio
async def test_get_message_content(gmail_tool):
    """Test getting message content."""
    content = await gmail_tool._get_message_content("msg1")
    
    assert content["subject"] == "Test Subject"
    assert content["from"] == "test@example.com"
    assert content["date"] == "Wed, 1 Jan 2024 12:00:00 GMT"
    assert content["content"] == "Test body content"

@pytest.mark.asyncio
async def test_gmail_tool_run(gmail_tool):
    """Test the run method."""
    result = await gmail_tool.run("test query", limit=2)
    
    assert result["status"] == "success"
    assert len(result["messages"]) == 2
    assert result["messages"][0]["subject"] == "Test Subject"
    assert result["messages"][0]["from"] == "test@example.com"
    assert result["messages"][0]["content"] == "Test body content"

@pytest.mark.asyncio
async def test_gmail_tool_no_messages(gmail_tool):
    """Test behavior when no messages are found."""
    # Mock empty response
    gmail_tool._service.users().messages().list().execute.return_value = {'messages': []}
    
    result = await gmail_tool.run("no results")
    
    assert result["status"] == "success"
    assert result["message"] == "No messages found"
    assert len(result["messages"]) == 0

@pytest.mark.asyncio
async def test_gmail_tool_error_handling(gmail_tool):
    """Test error handling."""
    # Mock an error
    gmail_tool._service.users().messages().list.side_effect = Exception("API Error")
    
    result = await gmail_tool.run("test query")
    
    assert result["status"] == "error"
    assert "API Error" in result["message"] 
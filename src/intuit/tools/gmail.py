"""
Gmail tool implementation for Intuit.
"""
import os
import base64
import logging
from typing import Dict, Any, List, Optional
from pydantic import Field, PrivateAttr
from google.oauth2.credentials import Credentials
from google_auth_oauthlib.flow import InstalledAppFlow
from google.auth.transport.requests import Request
from googleapiclient.discovery import build
from googleapiclient.errors import HttpError
from pathlib import Path

from .basetool import BaseTool

# Set up logging
logger = logging.getLogger(__name__)

class GmailTool(BaseTool):
    """Tool for interacting with Gmail."""
    
    name: str = Field(default="gmail")
    description: str = Field(
        default="Tool for reading and managing Gmail messages. Use this tool to search and read your Gmail messages. The search query supports Gmail's search operators (e.g., 'from:example@gmail.com', 'subject:meeting', 'has:attachment')."
    )
    
    _service: Optional[Any] = PrivateAttr(default=None)
    _credentials_file: str = PrivateAttr(default="credentials.json")
    _token_file: str = PrivateAttr(default="token.json")
    
    def __init__(self, credentials_file: Optional[str] = None, token_file: Optional[str] = None, service: Optional[Any] = None, **data):
        super().__init__(**data)
        if credentials_file:
            self._credentials_file = credentials_file
        if token_file:
            self._token_file = token_file
        if service:
            self._service = service
        else:
            self._initialize_service()
    
    def _initialize_service(self):
        """Initialize the Gmail service."""
        SCOPES = ['https://www.googleapis.com/auth/gmail.readonly']
        creds = None
        
        try:
            # Try to load existing token
            if os.path.exists(self._token_file):
                creds = Credentials.from_authorized_user_file(self._token_file, SCOPES)
            
            # If no valid credentials available, let the user log in
            if not creds or not creds.valid:
                if creds and creds.expired and creds.refresh_token:
                    creds.refresh(Request())
                else:
                    if not os.path.exists(self._credentials_file):
                        raise FileNotFoundError(
                            f"Credentials file not found: {self._credentials_file}. "
                            "Please download your OAuth 2.0 credentials from the Google Cloud Console "
                            "and save them as 'credentials.json'."
                        )
                    flow = InstalledAppFlow.from_client_secrets_file(
                        self._credentials_file, SCOPES)
                    creds = flow.run_local_server(port=0)
                
                # Save the credentials for the next run
                with open(self._token_file, 'w') as token:
                    token.write(creds.to_json())
            
            self._service = build('gmail', 'v1', credentials=creds)
            logger.info("Gmail service initialized successfully")
        except Exception as e:
            logger.error("Error initializing Gmail service: %s", str(e))
            raise
    
    async def _get_message_content(self, message_id: str) -> Dict[str, Any]:
        """Get the content of a specific message."""
        try:
            message = self._service.users().messages().get(userId='me', id=message_id).execute()
            headers = {header['name']: header['value'] for header in message['payload']['headers']}
            
            content = ""
            if 'body' in message['payload'] and 'data' in message['payload']['body']:
                content = base64.urlsafe_b64decode(message['payload']['body']['data']).decode()
            
            return {
                'subject': headers.get('Subject', ''),
                'from': headers.get('From', ''),
                'date': headers.get('Date', ''),
                'content': content
            }
        except HttpError as e:
            logger.error("Error getting message content: %s", str(e))
            raise
    
    async def _arun(self, query: str = None, limit: int = 5) -> Dict[str, Any]:
        """
        Execute Gmail operations asynchronously.
        
        Args:
            query: Search query for messages
            limit: Maximum number of messages to return
            
        Returns:
            Dict containing operation results
        """
        try:
            if not query:
                return {
                    "status": "error",
                    "message": "Search query is required"
                }
            
            # Search for messages
            results = self._service.users().messages().list(userId='me', q=query, maxResults=limit).execute()
            messages = results.get('messages', [])
            
            if not messages:
                return {
                    "status": "success",
                    "message": "No messages found",
                    "messages": []
                }
            
            # Get content for each message
            message_contents = []
            for msg in messages:
                content = await self._get_message_content(msg['id'])
                message_contents.append(content)
            
            return {
                "status": "success",
                "message": f"Found {len(message_contents)} messages",
                "messages": message_contents
            }
        except HttpError as e:
            error_message = str(e)
            if e.resp.status == 401:
                error_message = "Authentication failed. Please check your credentials and try again."
            elif e.resp.status == 403:
                error_message = "Permission denied. Please ensure you have the necessary Gmail API permissions."
            logger.error("Gmail API error: %s", error_message)
            return {
                "status": "error",
                "message": error_message
            }
        except Exception as e:
            logger.error("Error in Gmail operations: %s", str(e))
            return {
                "status": "error",
                "message": str(e)
            }
    
    async def run(self, query: str = None, limit: int = 5) -> Dict[str, Any]:
        """
        Execute Gmail operations.
        
        Args:
            query: Search query for messages
            limit: Maximum number of messages to return
            
        Returns:
            Dict containing operation results
        """
        return await self._arun(query=query, limit=limit) 
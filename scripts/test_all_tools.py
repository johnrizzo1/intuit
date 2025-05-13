#!/usr/bin/env python3
"""
Script to test all Intuit tools.
"""
import asyncio
import sys
import os
import logging
from pathlib import Path

# Add the src directory to the Python path
sys.path.insert(0, str(Path(__file__).parent.parent))

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

# Import tools
from intuit.tools.filesystem import FilesystemTool
from intuit.tools.web_search import WebSearchTool
from intuit.tools.gmail import GmailTool
from intuit.tools.weather import WeatherTool
from intuit.tools.calendar import CalendarTool
from intuit.tools.notes import NotesTool
from intuit.tools.reminders import RemindersTool
from intuit.tools.hackernews import HackerNewsTool

async def test_filesystem_tool():
    """Test the filesystem tool."""
    logger.info("Testing FilesystemTool...")
    tool = FilesystemTool()
    
    try:
        # Test listing the current directory
        result = await tool.run("list", path=".")
        if result["status"] == "success":
            logger.info("✅ FilesystemTool list: SUCCESS")
        else:
            logger.error(f"❌ FilesystemTool list: FAILED - {result['message']}")
        
        # Test reading a file
        result = await tool.run("read", path="README.md")
        if result["status"] == "success":
            logger.info("✅ FilesystemTool read: SUCCESS")
        else:
            logger.error(f"❌ FilesystemTool read: FAILED - {result['message']}")
        
        # Test getting file info
        result = await tool.run("info", path="README.md")
        if result["status"] == "success":
            logger.info("✅ FilesystemTool info: SUCCESS")
        else:
            logger.error(f"❌ FilesystemTool info: FAILED - {result['message']}")
        
        # Test searching files
        result = await tool.run("search", query="intuit")
        if result["status"] == "success":
            logger.info("✅ FilesystemTool search: SUCCESS")
        else:
            logger.error(f"❌ FilesystemTool search: FAILED - {result['message']}")
        
        return True
    except Exception as e:
        logger.error(f"❌ FilesystemTool: FAILED - {str(e)}")
        return False

async def test_web_search_tool():
    """Test the web search tool."""
    logger.info("Testing WebSearchTool...")
    tool = WebSearchTool()
    
    try:
        # Test searching the web
        result = await tool._arun(query="Python programming language")
        if "results" in result:
            logger.info("✅ WebSearchTool: SUCCESS")
            return True
        else:
            logger.error(f"❌ WebSearchTool: FAILED - {result.get('error', 'Unknown error')}")
            return False
    except Exception as e:
        logger.error(f"❌ WebSearchTool: FAILED - {str(e)}")
        return False

async def test_gmail_tool():
    """Test the Gmail tool."""
    logger.info("Testing GmailTool...")
    tool = GmailTool()
    
    try:
        # Test listing emails
        result = await tool._arun(query="is:unread")
        if "messages" in result or "emails" in result:
            logger.info("✅ GmailTool list: SUCCESS")
            return True
        else:
            logger.error(f"❌ GmailTool list: FAILED - {result.get('error', 'Unknown error')}")
            return False
    except Exception as e:
        logger.error(f"❌ GmailTool: FAILED - {str(e)}")
        return False

async def test_weather_tool():
    """Test the weather tool."""
    logger.info("Testing WeatherTool...")
    tool = WeatherTool()
    
    try:
        # Test getting weather
        result = await tool._arun(location="New York")
        if "current" in result or "forecast" in result:
            logger.info("✅ WeatherTool: SUCCESS")
            return True
        else:
            logger.error(f"❌ WeatherTool: FAILED - {result.get('error', 'Unknown error')}")
            return False
    except Exception as e:
        logger.error(f"❌ WeatherTool: FAILED - {str(e)}")
        return False

async def test_calendar_tool():
    """Test the calendar tool."""
    logger.info("Testing CalendarTool...")
    tool = CalendarTool()
    
    try:
        # Test adding a calendar event
        add_result = tool.add_event("Test event for tool verification")
        logger.info(f"Calendar add result: {add_result}")
        
        # Test listing calendar events
        list_result = tool.list_events()
        logger.info(f"Calendar list result: {list_result}")
        
        # If we got here without exceptions, consider it a success
        logger.info("✅ CalendarTool: SUCCESS")
        return True
    except Exception as e:
        logger.error(f"❌ CalendarTool: FAILED - {str(e)}")
        return False

async def test_notes_tool():
    """Test the notes tool."""
    logger.info("Testing NotesTool...")
    tool = NotesTool()
    
    try:
        # Test adding a note
        add_result = tool.add_note("Test note for tool verification")
        logger.info(f"Notes add result: {add_result}")
        
        # Test listing notes
        list_result = tool.list_notes()
        logger.info(f"Notes list result: {list_result}")
        
        # If we got here without exceptions, consider it a success
        logger.info("✅ NotesTool: SUCCESS")
        return True
    except Exception as e:
        logger.error(f"❌ NotesTool: FAILED - {str(e)}")
        return False

async def test_reminders_tool():
    """Test the reminders tool."""
    logger.info("Testing RemindersTool...")
    tool = RemindersTool()
    
    try:
        # Test adding a reminder
        add_result = tool.add_reminder("Test reminder for tool verification")
        logger.info(f"Reminders add result: {add_result}")
        
        # Test listing reminders
        list_result = tool.list_reminders()
        logger.info(f"Reminders list result: {list_result}")
        
        # If we got here without exceptions, consider it a success
        logger.info("✅ RemindersTool: SUCCESS")
        return True
    except Exception as e:
        logger.error(f"❌ RemindersTool: FAILED - {str(e)}")
        return False

async def test_hackernews_tool():
    """Test the Hacker News tool."""
    logger.info("Testing HackerNewsTool...")
    tool = HackerNewsTool()
    
    try:
        # Test getting top stories
        result = await tool._arun(action="top", limit=5)
        if "stories" in result:
            logger.info("✅ HackerNewsTool top: SUCCESS")
            return True
        else:
            logger.error(f"❌ HackerNewsTool top: FAILED - {result.get('error', 'Unknown error')}")
            return False
    except Exception as e:
        logger.error(f"❌ HackerNewsTool: FAILED - {str(e)}")
        return False

async def main():
    """Run all tool tests."""
    logger.info("Starting tool tests...")
    
    # Define the test functions
    tests = [
        test_filesystem_tool,
        test_web_search_tool,
        test_gmail_tool,
        test_weather_tool,
        test_calendar_tool,
        test_notes_tool,
        test_reminders_tool,
        test_hackernews_tool
    ]
    
    # Run the tests
    results = []
    for test in tests:
        try:
            result = await test()
            results.append(result)
        except Exception as e:
            logger.error(f"Error running test {test.__name__}: {str(e)}")
            results.append(False)
    
    # Print summary
    logger.info("\n--- Test Summary ---")
    total_tests = len(tests)
    passed_tests = sum(results)
    logger.info(f"Total tests: {total_tests}")
    logger.info(f"Passed tests: {passed_tests}")
    logger.info(f"Failed tests: {total_tests - passed_tests}")
    
    if passed_tests == total_tests:
        logger.info("✅ All tests passed!")
    else:
        logger.error("❌ Some tests failed.")

if __name__ == "__main__":
    asyncio.run(main())
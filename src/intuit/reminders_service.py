import asyncio
import logging
from datetime import datetime
from typing import Optional

import os
import json
from datetime import datetime
import asyncio
import logging
from typing import Optional
from pydantic import BaseModel, Field # Import BaseModel and Field

from .tools.reminders import RemindersTool, Reminder
from .voice import VoiceOutput

logger = logging.getLogger(__name__)

class Reminder(BaseModel): # Redefine Reminder model with triggered field
    """Pydantic model for a reminder."""
    id: str = Field(...)
    content: str = Field(...)
    timestamp: datetime = Field(default_factory=datetime.now)
    reminder_time: Optional[datetime] = Field(default=None)
    triggered: bool = Field(default=False) # Add triggered field

class ReminderService:
    """
    Background service to check and trigger reminders.
    """
    def __init__(self, reminders_tool: RemindersTool, voice_output: Optional[VoiceOutput] = None):
        self.reminders_tool = reminders_tool
        self.voice_output = voice_output
        self._running = False
        self._task: Optional[asyncio.Task] = None

    async def _check_reminders(self):
        """Periodically checks for reminders to trigger."""
        while self._running:
            logger.debug("Checking for reminders...")
            try:
                # Read and parse reminder files directly
                for filename in os.listdir(self.reminders_tool.data_dir):
                    if filename.endswith(".json"):
                        filepath = os.path.join(self.reminders_tool.data_dir, filename)
                        try:
                            with open(filepath, "r") as f:
                                reminder_data = json.load(f)
                                reminder = Reminder(**reminder_data)
    
                            # Check if reminder should be triggered
                            if reminder.reminder_time and not reminder.triggered and reminder.reminder_time <= datetime.now():
                                logger.info(f"Triggering reminder: {reminder.content}")
                                if self.voice_output:
                                    await self.voice_output.speak(f"Reminder: {reminder.content}")
    
                                # Mark reminder as triggered and save
                                reminder.triggered = True
                                with open(filepath, "w") as f:
                                    f.write(reminder.model_dump_json(indent=4))
    
                        except (json.JSONDecodeError, FileNotFoundError) as e:
                            logger.error(f"Error reading reminder file {filename}: {e}")
                            continue # Continue to next file on error
    
            except Exception as e:
                logger.error(f"Error in reminder checking task: {e}")
    
            await asyncio.sleep(60) # Check every 60 seconds

    def start(self):
        """Starts the background reminder checking task."""
        if not self._running:
            logger.info("Starting reminder service background task.")
            self._running = True
            self._task = asyncio.create_task(self._check_reminders())

    def stop(self):
        """Stops the background reminder checking task."""
        if self._running and self._task:
            logger.info("Stopping reminder service background task.")
            self._running = False
            self._task.cancel()
            try:
                asyncio.get_event_loop().run_until_complete(self._task)
            except asyncio.CancelledError:
                logger.info("Reminder service background task cancelled.")
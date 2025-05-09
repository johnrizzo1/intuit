# Calendar Tool Documentation

This document provides an overview and usage instructions for the Intuit Calendar Tool.

## Overview

The Calendar Tool allows users to manage their calendar events locally within the Intuit assistant. Events are stored as JSON files using Pydantic models for structured data.

## Features

- **Add Event:** Create a new calendar event with details and a timestamp.
- **List Events:** View all stored calendar events.
- **Search Events:** Find calendar events based on keywords in their details.
- **Delete Event:** Remove a calendar event using its unique ID.

## Usage

The Calendar Tool can be accessed via the Command Line Interface (CLI) and the Voice Interface.

### CLI Usage

The calendar tool is accessed using the `calendar` command, with subcommands for each action:

- **Add an event:**
  ```bash
  intuit calendar add "Meeting with the team on Friday at 2 PM"
  ```
  This will add a new event with the provided details and a timestamp of when the command was executed. The output will provide the unique ID of the added event.

- **List all events:**
  ```bash
  intuit calendar list
  ```
  This will display a list of all stored calendar events, including their IDs, details, and timestamps.

- **Search for events:**
  ```bash
  intuit calendar search "meeting"
  ```
  This will search for all events containing the keyword "meeting" in their details and display the matching events.

- **Delete an event:**
  ```bash
  intuit calendar delete [event_id]
  ```
  Replace `[event_id]` with the unique ID of the event you want to delete (obtained from the `list` or `add` commands).

### Voice Usage

When using the voice interface, you can naturally phrase your requests related to calendar management. The agent will interpret your request and use the Calendar Tool accordingly.

Examples:

- "Add a calendar event: Lunch with Sarah tomorrow at noon."
- "List my calendar events."
- "Search my calendar for 'presentation'."
- "Delete calendar event with ID [event_id]."

## Implementation Details

The Calendar Tool is implemented in Python using the `CalendarTool` class in `src/intuit/tools/calendar.py`. Calendar events are represented by the `CalendarEvent` Pydantic model and stored as JSON files in the `data/calendar/` directory. Each file is named using the unique ID of the event.

- **`CalendarEvent` Model:** Defines the structure of a calendar event with fields for `id` (string), `details` (string), and `timestamp` (datetime).
- **File Storage:** Events are serialized to JSON using `model_dump_json()` and saved as individual files. Deserialization is handled using `model_validate_json()`.
- **Tool Integration:** The `CalendarTool` is integrated with the main agent in `src/intuit/agent.py` and exposed via the CLI in `src/intuit/cli.py`.

## Future Enhancements

- Adding support for specifying date and time when adding events.
- Implementing reminder notifications.
- Adding options for editing existing events.
- Integrating with external calendar services (e.g., Google Calendar) as an alternative storage option.
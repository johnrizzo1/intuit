# Reminders Tool Documentation

This document provides an overview and usage instructions for the Intuit Reminders Tool.

## Overview

The Reminders Tool allows users to create, list, search, and delete personal reminders locally within the Intuit assistant. Reminders can optionally include a specific time for notification. Reminders are stored as JSON files using Pydantic models for structured data.

## Features

- **Add Reminder:** Create a new reminder with content and an optional reminder time.
- **List Reminders:** View all stored reminders.
- **Search Reminders:** Find reminders based on keywords in their content.
- **Delete Reminder:** Remove a reminder using its unique ID.
- **Trigger Reminders:** (Planned) Logic to notify the user when a reminder time is reached.

## Usage

The Reminders Tool can be accessed via the Command Line Interface (CLI) and the Voice Interface.

### CLI Usage

The reminders tool is accessed using the `reminders` command, with subcommands for each action:

- **Add a reminder:**

  ```bash
  intuit reminders add "Call Mom" --time "2025-12-25T18:00:00"
  ```

  This will add a new reminder with the provided content and a specific reminder time in ISO 8601 format. If `--time` is omitted, the reminder will be added without a specific time. The output will provide the unique ID of the added reminder.

- **List all reminders:**

  ```bash
  intuit reminders list
  ```

  This will display a list of all stored reminders, including their IDs, content, timestamps, and reminder times (if set).

- **Search for reminders:**

  ```bash
  intuit reminders search "call"
  ```

  This will search for all reminders containing the keyword "call" in their content and display the matching reminders.

- **Delete a reminder:**

  ```bash
  intuit reminders delete [reminder_id]
  ```

  Replace `[reminder_id]` with the unique ID of the reminder you want to delete (obtained from the `list` or `add` commands).

### Voice Usage

When using the voice interface, you can naturally phrase your requests related to reminders management. The agent will interpret your request and use the Reminders Tool accordingly. You should specify the reminder time clearly in your voice command.

Examples:

- "Add a reminder: Pick up dry cleaning tomorrow morning."
- "Remind me to call the doctor on Tuesday at 3 PM."
- "List my reminders."
- "Search my reminders for 'meeting'."
- "Delete reminder with ID [reminder_id]."

## Implementation Details

The Reminders Tool is implemented in Python using the `RemindersTool` class in `src/intuit/tools/reminders.py`. Reminders are represented by the `Reminder` Pydantic model and stored as JSON files in the `data/reminders/` directory. Each file is named using the unique ID of the reminder.

- **`Reminder` Model:** Defines the structure of a reminder with fields for `id` (string), `content` (string), `timestamp` (datetime), and `reminder_time` (optional datetime).
- **File Storage:** Reminders are serialized to JSON using `model_dump_json()` and saved as individual files. Deserialization is handled using `model_validate_json()`.
- **Time Parsing:** The CLI uses a helper function (`parse_reminder_time`) to convert the provided time string into a datetime object. The agent is configured to understand time specifications in voice commands and pass them to the tool.
- **Tool Integration:** The `RemindersTool` is integrated with the main agent in `src/intuit/agent.py` and exposed via the CLI in `src/intuit/cli.py`.

## Future Enhancements

- Implementing the background logic for triggering reminders at the specified time.
- Adding options for editing existing reminders.
- Implementing recurring reminders.
- Integrating with external calendar or reminder services.

# Notes Tool Documentation

This document provides an overview and usage instructions for the Intuit Notes Tool.

## Overview

The Notes Tool allows users to create, list, search, and delete personal notes locally within the Intuit assistant. Notes are stored as JSON files using Pydantic models for structured data.

## Features

- **Add Note:** Create a new note with content and a timestamp.
- **List Notes:** View all stored notes.
- **Search Notes:** Find notes based on keywords in their content.
- **Delete Note:** Remove a note using its unique ID.

## Usage

The Notes Tool can be accessed via the Command Line Interface (CLI) and the Voice Interface.

### CLI Usage

The notes tool is accessed using the `notes` command, with subcommands for each action:

- **Add a note:**
  ```bash
  intuit notes add "Remember to buy groceries tomorrow."
  ```
  This will add a new note with the provided content and a timestamp of when the command was executed. The output will provide the unique ID of the added note.

- **List all notes:**
  ```bash
  intuit notes list
  ```
  This will display a list of all stored notes, including their IDs, content, and timestamps.

- **Search for notes:**
  ```bash
  intuit notes search "groceries"
  ```
  This will search for all notes containing the keyword "groceries" in their content and display the matching notes.

- **Delete a note:**
  ```bash
  intuit notes delete [note_id]
  ```
  Replace `[note_id]` with the unique ID of the note you want to delete (obtained from the `list` or `add` commands).

### Voice Usage

When using the voice interface, you can naturally phrase your requests related to notes management. The agent will interpret your request and use the Notes Tool accordingly.

Examples:

- "Add a note: Ideas for the new project."
- "List my notes."
- "Search my notes for 'meeting minutes'."
- "Delete note with ID [note_id]."

## Implementation Details

The Notes Tool is implemented in Python using the `NotesTool` class in `src/intuit/tools/notes.py`. Notes are represented by the `Note` Pydantic model and stored as JSON files in the `data/notes/` directory. Each file is named using the unique ID of the note.

- **`Note` Model:** Defines the structure of a note with fields for `id` (string), `content` (string), and `timestamp` (datetime).
- **File Storage:** Notes are serialized to JSON using `model_dump_json()` and saved as individual files. Deserialization is handled using `model_validate_json()`.
- **Tool Integration:** The `NotesTool` is integrated with the main agent in `src/intuit/agent.py` and exposed via the CLI in `src/intuit/cli.py`.

## Future Enhancements

- Adding support for tagging and categorizing notes.
- Implementing synchronization with external note-taking services.
- Adding options for editing existing notes.
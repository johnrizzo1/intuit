import sys
import logging
from intuit.tools.calendar import CalendarTool
from intuit.tools.notes import NotesTool # Import NotesTool
from intuit.tools.reminders import RemindersTool # Import RemindersTool
from datetime import datetime # Import datetime for parsing reminder time

def parse_reminder_time(time_str: str) -> datetime:
    """Parses a string into a datetime object for reminder time."""
    # This is a basic example; a real implementation would need robust parsing
    try:
        # Assuming ISO 8601 format for simplicity (e.g., "2025-12-31T23:59:59")
        return datetime.fromisoformat(time_str)
    except ValueError:
        raise ValueError(f"Invalid time format: {time_str}. Please use ISO 8601 format (e.g., '2025-12-31T23:59:59').")

def main():
    """Main entry point for the CLI."""
    try:
        # Check if input is being piped
        if not sys.stdin.isatty():
            # Read from stdin
            input_text = sys.stdin.read()
            if not input_text.strip():
                print("Error: No input provided", file=sys.stderr)
                sys.exit(1)
            
            # Process the input
            try:
                result = process_input(input_text)
                print(result)
                sys.exit(0)
            except Exception as e:
                print(f"Error processing input: {str(e)}", file=sys.stderr)
                sys.exit(1)
        
        # If no input is piped, proceed with normal CLI
        parser = argparse.ArgumentParser(description="Intuit CLI")
        parser.add_argument("--version", action="version", version=f"%(prog)s version {__version__}")
        parser.add_argument("-v", "--verbose", action="store_true", help="Enable verbose logging")

        subparsers = parser.add_subparsers(dest="command", help="Available commands")

        # Search command
        search_parser = subparsers.add_parser("search", help="Search for information")
        search_parser.add_argument("query", help="The search query")

        # Chat command
        chat_parser = subparsers.add_parser("chat", help="Engage in a chat conversation")
        chat_parser.add_argument("query", help="The initial chat message")

        # Calendar command
        calendar_parser = subparsers.add_parser("calendar", help="Manage calendar events")
        calendar_subparsers = calendar_parser.add_subparsers(dest="action", help="Calendar actions")

        # Calendar add action
        calendar_add_parser = calendar_subparsers.add_parser("add", help="Add a new calendar event")
        calendar_add_parser.add_argument("event", help="Details of the calendar event")

        # Calendar list action
        calendar_list_parser = calendar_subparsers.add_parser("list", help="List calendar events")

        # Calendar search action
        calendar_search_parser = calendar_subparsers.add_parser("search", help="Search calendar events")
        calendar_search_parser.add_argument("keyword", help="Keyword to search for in events")

        # Calendar delete action
        calendar_delete_parser = calendar_subparsers.add_parser("delete", help="Delete a calendar event")
        calendar_delete_parser.add_argument("filename", help="Filename of the event to delete")

        # Notes command
        notes_parser = subparsers.add_parser("notes", help="Manage notes")
        notes_subparsers = notes_parser.add_subparsers(dest="action", help="Notes actions")

        # Notes add action
        notes_add_parser = notes_subparsers.add_parser("add", help="Add a new note")
        notes_add_parser.add_argument("content", help="Content of the note")

        # Notes list action
        notes_list_parser = notes_subparsers.add_parser("list", help="List notes")

        # Notes search action
        notes_search_parser = notes_subparsers.add_parser("search", help="Search notes")
        notes_search_parser.add_argument("keyword", help="Keyword to search for in notes")

        # Notes delete action
        notes_delete_parser = notes_subparsers.add_parser("delete", help="Delete a note")
        notes_delete_parser.add_argument("id", help="ID of the note to delete")

        # Reminders command
        reminders_parser = subparsers.add_parser("reminders", help="Manage reminders")
        reminders_subparsers = reminders_parser.add_subparsers(dest="action", help="Reminders actions")

        # Reminders add action
        reminders_add_parser = reminders_subparsers.add_parser("add", help="Add a new reminder")
        reminders_add_parser.add_argument("content", help="Content of the reminder")
        reminders_add_parser.add_argument("--time", type=parse_reminder_time, help="Optional reminder time (ISO 8601 format)")

        # Reminders list action
        reminders_list_parser = reminders_subparsers.add_parser("list", help="List reminders")

        # Reminders search action
        reminders_search_parser = reminders_subparsers.add_parser("search", help="Search reminders")
        reminders_search_parser.add_argument("keyword", help="Keyword to search for in reminders")

        # Reminders delete action
        reminders_delete_parser = reminders_subparsers.add_parser("delete", help="Delete a reminder")
        reminders_delete_parser.add_argument("id", help="ID of the reminder to delete")

        args = parser.parse_args()

        if args.verbose:
            logging.basicConfig(level=logging.DEBUG)
        else:
            logging.basicConfig(level=logging.INFO)

        calendar_tool = CalendarTool()
        notes_tool = NotesTool() # Initialize NotesTool
        reminders_tool = RemindersTool() # Initialize RemindersTool

        if args.command == "search":
            if not args.query:
                print("Error: Query is required for search command", file=sys.stderr)
                sys.exit(1)
            search(args.query)
        elif args.command == "chat":
            if not args.query:
                print("Error: Query is required for chat command", file=sys.stderr)
                sys.exit(1)
            chat(args.query)
        elif args.command == "calendar":
            if args.action == "add":
                if not args.event:
                    print("Error: Event details are required for add action", file=sys.stderr)
                    sys.exit(1)
                print(calendar_tool.add_event(args.event))
            elif args.action == "list":
                print(calendar_tool.list_events())
            elif args.action == "search":
                if not args.keyword:
                    print("Error: Keyword is required for search action", file=sys.stderr)
                    sys.exit(1)
                print(calendar_tool.search_events(args.keyword))
            elif args.action == "delete":
                if not args.filename:
                    print("Error: Filename is required for delete action", file=sys.stderr)
                    sys.exit(1)
                print(calendar_tool.delete_event(args.filename))
            else:
                print(f"Unknown calendar action: {args.action}", file=sys.stderr)
                sys.exit(1)
        elif args.command == "notes": # Handle notes command
            if args.action == "add":
                if not args.content:
                    print("Error: Note content is required for add action", file=sys.stderr)
                    sys.exit(1)
                print(notes_tool.add_note(args.content))
            elif args.action == "list":
                print(notes_tool.list_notes())
            elif args.action == "search":
                if not args.keyword:
                    print("Error: Keyword is required for search action", file=sys.stderr)
                    sys.exit(1)
                print(notes_tool.search_notes(args.keyword))
            elif args.action == "delete":
                if not args.id:
                    print("Error: Note ID is required for delete action", file=sys.stderr)
                    sys.exit(1)
                print(notes_tool.delete_note(args.id))
            else:
                print(f"Unknown notes action: {args.action}", file=sys.stderr)
                sys.exit(1)
        elif args.command == "reminders": # Handle reminders command
            if args.action == "add":
                if not args.content:
                    print("Error: Reminder content is required for add action", file=sys.stderr)
                    sys.exit(1)
                print(reminders_tool.add_reminder(args.content, args.time))
            elif args.action == "list":
                print(reminders_tool.list_reminders())
            elif args.action == "search":
                if not args.keyword:
                    print("Error: Keyword is required for search action", file=sys.stderr)
                    sys.exit(1)
                print(reminders_tool.search_reminders(args.keyword))
            elif args.action == "delete":
                if not args.id:
                    print("Error: Reminder ID is required for delete action", file=sys.stderr)
                    sys.exit(1)
                print(reminders_tool.delete_reminder(args.id))
            else:
                print(f"Unknown reminders action: {args.action}", file=sys.stderr)
                sys.exit(1)
        else:
            print(f"Unknown command: {args.command}", file=sys.stderr)
            sys.exit(1)
    except KeyboardInterrupt:
        print("\nOperation cancelled by user")
        sys.exit(130)
    except Exception as e:
        print(f"Error: {str(e)}", file=sys.stderr)
        sys.exit(1)
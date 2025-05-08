import sys
import logging

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
        args = parse_args()
        
        if args.version:
            print(f"intuit version {__version__}")
            return
        
        if args.verbose:
            logging.basicConfig(level=logging.DEBUG)
        else:
            logging.basicConfig(level=logging.INFO)
        
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
        else:
            print(f"Unknown command: {args.command}", file=sys.stderr)
            sys.exit(1)
            
    except KeyboardInterrupt:
        print("\nOperation cancelled by user")
        sys.exit(130)
    except Exception as e:
        print(f"Error: {str(e)}", file=sys.stderr)
        sys.exit(1) 
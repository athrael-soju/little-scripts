import argparse
import sys

from utils import Colors

from .command_registry import COMMAND_REGISTRY, CommandType
from .commands import ask, clear, setup_pipeline, status, upload
from .messaging import InteractiveHelp, SetupMessages, UIMessages


def interactive_mode():
    """Runs the CLI in interactive mode."""
    InteractiveHelp.show_welcome()
    InteractiveHelp.show_commands_help()

    # Initialize pipeline once for interactive mode
    SetupMessages.pipeline_initializing()
    _, vector_db, pipeline, openai_handler = setup_pipeline(include_openai=True)
    minio_handler = pipeline.minio_handler
    SetupMessages.pipeline_ready()

    while True:
        try:
            # Simple prompt for conversational search
            prompt = f"\n{Colors.BOLD}🤖 colpali>{Colors.ENDC} "
            user_input = input(prompt).strip()

            if not user_input:
                continue

            parts = user_input.split()
            command_name = parts[0].lower()

            # Handle completely removed commands
            if command_name in ["set-mode", "mode"]:
                UIMessages.info(
                    "Mode switching has been removed. All searches are now AI-powered by default."
                )
                UIMessages.tip(
                    "Just type your questions directly in this conversational mode."
                )
                continue
            elif command_name == "interactive":
                UIMessages.info(
                    "Interactive mode is now the default! You're already in conversational mode."
                )
                UIMessages.tip(
                    "Just type your questions directly or use the commands listed above."
                )
                continue
            elif command_name in ["search", "query"]:
                UIMessages.info(
                    "Separate search commands have been removed. Just type your questions directly!"
                )
                UIMessages.tip(
                    "No need for commands - the whole app is conversational now."
                )
                continue

            # Check if it's a valid command from the registry
            command = COMMAND_REGISTRY.get_command(command_name)
            if command:
                # Execute command based on type
                if command.command_type == CommandType.EXIT:
                    SetupMessages.goodbye()
                    break

                elif command.command_type == CommandType.HELP:
                    InteractiveHelp.show_commands_help()

                elif command.command_type == CommandType.UPLOAD:
                    _handle_upload_command(parts, pipeline)

                elif command.command_type == CommandType.CLEAR:
                    clear(vector_db, minio_handler, interactive=True)

                elif command.command_type == CommandType.STATUS:
                    status(vector_db, openai_handler, minio_handler, pipeline)
            else:
                # Treat unrecognized input as a conversational query
                query = user_input
                ask(
                    query,
                    pipeline,
                    vector_db,
                    openai_handler,
                    interactive=True,
                    use_openai=True,
                )

        except KeyboardInterrupt:
            SetupMessages.goodbye()
            break
        except Exception as e:
            UIMessages.error(f"Error: {e}")


def _handle_upload_command(parts, pipeline):
    """Handle upload commands."""
    file_path = None
    if len(parts) > 1:
        if parts[1] == "--file" and len(parts) > 2:
            file_path = " ".join(parts[2:])  # Handle paths with spaces
        else:
            file_path = " ".join(parts[1:])  # Handle paths with spaces
    upload(pipeline, file_path, interactive=True)


def main_cli():
    """Main CLI handler."""
    parser = argparse.ArgumentParser(
        description="🚀 ColPali AI-Powered Document Search (starts in conversational mode by default)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  %(prog)s                                                    # Start conversational mode (default)
  %(prog)s upload --file documents.txt                        # Upload documents and start conversation
  %(prog)s show-status                                        # Quick status check
  %(prog)s clear-collection                                   # Clear data

Note: For questions about your documents, just run the app and ask directly!
        """,
    )

    subparsers = parser.add_subparsers(dest="command", help="Available commands")

    # Register commands using the command registry
    for command in COMMAND_REGISTRY.get_all_commands():
        if command.command_type == CommandType.UPLOAD:
            parser_cmd = subparsers.add_parser(
                command.name, help=f"{command.emoji} {command.description}"
            )
            parser_cmd.add_argument(
                "--file", type=str, help="Path to a file to upload (optional)"
            )
        elif command.command_type in [CommandType.CLEAR, CommandType.STATUS]:
            subparsers.add_parser(
                command.name, help=f"{command.emoji} {command.description}"
            )

    args = parser.parse_args()

    # If no command is provided, start in conversational mode by default
    if not args.command:
        interactive_mode()
        return

    # For single command execution, setup pipeline once
    command = COMMAND_REGISTRY.get_command(args.command)
    if not command:
        UIMessages.error(f"Unknown command: {args.command}")
        sys.exit(1)

    # Enable OpenAI for status commands
    use_openai_for_setup = command.command_type == CommandType.STATUS

    _, vector_db, pipeline, openai_handler = setup_pipeline(
        include_openai=use_openai_for_setup
    )
    minio_handler = pipeline.minio_handler

    # Execute command
    success = True
    if command.command_type == CommandType.UPLOAD:
        success = upload(pipeline, args.file)
    elif command.command_type == CommandType.CLEAR:
        success = clear(vector_db, minio_handler)
    elif command.command_type == CommandType.STATUS:
        success = status(vector_db, openai_handler, minio_handler, pipeline)

    sys.exit(0 if success else 1)

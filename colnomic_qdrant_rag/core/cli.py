import argparse
import sys
from .commands import setup_pipeline, ask, upload, clear, status
from utils import colored_print, Colors


def interactive_mode():
    """Runs the CLI in interactive mode."""
    colored_print("🚀 Welcome to ColPali Binary Quant Interactive Mode!", Colors.HEADER)
    colored_print("=" * 60, Colors.HEADER)
    colored_print("🎯 Query Mode: Just type your questions directly!", Colors.OKBLUE)
    colored_print("📋 Available commands:", Colors.OKBLUE)
    colored_print(
        "  • <your question>     - Ask directly (current mode: basic)", Colors.OKCYAN
    )
    colored_print(
        "  • set-mode basic      - Switch to basic search mode", Colors.OKCYAN
    )
    colored_print(
        "  • set-mode conversational - Switch to AI conversational mode", Colors.OKCYAN
    )
    colored_print(
        "  • upload [--file path] - Upload and index documents", Colors.OKCYAN
    )
    colored_print("  • clear-collection    - Clear the collection", Colors.OKCYAN)
    colored_print("  • show-status         - Show system status", Colors.OKCYAN)
    colored_print("  • help                - Show this help", Colors.OKCYAN)
    colored_print("  • exit/quit           - Exit interactive mode", Colors.OKCYAN)
    colored_print("=" * 60, Colors.HEADER)
    colored_print(
        "💡 Tip: You can ask questions directly without typing commands first!",
        Colors.OKCYAN,
    )

    # Initialize pipeline once for interactive mode
    colored_print("\n🔧 Initializing pipeline for the session...", Colors.OKBLUE)
    _, vector_db, pipeline, openai_handler = setup_pipeline(include_openai=True)
    minio_handler = pipeline.minio_handler  # Get it from the pipeline
    colored_print("✅ Pipeline ready.", Colors.OKGREEN)

    # Current mode state
    current_mode = "basic"  # Default to basic mode
    mode_emoji = {"basic": "🔍", "conversational": "🤖"}
    mode_name = {"basic": "Basic", "conversational": "Conversational"}

    while True:
        try:
            # Dynamic prompt showing current mode
            prompt = f"\n{Colors.BOLD}{mode_emoji[current_mode]} colpali[{mode_name[current_mode]}]>{Colors.ENDC} "
            user_input = input(prompt).strip()

            if not user_input:
                continue

            parts = user_input.split()
            command = parts[0].lower()

            # Handle system commands
            if command in ["exit", "quit"]:
                colored_print("👋 Goodbye!", Colors.OKGREEN)
                break
            elif command == "help":
                colored_print(
                    "🎯 Query Mode: Just type your questions directly!", Colors.OKBLUE
                )
                colored_print("📋 Available commands:", Colors.OKBLUE)
                colored_print(
                    "  • <your question>     - Ask directly (current mode: "
                    + f"{mode_emoji[current_mode]} {mode_name[current_mode]})",
                    Colors.OKCYAN,
                )
                colored_print(
                    "  • set-mode basic      - Switch to basic search mode",
                    Colors.OKCYAN,
                )
                colored_print(
                    "  • set-mode conversational - Switch to AI conversational mode",
                    Colors.OKCYAN,
                )
                colored_print(
                    "  • upload [--file path] - Upload and index documents",
                    Colors.OKCYAN,
                )
                colored_print(
                    "  • clear-collection    - Clear the collection", Colors.OKCYAN
                )
                colored_print(
                    "  • show-status         - Show system status", Colors.OKCYAN
                )
                colored_print("  • help                - Show this help", Colors.OKCYAN)
                colored_print(
                    "  • exit/quit           - Exit interactive mode", Colors.OKCYAN
                )
                colored_print(
                    f"\n🎯 Current mode: {mode_emoji[current_mode]} {mode_name[current_mode]}",
                    Colors.OKGREEN,
                )
            elif command == "set-mode":
                if len(parts) > 1:
                    new_mode = parts[1].lower()
                    if new_mode in ["basic", "search"]:
                        current_mode = "basic"
                        colored_print("🔍 Switched to Basic mode", Colors.OKGREEN)
                        colored_print(
                            "💡 Your questions will use fast document search",
                            Colors.OKCYAN,
                        )
                    elif new_mode in ["conversational", "ai", "analyze", "analysis"]:
                        current_mode = "conversational"
                        colored_print(
                            "🤖 Switched to Conversational mode", Colors.OKGREEN
                        )
                        colored_print(
                            "💡 Your questions will include AI-powered responses and insights",
                            Colors.OKCYAN,
                        )
                        if not openai_handler or not openai_handler.is_available():
                            colored_print(
                                "⚠️  Note: OpenAI is not configured, AI features will be limited",
                                Colors.WARNING,
                            )
                    else:
                        colored_print(f"❌ Unknown mode: {new_mode}", Colors.FAIL)
                        colored_print(
                            "💡 Available modes: basic, conversational", Colors.OKCYAN
                        )
                else:
                    colored_print(
                        f"🎯 Current mode: {mode_emoji[current_mode]} {mode_name[current_mode]}",
                        Colors.OKGREEN,
                    )
                    colored_print(
                        "💡 Use 'set-mode basic' or 'set-mode conversational' to switch modes",
                        Colors.OKCYAN,
                    )
            elif command == "upload":
                # Handle both formats: "upload --file path" and "upload path"
                file_path = None
                if len(parts) > 1:
                    if parts[1] == "--file" and len(parts) > 2:
                        file_path = " ".join(parts[2:])  # Handle paths with spaces
                    else:
                        file_path = " ".join(parts[1:])  # Handle paths with spaces
                upload(pipeline, file_path, interactive=True)
            elif command == "clear-collection":
                clear(vector_db, interactive=True)
            elif command == "show-status":
                status(vector_db, openai_handler, minio_handler)
            # Backwards compatibility for old commands
            elif command == "mode":
                colored_print(
                    "💡 Use 'set-mode basic' or 'set-mode conversational' instead",
                    Colors.OKCYAN,
                )
                if len(parts) > 1:
                    new_mode = parts[1].lower()
                    if new_mode in ["basic", "search"]:
                        current_mode = "basic"
                        colored_print("🔍 Switched to Basic mode", Colors.OKGREEN)
                    elif new_mode in ["conversational", "ai"]:
                        current_mode = "conversational"
                        colored_print(
                            "🤖 Switched to Conversational mode", Colors.OKGREEN
                        )
            elif command == "clear":
                colored_print("💡 Use 'clear-collection' for clarity", Colors.OKCYAN)
                clear(vector_db, interactive=True)
            elif command == "status":
                colored_print("💡 Use 'show-status' for clarity", Colors.OKCYAN)
                status(vector_db, openai_handler, minio_handler)
            # Explicit ask/analyze commands (for backwards compatibility)
            elif command == "ask":
                if len(parts) > 1:
                    query = " ".join(parts[1:])
                    ask(query, pipeline, vector_db, openai_handler, interactive=True)
                else:
                    colored_print("❌ Usage: ask <query>", Colors.FAIL)
            elif command == "analyze":
                if len(parts) > 1:
                    query = " ".join(parts[1:])
                    ask(
                        query,
                        pipeline,
                        vector_db,
                        openai_handler,
                        interactive=True,
                        use_openai=True,
                    )
                else:
                    colored_print("❌ Usage: analyze <query>", Colors.FAIL)
            else:
                # Treat any unrecognized input as a query in the current mode
                query = user_input  # Use the full input as the query
                use_openai = current_mode == "conversational"
                ask(
                    query,
                    pipeline,
                    vector_db,
                    openai_handler,
                    interactive=True,
                    use_openai=use_openai,
                )

        except KeyboardInterrupt:
            colored_print("\n👋 Goodbye!", Colors.OKGREEN)
            break
        except Exception as e:
            colored_print(f"❌ Error: {e}", Colors.FAIL)


def main_cli():
    """Main CLI handler."""
    parser = argparse.ArgumentParser(
        description="🚀 ColPali Binary Quantization Retrieval System",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  %(prog)s ask "What are UFO sightings?"
  %(prog)s analyze "What do these UFO images show?"
  %(prog)s upload --file documents.txt
  %(prog)s upload  # Use default UFO dataset
  %(prog)s show-status
  %(prog)s clear-collection
  %(prog)s interactive  # Interactive mode
        """,
    )

    subparsers = parser.add_subparsers(dest="command", help="Available commands")

    # Ask command
    ask_parser = subparsers.add_parser(
        "ask", help="🔍 Search for documents using natural language queries"
    )
    ask_parser.add_argument(
        "query", type=str, help="The search query (e.g., 'UFO sightings in California')"
    )

    # Analyze command (with OpenAI)
    analyze_parser = subparsers.add_parser(
        "analyze", help="🧠 Search for documents with conversational AI responses"
    )
    analyze_parser.add_argument(
        "query",
        type=str,
        help="The search query for conversational response (e.g., 'What do these UFO images show?')",
    )

    # Upload command
    upload_parser = subparsers.add_parser(
        "upload", help="📂 Upload and index documents from file or default dataset"
    )
    upload_parser.add_argument(
        "--file",
        type=str,
        help="Path to a text file to upload (optional, uses UFO dataset if not provided)",
    )

    # Clear command
    subparsers.add_parser(
        "clear-collection",
        help="🗑️  Clear all documents from the collection (destructive operation)",
    )
    # Backwards compatibility alias
    subparsers.add_parser(
        "clear", help="🗑️  Clear all documents (alias for clear-collection)"
    )

    # Status command
    subparsers.add_parser(
        "show-status", help="📊 Show current system status and collection information"
    )
    # Backwards compatibility alias
    subparsers.add_parser(
        "status", help="📊 Show system status (alias for show-status)"
    )

    # Interactive command
    subparsers.add_parser(
        "interactive", help="🚀 Start interactive mode for easier CLI usage"
    )

    args = parser.parse_args()

    if args.command == "interactive":
        interactive_mode()
        return

    if not args.command:
        parser.print_help()
        return

    # For single command execution, setup pipeline once
    # Enable OpenAI for commands that need it or for status checking
    use_openai_for_setup = args.command in ["analyze", "show-status", "status"]
    _, vector_db, pipeline, openai_handler = setup_pipeline(
        include_openai=use_openai_for_setup
    )
    minio_handler = pipeline.minio_handler

    if args.command == "ask":
        success = ask(args.query, pipeline, vector_db, openai_handler)
        sys.exit(0 if success else 1)
    elif args.command == "analyze":
        success = ask(args.query, pipeline, vector_db, openai_handler, use_openai=True)
        sys.exit(0 if success else 1)
    elif args.command == "upload":
        success = upload(pipeline, args.file)
        sys.exit(0 if success else 1)
    elif args.command in ["clear-collection", "clear"]:
        success = clear(vector_db)
        sys.exit(0 if success else 1)
    elif args.command in ["show-status", "status"]:
        success = status(vector_db, openai_handler, minio_handler)
        sys.exit(0 if success else 1)

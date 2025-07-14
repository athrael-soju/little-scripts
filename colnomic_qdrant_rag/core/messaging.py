"""
Unified messaging system for consistent CLI output formatting.
This module provides centralized message formatting and replaces scattered colored_print calls.
"""

from typing import Dict, Optional

from utils import Colors, colored_print

from .command_registry import COMMAND_REGISTRY


class MessageType:
    """Standard message types with consistent formatting."""

    SUCCESS = "success"
    ERROR = "error"
    WARNING = "warning"
    INFO = "info"
    HEADER = "header"
    TIP = "tip"
    STATUS = "status"


class UIMessages:
    """Centralized UI messaging system."""

    @staticmethod
    def success(message: str, details: Optional[str] = None):
        """Display a success message."""
        colored_print(f"✅ {message}", Colors.OKGREEN)
        if details:
            colored_print(f"   {details}", Colors.OKCYAN)

    @staticmethod
    def error(message: str, details: Optional[str] = None):
        """Display an error message."""
        colored_print(f"❌ {message}", Colors.FAIL)
        if details:
            colored_print(f"   {details}", Colors.OKCYAN)

    @staticmethod
    def warning(message: str, details: Optional[str] = None):
        """Display a warning message."""
        colored_print(f"⚠️  {message}", Colors.WARNING)
        if details:
            colored_print(f"   {details}", Colors.OKCYAN)

    @staticmethod
    def info(message: str, details: Optional[str] = None):
        """Display an info message."""
        colored_print(f"🔍 {message}", Colors.OKBLUE)
        if details:
            colored_print(f"   {details}", Colors.OKCYAN)

    @staticmethod
    def header(message: str, separator: bool = True):
        """Display a header message."""
        colored_print(message, Colors.HEADER)
        if separator:
            colored_print("=" * 60, Colors.HEADER)

    @staticmethod
    def tip(message: str):
        """Display a tip message."""
        colored_print(f"💡 {message}", Colors.OKCYAN)

    @staticmethod
    def status_item(label: str, value: str, status_color: str = Colors.OKCYAN):
        """Display a status item with consistent formatting."""
        colored_print(f"├─ {label:<20} {value}", status_color)

    @staticmethod
    def status_item_last(label: str, value: str, status_color: str = Colors.OKCYAN):
        """Display the last status item with consistent formatting."""
        colored_print(f"└─ {label:<20} {value}", status_color)

    @staticmethod
    def connection_status(
        service: str, connected: bool, details: Optional[Dict[str, str]] = None
    ):
        """Display connection status for a service."""
        if connected:
            colored_print(f"✅ {service:<20} Connected", Colors.OKGREEN)
        else:
            colored_print(f"❌ {service:<20} Disconnected", Colors.FAIL)

        if details:
            for key, value in details.items():
                colored_print(f"   └─ {key:<15} {value}", Colors.OKCYAN)

    @staticmethod
    def processing_start(message: str):
        """Display a processing start message."""
        colored_print(f"🔧 {message}", Colors.OKBLUE)

    @staticmethod
    def processing_complete(message: str):
        """Display a processing completion message."""
        colored_print(f"✅ {message}", Colors.OKGREEN)


class InteractiveHelp:
    """Interactive mode help system."""

    @staticmethod
    def show_welcome():
        """Display the welcome message for interactive mode."""
        UIMessages.header("🚀 Welcome to ColPali Interactive Search!")
        UIMessages.info(
            "🤖 AI-Powered Document Search: Just type your questions directly!"
        )
        UIMessages.tip(
            "Ask questions about your documents and get intelligent responses!"
        )

    @staticmethod
    def show_commands_help():
        """Display available commands in conversational mode."""
        commands = COMMAND_REGISTRY.get_interactive_commands()

        colored_print("📋 Available commands:", Colors.OKBLUE)

        for cmd in commands:
            colored_print(f"  • {cmd.name:<20} - {cmd.help_text}", Colors.OKCYAN)

    @staticmethod
    def show_next_steps(has_documents: bool):
        """Display next steps based on system state."""
        UIMessages.info("💡 Next Steps:")
        if has_documents:
            UIMessages.tip(
                "Just type your questions directly to search documents with AI"
            )
        else:
            UIMessages.tip("Use 'upload <file>' to add documents")


class StatusDisplay:
    """System status display utilities."""

    @staticmethod
    def show_system_status_header():
        """Display system status header."""
        UIMessages.header("📊 System Status")

    @staticmethod
    def show_connection_status_header():
        """Display connection status header."""
        UIMessages.header("🔌 Connection Status")

    @staticmethod
    def show_collection_status_header():
        """Display collection status header."""
        UIMessages.header("🗄️  Collection Status")

    @staticmethod
    def show_processing_status_header():
        """Display processing status header."""
        UIMessages.header("📤 Image Processing Status")

    @staticmethod
    def show_model_config_header():
        """Display model configuration header."""
        UIMessages.header("🤖 Model Configuration")

    @staticmethod
    def show_system_summary_header():
        """Display system summary header."""
        UIMessages.header("📋 System Summary")

    @staticmethod
    def show_search_results_header(query: str, count: int):
        """Display search results header."""
        UIMessages.success(f"Found {count} results for '{query}':")
        UIMessages.header("", separator=True)

    @staticmethod
    def show_search_timing(search_time: float):
        """Display search timing information."""
        UIMessages.info(f"Search completed in {search_time:.2f}s")


class ValidationMessages:
    """Validation and error message utilities."""

    @staticmethod
    def query_empty():
        """Display empty query error."""
        UIMessages.error("Query cannot be empty")

    @staticmethod
    def query_too_short():
        """Display query too short error."""
        UIMessages.warning("Query must be at least 2 characters long")

    @staticmethod
    def file_not_found(file_path: str):
        """Display file not found error."""
        UIMessages.error(f"File '{file_path}' does not exist")

    @staticmethod
    def not_a_file(file_path: str):
        """Display not a file error."""
        UIMessages.error(f"'{file_path}' is not a file")

    @staticmethod
    def collection_empty():
        """Display empty collection warning."""
        UIMessages.warning("Collection is empty or does not exist.")

    @staticmethod
    def no_results_found():
        """Display no results found message."""
        UIMessages.warning("📭 No results found.")

    @staticmethod
    def interactive_tip_upload():
        """Display interactive tip for uploading."""
        UIMessages.tip(
            "Use 'upload' to add documents first, then ask your questions directly"
        )

    @staticmethod
    def interactive_tip_search():
        """Display interactive tip for searching."""
        UIMessages.tip("Try different keywords or check if documents are indexed")

    @staticmethod
    def unknown_mode(mode: str):
        """Display unknown mode error."""
        UIMessages.error(f"Unknown mode: {mode}")
        UIMessages.tip("Available modes: basic, conversational")

    @staticmethod
    def command_usage(command: str, usage: str):
        """Display command usage error."""
        UIMessages.error(f"Usage: {command} {usage}")


class SetupMessages:
    """Setup and initialization message utilities."""

    @staticmethod
    def pipeline_initializing():
        """Display pipeline initialization message."""
        UIMessages.processing_start("Initializing pipeline for the session...")

    @staticmethod
    def pipeline_ready():
        """Display pipeline ready message."""
        UIMessages.processing_complete("Pipeline ready.")

    @staticmethod
    def service_configured(service: str):
        """Display service configured message."""
        UIMessages.success(f"🔗 {service} connection configured")

    @staticmethod
    def service_failed(service: str, error: str):
        """Display service failure message."""
        UIMessages.warning(f"{service} setup failed: {error}")

    @staticmethod
    def service_disabled(service: str, reason: str):
        """Display service disabled message."""
        UIMessages.tip(f"{service} {reason}")

    @staticmethod
    def goodbye():
        """Display goodbye message."""
        UIMessages.success("👋 Goodbye!")

from dataclasses import dataclass
from enum import Enum
from typing import Dict, List, Optional


class CommandType(Enum):
    """Types of commands available in the CLI."""

    UPLOAD = "upload"
    CLEAR = "clear"
    STATUS = "status"
    HELP = "help"
    EXIT = "exit"


@dataclass
class Command:
    """Represents a CLI command with its properties."""

    name: str
    aliases: List[str]
    command_type: CommandType
    description: str
    help_text: str
    emoji: str
    requires_args: bool = False


class CommandRegistry:
    """Centralized registry for all CLI commands."""

    def __init__(self):
        self._commands: Dict[str, Command] = {}
        self._aliases: Dict[str, str] = {}
        self._register_commands()

    def _register_commands(self):
        """Register all available commands."""
        commands = [
            Command(
                name="upload",
                aliases=["upload"],
                command_type=CommandType.UPLOAD,
                description="Upload and index documents from file or default dataset",
                help_text="Upload and index documents",
                emoji="📂",
                requires_args=False,
            ),
            Command(
                name="clear-data",
                aliases=["clear-data"],
                command_type=CommandType.CLEAR,
                description="Clear all data from the system",
                help_text="Clear qdrant collection and minio bucket",
                emoji="🗑️",
                requires_args=False,
            ),
            Command(
                name="show-status",
                aliases=["show-status"],
                command_type=CommandType.STATUS,
                description="Show current system status",
                help_text="Show system status",
                emoji="📊",
                requires_args=False,
            ),
            Command(
                name="help",
                aliases=["help"],
                command_type=CommandType.HELP,
                description="Show help information",
                help_text="Show this help",
                emoji="❓",
                requires_args=False,
            ),
            Command(
                name="exit",
                aliases=["exit", "quit"],
                command_type=CommandType.EXIT,
                description="Exit conversational mode",
                help_text="Exit conversational mode",
                emoji="👋",
                requires_args=False,
            ),
        ]

        for command in commands:
            self._register_command(command)

    def _register_command(self, command: Command):
        """Register a single command and its aliases."""
        self._commands[command.name] = command

        # Register primary aliases
        for alias in command.aliases:
            self._aliases[alias] = command.name

    def get_command(self, name: str) -> Optional[Command]:
        """Get a command by name or alias."""
        canonical_name = self._aliases.get(name)
        if canonical_name:
            return self._commands.get(canonical_name)
        return None

    def get_canonical_name(self, name: str) -> Optional[str]:
        """Get the canonical name for a command alias."""
        return self._aliases.get(name)

    def get_all_commands(self) -> List[Command]:
        """Get all registered commands."""
        return list(self._commands.values())

    def get_commands_by_type(self, command_type: CommandType) -> List[Command]:
        """Get all commands of a specific type."""
        return [
            cmd for cmd in self._commands.values() if cmd.command_type == command_type
        ]

    def get_interactive_commands(self) -> List[Command]:
        """Get commands that are relevant for interactive mode."""
        excluded_types = {CommandType.HELP}
        return [
            cmd
            for cmd in self._commands.values()
            if cmd.command_type not in excluded_types
        ]


# Global registry instance
COMMAND_REGISTRY = CommandRegistry()

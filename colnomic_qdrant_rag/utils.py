from pathlib import Path


# Color codes for better output
class Colors:
    HEADER = "\033[95m"
    OKBLUE = "\033[94m"
    OKCYAN = "\033[96m"
    OKGREEN = "\033[92m"
    WARNING = "\033[93m"
    FAIL = "\033[91m"
    ENDC = "\033[0m"
    BOLD = "\033[1m"
    UNDERLINE = "\033[4m"


def colored_print(message, color=Colors.ENDC):
    """Print colored message to console."""
    print(f"{color}{message}{Colors.ENDC}")


def validate_query(query):
    """Validate the search query."""
    if not query or query.strip() == "":
        colored_print("❌ Error: Query cannot be empty", Colors.FAIL)
        return False
    if len(query.strip()) < 2:
        colored_print(
            "❌ Error: Query must be at least 2 characters long", Colors.WARNING
        )
        return False
    return True


def validate_file_path(file_path):
    """Validate the file path exists."""
    if not file_path:
        return True  # No file path is valid (uses default dataset)

    path = Path(file_path)
    if not path.exists():
        colored_print(f"❌ Error: File '{file_path}' does not exist", Colors.FAIL)
        return False
    if not path.is_file():
        colored_print(f"❌ Error: '{file_path}' is not a file", Colors.FAIL)
        return False
    return True

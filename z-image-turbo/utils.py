"""Utility functions for Z-Image-Turbo application."""


def get_resolution(resolution):
    """
    Extract width and height from resolution string.

    Args:
        resolution (str): Resolution string in format "WIDTHxHEIGHT" or "WIDTH x HEIGHT"

    Returns:
        tuple: (width, height) as integers
    """
    # Remove all whitespace and normalize separator
    normalized = resolution.replace(" ", "").replace("×", "x")

    # Split by 'x' and parse dimensions
    if "x" in normalized:
        parts = normalized.split("x", 1)
        if len(parts) == 2:
            try:
                # Extract only digits from each part
                width = int("".join(c for c in parts[0] if c.isdigit()))
                height = int("".join(c for c in parts[1] if c.isdigit()))
                return width, height
            except ValueError:
                pass

    return 1024, 1024

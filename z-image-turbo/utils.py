"""Utility functions for Z-Image-Turbo application."""

import re


def get_resolution(resolution):
    """
    Extract width and height from resolution string.

    Args:
        resolution (str): Resolution string in format "WIDTHxHEIGHT" or "WIDTH x HEIGHT"

    Returns:
        tuple: (width, height) as integers
    """
    match = re.search(r"(\d+)\s*[×x]\s*(\d+)", resolution)
    if match:
        return int(match.group(1)), int(match.group(2))
    return 1024, 1024

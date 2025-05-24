"""Logging utilities using loguru."""

from loguru import logger
import sys
from typing import Optional


def get_logger(name: Optional[str] = None, level: str = "INFO") -> logger:
    """
    Get a configured logger instance.
    
    Args:
        name: Optional logger name
        level: Logging level (DEBUG, INFO, WARNING, ERROR, CRITICAL)
        
    Returns:
        Configured logger instance
    """
    logger.remove()  # Remove default handler
    logger.add(
        sys.stdout,
        level=level,
        format="<green>{time:YYYY-MM-DD HH:mm:ss}</green> | "
               "<level>{level: <8}</level> | "
               "<cyan>{name}</cyan>:<cyan>{function}</cyan>:<cyan>{line}</cyan> - "
               "<level>{message}</level>",
        colorize=True
    )
    
    if name:
        return logger.bind(name=name)
    return logger 
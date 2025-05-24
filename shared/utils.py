"""
Common utility functions for AI scripts.
"""

import os
import json
import logging
from typing import Any, Dict, Optional
from pathlib import Path
from dotenv import load_dotenv


def setup_logging(name: str, level: str = "INFO") -> logging.Logger:
    """
    Set up logging for a script.
    
    Args:
        name: Logger name (usually __name__)
        level: Logging level (DEBUG, INFO, WARNING, ERROR, CRITICAL)
    
    Returns:
        Configured logger instance
    """
    logging.basicConfig(
        level=getattr(logging, level.upper()),
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S"
    )
    return logging.getLogger(name)


def load_config(config_path: Optional[str] = None) -> Dict[str, Any]:
    """
    Load configuration from environment variables and optional config file.
    
    Args:
        config_path: Path to config JSON file (optional)
    
    Returns:
        Dictionary containing configuration
    """
    # Load environment variables
    load_dotenv()
    
    config = {
        "openai_api_key": os.getenv("OPENAI_API_KEY"),
        "anthropic_api_key": os.getenv("ANTHROPIC_API_KEY"),
    }
    
    # Load additional config from file if provided
    if config_path and Path(config_path).exists():
        with open(config_path, "r") as f:
            file_config = json.load(f)
            config.update(file_config)
    
    return config


def save_json(data: Any, filepath: str) -> None:
    """
    Save data to a JSON file.
    
    Args:
        data: Data to save
        filepath: Path to save the file
    """
    Path(filepath).parent.mkdir(parents=True, exist_ok=True)
    with open(filepath, "w") as f:
        json.dump(data, f, indent=2, default=str)


def load_json(filepath: str) -> Any:
    """
    Load data from a JSON file.
    
    Args:
        filepath: Path to the JSON file
    
    Returns:
        Loaded data
    """
    with open(filepath, "r") as f:
        return json.load(f)


def ensure_dir(directory: str) -> Path:
    """
    Ensure a directory exists, create if it doesn't.
    
    Args:
        directory: Directory path
    
    Returns:
        Path object for the directory
    """
    path = Path(directory)
    path.mkdir(parents=True, exist_ok=True)
    return path


def get_script_dir() -> Path:
    """
    Get the directory of the calling script.
    
    Returns:
        Path object for the script's directory
    """
    import inspect
    frame = inspect.currentframe().f_back
    return Path(frame.f_globals["__file__"]).parent 
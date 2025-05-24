"""Helper functions for common operations."""

from typing import Any, Dict, Optional
from pydantic import BaseModel, ValidationError
import json


def format_response(data: Any, status: str = "success", message: Optional[str] = None) -> Dict[str, Any]:
    """
    Format a standardized response.
    
    Args:
        data: The response data
        status: Response status (success, error, warning)
        message: Optional message
        
    Returns:
        Formatted response dictionary
    """
    response = {
        "status": status,
        "data": data
    }
    
    if message:
        response["message"] = message
        
    return response


def validate_data(data: Dict[str, Any], model_class: BaseModel) -> tuple[bool, Optional[BaseModel], Optional[str]]:
    """
    Validate data against a Pydantic model.
    
    Args:
        data: Data to validate
        model_class: Pydantic model class
        
    Returns:
        Tuple of (is_valid, validated_model, error_message)
    """
    try:
        validated_model = model_class(**data)
        return True, validated_model, None
    except ValidationError as e:
        return False, None, str(e)


def safe_json_loads(json_string: str) -> tuple[bool, Optional[Dict[str, Any]], Optional[str]]:
    """
    Safely parse JSON string.
    
    Args:
        json_string: JSON string to parse
        
    Returns:
        Tuple of (is_valid, parsed_data, error_message)
    """
    try:
        data = json.loads(json_string)
        return True, data, None
    except json.JSONDecodeError as e:
        return False, None, str(e) 
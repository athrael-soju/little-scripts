"""
Unified error handling system for consistent error management.
This module provides centralized error handling patterns.
"""

import sys
from functools import wraps
from typing import Any, Callable, Dict, Optional

from .messaging import SetupMessages, UIMessages, ValidationMessages


class ErrorSeverity:
    """Error severity levels."""

    WARNING = "warning"
    ERROR = "error"
    CRITICAL = "critical"


class ApplicationError(Exception):
    """Base exception for application errors."""

    def __init__(
        self,
        message: str,
        severity: str = ErrorSeverity.ERROR,
        details: Optional[str] = None,
    ):
        super().__init__(message)
        self.severity = severity
        self.details = details


class ValidationError(ApplicationError):
    """Exception for validation errors."""

    def __init__(self, message: str, details: Optional[str] = None):
        super().__init__(message, ErrorSeverity.WARNING, details)


class ServiceError(ApplicationError):
    """Exception for service-related errors."""

    def __init__(self, service: str, message: str, details: Optional[str] = None):
        super().__init__(f"{service} error: {message}", ErrorSeverity.ERROR, details)
        self.service = service


class CriticalError(ApplicationError):
    """Exception for critical system errors."""

    def __init__(self, message: str, details: Optional[str] = None):
        super().__init__(message, ErrorSeverity.CRITICAL, details)


class ErrorHandler:
    """Centralized error handling utilities."""

    @staticmethod
    def handle_error(error: Exception, context: Optional[str] = None) -> bool:
        """
        Handle an error with appropriate messaging and return success status.

        Args:
            error: The exception to handle
            context: Optional context information

        Returns:
            bool: False for errors, True for warnings
        """
        if isinstance(error, ApplicationError):
            if error.severity == ErrorSeverity.WARNING:
                UIMessages.warning(str(error), error.details)
                return True
            elif error.severity == ErrorSeverity.ERROR:
                UIMessages.error(str(error), error.details)
                return False
            elif error.severity == ErrorSeverity.CRITICAL:
                UIMessages.error(str(error), error.details)
                if context:
                    UIMessages.error(f"Context: {context}")
                sys.exit(1)
        else:
            # Handle generic exceptions
            message = str(error)
            if context:
                message = f"{context}: {message}"
            UIMessages.error(message)
            return False


def safe_execution(context: Optional[str] = None, return_on_error: Any = False):
    """
    Decorator for safe function execution with error handling.

    Args:
        context: Context information for error messages
        return_on_error: Value to return on error (default: False)
    """

    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            try:
                return func(*args, **kwargs)
            except Exception as e:
                ErrorHandler.handle_error(e, context or func.__name__)
                return return_on_error

        return wrapper

    return decorator


class ServiceManager:
    """Manager for service initialization."""

    def __init__(self):
        self.services: Dict[str, Any] = {}
        self.service_status: Dict[str, bool] = {}

    def register_service(self, name: str, factory: Callable) -> Any:
        """
        Register and initialize a service.

        Args:
            name: Service name
            factory: Factory function to create the service

        Returns:
            The initialized service

        Raises:
            ServiceError: If service initialization fails
        """
        try:
            service = factory()
            self.services[name] = service
            self.service_status[name] = True
            SetupMessages.service_configured(name)
            return service
        except Exception as e:
            self.service_status[name] = False
            raise ServiceError(name, str(e))

    def get_service(self, name: str) -> Optional[Any]:
        """Get a registered service."""
        return self.services.get(name)

    def is_service_available(self, name: str) -> bool:
        """Check if a service is available."""
        return self.service_status.get(name, False)

    def get_available_services(self) -> list:
        """Get list of available services."""
        return [name for name, status in self.service_status.items() if status]


class ValidationManager:
    """Centralized validation with consistent error handling."""

    @staticmethod
    def validate_query(query: str) -> bool:
        """Validate a search query."""
        if not query or query.strip() == "":
            ValidationMessages.query_empty()
            return False
        if len(query.strip()) < 2:
            ValidationMessages.query_too_short()
            return False
        return True

    @staticmethod
    def validate_file_path(file_path: Optional[str]) -> bool:
        """Validate a file path."""
        if not file_path:
            return True  # No file path is valid (uses default dataset)

        from pathlib import Path

        path = Path(file_path)

        if not path.exists():
            ValidationMessages.file_not_found(file_path)
            return False

        if not path.is_file():
            ValidationMessages.not_a_file(file_path)
            return False

        return True

    @staticmethod
    def validate_collection_state(vector_db, collection_name: str) -> bool:
        """Validate that collection exists and has documents."""
        try:
            if not vector_db.collection_exists():
                ValidationMessages.collection_empty()
                return False

            collection_info = vector_db.client.get_collection(collection_name)
            if collection_info.points_count == 0:
                ValidationMessages.collection_empty()
                return False

            return True
        except Exception as e:
            raise ServiceError(
                "Vector Database", f"Failed to check collection state: {e}"
            )

    @staticmethod
    def validate_mode(mode: str) -> bool:
        """Validate a search mode."""
        valid_modes = ["basic", "search", "conversational", "ai", "analyze", "analysis"]
        if mode.lower() not in valid_modes:
            ValidationMessages.unknown_mode(mode)
            return False
        return True


# Global service manager instance
SERVICE_MANAGER = ServiceManager()

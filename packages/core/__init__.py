"""Core business logic package for the monorepo."""

__version__ = "0.1.0"

from .models import User, Task
from .services import UserService, TaskService

__all__ = ["User", "Task", "UserService", "TaskService"] 
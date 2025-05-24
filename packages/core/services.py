"""Core business logic services."""

from typing import List, Optional, Dict, Any
from datetime import datetime
from .models import User, Task, TaskCreate, TaskUpdate, TaskStatus


class UserService:
    """Service for user-related operations."""
    
    def __init__(self):
        self._users: Dict[int, User] = {}
        self._next_id = 1
    
    def create_user(self, username: str, email: str, full_name: Optional[str] = None) -> User:
        """Create a new user."""
        user = User(
            id=self._next_id,
            username=username,
            email=email,
            full_name=full_name,
            created_at=datetime.now()
        )
        self._users[self._next_id] = user
        self._next_id += 1
        return user
    
    def get_user(self, user_id: int) -> Optional[User]:
        """Get a user by ID."""
        return self._users.get(user_id)
    
    def get_user_by_username(self, username: str) -> Optional[User]:
        """Get a user by username."""
        for user in self._users.values():
            if user.username == username:
                return user
        return None
    
    def list_users(self, active_only: bool = True) -> List[User]:
        """List all users."""
        users = list(self._users.values())
        if active_only:
            users = [user for user in users if user.is_active]
        return users
    
    def deactivate_user(self, user_id: int) -> bool:
        """Deactivate a user."""
        user = self._users.get(user_id)
        if user:
            user.is_active = False
            return True
        return False


class TaskService:
    """Service for task-related operations."""
    
    def __init__(self, user_service: UserService):
        self._tasks: Dict[int, Task] = {}
        self._next_id = 1
        self._user_service = user_service
    
    def create_task(self, task_data: TaskCreate) -> Task:
        """Create a new task."""
        task = Task(
            id=self._next_id,
            title=task_data.title,
            description=task_data.description,
            assigned_to=task_data.assigned_to,
            due_date=task_data.due_date,
            created_at=datetime.now(),
            updated_at=datetime.now()
        )
        self._tasks[self._next_id] = task
        self._next_id += 1
        return task
    
    def get_task(self, task_id: int) -> Optional[Task]:
        """Get a task by ID."""
        return self._tasks.get(task_id)
    
    def update_task(self, task_id: int, task_data: TaskUpdate) -> Optional[Task]:
        """Update a task."""
        task = self._tasks.get(task_id)
        if not task:
            return None
        
        update_data = task_data.dict(exclude_unset=True)
        for field, value in update_data.items():
            setattr(task, field, value)
        
        task.updated_at = datetime.now()
        return task
    
    def list_tasks(self, status: Optional[TaskStatus] = None, assigned_to: Optional[int] = None) -> List[Task]:
        """List tasks with optional filtering."""
        tasks = list(self._tasks.values())
        
        if status:
            tasks = [task for task in tasks if task.status == status]
        
        if assigned_to:
            tasks = [task for task in tasks if task.assigned_to == assigned_to]
        
        return tasks
    
    def assign_task(self, task_id: int, user_id: int) -> bool:
        """Assign a task to a user."""
        task = self._tasks.get(task_id)
        user = self._user_service.get_user(user_id)
        
        if task and user and user.is_active:
            task.assigned_to = user_id
            task.updated_at = datetime.now()
            return True
        return False
    
    def complete_task(self, task_id: int) -> bool:
        """Mark a task as completed."""
        task = self._tasks.get(task_id)
        if task:
            task.status = TaskStatus.COMPLETED
            task.updated_at = datetime.now()
            return True
        return False 
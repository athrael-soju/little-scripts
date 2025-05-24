"""Tests for the core package."""

import pytest
from datetime import datetime

from core import UserService, TaskService, TaskCreate, TaskUpdate, TaskStatus


class TestUserService:
    """Test cases for UserService."""
    
    def setup_method(self):
        """Set up test dependencies."""
        self.user_service = UserService()
    
    def test_create_user(self):
        """Test user creation."""
        user = self.user_service.create_user("testuser", "test@example.com", "Test User")
        
        assert user.id == 1
        assert user.username == "testuser"
        assert user.email == "test@example.com"
        assert user.full_name == "Test User"
        assert user.is_active is True
        assert isinstance(user.created_at, datetime)
    
    def test_get_user(self):
        """Test getting a user by ID."""
        user = self.user_service.create_user("testuser", "test@example.com")
        retrieved_user = self.user_service.get_user(1)
        
        assert retrieved_user == user
        assert self.user_service.get_user(999) is None
    
    def test_get_user_by_username(self):
        """Test getting a user by username."""
        user = self.user_service.create_user("testuser", "test@example.com")
        retrieved_user = self.user_service.get_user_by_username("testuser")
        
        assert retrieved_user == user
        assert self.user_service.get_user_by_username("nonexistent") is None
    
    def test_list_users(self):
        """Test listing users."""
        user1 = self.user_service.create_user("user1", "user1@example.com")
        user2 = self.user_service.create_user("user2", "user2@example.com")
        
        users = self.user_service.list_users()
        assert len(users) == 2
        assert user1 in users
        assert user2 in users
    
    def test_deactivate_user(self):
        """Test user deactivation."""
        user = self.user_service.create_user("testuser", "test@example.com")
        
        # Deactivate user
        success = self.user_service.deactivate_user(1)
        assert success is True
        assert user.is_active is False
        
        # Try to deactivate non-existent user
        success = self.user_service.deactivate_user(999)
        assert success is False


class TestTaskService:
    """Test cases for TaskService."""
    
    def setup_method(self):
        """Set up test dependencies."""
        self.user_service = UserService()
        self.task_service = TaskService(self.user_service)
        self.user = self.user_service.create_user("testuser", "test@example.com")
    
    def test_create_task(self):
        """Test task creation."""
        task_data = TaskCreate(
            title="Test task",
            description="A test task",
            assigned_to=self.user.id
        )
        task = self.task_service.create_task(task_data)
        
        assert task.id == 1
        assert task.title == "Test task"
        assert task.description == "A test task"
        assert task.assigned_to == self.user.id
        assert task.status == TaskStatus.PENDING
        assert isinstance(task.created_at, datetime)
        assert isinstance(task.updated_at, datetime)
    
    def test_get_task(self):
        """Test getting a task by ID."""
        task_data = TaskCreate(title="Test task")
        task = self.task_service.create_task(task_data)
        retrieved_task = self.task_service.get_task(1)
        
        assert retrieved_task == task
        assert self.task_service.get_task(999) is None
    
    def test_update_task(self):
        """Test task update."""
        task_data = TaskCreate(title="Original title")
        task = self.task_service.create_task(task_data)
        
        update_data = TaskUpdate(title="Updated title", status=TaskStatus.IN_PROGRESS)
        updated_task = self.task_service.update_task(1, update_data)
        
        assert updated_task is not None
        assert updated_task.title == "Updated title"
        assert updated_task.status == TaskStatus.IN_PROGRESS
        assert updated_task.updated_at > updated_task.created_at
    
    def test_list_tasks(self):
        """Test listing tasks."""
        task1_data = TaskCreate(title="Task 1", assigned_to=self.user.id)
        task2_data = TaskCreate(title="Task 2")
        
        task1 = self.task_service.create_task(task1_data)
        task2 = self.task_service.create_task(task2_data)
        
        # List all tasks
        all_tasks = self.task_service.list_tasks()
        assert len(all_tasks) == 2
        
        # Filter by assigned user
        assigned_tasks = self.task_service.list_tasks(assigned_to=self.user.id)
        assert len(assigned_tasks) == 1
        assert assigned_tasks[0] == task1
        
        # Filter by status
        pending_tasks = self.task_service.list_tasks(status=TaskStatus.PENDING)
        assert len(pending_tasks) == 2
    
    def test_assign_task(self):
        """Test task assignment."""
        task_data = TaskCreate(title="Test task")
        task = self.task_service.create_task(task_data)
        
        # Assign to existing user
        success = self.task_service.assign_task(1, self.user.id)
        assert success is True
        assert task.assigned_to == self.user.id
        
        # Try to assign to non-existent user
        success = self.task_service.assign_task(1, 999)
        assert success is False
    
    def test_complete_task(self):
        """Test task completion."""
        task_data = TaskCreate(title="Test task")
        task = self.task_service.create_task(task_data)
        
        # Complete existing task
        success = self.task_service.complete_task(1)
        assert success is True
        assert task.status == TaskStatus.COMPLETED
        
        # Try to complete non-existent task
        success = self.task_service.complete_task(999)
        assert success is False 
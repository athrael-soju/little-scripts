"""Main CLI application for task management."""

import click
from rich.console import Console
from rich.table import Table
from rich.panel import Panel
from typing import Optional

from core import UserService, TaskService, TaskCreate, TaskUpdate, TaskStatus
from utils import get_logger

console = Console()
logger = get_logger("cli-tool")

# Global services
user_service = UserService()
task_service = TaskService(user_service)


@click.group()
@click.version_option(version="0.1.0")
def cli():
    """Simple task management CLI tool."""
    pass


@cli.group()
def users():
    """User management commands."""
    pass


@users.command("create")
@click.option("--username", "-u", required=True, help="Username")
@click.option("--email", "-e", required=True, help="Email address")
@click.option("--full-name", "-f", help="Full name")
def create_user(username: str, email: str, full_name: Optional[str]):
    """Create a new user."""
    try:
        user = user_service.create_user(username, email, full_name)
        console.print(Panel(f"✅ User created successfully!\nID: {user.id}\nUsername: {user.username}", 
                          title="Success", border_style="green"))
        logger.info(f"Created user: {username}")
    except Exception as e:
        console.print(Panel(f"❌ Error creating user: {e}", title="Error", border_style="red"))
        logger.error(f"Failed to create user: {e}")


@users.command("list")
def list_users():
    """List all users."""
    users_list = user_service.list_users()
    
    if not users_list:
        console.print("No users found.")
        return
    
    table = Table(title="Users")
    table.add_column("ID", style="cyan")
    table.add_column("Username", style="magenta")
    table.add_column("Email", style="green")
    table.add_column("Full Name", style="yellow")
    table.add_column("Status", style="red")
    
    for user in users_list:
        status = "Active" if user.is_active else "Inactive"
        table.add_row(
            str(user.id),
            user.username,
            user.email,
            user.full_name or "N/A",
            status
        )
    
    console.print(table)


@cli.group()
def tasks():
    """Task management commands."""
    pass


@tasks.command("create")
@click.option("--title", "-t", required=True, help="Task title")
@click.option("--description", "-d", help="Task description")
@click.option("--assigned-to", "-a", type=int, help="Assign to user ID")
def create_task(title: str, description: Optional[str], assigned_to: Optional[int]):
    """Create a new task."""
    try:
        task_data = TaskCreate(
            title=title,
            description=description,
            assigned_to=assigned_to
        )
        task = task_service.create_task(task_data)
        console.print(Panel(f"✅ Task created successfully!\nID: {task.id}\nTitle: {task.title}", 
                          title="Success", border_style="green"))
        logger.info(f"Created task: {title}")
    except Exception as e:
        console.print(Panel(f"❌ Error creating task: {e}", title="Error", border_style="red"))
        logger.error(f"Failed to create task: {e}")


@tasks.command("list")
@click.option("--status", "-s", type=click.Choice(['pending', 'in_progress', 'completed', 'cancelled']), 
              help="Filter by status")
@click.option("--assigned-to", "-a", type=int, help="Filter by assigned user ID")
def list_tasks(status: Optional[str], assigned_to: Optional[int]):
    """List tasks with optional filtering."""
    task_status = TaskStatus(status) if status else None
    tasks_list = task_service.list_tasks(status=task_status, assigned_to=assigned_to)
    
    if not tasks_list:
        console.print("No tasks found.")
        return
    
    table = Table(title="Tasks")
    table.add_column("ID", style="cyan")
    table.add_column("Title", style="magenta")
    table.add_column("Status", style="green")
    table.add_column("Assigned To", style="yellow")
    table.add_column("Created", style="blue")
    
    for task in tasks_list:
        assigned_user = "N/A"
        if task.assigned_to:
            user = user_service.get_user(task.assigned_to)
            assigned_user = user.username if user else f"ID:{task.assigned_to}"
        
        created_str = task.created_at.strftime("%Y-%m-%d %H:%M") if task.created_at else "N/A"
        
        table.add_row(
            str(task.id),
            task.title,
            task.status.value,
            assigned_user,
            created_str
        )
    
    console.print(table)


@tasks.command("complete")
@click.argument("task_id", type=int)
def complete_task(task_id: int):
    """Mark a task as completed."""
    if task_service.complete_task(task_id):
        console.print(Panel(f"✅ Task {task_id} marked as completed!", 
                          title="Success", border_style="green"))
        logger.info(f"Completed task: {task_id}")
    else:
        console.print(Panel(f"❌ Task {task_id} not found!", 
                          title="Error", border_style="red"))


if __name__ == "__main__":
    cli() 
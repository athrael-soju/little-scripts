"""Main FastAPI application for task management."""

from fastapi import FastAPI, HTTPException, Depends
from fastapi.responses import HTMLResponse
from typing import List, Optional

from core import UserService, TaskService, User, Task, TaskCreate, TaskUpdate, TaskStatus
from utils import get_logger, format_response

# Initialize services
user_service = UserService()
task_service = TaskService(user_service)
logger = get_logger("web-app")

app = FastAPI(
    title="Task Management API",
    description="A simple task management system",
    version="0.1.0"
)


@app.get("/", response_class=HTMLResponse)
async def root():
    """Root endpoint with basic HTML interface."""
    html_content = """
    <!DOCTYPE html>
    <html>
    <head>
        <title>Task Management API</title>
        <style>
            body { font-family: Arial, sans-serif; margin: 40px; background-color: #f5f5f5; }
            .container { max-width: 800px; margin: 0 auto; background: white; padding: 30px; border-radius: 10px; box-shadow: 0 2px 10px rgba(0,0,0,0.1); }
            h1 { color: #333; text-align: center; }
            .section { margin: 30px 0; padding: 20px; border: 1px solid #ddd; border-radius: 5px; }
            .endpoint { background: #f8f9fa; padding: 10px; margin: 10px 0; border-radius: 3px; }
            code { background: #e9ecef; padding: 2px 5px; border-radius: 3px; }
            .method { font-weight: bold; color: #007bff; }
        </style>
    </head>
    <body>
        <div class="container">
            <h1>🚀 Task Management API</h1>
            <p>Welcome to the Task Management API! This is a simple web application built with FastAPI.</p>
            
            <div class="section">
                <h2>📚 API Documentation</h2>
                <p>Explore the interactive API documentation:</p>
                <ul>
                    <li><a href="/docs" target="_blank">Swagger UI (/docs)</a></li>
                    <li><a href="/redoc" target="_blank">ReDoc (/redoc)</a></li>
                </ul>
            </div>
            
            <div class="section">
                <h2>👥 User Endpoints</h2>
                <div class="endpoint">
                    <span class="method">POST</span> <code>/users/</code> - Create a new user
                </div>
                <div class="endpoint">
                    <span class="method">GET</span> <code>/users/</code> - List all users
                </div>
                <div class="endpoint">
                    <span class="method">GET</span> <code>/users/{user_id}</code> - Get user by ID
                </div>
            </div>
            
            <div class="section">
                <h2>📋 Task Endpoints</h2>
                <div class="endpoint">
                    <span class="method">POST</span> <code>/tasks/</code> - Create a new task
                </div>
                <div class="endpoint">
                    <span class="method">GET</span> <code>/tasks/</code> - List all tasks
                </div>
                <div class="endpoint">
                    <span class="method">GET</span> <code>/tasks/{task_id}</code> - Get task by ID
                </div>
                <div class="endpoint">
                    <span class="method">PUT</span> <code>/tasks/{task_id}</code> - Update task
                </div>
                <div class="endpoint">
                    <span class="method">POST</span> <code>/tasks/{task_id}/complete</code> - Mark task as completed
                </div>
            </div>
        </div>
    </body>
    </html>
    """
    return HTMLResponse(content=html_content)


# User endpoints
@app.post("/users/", response_model=dict)
async def create_user(username: str, email: str, full_name: Optional[str] = None):
    """Create a new user."""
    try:
        user = user_service.create_user(username, email, full_name)
        logger.info(f"Created user: {username}")
        return format_response(user.dict(), message="User created successfully")
    except Exception as e:
        logger.error(f"Failed to create user: {e}")
        raise HTTPException(status_code=400, detail=str(e))


@app.get("/users/", response_model=dict)
async def list_users(active_only: bool = True):
    """List all users."""
    users = user_service.list_users(active_only)
    return format_response([user.dict() for user in users])


@app.get("/users/{user_id}", response_model=dict)
async def get_user(user_id: int):
    """Get a user by ID."""
    user = user_service.get_user(user_id)
    if not user:
        raise HTTPException(status_code=404, detail="User not found")
    return format_response(user.dict())


# Task endpoints
@app.post("/tasks/", response_model=dict)
async def create_task(task_data: TaskCreate):
    """Create a new task."""
    try:
        task = task_service.create_task(task_data)
        logger.info(f"Created task: {task_data.title}")
        return format_response(task.dict(), message="Task created successfully")
    except Exception as e:
        logger.error(f"Failed to create task: {e}")
        raise HTTPException(status_code=400, detail=str(e))


@app.get("/tasks/", response_model=dict)
async def list_tasks(
    status: Optional[TaskStatus] = None,
    assigned_to: Optional[int] = None
):
    """List all tasks with optional filtering."""
    tasks = task_service.list_tasks(status=status, assigned_to=assigned_to)
    return format_response([task.dict() for task in tasks])


@app.get("/tasks/{task_id}", response_model=dict)
async def get_task(task_id: int):
    """Get a task by ID."""
    task = task_service.get_task(task_id)
    if not task:
        raise HTTPException(status_code=404, detail="Task not found")
    return format_response(task.dict())


@app.put("/tasks/{task_id}", response_model=dict)
async def update_task(task_id: int, task_data: TaskUpdate):
    """Update a task."""
    task = task_service.update_task(task_id, task_data)
    if not task:
        raise HTTPException(status_code=404, detail="Task not found")
    logger.info(f"Updated task: {task_id}")
    return format_response(task.dict(), message="Task updated successfully")


@app.post("/tasks/{task_id}/complete", response_model=dict)
async def complete_task(task_id: int):
    """Mark a task as completed."""
    success = task_service.complete_task(task_id)
    if not success:
        raise HTTPException(status_code=404, detail="Task not found")
    logger.info(f"Completed task: {task_id}")
    return format_response({"task_id": task_id}, message="Task marked as completed")


@app.get("/health")
async def health_check():
    """Health check endpoint."""
    return {"status": "healthy", "service": "task-management-api"}


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000) 
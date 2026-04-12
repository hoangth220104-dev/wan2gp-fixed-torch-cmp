"""Pydantic request/response models"""

from typing import Optional, Dict, Any
from pydantic import BaseModel, Field


class TaskResponse(BaseModel):
    """Response when submitting a task"""
    task_id: str
    status: str
    message: str


class TaskStatus(BaseModel):
    """Task status response"""
    task_id: str
    status: str  # "queued", "processing", "completed", "failed", "cancelled"
    progress: Optional[float] = None  # 0-100
    current_step: Optional[int] = None
    total_steps: Optional[int] = None
    error: Optional[str] = None
    created_at: str
    completed_at: Optional[str] = None
    result: Optional[Dict[str, Any]] = None


class HealthResponse(BaseModel):
    """Health check response"""
    status: str
    model_loaded: bool
    model_type: Optional[str] = None
    gpu_available: bool
    queue_size: int


class ModelsResponse(BaseModel):
    """Available models response"""
    current_model: Optional[str] = None
    available_models: list[str] = ["ltx2_19B", "ltx2_22B"]

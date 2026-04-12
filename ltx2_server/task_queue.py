"""Task queue management"""

import uuid
from pathlib import Path
from typing import Optional, Dict, Any
from datetime import datetime


class TaskQueue:
    """Manages video generation tasks"""
    
    def __init__(self):
        self.tasks: Dict[str, Dict[str, Any]] = {}
        self.current_task_id: Optional[str] = None
    
    @property
    def has_active_task(self) -> bool:
        """Check if there's an active (queued/processing) task"""
        return any(
            t["status"] in ("queued", "processing")
            for t in self.tasks.values()
        )
    
    @property
    def active_tasks_count(self) -> int:
        """Count active tasks"""
        return sum(
            1 for t in self.tasks.values()
            if t["status"] in ("queued", "processing")
        )
    
    def create_task(
        self,
        params: Dict[str, Any],
    ) -> str:
        """
        Create a new task and return task_id.
        
        Args:
            params: Generation parameters
            
        Returns:
            task_id
        """
        task_id = str(uuid.uuid4())
        
        num_steps = params.get("num_inference_steps") or 40
        
        self.tasks[task_id] = {
            "task_id": task_id,
            "status": "queued",
            "progress": 0.0,
            "current_step": 0,
            "total_steps": num_steps,
            "created_at": datetime.now().isoformat(),
            "completed_at": None,
            "error": None,
            "result": None,
            "params": params,
        }
        
        return task_id
    
    def update_progress(
        self,
        task_id: str,
        current_step: int,
        total_steps: int,
        progress: float,
    ) -> None:
        """Update task progress"""
        if task_id not in self.tasks:
            return
        
        task = self.tasks[task_id]
        task["current_step"] = current_step
        task["total_steps"] = total_steps
        task["progress"] = progress
    
    def set_processing(self, task_id: str) -> None:
        """Mark task as processing"""
        if task_id in self.tasks:
            self.tasks[task_id]["status"] = "processing"
            self.current_task_id = task_id
    
    def set_completed(
        self,
        task_id: str,
        video_path: str,
        filename: str,
        seed: int,
        generation_time: float,
    ) -> None:
        """Mark task as completed"""
        if task_id not in self.tasks:
            return
        
        task = self.tasks[task_id]
        task["status"] = "completed"
        task["progress"] = 100.0
        task["completed_at"] = datetime.now().isoformat()
        task["result"] = {
            "video_path": video_path,
            "video_url": f"/api/v1/tasks/{task_id}/video",
            "filename": filename,
            "seed": seed,
            "generation_time": round(generation_time, 2),
        }
    
    def set_failed(self, task_id: str, error: str) -> None:
        """Mark task as failed"""
        if task_id not in self.tasks:
            return
        
        task = self.tasks[task_id]
        task["status"] = "failed"
        task["error"] = error
        task["completed_at"] = datetime.now().isoformat()
    
    def set_cancelled(self, task_id: str) -> None:
        """Mark task as cancelled"""
        if task_id not in self.tasks:
            return
        
        task = self.tasks[task_id]
        task["status"] = "cancelled"
        task["completed_at"] = datetime.now().isoformat()
    
    def get_task(self, task_id: str) -> Optional[Dict[str, Any]]:
        """Get task by ID"""
        return self.tasks.get(task_id)
    
    def get_task_params(self, task_id: str) -> Optional[Dict[str, Any]]:
        """Get task parameters"""
        task = self.tasks.get(task_id)
        return task.get("params") if task else None
    
    def cleanup_uploads(self, task_id: str) -> None:
        """Clean up uploaded files for a task"""
        task = self.tasks.get(task_id)
        if not task:
            return
        
        params = task.get("params", {})
        for key in ["image_start_path", "image_end_path", "audio_guide_path"]:
            path = params.get(key)
            if path and Path(path).exists():
                Path(path).unlink()
                print(f"  Cleaned up: {path}")

"""API routes"""

import uuid
import asyncio
from pathlib import Path
from typing import Optional

from fastapi import APIRouter, HTTPException, UploadFile, File, Form
from fastapi.responses import FileResponse
from PIL import Image

from ltx2_server.models import (
    TaskResponse,
    TaskStatus,
    HealthResponse,
    ModelsResponse,
)
from ltx2_server.model_manager import ModelManager
from ltx2_server.task_queue import TaskQueue
from ltx2_server.generation import generate_video, save_video_result


router = APIRouter(prefix="/api/v1")


# ===== Validation Helpers =====

def is_valid_image(file_path: str) -> bool:
    """Check if file is a valid image"""
    try:
        with Image.open(file_path) as img:
            img.verify()
        return True
    except Exception:
        return False


def is_valid_audio(file_path: str) -> bool:
    """Check if file is a valid audio file"""
    try:
        import soundfile as sf
        info = sf.info(file_path)
        return info.channels > 0
    except Exception:
        return False

# Global instances (injected at startup)
model_manager: ModelManager
task_queue: TaskQueue
config = None


def init_globals(manager: ModelManager, queue: TaskQueue, cfg):
    """Initialize global references"""
    global model_manager, task_queue, config
    model_manager = manager
    task_queue = queue
    config = cfg


@router.get("/health", response_model=HealthResponse)
async def health_check():
    """Health check endpoint"""
    return HealthResponse(
        status="healthy",
        model_loaded=model_manager.is_loaded,
        model_type=model_manager.model_type,
        gpu_available=True,  # Assumes GPU is available if model is loaded
        queue_size=task_queue.active_tasks_count,
    )


@router.post("/generate", response_model=TaskResponse)
async def generate_video_endpoint(
    prompt: str = Form(..., description="Text prompt for video generation"),
    negative_prompt: Optional[str] = Form("", description="Negative prompt"),
    image_start: Optional[UploadFile] = File(None, description="Starting image"),
    image_end: Optional[UploadFile] = File(None, description="Ending image"),
    audio_guide: Optional[UploadFile] = File(None, description="Audio guide file"),
    input_video: Optional[UploadFile] = File(None, description="Input video to continue from"),
    width: int = Form(768, description="Video width (divisible by 64)"),
    height: int = Form(512, description="Video height (divisible by 64)"),
    num_frames: int = Form(121, description="Number of frames"),
    fps: float = Form(24.0, description="Frames per second"),
    num_inference_steps: str = Form("", description="Denoising steps"),
    guidance_scale: str = Form("", description="CFG scale"),
    seed: str = Form("", description="Random seed"),
    input_video_strength: str = Form("1.0", description="Input video influence strength (0.0-1.0)"),
    denoising_strength: str = Form("1.0", description="Denoising strength (0.0-1.0)"),
    prefix_frames_count: int = Form(0, description="Number of prefix frames from input video"),
    attention: Optional[str] = Form(None, description="Attention mode"),
    sliding_window_size: int = Form(481, description="Sliding window size"),
    sliding_window_overlap: int = Form(17, description="Sliding window overlap"),
):
    """Submit a video generation task"""

    if not model_manager.is_loaded:
        raise HTTPException(status_code=503, detail="Model not loaded")

    if task_queue.has_active_task:
        raise HTTPException(
            status_code=429,
            detail="A task is already running. Please wait for it to complete."
        )

    # Adjust dimensions to be divisible by 64
    import numpy as np
    if width % 64 != 0:
        width = int(np.ceil(width / 64) * 64)
    if height % 64 != 0:
        height = int(np.ceil(height / 64) * 64)

    # Parse optional numeric fields (accept empty string as None)
    parsed_steps = int(num_inference_steps) if num_inference_steps.strip() else None
    parsed_guidance = float(guidance_scale) if guidance_scale.strip() else None
    parsed_seed = int(seed) if seed.strip() else None
    parsed_video_strength = float(input_video_strength) if input_video_strength.strip() else 1.0
    parsed_denoising = float(denoising_strength) if denoising_strength.strip() else 1.0

    # Save uploaded files
    upload_dir = Path(config.upload_dir)
    image_start_path = await _save_upload(image_start, upload_dir)
    image_end_path = await _save_upload(image_end, upload_dir)
    audio_guide_path = await _save_upload(audio_guide, upload_dir)
    input_video_path = await _save_upload(input_video, upload_dir)

    # Validate uploaded files
    if image_start_path and not is_valid_image(image_start_path):
        raise HTTPException(status_code=400, detail="Invalid image file for start image")
    if image_end_path and not is_valid_image(image_end_path):
        raise HTTPException(status_code=400, detail="Invalid image file for end image")
    if audio_guide_path and not is_valid_audio(audio_guide_path):
        raise HTTPException(status_code=400, detail="Invalid audio file")

    # Debug logging
    print(f"\n[API] Received generation request:")
    print(f"  Prompt: {prompt[:80]}...")
    print(f"  image_start: {image_start_path}")
    print(f"  image_end: {image_end_path}")
    print(f"  audio_guide: {audio_guide_path}")

    # Build params
    params = {
        "prompt": prompt,
        "negative_prompt": negative_prompt or "",
        "image_start_path": image_start_path,
        "image_end_path": image_end_path,
        "audio_guide_path": audio_guide_path,
        "input_video_path": input_video_path,
        "width": width,
        "height": height,
        "num_frames": num_frames,
        "fps": fps,
        "num_inference_steps": parsed_steps,
        "guidance_scale": parsed_guidance,
        "seed": parsed_seed,
        "input_video_strength": parsed_video_strength,
        "denoising_strength": parsed_denoising,
        "prefix_frames_count": prefix_frames_count,
        "attention": attention,
        "sliding_window_size": sliding_window_size,
        "sliding_window_overlap": sliding_window_overlap,
    }
    
    # Create task
    task_id = task_queue.create_task(params)
    
    # Start background processing
    asyncio.create_task(_process_task(task_id))
    
    return TaskResponse(
        task_id=task_id,
        status="queued",
        message="Task submitted successfully",
    )


@router.get("/tasks/{task_id}", response_model=TaskStatus)
async def get_task_status(task_id: str):
    """Get task status"""
    task = task_queue.get_task(task_id)
    if not task:
        raise HTTPException(status_code=404, detail="Task not found")
    
    return TaskStatus(
        task_id=task["task_id"],
        status=task["status"],
        progress=task["progress"],
        current_step=task["current_step"],
        total_steps=task["total_steps"],
        error=task["error"],
        created_at=task["created_at"],
        completed_at=task["completed_at"],
        result=task["result"],
    )


@router.get("/tasks/{task_id}/video")
async def download_video(task_id: str):
    """Download generated video"""
    task = task_queue.get_task(task_id)
    if not task:
        raise HTTPException(status_code=404, detail="Task not found")
    
    if task["status"] != "completed":
        raise HTTPException(status_code=400, detail=f"Task not completed (status: {task['status']})")
    
    video_path = task["result"]["video_path"]
    
    if not Path(video_path).exists():
        raise HTTPException(status_code=404, detail="Video file not found")
    
    return FileResponse(
        path=video_path,
        media_type="video/mp4",
        filename=task["result"]["filename"],
    )


@router.delete("/tasks/{task_id}")
async def cancel_task(task_id: str):
    """Cancel a task"""
    task = task_queue.get_task(task_id)
    if not task:
        raise HTTPException(status_code=404, detail="Task not found")
    
    if task["status"] in ("completed", "failed"):
        raise HTTPException(status_code=400, detail="Task already completed")
    
    task_queue.set_cancelled(task_id)
    return {"message": "Task cancelled"}


@router.get("/models", response_model=ModelsResponse)
async def get_models():
    """Get available models"""
    return ModelsResponse(
        current_model=model_manager.model_type,
        available_models=["ltx2_19B", "ltx2_22B"],
    )


async def _process_task(task_id: str):
    """Process a generation task in background"""
    task_queue.set_processing(task_id)

    try:
        params = task_queue.get_task_params(task_id)

        # Progress callback
        def on_progress(current_step, total_steps, progress):
            task_queue.update_progress(task_id, current_step, total_steps, progress)

        # Generate video in a separate thread to avoid blocking the event loop
        result, seed, gen_time = await asyncio.to_thread(
            generate_video,
            model_manager=model_manager,
            prompt=params["prompt"],
            negative_prompt=params["negative_prompt"],
            image_start_path=params["image_start_path"],
            image_end_path=params["image_end_path"],
            audio_guide_path=params["audio_guide_path"],
            input_video_path=params["input_video_path"],
            width=params["width"],
            height=params["height"],
            num_frames=params["num_frames"],
            fps=params["fps"],
            num_inference_steps=params["num_inference_steps"],
            guidance_scale=params["guidance_scale"],
            seed=params["seed"],
            input_video_strength=params.get("input_video_strength", 1.0),
            denoising_strength=params.get("denoising_strength", 1.0),
            prefix_frames_count=params.get("prefix_frames_count", 0),
            attention=params.get("attention"),
            sliding_window_size=params["sliding_window_size"],
            sliding_window_overlap=params["sliding_window_overlap"],
            task_id=task_id,
            progress_callback=on_progress,
        )

        # Save video (also synchronous, run in thread)
        video_path, filename = await asyncio.to_thread(
            save_video_result,
            result=result,
            seed=seed,
            output_dir=config.output_dir,
            prompt=params["prompt"],
            fps=params["fps"],
        )

        # Mark completed
        task_queue.set_completed(task_id, video_path, filename, seed, gen_time)
        print(f"\n✓ Task {task_id} completed: {filename}")

    except Exception as e:
        print(f"\n✗ Task {task_id} failed: {e}")
        import traceback
        traceback.print_exc()
        task_queue.set_failed(task_id, str(e))
    
    finally:
        # Cleanup uploaded files
        task_queue.cleanup_uploads(task_id)
        task_queue.current_task_id = None


async def _save_upload(file: Optional[UploadFile], upload_dir: Path) -> Optional[str]:
    """Save uploaded file and return path"""
    if not file:
        return None

    # Read content first to check if it's actually a file
    content = await file.read()

    # If the file is empty (0 bytes), skip it entirely
    if len(content) == 0:
        print(f"  Skipping empty upload: {file.filename or 'unknown'}")
        return None

    # Handle missing or empty filename
    original_filename = file.filename or ""
    if original_filename.strip():
        filename = f"{uuid.uuid4()}_{original_filename}"
    else:
        # Generate a name based on content type
        content_type = file.content_type or "application/octet-stream"
        ext = content_type.split("/")[-1] if "/" in content_type else "bin"
        # Map common content types to file extensions
        ext_map = {
            "jpeg": "jpg",
            "png": "png",
            "gif": "gif",
            "webp": "webp",
            "mpeg": "mp3",
            "mp4": "mp4",
            "wav": "wav",
            "octet-stream": "bin",
        }
        ext = ext_map.get(ext, ext)
        filename = f"{uuid.uuid4()}.{ext}"

    file_path = upload_dir / filename

    with open(file_path, "wb") as f:
        f.write(content)

    print(f"  Saved upload: {filename} ({len(content)} bytes)")
    return str(file_path)

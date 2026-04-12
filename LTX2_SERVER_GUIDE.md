# LTX-2 FastAPI Server

Complete FastAPI server with web UI for LTX-2 video generation.

## Project Structure

```
ltx2_server/                        # Server package
├── __init__.py                    # Package initialization
├── config.py                      # Server configuration (ServerConfig dataclass)
├── models.py                      # Pydantic request/response models
├── model_manager.py               # LTX-2 model loading/unloading (ModelManager class)
├── generation.py                  # Video generation logic (generate_video, save_video_result)
├── task_queue.py                  # Task queue management (TaskQueue class)
├── routes.py                      # API endpoints (FastAPI router)
├── main.py                        # FastAPI app factory (create_app)
├── README_UI.md                   # UI documentation
├── static/                        # Web UI files
│   ├── index.html                 # Main HTML structure
│   ├── style.css                  # Modern dark theme styles
│   └── app.js                     # Frontend application logic
ltx2_server.py                     # CLI entry point (python ltx2_server.py)
```

### Component Responsibilities

| File | Purpose | Lines |
|------|---------|-------|
| `config.py` | Configuration dataclass with server settings | 30 |
| `models.py` | Pydantic models for API request/response | 36 |
| `model_manager.py` | Load/unload LTX-2 model, manage lifecycle | 145 |
| `generation.py` | Call LTX-2 generate(), save video output | 191 |
| `task_queue.py` | Track task status, progress, results | 147 |
| `routes.py` | FastAPI endpoints (/generate, /tasks, /health) | 260 |
| `main.py` | Create FastAPI app, wire components, serve UI | 139 |
| `ltx2_server.py` | CLI wrapper with argparse + uvicorn | 141 |
| `static/index.html` | Web UI structure with forms and video player | 350 |
| `static/style.css` | Modern dark theme with responsive design | 550 |
| `static/app.js` | Frontend logic, API calls, progress polling | 290 |

## Features

✅ Async task-based video generation  
✅ Progress tracking with real-time updates  
✅ File upload support (images, audio)  
✅ Automatic video saving and download  
✅ Swagger API documentation  
✅ Health check endpoint  
✅ Single model instance management  

## Quick Start

### 1. Install Dependencies

```bash
pip install -r requirements_fastapi.txt
```

### 2. Start the Server

```bash
# Default settings (ltx2_22B, port 8000)
python ltx2_server.py

# Custom settings
python ltx2_server.py --model_type ltx2_19B --port 8001 --profile 1

# Development mode with auto-reload
python ltx2_server.py --reload
```

### 3. Access API Documentation

Once the server is running, visit:
- **Swagger UI**: http://localhost:8000/docs
- **ReDoc**: http://localhost:8000/redoc

## API Endpoints

### Health Check

```bash
GET /api/v1/health
```

**Response:**
```json
{
  "status": "healthy",
  "model_loaded": true,
  "model_type": "ltx2_22B",
  "gpu_available": true,
  "queue_size": 0
}
```

### Submit Video Generation

```bash
POST /api/v1/generate
Content-Type: multipart/form-data
```

**Parameters:**
- `prompt` (required): Text prompt for video generation
- `negative_prompt`: Negative prompt (default: "")
- `image_start`: Starting image file (optional)
- `image_end`: Ending image file (optional)
- `audio_guide`: Audio file for audio-guided generation (optional)
- `width`: Video width, must be divisible by 64 (default: 768)
- `height`: Video height, must be divisible by 64 (default: 512)
- `num_frames`: Number of frames (default: 121)
- `fps`: Frames per second (default: 24.0)
- `num_inference_steps`: Denoising steps (default: auto)
- `guidance_scale`: CFG scale (default: 4.0)
- `seed`: Random seed (default: random)
- `profile`: Memory profile -1 to 4 (default: -1)
- `vram_safety_coefficient`: VRAM safety 0.0-0.95 (default: 0.85)
- `attention`: Attention mode: sdpa/sage/sage2/flash (default: auto)

**Example (curl):**
```bash
curl -X POST http://localhost:8000/api/v1/generate \
  -F "prompt=A beautiful sunset over the ocean" \
  -F "width=768" \
  -F "height=512" \
  -F "num_frames=121" \
  -F "seed=42"
```

**Response:**
```json
{
  "task_id": "abc123-def456-...",
  "status": "queued",
  "message": "Task submitted successfully"
}
```

### Check Task Status

```bash
GET /api/v1/tasks/{task_id}
```

**Response (processing):**
```json
{
  "task_id": "abc123-def456-...",
  "status": "processing",
  "progress": 45.0,
  "current_step": 18,
  "total_steps": 40,
  "error": null,
  "created_at": "2026-04-12T10:30:00",
  "completed_at": null,
  "result": null
}
```

**Response (completed):**
```json
{
  "task_id": "abc123-def456-...",
  "status": "completed",
  "progress": 100.0,
  "current_step": 40,
  "total_steps": 40,
  "error": null,
  "created_at": "2026-04-12T10:30:00",
  "completed_at": "2026-04-12T10:35:00",
  "result": {
    "video_path": "output/ltx2_20260412_103500_a_beautiful_sunset_over_the_ocean_seed42.mp4",
    "video_url": "/api/v1/tasks/abc123-def456-.../video",
    "filename": "ltx2_20260412_103500_a_beautiful_sunset_over_the_ocean_seed42.mp4",
    "seed": 42,
    "generation_time": 125.5
  }
}
```

### Download Video

```bash
GET /api/v1/tasks/{task_id}/video
```

Returns the video file as a download response.

### Cancel Task

```bash
DELETE /api/v1/tasks/{task_id}
```

**Note:** Actual cancellation is not yet implemented. This marks the task as cancelled but doesn't interrupt generation.

### List Models

```bash
GET /api/v1/models
```

**Response:**
```json
{
  "current_model": "ltx2_22B",
  "available_models": ["ltx2_19B", "ltx2_22B"]
}
```

## Python Client Example

```python
import requests
import time

# Submit task
response = requests.post(
    "http://localhost:8000/api/v1/generate",
    data={
        "prompt": "A cat playing with a ball of yarn",
        "width": 768,
        "height": 512,
        "num_frames": 121,
        "seed": 12345,
    }
)
task_id = response.json()["task_id"]

# Poll for completion
while True:
    status_resp = requests.get(f"http://localhost:8000/api/v1/tasks/{task_id}")
    status = status_resp.json()
    
    print(f"Status: {status['status']}, Progress: {status.get('progress', 0)}%")
    
    if status["status"] == "completed":
        # Download video
        video_resp = requests.get(f"http://localhost:8000/api/v1/tasks/{task_id}/video")
        with open("output.mp4", "wb") as f:
            f.write(video_resp.content)
        print("Video downloaded to output.mp4")
        break
    elif status["status"] == "failed":
        print(f"Task failed: {status['error']}")
        break
    
    time.sleep(5)
```

## Command-Line Arguments

| Argument | Description | Default |
|----------|-------------|---------|
| `--model_type` | Model variant: ltx2_19B or ltx2_22B | ltx2_22B |
| `--transformer_path` | Path to transformer safetensors file | Auto-detect |
| `--gemma_path` | Path to Gemma text encoder file | Auto-detect |
| `--profile` | Memory profile (-1 to 4) | -1 |
| `--vram_safety_coefficient` | VRAM safety coefficient (0.0-0.95) | 0.85 |
| `--output_dir` | Output directory for videos | output |
| `--host` | Server host | 0.0.0.0 |
| `--port` | Server port | 8000 |
| `--reload` | Enable auto-reload (development) | False |

## Configuration

### Using Config Files

You can configure the server via environment variables or the `ServerConfig` class in `config.py`:

```python
from ltx2_server.config import ServerConfig

config = ServerConfig(
    model_type="ltx2_22B",
    transformer_path="/path/to/ltx2_22B.safetensors",  # Optional
    gemma_path="/path/to/gemma-3-12b-it.safetensors",  # Optional
    profile=1,
    vram_safety_coefficient=0.85,
    output_dir="output",
)
```

### Auto-Detection vs Explicit Paths

**Auto-Detection (Default):**
- Leave `transformer_path` and `gemma_path` empty
- Server will auto-detect from model definitions
- Works when models are in standard locations

**Explicit Paths:**
- Set paths to override auto-detection
- Useful for custom model locations or fine-tuned models
- Both CLI and programmatic configuration supported

**CLI Example:**
```bash
python ltx2_server.py \
  --model_type ltx2_22B \
  --transformer_path /models/custom/ltx2_22B.safetensors \
  --gemma_path /models/custom/gemma.safetensors \
  --port 8000
```

## Limitations

- **Single task at a time**: Only one video generation can run at a time due to VRAM constraints
- **No model hot-swapping**: Must restart server to change model
- **Task cancellation not implemented**: Cannot interrupt running generations
- **No authentication**: Anyone can submit tasks (add middleware if needed)

## Architecture

```
┌─────────────────────────────────────────────────────────┐
│                      Client Request                      │
│              POST /api/v1/generate                       │
└────────────────────┬────────────────────────────────────┘
                     │
                     ▼
┌─────────────────────────────────────────────────────────┐
│                   routes.py                              │
│  ┌───────────────────────────────────────────────────┐  │
│  │ generate_video_endpoint()                         │  │
│  │  - Validate request                              │  │
│  │  - Save uploaded files                            │  │
│  │  - Create task in TaskQueue                       │  │
│  │  - Start background _process_task()               │  │
│  └───────────────────────────────────────────────────┘  │
└────────────────────┬────────────────────────────────────┘
                     │
                     ▼
┌─────────────────────────────────────────────────────────┐
│                  task_queue.py                           │
│  ┌───────────────────────────────────────────────────┐  │
│  │ TaskQueue.create_task()                           │  │
│  │  - Generate UUID                                  │  │
│  │  - Store params with "queued" status              │  │
│  └───────────────────────────────────────────────────┘  │
└────────────────────┬────────────────────────────────────┘
                     │
                     ▼
┌─────────────────────────────────────────────────────────┐
│                  routes._process_task()                  │
│  ┌───────────────────────────────────────────────────┐  │
│  │ 1. Set status to "processing"                     │  │
│  │ 2. Call generation.generate_video()               │  │
│  │ 3. Call generation.save_video_result()            │  │
│  │ 4. Set status to "completed"                      │  │
│  │ 5. Cleanup uploaded files                         │  │
│  └───────────────────────────────────────────────────┘  │
└────────────────────┬────────────────────────────────────┘
                     │
          ┌──────────┴──────────┐
          ▼                     ▼
┌──────────────────┐   ┌──────────────────┐
│  generation.py   │   │  model_manager.py│
│ ┌──────────────┐ │   │ ┌──────────────┐│
│ │generate_video│ │   │ │  ModelManager││
│ │  - Load imgs │ │   │ │  - LTX2 inst ││
│ │  - Setup attn│ │   │ │  - offload   ││
│ │  - Call model│ │   │ │  - profile() ││
│ │  - Progress  │ │   │ └──────────────┘│
│ └──────────────┘ │   └────────┬─────────┘
│ ┌──────────────┐ │            │
│ │save_video_rlt│ │            │
│ │  - save mp4  │ │            │
│ └──────────────┘ │            │
└──────────────────┘            │
                                │
                     ┌──────────┘
                     ▼
          ┌──────────────────────┐
          │   LTX-2 Model GPU   │
          │   Video Generation   │
          └──────────────────────┘
```

### Data Flow

## Troubleshooting

### Out of Memory Error
- Use higher memory profile: `--profile 3`
- Reduce resolution: `width=512 height=512`
- Reduce frames: `num_frames=49`
- Increase VRAM safety: `--vram_safety_coefficient 0.90`

### Model Not Found
Ensure model files are in the standard locations as defined by the main project. Use the same model setup as `run_ltx2.py`.

### Port Already in Use
Change port: `python ltx2_server.py --port 8001`

## Next Steps

Potential enhancements:
- [ ] Task cancellation (interrupt generation)
- [ ] Multiple model support (switch without restart)
- [ ] Authentication middleware
- [ ] Rate limiting
- [ ] Batch processing
- [ ] WebSocket for real-time progress
- [ ] Task queue with priority
- [ ] Video upscaling endpoint

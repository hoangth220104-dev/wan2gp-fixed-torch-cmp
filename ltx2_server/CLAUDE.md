# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

`ltx2_server/` is a FastAPI server for LTX-2 text/image-to-video generation. It wraps the core `models.ltx2` and `shared` modules from the parent directory with an async REST API, task queue, and web UI.

## Running the Server

```bash
# Install dependencies
pip install -r requirements_fastapi.txt

# Start server (default: ltx2_22B, port 8000)
python ltx2_server.py

# Development with auto-reload
python ltx2_server.py --reload

# Custom model/settings
python ltx2_server.py --model_type ltx2_19B --port 8001 --profile 1
```

## Architecture

**Request flow:**
1. `routes.py` → `generate_video_endpoint()` validates request, saves uploads, creates task in `TaskQueue`
2. `_process_task()` runs in background via `asyncio.create_task()` — runs generation off the event loop via `asyncio.to_thread()`
3. `generation.py` → `generate_video()` calls `model_manager.generate()` which delegates to `ltx2_instance.generate()`
4. `model_manager.py` → `ModelManager` wraps LTX-2 model loading/unloading via `_load_ltx2_model()`, uses `mmgp.offload.profile()` for memory management
5. On completion, `save_video_result()` muxes audio into video if present

**Single-task constraint:** Only one video generation runs at a time. `task_queue.has_active_task` blocks new submissions while a task is queued or processing.

**Key modules:**
- `model_manager.py`: LTX-2 lifecycle — load/unload, references `models.ltx2.ltx2.LTX2` and `mmgp.offload`
- `lora_manager.py`: LoRA weight management, attaches to `transformer` via `_attach_lora_preprocessor()`
- `task_queue.py`: In-memory task registry (UUID-keyed dict), progress tracking, upload cleanup
- `generation.py`: Parameter normalization, input loading (images/video/audio), video saving with audio muxing
- `routes.py`: FastAPI endpoints, global `model_manager`/`task_queue` injected via `init_globals()`

**Global singletons:** `routes.py` uses module-level globals (`model_manager`, `task_queue`, `config`) set via `init_globals()` at startup — not ideal for testing but matches the current design.

**Dependencies:** This package imports from parent-level packages (`models.ltx2`, `shared`, `mmgp`). The project root must be in `sys.path` at runtime, which `main.py` handles.

## API Endpoints

| Method | Path | Purpose |
|--------|------|---------|
| POST | `/api/v1/generate` | Submit generation task |
| GET | `/api/v1/tasks/{task_id}` | Poll task status |
| GET | `/api/v1/tasks/{task_id}/video` | Download completed video |
| DELETE | `/api/v1/tasks/{task_id}` | Cancel task (marks cancelled, no actual interruption) |
| GET | `/api/v1/health` | Health check |
| GET | `/api/v1/models` | List available models |
| POST | `/api/v1/loras/load` | Load LoRA |
| GET | `/api/v1/loras` | List loaded LoRAs |
| POST | `/api/v1/loras/multiplier` | Update LoRA strength |
| POST | `/api/v1/loras/{path}/activate` | Activate LoRA |
| POST | `/api/v1/loras/{path}/deactivate` | Deactivate LoRA |
| DELETE | `/api/v1/loras/{path}` | Unload LoRA |

## Limitations

- Task cancellation is标记-only; the running generation is not actually interrupted
- No authentication on any endpoint
- VRAM safety coefficient (default 0.1) and profile settings affect GPU memory usage

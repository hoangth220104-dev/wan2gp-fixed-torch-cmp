"""FastAPI application entry point"""

import os
import sys
import json
from pathlib import Path
from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse

# Add project root to path
project_root = Path(__file__).parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

# Set environment variables
os.environ["GRADIO_LANG"] = "en"
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

import warnings
warnings.filterwarnings('ignore')

from ltx2_server.config import ServerConfig
from ltx2_server.model_manager import ModelManager
from ltx2_server.task_queue import TaskQueue
from ltx2_server.routes import router, init_globals


# Global config instance
_config: ServerConfig = None


def get_config() -> ServerConfig:
    """Get config from environment or create default"""
    global _config
    
    if _config is None:
        # Try to load from environment
        config_json = os.environ.get("LTX2_SERVER_CONFIG")
        if config_json:
            _config = ServerConfig(**json.loads(config_json))
        else:
            _config = ServerConfig()
    
    return _config


def set_config(config: ServerConfig):
    """Set config and update environment"""
    global _config
    _config = config
    os.environ["LTX2_SERVER_CONFIG"] = json.dumps({
        "model_type": config.model_type,
        "transformer_path": config.transformer_path,
        "gemma_path": config.gemma_path,
        "lora_dir": config.lora_dir,
        "profile": config.profile,
        "vram_safety_coefficient": config.vram_safety_coefficient,
        "output_dir": config.output_dir,
        "host": config.host,
        "port": config.port,
        "upload_dir": config.upload_dir,
    })


def create_app() -> FastAPI:
    """Create and configure FastAPI application (factory for uvicorn)"""
    config = get_config()
    return _create_app_with_config(config)


def create_app_with_config(config: ServerConfig) -> FastAPI:
    """Create app with explicit config (for CLI usage)"""
    set_config(config)
    return _create_app_with_config(config)


def _create_app_with_config(config: ServerConfig) -> FastAPI:
    """Internal: create app with config"""
    
    model_manager = ModelManager()
    task_queue = TaskQueue()
    
    @asynccontextmanager
    async def lifespan(app: FastAPI):
        """Startup and shutdown"""
        print("=" * 60)
        print("LTX-2 FastAPI Server Starting")
        print(f"  Model: {config.model_type}")
        print(f"  Profile: {config.profile}")
        print(f"  VRAM Safety: {config.vram_safety_coefficient}")
        print("=" * 60)
        
        try:
            model_manager.load(config)
        except Exception as e:
            print(f"\n✗ Failed to load model: {e}")
            raise
        
        yield
        
        print("\nShutting down...")
        model_manager.unload()
        print("Cleanup complete")
    
    # Initialize route globals
    init_globals(model_manager, task_queue, config)
    
    # Create app
    app = FastAPI(
        title="LTX-2 Video Generation API",
        description="FastAPI server for LTX-2 text/image-to-video generation",
        version="1.0.0",
        lifespan=lifespan,
    )
    
    # Register API routes
    app.include_router(router)
    
    # Mount static files
    static_path = Path(__file__).parent / "static"
    if static_path.exists():
        app.mount("/static", StaticFiles(directory=str(static_path)), name="static")
        
        # Serve index.html at root
        @app.get("/")
        async def serve_index():
            return FileResponse(str(static_path / "index.html"))
        
        # Catch-all route for SPA (Single Page Application)
        @app.get("/{full_path:path}")
        async def serve_spa(full_path: str):
            file_path = static_path / full_path
            if file_path.exists() and file_path.is_file():
                return FileResponse(str(file_path))
            return FileResponse(str(static_path / "index.html"))
    
    return app

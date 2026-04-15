"""Server configuration management"""

import os
from pathlib import Path
from dataclasses import dataclass, field


@dataclass
class ServerConfig:
    """Server configuration"""
    # Model configuration
    model_type: str = "ltx2_22B"
    transformer_path: str = ""  # Path to transformer safetensors file(s)
    gemma_path: str = ""  # Path to Gemma text encoder safetensors file
    lora_dir: str = ""  # Directory containing LoRA weights (auto-loaded on startup)
    profile: int = 1
    vram_safety_coefficient: float = 0.1

    # Server configuration
    output_dir: str = "output"
    host: str = "0.0.0.0"
    port: int = 8000
    reload: bool = False
    upload_dir: str = "uploads"
    
    def __post_init__(self):
        """Create directories if they don't exist"""
        Path(self.output_dir).mkdir(parents=True, exist_ok=True)
        Path(self.upload_dir).mkdir(parents=True, exist_ok=True)
    
    @property
    def num_inference_steps_default(self) -> int:
        """Default inference steps based on model type"""
        return 40 if "19B" in self.model_type else 30
